"""Dataset utilities for SO-100 teleop imitation learning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


COLOR_PERMUTATIONS: tuple[tuple[int, int, int], ...] = tuple(permutations(range(3)))


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def _parse_key_spec(spec: str) -> tuple[str, slice]:
    """Parse a key spec like ``"state_cube[:3]"`` into (key, col_slice).

    Supports slicing notations: ``key``, ``key[:N]``, ``key[M:]``, ``key[M:N]``.
    Returns the array name and a column slice to apply on axis=1.
    """
    if "[" not in spec:
        return spec, slice(None)
    name, rest = spec.split("[", 1)
    rest = rest.rstrip("]")
    parts = rest.split(":")
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return name, slice(start, stop)
    raise ValueError(
        f"Invalid key spec: {spec!r}  (expected 'key', 'key[:N]', 'key[M:]', or 'key[M:N]')"
    )


def _slice_width(width: int, col_slice: slice) -> int:
    """Return the resulting width after applying ``col_slice``."""
    return len(range(width)[col_slice])


@dataclass(frozen=True)
class ColorPermutationSpec:
    """Resolved state slices needed for multicube color-permutation augmentation."""

    cube_slices: tuple[slice, slice, slice]
    goal_slice: slice


def resolve_state_feature_slices(
    state_keys: list[str],
    key_widths: dict[str, int],
) -> dict[str, list[slice]]:
    """Resolve concatenated state-vector slices for the configured ``state_keys``."""
    offset = 0
    resolved: dict[str, list[slice]] = {}

    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        if name not in key_widths:
            raise ValueError(f"State key {name!r} was not found in the zarr data arrays.")

        width = _slice_width(key_widths[name], col_slice)
        current = slice(offset, offset + width)
        resolved.setdefault(name, []).append(current)
        offset += width

    return resolved


def build_color_permutation_spec(
    state_keys: list[str],
    key_widths: dict[str, int],
) -> ColorPermutationSpec:
    """Validate multicube augmentation inputs and resolve their state slices."""
    required_keys = (
        "original_pos_cube_red",
        "original_pos_cube_green",
        "original_pos_cube_blue",
        "state_goal",
    )
    resolved = resolve_state_feature_slices(state_keys, key_widths)

    missing = [name for name in required_keys if name not in resolved]
    if missing:
        raise ValueError(
            "Multicube color augmentation requires state_keys to include "
            f"{list(required_keys)}. Missing: {missing}."
        )

    duplicated = [name for name in required_keys if len(resolved[name]) != 1]
    if duplicated:
        raise ValueError(
            "Multicube color augmentation expects exactly one state-key entry for "
            f"each required input. Ambiguous entries: {duplicated}."
        )

    cube_slices = (
        resolved["original_pos_cube_red"][0],
        resolved["original_pos_cube_green"][0],
        resolved["original_pos_cube_blue"][0],
    )
    cube_widths = [sl.stop - sl.start for sl in cube_slices]
    if len(set(cube_widths)) != 1:
        raise ValueError(
            "Multicube color augmentation requires the red/green/blue cube state "
            f"slices to have identical widths after slicing. Got: {cube_widths}."
        )

    goal_slice = resolved["state_goal"][0]
    goal_width = goal_slice.stop - goal_slice.start
    if goal_width != 3:
        raise ValueError(
            "Multicube color augmentation requires state_goal to remain width 3. "
            f"Got width {goal_width}."
        )

    return ColorPermutationSpec(cube_slices=cube_slices, goal_slice=goal_slice)


def permute_color_conditioned_state(
    state: torch.Tensor,
    permutation: tuple[int, int, int],
    spec: ColorPermutationSpec,
    *,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a copy of ``state`` with multicube color channels permuted."""
    if state.ndim != 1:
        raise ValueError(f"Expected a flat state vector, got shape {tuple(state.shape)}.")
    if (state_mean is None) != (state_std is None):
        raise ValueError("state_mean and state_std must either both be set or both be None.")

    out = state.clone()
    cube_values = [state[sl].clone() for sl in spec.cube_slices]
    goal_values = state[spec.goal_slice].clone()

    if state_mean is None or state_std is None:
        for dest_idx, src_idx in enumerate(permutation):
            out[spec.cube_slices[dest_idx]] = cube_values[src_idx]
        out[spec.goal_slice] = goal_values[list(permutation)]
        return out

    cube_means = [state_mean[sl] for sl in spec.cube_slices]
    cube_stds = [state_std[sl] for sl in spec.cube_slices]
    for dest_idx, src_idx in enumerate(permutation):
        raw = cube_values[src_idx] * cube_stds[src_idx] + cube_means[src_idx]
        out[spec.cube_slices[dest_idx]] = (raw - cube_means[dest_idx]) / cube_stds[dest_idx]

    goal_mean = state_mean[spec.goal_slice]
    goal_std = state_std[spec.goal_slice]
    raw_goal = goal_values * goal_std + goal_mean
    out[spec.goal_slice] = (raw_goal[list(permutation)] - goal_mean) / goal_std
    return out


def load_zarr(
    zarr_path: Path,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states, actions, and episode_ends from a processed .zarr.

    Args:
        zarr_path: Path to the processed .zarr store.
        state_keys: List of data array key specs to concatenate as the state.
            Each entry can include an optional column slice, e.g.
            ``["state_ee_xyz", "state_cube[:3]"]``.
            If ``None``, falls back to the ``state_key`` attribute in the zarr metadata.
        action_keys: List of data array key specs to concatenate as the action.
            Supports column slicing, e.g. ``["action_ee_xyz", "action_gripper"]``.
            If ``None``, falls back to the ``action_key`` attribute in the zarr metadata.

    Returns:
        states, actions, episode_ends
    """
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]

    # ── states: concatenate one or more arrays ────────────────────────
    if state_keys is None:
        sk = root.attrs.get("state_key", "state")
        state_keys = [sk]

    state_parts: list[np.ndarray] = []
    for spec in state_keys:
        name, col_slice = _parse_key_spec(spec)
        arr = np.asarray(data[name][:], dtype=np.float32)
        state_parts.append(arr[:, col_slice] if col_slice != slice(None) else arr)
    states = (
        np.concatenate(state_parts, axis=1) if len(state_parts) > 1 else state_parts[0]
    )

    # ── actions: concatenate one or more arrays ───────────────────────
    if action_keys is None:
        ak = root.attrs.get("action_key", "action")
        action_keys = [ak]

    action_parts: list[np.ndarray] = []
    for spec in action_keys:
        act_name, act_slice = _parse_key_spec(spec)
        arr = np.asarray(data[act_name][:], dtype=np.float32)
        action_parts.append(arr[:, act_slice] if act_slice != slice(None) else arr)
    actions = (
        np.concatenate(action_parts, axis=1)
        if len(action_parts) > 1
        else action_parts[0]
    )

    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

    return states, actions, episode_ends


def load_and_merge_zarrs(
    zarr_paths: list[Path],
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate data from multiple processed .zarr stores.

    Each zarr store is loaded independently via :func:`load_zarr` and the
    results are concatenated.  Episode-end indices are shifted so they remain
    globally correct after concatenation.

    Returns the same ``(states, actions, episode_ends)`` tuple
    as :func:`load_zarr`.
    """
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_ep_ends: list[np.ndarray] = []
    offset = 0

    for zp in zarr_paths:
        states, actions, ep_ends = load_zarr(
            zp, state_keys=state_keys, action_keys=action_keys,
        )
        all_states.append(states)
        all_actions.append(actions)
        all_ep_ends.append(ep_ends + offset)
        offset += states.shape[0]

    merged_states = np.concatenate(all_states, axis=0)
    merged_actions = np.concatenate(all_actions, axis=0)
    merged_ep_ends = np.concatenate(all_ep_ends, axis=0)

    return merged_states, merged_actions, merged_ep_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """Return flat indices where a full action chunk of length ``chunk_size`` fits.

    For each episode [start, end) we keep indices start … (end - chunk_size).
    """
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class SO100ChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs with a sliding window of size H.

    Each sample consists of:
        state:        (state_dim,)             - state at timestep t
        action_chunk: (chunk_size, action_dim) - actions [t, t+1, …, t+H-1]
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        state = self.states[t]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        state_t = torch.from_numpy(state).float()
        action_t = torch.from_numpy(action_chunk).float()

        return state_t, action_t


class ColorPermutationAugmentedDataset(Dataset):
    """Virtual training-set expansion over all 6 multicube color permutations."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        spec: ColorPermutationSpec,
        normalizer: Normalizer | None,
    ) -> None:
        self.dataset = dataset
        self.spec = spec
        self.num_permutations = len(COLOR_PERMUTATIONS)
        self.state_mean = (
            torch.from_numpy(normalizer.state_mean).float()
            if normalizer is not None
            else None
        )
        self.state_std = (
            torch.from_numpy(normalizer.state_std).float()
            if normalizer is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.dataset) * self.num_permutations

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_idx = idx // self.num_permutations
        perm_idx = idx % self.num_permutations
        state_t, action_t = self.dataset[base_idx]
        state_aug = permute_color_conditioned_state(
            state_t,
            COLOR_PERMUTATIONS[perm_idx],
            self.spec,
            state_mean=self.state_mean,
            state_std=self.state_std,
        )
        return state_aug, action_t
