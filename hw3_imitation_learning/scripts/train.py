"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    ColorPermutationAugmentedDataset,
    Normalizer,
    SO100ChunkDataset,
    build_color_permutation_spec,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split

EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2 # try 0.2
GRAD_CLIP_NORM = 1.0
DEFAULT_D_MODEL = 256 # 128
DEFAULT_DEPTH = 5  # 3
EARLY_STOPPING_PATIENCE = 20


def load_checkpoint_metadata(ckpt_path: Path, device: torch.device) -> dict:
    """Load checkpoint metadata for model construction or initialization."""
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def resolve_model_hparams(
    args: argparse.Namespace,
    checkpoint_meta: dict | None,
) -> tuple[int, int]:
    """Choose model width/depth, preferring explicit args over checkpoint metadata."""
    if args.d_model is not None:
        d_model = args.d_model
    elif checkpoint_meta is not None and "d_model" in checkpoint_meta:
        d_model = int(checkpoint_meta["d_model"])
    else:
        d_model = DEFAULT_D_MODEL

    if args.depth is not None:
        depth = args.depth
    elif checkpoint_meta is not None and "depth" in checkpoint_meta:
        depth = int(checkpoint_meta["depth"])
    else:
        depth = DEFAULT_DEPTH

    return d_model, depth


def initialize_model_from_checkpoint(
    model: BasePolicy,
    checkpoint: dict,
    *,
    load_mode: str,
) -> None:
    """Initialize model weights from a checkpoint."""
    state_dict = checkpoint["model_state_dict"]
    if load_mode == "exact":
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError(
                "Exact checkpoint loading failed. If you are finetuning across "
                "different state dimensions, use "
                "--checkpoint-load-mode compatible."
            ) from exc
        print("Loaded model weights from checkpoint (exact).")
        return

    if load_mode != "compatible":
        raise ValueError(f"Unknown checkpoint load mode: {load_mode}")

    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None or target.shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value

    if not compatible:
        raise ValueError(
            "No compatible weights could be loaded from the checkpoint. "
            "Check d_model/depth/chunk_size/action space compatibility."
        )

    model.load_state_dict(compatible, strict=False)
    print(
        "Loaded compatible checkpoint weights: "
        f"{len(compatible)}/{len(model_state)} tensors matched."
    )
    if skipped:
        print(f"  Skipped mismatched tensors: {', '.join(skipped)}")

    left_fresh = sorted(set(model_state) - set(compatible))
    if left_fresh:
        print(f"  Left at fresh init: {', '.join(left_fresh)}")


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device, non_blocking=True)
        action_chunks = action_chunks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device, non_blocking=True)
        action_chunks = action_chunks.to(device, non_blocking=True)

        loss = model.compute_loss(states, action_chunks)
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional processed .zarr stores to merge, e.g. DAgger data.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR}).")
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Hidden width. Defaults to the checkpoint value when --checkpoint is set, otherwise the script default.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of residual MLP blocks. Defaults to the checkpoint value when --checkpoint is set, otherwise the script default.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience in epochs without validation improvement (default: {EARLY_STOPPING_PATIENCE}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint file to initialize model weights from.",
    )
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=["exact", "compatible"],
        default="exact",
        help="How to load --checkpoint weights: 'exact' for full resume, 'compatible' for partial finetuning when shapes differ.",
    )
    parser.add_argument(
        "--multicube-color-augment",
        action="store_true",
        help="Enable train-only virtual 6x color-permutation augmentation for multicube state inputs.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.d_model is not None and args.d_model < 32:
        raise ValueError("--d-model must be >= 32")
    if args.depth is not None and args.depth < 1:
        raise ValueError("--depth must be >= 1")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    if not 0.0 <= VAL_SPLIT < 1.0:
        raise ValueError("--val-split must be in [0, 1)")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_meta: dict | None = None
    if args.checkpoint is not None:
        checkpoint_meta = load_checkpoint_metadata(args.checkpoint, device)
        print(
            "Checkpoint init source: "
            f"{args.checkpoint} "
            f"(policy={checkpoint_meta.get('policy_type', '?')}, "
            f"d_model={checkpoint_meta.get('d_model', '?')}, "
            f"depth={checkpoint_meta.get('depth', '?')})"
        )

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )

    ref_zarr = zarr_lib.open_group(str(zarr_paths[0]), mode="r")
    data_group = ref_zarr["data"]
    state_keys = args.state_keys or [str(ref_zarr.attrs.get("state_key", "state"))]
    action_keys = args.action_keys or [str(ref_zarr.attrs.get("action_key", "action"))]
    state_key_widths = {
        name: int(data_group[name].shape[1])
        for name in {spec.split("[", 1)[0] for spec in state_keys}
    }

    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")
    print(f"  state_keys={state_keys}")
    print(f"  action_keys={action_keys}")

    if len(dataset) < 2:
        raise ValueError("Dataset is too small to split into train/val sets.")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  train_samples={n_train}, val_samples={n_val}")

    data_augmentation: dict[str, str] = {}
    if args.multicube_color_augment:
        color_spec = build_color_permutation_spec(state_keys, state_key_widths)
        train_ds = ColorPermutationAugmentedDataset(
            train_ds,
            spec=color_spec,
            normalizer=normalizer,
        )
        data_augmentation["multicube_color_permutation"] = "virtual_6x"
        print("  multicube_color_augment=enabled (virtual x6 train-set expansion)")
        print(f"  effective_train_samples={len(train_ds)}")
    else:
        print("  multicube_color_augment=disabled")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    # ── model ─────────────────────────────────────────────────────────
    d_model, depth = resolve_model_hparams(args, checkpoint_meta)
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=d_model,
        depth=depth,
    ).to(device)
    print(f"Model config: d_model={d_model}, depth={depth}")

    if checkpoint_meta is not None:
        initialize_model_from_checkpoint(
            model,
            checkpoint_meta,
            load_mode=args.checkpoint_load_mode,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")
    epochs_without_improvement = 0

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    best_save_name = f'best_model_{action_space}_{args.policy}.pt'
    last_save_name = f'last_model_{action_space}_{args.policy}.pt'

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        best_save_name = f'best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt'
        last_save_name = f'last_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt'

    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    best_save_path = ckpt_dir / best_save_name
    last_save_path = ckpt_dir / last_save_name
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

    grad_clip_norm = GRAD_CLIP_NORM if GRAD_CLIP_NORM > 0 else None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": {
                "state_mean": normalizer.state_mean,
                "state_std": normalizer.state_std,
                "action_mean": normalizer.action_mean,
                "action_std": normalizer.action_std,
            },
            "chunk_size": args.chunk_size,
            "d_model": d_model,
            "depth": depth,
            "policy_type": args.policy,
            "state_keys": state_keys,
            "action_keys": action_keys,
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "data_augmentation": data_augmentation,
            "init_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "checkpoint_load_mode": args.checkpoint_load_mode if args.checkpoint is not None else None,
        }
        torch.save(checkpoint, last_save_path)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, best_save_path)
            tag = " saved"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | lr {lr:.2e}"
            f" | patience {epochs_without_improvement}/{args.patience}{tag}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: no validation improvement for "
                f"{args.patience} consecutive epoch(s)."
            )
            break

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_save_path}")
    print(f"Last checkpoint: {last_save_path}")


if __name__ == "__main__":
    main()
