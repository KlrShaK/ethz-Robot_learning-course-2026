"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


DEFAULT_D_MODEL = 128
DEFAULT_DEPTH = 5
DROPOUT_P = 0.5


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


def _init_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual block for low-dimensional state inputs."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(DROPOUT_P)
        self.fc2 = nn.Linear(d_model, d_model)

        _init_linear(self.fc1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class ObstaclePolicy(BasePolicy):
    """Residual MLP policy for single-cube obstacle imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = DEFAULT_D_MODEL,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        if d_model < 32:
            raise ValueError('d_model must be >= 32')
        if depth < 1:
            raise ValueError('depth must be >= 1')

        self.d_model = d_model
        self.depth = depth

        self.input_proj = nn.Linear(state_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([ResidualMLPBlock(d_model) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, chunk_size * action_dim)

        _init_linear(self.input_proj)
        _init_linear(self.output_proj)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        squeeze_batch = state.ndim == 1
        if squeeze_batch:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f'Expected state with shape (B, D), got {tuple(state.shape)}')

        x = self.input_proj(state)
        x = nn.functional.silu(self.input_norm(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        out = self.output_proj(x)
        out = out.reshape(state.shape[0], self.chunk_size, self.action_dim)
        return out.squeeze(0) if squeeze_batch else out

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self(state)


class MultiTaskPolicy(ObstaclePolicy):
    """Goal-conditioned policy for the multicube scene.

    The goal information is assumed to be part of the input state vector,
    so the same residual MLP architecture works well here too.
    """


PolicyType: TypeAlias = Literal['obstacle', 'multitask']


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = DEFAULT_D_MODEL,
    depth: int = DEFAULT_DEPTH,
) -> BasePolicy:
    if policy_type == 'obstacle':
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == 'multitask':
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    raise ValueError(f'Unknown policy type: {policy_type}')
