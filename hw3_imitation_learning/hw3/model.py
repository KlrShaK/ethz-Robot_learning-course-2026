"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int, 
        action_dim: int, 
        chunk_size: int, 
        d_model: int = 128, 
        depth: int = 3
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.d_model = d_model
        self.depth = depth    

        layers: list[nn.Module] = [
                nn.Linear(state_dim, d_model),
                nn.ReLU(),
            ]

        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                ]
            )

        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        out = self.net(state)
        return out.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def compute_loss(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
