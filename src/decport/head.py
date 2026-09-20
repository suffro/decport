"""Backbone-agnostic decision head."""

from __future__ import annotations

from torch import Tensor, nn


class DecisionHead(nn.Module):
    """Score a candidate represented in the shared DecPort space."""

    def __init__(self, shared_size: int = 256) -> None:
        super().__init__()
        if shared_size <= 0:
            raise ValueError("shared_size must be positive")
        self.shared_size = shared_size
        self.network = nn.Sequential(
            nn.LayerNorm(shared_size),
            nn.Linear(shared_size, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.shape[-1] != self.shared_size:
            raise ValueError(
                f"expected shared width {self.shared_size}, got {latent.shape[-1]}"
            )
        parameter = next(self.parameters())
        latent = latent.to(device=parameter.device, dtype=parameter.dtype)
        return self.network(latent).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
