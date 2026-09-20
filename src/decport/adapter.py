"""Lightweight backbone-specific mapping into DecPort's latent space."""

from __future__ import annotations

from torch import Tensor, nn


class BackboneAdapter(nn.Module):
    """Map one backbone's native hidden width into the shared space."""

    def __init__(self, input_size: int, shared_size: int = 256) -> None:
        super().__init__()
        if input_size <= 0 or shared_size <= 0:
            raise ValueError("input_size and shared_size must be positive")
        self.input_size = input_size
        self.shared_size = shared_size
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, shared_size),
            nn.GELU(),
            nn.Linear(shared_size, shared_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.input_size:
            raise ValueError(
                f"expected hidden width {self.input_size}, got {hidden.shape[-1]}"
            )
        parameter = next(self.parameters())
        hidden = hidden.to(device=parameter.device, dtype=parameter.dtype)
        return self.network(hidden)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
