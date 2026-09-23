"""Lightweight backbone-specific mapping into DecPort's latent space."""

from __future__ import annotations

from torch import Tensor, nn


class BackboneAdapter(nn.Module):
    """Map one backbone's native hidden width into the shared space.

    ``hidden_size`` is the intermediate width and defaults to ``shared_size``, the original v0.1
    architecture. A narrower intermediate width keeps the adapter lightweight when the frozen
    decision core reads a wide representation.
    """

    def __init__(
        self,
        input_size: int,
        shared_size: int = 256,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = shared_size if hidden_size is None else hidden_size
        if input_size <= 0 or shared_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size, shared_size, and hidden_size must be positive")
        self.input_size = input_size
        self.shared_size = shared_size
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, shared_size),
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
