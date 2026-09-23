"""Frozen external decision cores with typed Choice, Noul, and Score semantics.

A decision core is the frozen, backbone-independent part of a Jev-like decision system. It scores
one representation per candidate, assembles those scalars into typed logits, and applies its own
saved calibration. DecPort ports a core to another backbone by training only an adapter that
produces representations the unchanged core can read.

Typed semantics follow the Jev wire format used by Open-Jev and JevBench:

- ``choice`` and ``score`` score one representation per candidate; logits follow answer-key order.
- ``noul`` scores one representation and uses the logits ``[0, s]`` ordered ``(false, true)``,
  so ``P(true) = sigmoid(s / temperature)``. It is not a two-candidate Choice.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from decport.schema import DecisionExample

DECISION_KINDS = ("choice", "noul", "score")
NOUL_KEYS = ("false", "true")


@dataclass(frozen=True, slots=True)
class TypedDecision:
    """One typed Jev-wire question over a state.

    ``options`` are the ordered answer keys and therefore the typed-logit order: Choice candidate
    names, Noul ``("false", "true")``, or Score level indices ``"0"`` to ``"n-1"``. ``answer`` is
    one of those keys on evaluation records and ``None`` on transfer inputs.
    """

    state: object
    kind: str
    instructions: object
    criteria: object = None
    answer: str | None = None
    dataset: str | None = None
    task_family: str | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, (str, dict, list)):
            raise TypeError("state must be text, a JSON object, or an array")
        json.dumps(self.state, allow_nan=False)
        if self.kind not in DECISION_KINDS:
            raise ValueError(f"kind must be one of {DECISION_KINDS}")
        if isinstance(self.instructions, str):
            if not self.instructions.strip():
                raise ValueError("instructions must be non-empty")
        elif not isinstance(self.instructions, (dict, list)):
            raise TypeError("instructions must be text, a JSON object, or an array")
        if self.kind == "choice":
            if not isinstance(self.criteria, Mapping) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("Choice requires between 1 and 255 named candidates")
            if any(not isinstance(key, str) or not key.strip() for key in self.criteria):
                raise ValueError("Choice candidate names must be non-empty strings")
            object.__setattr__(self, "criteria", dict(self.criteria))
        elif self.kind == "score":
            if not isinstance(self.criteria, (list, tuple)) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("Score requires 2 to 10 ordered levels")
            object.__setattr__(self, "criteria", tuple(self.criteria))
        elif self.criteria is not None:
            if not isinstance(self.criteria, Mapping) or set(self.criteria) != {"true", "false"}:
                raise ValueError("Noul criteria must describe exactly true and false")
            object.__setattr__(self, "criteria", dict(self.criteria))
        json.dumps(self.question(), allow_nan=False)
        if self.answer is not None and self.answer not in self.options:
            raise ValueError("answer must be one of the decision's answer keys")
        for name in ("dataset", "task_family", "decision_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")

    @property
    def options(self) -> tuple[str, ...]:
        if self.kind == "choice":
            return tuple(self.criteria)  # type: ignore[arg-type]
        if self.kind == "score":
            return tuple(str(index) for index in range(len(self.criteria)))  # type: ignore[arg-type]
        return NOUL_KEYS

    @property
    def decision_type(self) -> str:
        return self.kind

    @property
    def candidate_count(self) -> int:
        """Number of representations the core scores for this decision."""

        return 1 if self.kind == "noul" else len(self.options)

    @property
    def answer_index(self) -> int:
        if self.answer is None:
            raise ValueError("an unlabeled decision has no answer index")
        return self.options.index(self.answer)

    def question(self) -> dict[str, object]:
        """Return the typed wire question, which never contains the answer."""

        question: dict[str, object] = {"type": self.kind, "instructions": self.instructions}
        if self.kind == "score":
            question["criteria"] = list(self.criteria)  # type: ignore[arg-type]
        elif self.criteria is not None:
            question["criteria"] = dict(self.criteria)  # type: ignore[arg-type]
        return question

    def without_answer(self) -> TypedDecision:
        return replace(self, answer=None)


def typed_from_example(example: DecisionExample) -> TypedDecision:
    """Convert a canonical DecPort record into its explicit typed decision.

    Boolean records become Noul questions with ``yes -> true``. Score options must already be in
    ordinal order; their level indices become the answer keys.
    """

    if example.decision_type == "choice":
        return TypedDecision(
            state=example.state,
            kind="choice",
            instructions=example.question,
            criteria={option: None for option in example.options},
            answer=example.answer,
            dataset=example.dataset,
            task_family=example.task_family,
        )
    if example.decision_type == "boolean":
        if set(example.options) != {"yes", "no"}:
            raise ValueError("a Boolean record must have exactly the options yes and no")
        answer = None
        if example.answer is not None:
            answer = "true" if example.answer == "yes" else "false"
        return TypedDecision(
            state=example.state,
            kind="noul",
            instructions=example.question,
            answer=answer,
            dataset=example.dataset,
            task_family=example.task_family,
        )
    if example.decision_type == "score":
        answer = None if example.answer is None else str(example.answer_index)
        return TypedDecision(
            state=example.state,
            kind="score",
            instructions=example.question,
            criteria=tuple(example.options),
            answer=answer,
            dataset=example.dataset,
            task_family=example.task_family,
        )
    raise ValueError(f"unsupported decision type: {example.decision_type!r}")


def typed_logits(kind: str, candidate_scores: Tensor) -> Tensor:
    """Assemble ``(batch, candidates)`` scalars into typed logits in answer-key order.

    Shared by frozen cores and by source-side core training, so both use identical semantics.
    """

    if candidate_scores.ndim != 2:
        raise ValueError("candidate scores must have shape (batch, candidates)")
    if kind == "noul":
        if candidate_scores.shape[1] != 1:
            raise ValueError("a Noul decision scores exactly one candidate")
        return torch.cat([torch.zeros_like(candidate_scores), candidate_scores], dim=1)
    if kind in ("choice", "score"):
        return candidate_scores
    raise ValueError(f"kind must be one of {DECISION_KINDS}")


def mlp_decision_network(
    input_size: int, hidden_sizes: tuple[int, int] = (512, 128)
) -> nn.Sequential:
    """``LayerNorm → Linear → GELU → Linear → GELU → Linear(·, 1)``: one scalar per candidate."""

    if input_size <= 0 or len(hidden_sizes) != 2 or min(hidden_sizes) <= 0:
        raise ValueError("input_size and both hidden sizes must be positive")
    first, second = hidden_sizes
    return nn.Sequential(
        nn.LayerNorm(input_size),
        nn.Linear(input_size, first),
        nn.GELU(),
        nn.Linear(first, second),
        nn.GELU(),
        nn.Linear(second, 1),
    )


class FrozenDecisionCore(nn.Module, ABC):
    """Frozen per-candidate scorer with typed assembly and its own saved calibration."""

    def __init__(self, input_size: int, temperature: float) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.input_size = input_size
        self.temperature = float(temperature)

    @property
    def shared_size(self) -> int:
        """Representation width expected from an adapter; used by DecPort composition."""

        return self.input_size

    @abstractmethod
    def score_candidates(self, representation: Tensor) -> Tensor:
        """Return one float32 scalar per representation."""

    def forward(self, representation: Tensor) -> Tensor:
        if representation.shape[-1] != self.input_size:
            raise ValueError(
                f"expected core input width {self.input_size}, got {representation.shape[-1]}"
            )
        return self.score_candidates(representation)

    def typed_logits(self, kind: str, candidate_scores: Tensor) -> Tensor:
        """Assemble ``(batch, candidates)`` scalars into typed logits in answer-key order."""

        return typed_logits(kind, candidate_scores)

    def calibrated_logits(self, kind: str, candidate_scores: Tensor) -> Tensor:
        """Typed logits whose softmax is the core's calibrated probability distribution."""

        return self.typed_logits(kind, candidate_scores) / self.temperature

    def train(self, mode: bool = True) -> FrozenDecisionCore:
        """A frozen core never enters training mode."""

        super().train(False)
        return self

    def assert_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("the decision core must remain frozen")

    def state_sha256(self) -> str:
        """Digest of every core tensor plus the calibration temperature."""

        digest = hashlib.sha256(repr(self.temperature).encode("utf-8"))
        for key, value in sorted(self.state_dict().items()):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class LinearDecisionCore(FrozenDecisionCore):
    """Frozen float32 scalar linear readout, the form of Open-Jev's trained decision head."""

    def __init__(
        self,
        input_size: int,
        temperature: float,
        *,
        weight: Tensor,
        bias: Tensor,
    ) -> None:
        super().__init__(input_size, temperature)
        if tuple(weight.shape) != (1, input_size) or tuple(bias.shape) != (1,):
            raise ValueError("readout weight/bias must have shapes (1, input_size) and (1,)")
        if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError("readout parameters must be finite")
        self.readout = nn.Linear(input_size, 1, dtype=torch.float32)
        with torch.no_grad():
            self.readout.weight.copy_(weight.float())
            self.readout.bias.copy_(bias.float())
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Tensor], temperature: float
    ) -> LinearDecisionCore:
        """Build from an ``nn.Linear(hidden, 1)`` state dict with ``weight`` and ``bias``."""

        if set(state) != {"weight", "bias"}:
            raise ValueError("a linear decision head needs exactly weight and bias tensors")
        weight = state["weight"]
        if weight.ndim != 2 or weight.shape[0] != 1:
            raise ValueError("decision head weight must have shape (1, hidden)")
        return cls(int(weight.shape[1]), temperature, weight=weight, bias=state["bias"])

    @classmethod
    def random_control(cls, reference: LinearDecisionCore, *, seed: int) -> LinearDecisionCore:
        """Frozen random readout direction with the reference norm, bias, and calibration."""

        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn((1, reference.input_size), generator=generator)
        norm = reference.readout.weight.detach().cpu().norm()
        weight = direction / direction.norm() * norm
        return cls(
            reference.input_size,
            reference.temperature,
            weight=weight,
            bias=reference.readout.bias.detach().cpu().clone(),
        )

    def score_candidates(self, representation: Tensor) -> Tensor:
        weight = self.readout.weight
        representation = representation.to(device=weight.device, dtype=torch.float32)
        return self.readout(representation).squeeze(-1)


class MLPDecisionCore(FrozenDecisionCore):
    """Frozen float32 nonlinear core learned once on the source side (decision 0007)."""

    def __init__(
        self,
        input_size: int,
        temperature: float,
        *,
        state: Mapping[str, Tensor],
        hidden_sizes: tuple[int, int] = (512, 128),
    ) -> None:
        super().__init__(input_size, temperature)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.network = mlp_decision_network(input_size, self.hidden_sizes)
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("core parameters must be finite")
        self.network.load_state_dict({key: value.float() for key, value in state.items()})
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def random_control(
        cls,
        reference: MLPDecisionCore,
        *,
        seed: int,
        representations: Tensor,
    ) -> tuple[MLPDecisionCore, dict[str, list[dict[str, float]]]]:
        """Same architecture and temperature, random weights, layer-wise matched activation scale.

        Every tensor starts as a Gaussian draw rescaled to the reference tensor's Frobenius norm.
        Then, in order, each Linear layer gets one scalar gain and one scalar shift so that the mean
        and standard deviation of its pre-activations over ``representations`` (source inputs, no
        labels) equal the reference core's. No learned direction is copied. Per-tensor norm matching
        alone is not enough: a trained core's output scale comes largely from layer alignment, and
        the core's input LayerNorm stops an adapter from compensating.

        Returns the core and the matched per-layer statistics.
        """

        generator = torch.Generator().manual_seed(seed)
        state = {}
        for key, value in sorted(reference.network.state_dict().items()):
            draw = torch.randn(value.shape, generator=generator)
            state[key] = draw / draw.norm() * value.detach().cpu().float().norm()
        device = next(reference.network.parameters()).device
        network = mlp_decision_network(reference.input_size, reference.hidden_sizes).to(device)
        network.load_state_dict(state)
        learned_input = random_input = representations.to(device=device, dtype=torch.float32)
        statistics: list[dict[str, float]] = []
        with torch.no_grad():
            for learned_layer, random_layer in zip(reference.network, network, strict=True):
                learned_output = learned_layer(learned_input)
                if isinstance(random_layer, nn.Linear):
                    drawn = random_layer(random_input).double()
                    target = learned_output.double()
                    gain = float(target.std() / drawn.std())
                    shift = float(target.mean() - gain * drawn.mean())
                    random_layer.weight.mul_(gain)
                    random_layer.bias.mul_(gain).add_(shift)
                    statistics.append(
                        {"gain": gain, "shift": shift, "mean": float(target.mean()),
                         "std": float(target.std())}
                    )
                learned_input, random_input = learned_output, random_layer(random_input)
        core = cls(
            reference.input_size,
            reference.temperature,
            state={key: value.detach().cpu() for key, value in network.state_dict().items()},
            hidden_sizes=reference.hidden_sizes,
        )
        return core, {"linear_layers": statistics}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def score_candidates(self, representation: Tensor) -> Tensor:
        parameter = next(self.network.parameters())
        representation = representation.to(device=parameter.device, dtype=torch.float32)
        return self.network(representation).squeeze(-1)


class ScalarIdentityCore(FrozenDecisionCore):
    """Parameter-free core for modules that already emit one scalar per candidate.

    It keeps typed assembly and one calibration constant, so a target-specific module is trained and
    scored through exactly the same typed path as an adapter plus a shared core.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__(1, temperature)
        self.eval()

    def score_candidates(self, representation: Tensor) -> Tensor:
        return representation.float().squeeze(-1)
