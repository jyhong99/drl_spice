from __future__ import annotations

"""
env.types
=========

Core type aliases and dataclasses used by the environment layer.

Shared simulation contracts (circuit/knob/domain/status/result) are sourced
from `ngspice.types` to avoid duplicate definitions across packages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Union, Protocol

from ngspice.types import (
    CircuitType,
    KnobValue,
    PathLike,
    CircuitNames,
    AnalysisKnobSpec,
    AnalysisKnobSet,
    ParamSpec,
    ParamKeys,
    ParamDomain,
    SimStatus,
    SimResult,
)

from .patterns import NOISE_KEY, SPARAM_KEY


# =============================================================================
# Type aliases
# =============================================================================

ActionMode = Literal["delta"]

Scale = Literal["log", "linear"]
Ineq = Literal["lte", "gte"]

AnalysisKind = Literal["sp", "noise", "linearity", "fft"]

DoneReason = Literal["running", "time_limit", "non_convergent"]
ResetMode = Literal["random", "continue_last"]

EnvInfo = Dict[str, Any]
EvalOut = Dict[str, Any]
SimInfo = Dict[str, Any]
ResetOptions = Dict[str, Any]

TemplatePath = Union[str, Path]
TemplateMap = Dict[str, Path]
KernelEnv = Dict[str, str]


# =============================================================================
# Performance spec (SPEC) + references (env-specific)
# =============================================================================

@dataclass(frozen=True)
class SpecItem:
    """
    One immutable performance-constraint item.

    Parameters
    ----------
    metric : str
        Metric name (e.g., ``"S21_dB"``).
    ineq : {"lte", "gte"}
        Inequality direction. ``"lte"`` means the metric should be less than or
        equal to ``value``; ``"gte"`` means greater than or equal.
    value : float
        Target threshold value for the metric.
    """

    metric: str
    ineq: Ineq
    value: float


@dataclass(frozen=True)
class PerfSpec:
    """
    Immutable performance specification with normalization references.

    Parameters
    ----------
    circuit_type : CircuitType
        Circuit identifier this specification is defined for.
    items : tuple[SpecItem, ...]
        Ordered tuple of constraint entries.
    references : dict[str, float]
        Per-metric baseline values used by environment-side normalization.

    Notes
    -----
    Every metric present in ``items`` must exist in ``references``. This is
    enforced during initialization.
    """

    circuit_type: CircuitType
    items: Tuple[SpecItem, ...]
    references: Dict[str, float]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Convert spec items into a serializable dictionary.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping:
            ``metric -> {"type": ineq, "value": threshold}``.

        Notes
        -----
        This shape is used by reward/constraint modules that expect dict-based
        spec access while preserving immutable dataclass storage internally.
        """
        return {it.metric: {"type": it.ineq, "value": float(it.value)} for it in self.items}

    def target_vector(self, order: Tuple[str, ...]) -> Tuple[float, ...]:
        """
        Build an ordered target-value vector.

        Parameters
        ----------
        order : tuple[str, ...]
            Desired metric order.

        Returns
        -------
        tuple[float, ...]
            Target threshold values in the exact order requested.

        Raises
        ------
        KeyError
            If ``order`` references a metric not present in ``items``.
        """
        d = {it.metric: float(it.value) for it in self.items}
        return tuple(d[k] for k in order)

    def reference_vector(self, order: Tuple[str, ...]) -> Tuple[float, ...]:
        """
        Build an ordered reference-value vector.

        Parameters
        ----------
        order : tuple[str, ...]
            Desired metric order.

        Returns
        -------
        tuple[float, ...]
            Reference values aligned with ``order``.

        Raises
        ------
        KeyError
            If ``order`` references a metric missing from ``references``.
        """
        return tuple(float(self.references[k]) for k in order)

    def __post_init__(self) -> None:
        """
        Validate inequality tokens and reference completeness.

        Raises
        ------
        ValueError
            If an item has an unsupported inequality token or if a metric in
            ``items`` has no corresponding entry in ``references``.
        """
        for it in self.items:
            if it.ineq not in ("lte", "gte"):
                raise ValueError(f"invalid ineq: {it.ineq} for metric={it.metric}")
        for it in self.items:
            if it.metric not in self.references:
                raise ValueError(f"[{self.circuit_type}] references missing key: {it.metric}")


# =============================================================================
# Simulator protocol + environment config
# =============================================================================

class SimulatorProtocol(Protocol):
    """
    Protocol for simulator backends used by the environment.
    """

    def simulate(
        self,
        *,
        design_variables_config: Dict[str, float],
        analyses: Tuple[str, ...],
        enable_linearity: bool,
    ) -> SimResult:
        """
        Run circuit simulation for a given design point.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Physical design variable configuration (decoded from environment
            state).
        analyses : tuple[str, ...]
            Analysis kinds to execute (e.g., ``"sp"``, ``"noise"``).
        enable_linearity : bool
            Whether linearity-related analysis should be included.

        Returns
        -------
        SimResult
            Simulation result payload containing status, performance vector,
            optional auxiliary metrics, and run metadata.
        """
        ...


@dataclass(frozen=True)
class EnvConfig:
    """
    High-level immutable environment configuration.

    Parameters
    ----------
    analyses : tuple[str, ...], default=("sp", "noise")
        Analysis kinds requested from the simulator each environment evaluation.
    require_k : bool, default=False
        If ``True``, missing ``K_min`` in simulator aux data is treated as
        non-convergence.
    enable_linearity : bool, default=False
        Enables linearity-aware metric/spec behavior across the environment.
    """

    analyses: Tuple[str, ...] = (SPARAM_KEY, NOISE_KEY)
    require_k: bool = False
    enable_linearity: bool = False


__all__ = [
    "ActionMode",
    "Scale",
    "Ineq",
    "AnalysisKind",
    "DoneReason",
    "ResetMode",
    "PathLike",
    "EnvInfo",
    "EvalOut",
    "SimInfo",
    "ResetOptions",
    "TemplatePath",
    "TemplateMap",
    "KernelEnv",
    "SimulatorProtocol",
    "SpecItem",
    "PerfSpec",
    "EnvConfig",
    "CircuitType",
    "CircuitNames",
    "AnalysisKnobSpec",
    "AnalysisKnobSet",
    "KnobValue",
    "ParamSpec",
    "ParamKeys",
    "ParamDomain",
    "SimStatus",
    "SimResult",
]
