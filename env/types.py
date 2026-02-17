from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

from .patterns import NOISE_KEY, SPARAM_KEY, STORE_OFF
from .utils.common_utils import _require_positive_int
from .utils.logger_utils import _require_store_arrays


# =============================================================================
# Type aliases
# =============================================================================

ActionMode = Literal["delta"]

Scale = Literal["log", "linear"]
Ineq = Literal["lte", "gte"]

CircuitType = Literal["CS", "CGCS"]
AnalysisKind = Literal["sp", "noise", "linearity", "fft"]

DoneReason = Literal["running", "time_limit", "non_convergent"]
ResetMode = Literal["random", "continue_last"]

PathLike = Union[str, Path]

KnobValue = Union[float, int, str]
KnobDict = Dict[str, KnobValue]

EnvInfo = Dict[str, Any]
EvalOut = Dict[str, Any]
SimInfo = Dict[str, Any]
ResetOptions = Dict[str, Any]

TemplatePath = Union[str, Path]
TemplateMap = Dict[str, Path]
KernelEnv = Dict[str, str]


# =============================================================================
# Circuit naming contracts
# =============================================================================

@dataclass(frozen=True)
class CircuitNames:
    """
    Named groups for circuit symbols used by the environment.

    This is a small schema object describing what the environment considers
    "design variables" and "devices" for a given circuit topology.

    Parameters
    ----------
    designvar_names : tuple[str, ...]
        Ordered names of design variables (the optimization/search variables).
        These names typically map onto template `.param` entries or other
        controllable quantities.
    device_names : tuple[str, ...]
        Ordered names of devices (e.g., transistor instance names) that may be
        referenced by post-processing or auxiliary measurements.

    Notes
    -----
    - This object is not a parameter domain by itself; it is a naming contract.
    - Ordering matters when you align vectors with names.
    """
    designvar_names: Tuple[str, ...]
    device_names: Tuple[str, ...]


# =============================================================================
# 0) Analysis knobs (outside-tunable non-design variables)
# =============================================================================

@dataclass(frozen=True)
class AnalysisKnobSpec:
    """
    Specification for one analysis knob (a simulation control parameter).

    Analysis knobs are **not** part of the design-variable vector `x`.
    Instead, they represent tunable simulation-side parameters that can be used
    to patch a template block such as:

        ** analysis_knobs_start
        .param <name> = <value>
        ** analysis_knobs_end

    Parameters
    ----------
    name : str
        Knob name used in the netlist/template (e.g., "temp", "vdd", "q_factor").
    default : object
        Default knob value. Kept flexible to support:
        - float / int
        - spice-friendly strings (e.g., "1p", "10n", "tt")
    description : str, default=""
        Human-readable description of the knob.

    Notes
    -----
    - Values are eventually serialized into a netlist line, so keep them numeric
      or Spice-compatible.
    - Knobs are usually used for "scenario control" or "analysis control"
      rather than sizing/optimization variables.
    """
    name: str
    default: object
    description: str = ""


@dataclass(frozen=True)
class AnalysisKnobSet:
    """
    Bundle of analysis knobs for a given circuit type.

    Parameters
    ----------
    circuit_type : CircuitType
        Circuit topology identifier (e.g., "CS", "CGCS").
    specs : tuple[AnalysisKnobSpec, ...]
        Ordered knob specifications for this circuit type.

    Notes
    -----
    - You can extend this later to be per-analysis-kind if you want by adding
      an `analysis_kind` discriminator field.
    """
    circuit_type: CircuitType
    specs: Tuple[AnalysisKnobSpec, ...]

    def defaults(self) -> Dict[str, object]:
        """
        Build a dict of knob defaults keyed by knob name.

        Returns
        -------
        dict[str, object]
            Mapping `{knob_name: default_value}`.
        """
        return {s.name: s.default for s in self.specs}


# =============================================================================
# 1) Design-parameter domain (BOUND)
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    """
    Specification for one design parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    p_min : float
        Lower bound in physical units.
    p_max : float
        Upper bound in physical units.
    scale : Scale, default="log"
        Interpolation scale when decoding normalized parameters:
        - "linear": linear interpolation between bounds
        - "log": log10-space interpolation between positive bounds
    round_sig_k : int or None, default=4
        If not None, round to `k` significant digits after interpolation.
        Useful for producing human-friendly parameters and stabilizing netlists.
    step : float or None, default=None
        Optional snapping step. If provided, values can be quantized onto a grid
        after interpolation (and optional significant-digit rounding).

    Notes
    -----
    - If `scale="log"`, you must ensure bounds are strictly positive.
    - The environment typically normalizes design variables to a compact range,
      then decodes into these physical bounds.
    """
    name: str
    p_min: float
    p_max: float
    scale: Scale = "log"
    round_sig_k: Optional[int] = 4
    step: Optional[float] = None


@dataclass(frozen=True)
class ParamKeys:
    """
    Ordered key container for design parameters.

    Parameters
    ----------
    keys : tuple[str, ...]
        Ordered parameter names.

    Notes
    -----
    This wrapper exists to make it explicit when ordering matters and to provide
    small convenience methods.
    """
    keys: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.keys)

    def as_tuple(self) -> Tuple[str, ...]:
        """
        Return the keys as a plain tuple.

        Returns
        -------
        tuple[str, ...]
            Parameter keys.
        """
        return self.keys


@dataclass(frozen=True)
class ParamDomain:
    """
    Design-parameter domain for a circuit type.

    This defines the ordering, bounds, scaling rules, and fixed values for a
    circuit topology.

    Parameters
    ----------
    circuit_type : CircuitType
        Circuit topology identifier.
    keys : ParamKeys
        Ordered parameter keys.
    specs : tuple[ParamSpec, ...]
        Parameter specs in the same order as `keys`.
    fixed_values_default : dict[str, float]
        Default fixed parameter values that are not optimized.

    Raises
    ------
    ValueError
        If the length or ordering of `keys` and `specs` is inconsistent.

    Notes
    -----
    The domain assumes a one-to-one correspondence between:
    - `keys.keys[i]`
    - `specs[i].name`
    """
    circuit_type: CircuitType
    keys: ParamKeys
    specs: Tuple[ParamSpec, ...]
    fixed_values_default: Dict[str, float]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.specs):
            raise ValueError(
                f"[{self.circuit_type}] len(keys) != len(specs): {len(self.keys)} vs {len(self.specs)}"
            )

        spec_names = tuple(s.name for s in self.specs)
        if self.keys.as_tuple() != spec_names:
            raise ValueError(
                f"[{self.circuit_type}] parameter order mismatch!\n"
                f"keys:  {self.keys.as_tuple()}\n"
                f"specs: {spec_names}"
            )


# =============================================================================
# 2) Performance spec (SPEC) + references
# =============================================================================

@dataclass(frozen=True)
class SpecItem:
    """
    One performance constraint item.

    Parameters
    ----------
    metric : str
        Metric name (e.g., "S21_dB", "NF_dB", "PD_mW").
    ineq : Ineq
        Inequality direction:
        - "lte": metric <= value
        - "gte": metric >= value
    value : float
        Constraint threshold value.
    """
    metric: str
    ineq: Ineq
    value: float


@dataclass(frozen=True)
class PerfSpec:
    """
    Performance specification and references for normalization.

    Parameters
    ----------
    circuit_type : CircuitType
        Circuit topology identifier.
    items : tuple[SpecItem, ...]
        Specification items defining constraints.
    references : dict[str, float]
        Reference values used for normalization/scoring. All metrics in `items`
        must exist in this dict.

    Methods
    -------
    as_dict()
        Return a dict representation of items suitable for config export.
    target_vector(order)
        Build target vector aligned to `order`.
    reference_vector(order)
        Build reference vector aligned to `order`.

    Raises
    ------
    ValueError
        If an invalid inequality is used or references are missing for an item.

    Notes
    -----
    You typically normalize with something like:
        denom = spec - ref
        f = (o - ref) / denom
    Hence `references` must be well-defined for each metric.
    """
    circuit_type: CircuitType
    items: Tuple[SpecItem, ...]
    references: Dict[str, float]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Convert items into a dict mapping.

        Returns
        -------
        dict[str, dict[str, float]]
            e.g. {"S21_dB": {"type": "gte", "value": 20.0}, ...}
        """
        return {it.metric: {"type": it.ineq, "value": float(it.value)} for it in self.items}

    def target_vector(self, order: Tuple[str, ...]) -> Tuple[float, ...]:
        """
        Build a target vector aligned to a given metric order.

        Parameters
        ----------
        order : tuple[str, ...]
            Metric key order.

        Returns
        -------
        tuple[float, ...]
            Target values in the given order.
        """
        d = {it.metric: float(it.value) for it in self.items}
        return tuple(d[k] for k in order)

    def reference_vector(self, order: Tuple[str, ...]) -> Tuple[float, ...]:
        """
        Build a reference vector aligned to a given metric order.

        Parameters
        ----------
        order : tuple[str, ...]
            Metric key order.

        Returns
        -------
        tuple[float, ...]
            Reference values in the given order.
        """
        return tuple(float(self.references[k]) for k in order)

    def __post_init__(self) -> None:
        for it in self.items:
            if it.ineq not in ("lte", "gte"):
                raise ValueError(f"invalid ineq: {it.ineq} for metric={it.metric}")

        # Contract: every metric in items must exist in references.
        for it in self.items:
            if it.metric not in self.references:
                raise ValueError(f"[{self.circuit_type}] references missing key: {it.metric}")


# =============================================================================
# Logger settings
# =============================================================================

@dataclass
class LoggerSettings:
    """
    Configuration options for the CSV-based run logger.

    Parameters
    ----------
    store_arrays : str, default=STORE_OFF
        Storage mode for array-like values (obs/action) in step events.
        Typically:
        - STORE_OFF: do not store arrays
        - STORE_INLINE: store arrays as JSON in CSV cells (large files)
    flush_every_steps : int, default=1
        Flush interval in number of steps. Smaller values reduce data loss risk
        on crashes but increase I/O overhead.

    Notes
    -----
    - Validation uses `_require_store_arrays` and `_require_positive_int`.
    - Flushing may also trigger fsync depending on writer implementation.
    """
    store_arrays: str = STORE_OFF
    flush_every_steps: int = 1

    def __post_init__(self) -> None:
        self.store_arrays = _require_store_arrays(self.store_arrays)
        self.flush_every_steps = _require_positive_int("flush_every_steps", self.flush_every_steps)


# =============================================================================
# Simulation result contracts
# =============================================================================

class SimStatus(str, Enum):
    """
    Simulation outcome status code.

    Attributes
    ----------
    OK
        Simulation completed successfully and outputs are valid.
    TIMEOUT
        Simulation exceeded timeout.
    NGSPICE_FAIL
        ngspice process failed (non-zero exit or fatal error).
    NO_OUTPUT
        ngspice finished but expected output files are missing.
    PARSE_FAIL
        Output files exist but parsing failed.
    NAN
        Outputs contain NaN/Inf where finite values were expected.
    VALIDATION
        Output failed validation checks (schema, bounds, etc.).
    UNKNOWN
        Any other failure mode not categorized above.
    """
    OK = "ok"
    TIMEOUT = "timeout"
    NGSPICE_FAIL = "ngspice_fail"
    NO_OUTPUT = "no_output"
    PARSE_FAIL = "parse_fail"
    NAN = "nan"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SimResult:
    """
    Result bundle returned from a simulation run.

    Parameters
    ----------
    performances : np.ndarray or None
        Performance vector (e.g., [S11_dB, S21_dB, S22_dB, NF_dB, PD_mW, ...]).
        None if simulation did not produce usable outputs.
    aux : dict[str, float]
        Auxiliary scalar metrics (e.g., K_min, bandwidth, IIP3, etc.).
    status : SimStatus
        Simulation status code.
    detail : str
        Human-readable details (error messages, reason strings).
    run_dir : str
        Path to the run directory where artifacts/logs were produced.

    Notes
    -----
    - `performances` may be None even if `aux` is non-empty (partial outputs).
    - `run_dir` is kept as string for JSON/log friendliness.
    """
    performances: Optional[np.ndarray]
    aux: Dict[str, float]
    status: SimStatus
    detail: str
    run_dir: str


# =============================================================================
# Environment config (high-level knobs)
# =============================================================================

@dataclass(frozen=True)
class EnvConfig:
    """
    High-level environment configuration.

    Parameters
    ----------
    analyses : tuple[str, ...], default=(SPARAM_KEY, NOISE_KEY)
        Enabled analysis kinds for each step/evaluation.
    require_k : bool, default=False
        If True, enforce stability constraints based on K-factor (or similar).
    enable_linearity : bool, default=False
        If True, include linearity-related metrics/analysis stages.
    log_enabled : bool, default=True
        Enable run logging.
    log_store_physical_params : bool, default=True
        If True, log decoded physical design parameters (not only normalized x).
    log_store_eval : bool, default=True
        If True, log evaluation outputs (constraints/objective/reward).
    log_store_obs : bool, default=False
        If True, store observation vectors in step logs (can be large).

    Notes
    -----
    - `analyses` should use keys consistent with your template map, e.g.,
      SPARAM_KEY / NOISE_KEY / LINEARITY_KEY.
    """
    analyses: Tuple[str, ...] = (SPARAM_KEY, NOISE_KEY)
    require_k: bool = False
    enable_linearity: bool = False
    log_enabled: bool = True
    log_store_physical_params: bool = True
    log_store_eval: bool = True
    log_store_obs: bool = False
