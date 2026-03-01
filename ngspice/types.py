from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union


# =============================================================================
# Types
# =============================================================================

PathLike = Union[str, Path]
KV = Mapping[str, Any]
KnobValue = Union[float, int, str]
CircuitType = Literal["CS", "CGCS"]


# =============================================================================
# Simulation-facing circuit contracts
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
    """
    circuit_type: CircuitType
    specs: Tuple[AnalysisKnobSpec, ...]

    def defaults(self) -> Dict[str, object]:
        """
        Build a dict of knob defaults keyed by knob name.
        """
        return {s.name: s.default for s in self.specs}


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
    scale : Literal["log", "linear"], default="log"
        Interpolation scale when decoding normalized parameters.
    round_sig_k : int or None, default=4
        If not None, round to `k` significant digits after interpolation.
    step : float or None, default=None
        Optional snapping step after interpolation.
    """
    name: str
    p_min: float
    p_max: float
    scale: Literal["log", "linear"] = "log"
    round_sig_k: Optional[int] = 4
    step: Optional[float] = None


@dataclass(frozen=True)
class ParamKeys:
    """
    Ordered key container for design parameters.
    """
    keys: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.keys)

    def as_tuple(self) -> Tuple[str, ...]:
        return self.keys


@dataclass(frozen=True)
class ParamDomain:
    """
    Design-parameter domain for a circuit type.
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
# Lightweight structs (data contracts)
# =============================================================================

@dataclass(frozen=True)
class Marker:
    """
    Delimiter markers used to locate a block of interest inside a text stream.

    Typically used when parsing ngspice logs/RAW-ascii outputs that contain a
    well-defined section bracketed by two sentinel strings (e.g., "Variables:"
    ... "Values:" or similar).

    Parameters
    ----------
    start:
        Start sentinel string. Parsing logic usually searches for the first
        occurrence of a line that "matches" or "starts with" this value.
    end:
        End sentinel string. Parsing logic usually stops when this sentinel is
        reached (or just before it), depending on the format.

    Notes
    -----
    - This is a pure data container; it does not perform matching by itself.
    - You may want to treat markers as case-sensitive unless your parser
      explicitly normalizes case/whitespace.
    """
    start: str
    end: str

    def __post_init__(self) -> None:
        """
        Validate marker boundaries.

        Raises
        ------
        ValueError
            If either marker boundary is empty.
        """
        if not self.start or not self.end:
            raise ValueError("Marker start/end must be non-empty strings")


@dataclass(frozen=True)
class CircuitNames:
    """
    Canonical naming contract for a circuit instance.

    This groups two related name lists:
    1) design variables (continuous/optimizable parameters)
    2) device names (M1, M2, R1, etc. or hierarchy-qualified identifiers)

    Parameters
    ----------
    designvar_names:
        Ordered tuple of design variable names (e.g., ("R_d2", "L_g", ...)).
        The order is significant when you serialize vectors or map actions to
        parameters.
    device_names:
        Ordered tuple of device instance names (e.g., ("M1", "M2", "R1", ...)).
        Often used for operating point extraction or device-level metrics.

    Notes
    -----
    - Prefer tuples (immutable) for deterministic ordering and hashability.
    - Keep naming consistent with your netlist/templates to avoid schema drift.
    """
    designvar_names: tuple[str, ...]
    device_names: tuple[str, ...]


@dataclass(frozen=True)
class RawAscii:
    """
    Parsed header/body skeleton for an ngspice ASCII RAW-like dump.

    Many ngspice configurations can emit an ASCII representation of a RAW file
    (or "raw-like" text) that includes:
      - number of variables (n_var)
      - number of points (n_pts)
      - variable names (var_names)
      - value lines (value_lines) containing numeric samples

    This struct stores the minimally processed artifacts; conversion to a
    DataFrame/ndarray is usually performed by a dedicated reader.

    Parameters
    ----------
    n_var:
        Number of variables (columns) declared in the header.
    n_pts:
        Number of points/samples (rows) declared in the header.
    var_names:
        Variable names as parsed from the header, in the order they appear.
        List is used because upstream parsing often accumulates incrementally.
        (Immutability is still ensured by `frozen=True`.)
    value_lines:
        Raw text lines for values. These may need additional normalization
        (e.g., whitespace/csv splitting, continuation lines, etc.).

    Notes
    -----
    - This object is intentionally low-level to preserve the original text for
      robust downstream parsing and error reporting.
    - Consider validating (n_var == len(var_names)) and (n_pts == inferred rows)
      in the next parsing stage where you have full context.
    """
    n_var: int
    n_pts: int
    var_names: list[str]
    value_lines: list[str]


@dataclass(frozen=True)
class NgspiceRunResult:
    """
    Result record for a single ngspice subprocess invocation.

    Captures process return code, captured stdio, runtime, and execution context.

    Parameters
    ----------
    returncode:
        Process return code (0 is typically success; non-zero indicates failure).
        Note that ngspice may still produce partial outputs even with a non-zero
        return code depending on the failure mode.
    stdout:
        Captured standard output (decoded as text).
    stderr:
        Captured standard error (decoded as text).
    elapsed_sec:
        Wall-clock execution time in seconds.
    cmd:
        The exact command vector executed (argv-style). Stored as a tuple for
        immutability and stable logging.
    cwd:
        Working directory used for the subprocess execution.

    Notes
    -----
    - Keep stdout/stderr as plain text (not truncated) at this layer; truncation
      is better handled by a presentation helper (e.g., tail()).
    """
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    cmd: tuple[str, ...]
    cwd: str


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
        Performance vector (or ``None`` on failure).
    aux : dict[str, float]
        Auxiliary scalar metrics.
    status : SimStatus
        Simulation status code.
    detail : str
        Human-readable details.
    run_dir : str
        Path to the run directory where artifacts/logs were produced.
    """

    performances: Optional[Any]
    aux: Dict[str, float]
    status: SimStatus
    detail: str
    run_dir: str


# =============================================================================
# Kernel-specific context (for error reporting / debugging)
# =============================================================================

@dataclass()
class KernelContext:
    """
    Mutable execution context used for error reporting and debugging.

    This object is meant to be attached to exceptions or propagated through
    failure-handling paths so you can print actionable diagnostics (command,
    netlist content, stdio snippets, timing).

    Parameters
    ----------
    cmd:
        Command vector used to invoke ngspice (argv-style).
    cwd:
        Working directory for the invocation.
    netlist:
        Rendered netlist text passed to ngspice (or path-resolved content).
        Storing the full netlist here is helpful for post-mortem debugging but
        can be large; consider keeping it only in debug mode if size is a concern.
    elapsed_sec:
        Wall-clock runtime in seconds. Default is 0.0 and can be filled in after
        execution completes (success or failure).
    stdout:
        Captured standard output text. Can be populated progressively.
    stderr:
        Captured standard error text. Can be populated progressively.

    Notes
    -----
    - Unlike `NgspiceRunResult`, this context is mutable because it is often
      assembled over time across multiple stages (render -> run -> parse).
    """

    cmd: tuple[str, ...]
    cwd: str
    netlist: str
    elapsed_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""

    def tail(self, limit: int = 2000) -> str:
        """
        Return a tail snippet of combined stderr+stdout for compact diagnostics.

        The method concatenates stderr first (usually most informative) followed
        by stdout, and then returns the last `limit` characters. This is useful
        to embed in exception messages or logs without dumping full outputs.

        Parameters
        ----------
        limit:
            Maximum number of characters to return from the end of the combined
            stream.

        Returns
        -------
        str
            A possibly-truncated tail string of diagnostic output.

        Notes
        -----
        - If both streams are empty, the return value is a single newline.
        - Truncation is character-based, not line-based.
        """
        if limit < 0:
            raise ValueError(f"limit must be >= 0 (got {limit})")
        if limit == 0:
            return ""
        s = (self.stderr or "") + "\n" + (self.stdout or "")
        return s[-limit:] if len(s) > limit else s
