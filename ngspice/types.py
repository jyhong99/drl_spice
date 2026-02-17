from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union  


# =============================================================================
# Types
# =============================================================================

PathLike = Union[str, Path]
KV = Mapping[str, Any]


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
        s = (self.stderr or "") + "\n" + (self.stdout or "")
        return s[-limit:] if len(s) > limit else s
