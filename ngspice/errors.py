from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ngspice.types import KernelContext, Marker


# =============================================================================
# RAW / parsing
# =============================================================================

class RawParseError(ValueError):
    """
    Raised when parsing ngspice RAW-ascii (or RAW-like text) fails.

    This exception is intended for *format/content* failures during parsing,
    not for subprocess execution failures (timeouts, non-zero exit codes, etc.).

    Notes
    -----
    - Prefer raising this when the input exists but violates the expected
      schema/structure (missing headers, malformed numeric lines, etc.).
    """


# =============================================================================
# Kernel / execution errors
# =============================================================================

class NgspiceError(RuntimeError):
    """
    Base class for ngspice-related runtime failures.

    This is the umbrella category for errors that occur while running ngspice
    or handling its outputs (timeouts, missing artifacts, non-zero exits, etc.).
    """


class KernelError(NgspiceError):
    """
    Base class for errors raised by `Kernel.run`.

    Parameters
    ----------
    message:
        Human-readable error message.
    ctx:
        Optional `KernelContext` containing command/cwd/netlist/stdout/stderr and
        timing information for debugging.

    Attributes
    ----------
    ctx:
        The execution context (may be None if not available).
    """

    def __init__(self, message: str, *, ctx: Optional[KernelContext] = None) -> None:
        super().__init__(message)
        self.ctx = ctx


class KernelNetlistNotFound(KernelError, FileNotFoundError):
    """
    Raised when the netlist path passed to the kernel does not exist.

    Notes
    -----
    - This is a *kernel-layer* error. If you also have netlist/circuit parsing
      errors, keep them in separate exception types to avoid ambiguity.
    """


class NgspiceExecutableNotFound(KernelError, FileNotFoundError):
    """
    Raised when the ngspice executable cannot be found or launched.

    Typical causes include:
    - ngspice not installed,
    - wrong `ngspice_bin` path,
    - PATH not configured in the execution environment.
    """


class NgspiceTimeout(KernelError):
    """
    Raised when ngspice execution exceeds the configured timeout.

    Notes
    -----
    - The associated `ctx` may contain partial stdout/stderr when available.
    """


class NgspiceMissingOutputs(KernelError):
    """
    Raised when ngspice finishes but expected output files are missing.

    Parameters
    ----------
    message:
        Human-readable error message (often includes a tail of logs).
    ctx:
        Execution context (required for this error type).
    missing:
        Sequence of missing file paths (typically stringified paths).

    Attributes
    ----------
    missing:
        List of missing output paths.
    """

    def __init__(self, message: str, *, ctx: KernelContext, missing: Sequence[str]) -> None:
        super().__init__(message, ctx=ctx)
        self.missing = list(missing)


class NgspiceNonZeroExit(KernelError):
    """
    Raised when ngspice returns a non-zero exit code (and `check=True`).

    Parameters
    ----------
    message:
        Human-readable error message.
    ctx:
        Execution context at failure time.
    returncode:
        Process exit code.

    Attributes
    ----------
    returncode:
        Integer exit status from the ngspice subprocess.
    """

    def __init__(self, message: str, *, ctx: KernelContext, returncode: int) -> None:
        super().__init__(message, ctx=ctx)
        self.returncode = int(returncode)


# =============================================================================
# Text I/O errors
# =============================================================================

class TextIOError(IOError):
    """
    Base class for text I/O failures.

    Intended for small helper utilities (read/write text files) so that upper
    layers can catch a single category for all text I/O problems.
    """


class TextFileNotFound(TextIOError, FileNotFoundError):
    """
    Raised when a text file path does not exist.

    Parameters
    ----------
    path:
        Missing file path.

    Attributes
    ----------
    path:
        The missing path.
    """

    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")
        self.path = path


class TextReadError(TextIOError):
    """
    Raised when reading a text file fails.

    Parameters
    ----------
    path:
        File that failed to read.
    cause:
        Original exception (permission error, decoding error, etc.).

    Attributes
    ----------
    path:
        Path to the file.
    cause:
        Original exception instance for debugging.
    """

    def __init__(self, path: Path, cause: Exception):
        super().__init__(f"Failed to read text file: {path}")
        self.path = path
        self.cause = cause


class TextWriteError(TextIOError):
    """
    Raised when writing a text file fails.

    Parameters
    ----------
    path:
        File that failed to write.
    cause:
        Original exception (permission error, disk full, etc.).

    Attributes
    ----------
    path:
        Path to the file.
    cause:
        Original exception instance for debugging.
    """

    def __init__(self, path: Path, cause: Exception):
        super().__init__(f"Failed to write text file: {path}")
        self.path = path
        self.cause = cause


# =============================================================================
# Data schema / parsing contracts (DataFrame columns etc.)
# =============================================================================

class DataSchemaError(ValueError):
    """
    Raised when an expected schema (columns/types) is missing or invalid.

    Intended for DataFrame parsing/validation layers where correctness depends
    on explicit column names (e.g., frequency column, S-parameters, etc.).
    """


class MissingColumnError(DataSchemaError):
    """
    Raised when a required column cannot be found among multiple candidates.

    Parameters
    ----------
    tried:
        Candidate column names that were attempted, in order.
    columns:
        Actual DataFrame columns present.

    Attributes
    ----------
    tried:
        Tuple of candidate names.
    columns:
        Full list of columns observed.
    """

    def __init__(self, tried: Sequence[str], columns: Sequence[str]) -> None:
        preview = list(columns)[:20]
        super().__init__(f"Missing required column; tried {tuple(tried)}. columns={preview}")
        self.tried = tuple(tried)
        self.columns = list(columns)


# =============================================================================
# Circuit / netlist structure errors
# =============================================================================

class CircuitError(ValueError):
    """
    Base class for netlist/circuit parsing and structural errors.

    Use this for:
    - missing marker blocks,
    - unexpected template structure,
    - parsing failures in circuit-specific logic (not subprocess execution).
    """


class CircuitNetlistNotFound(CircuitError, FileNotFoundError):
    """
    Raised when a circuit/netlist file is missing at the circuit layer.

    This is distinct from `KernelNetlistNotFound`:
    - Circuit-layer: cannot locate/parse the netlist artifact as input.
    - Kernel-layer: execution input file missing at runtime.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(f"Netlist file not found: {path}")
        self.path = path


class NetlistMarkerError(CircuitError):
    """
    Raised when required marker blocks are missing or malformed in a netlist.

    Examples
    --------
    - start marker exists but end marker missing,
    - markers exist but in wrong order,
    - markers appear multiple times unexpectedly.
    """


class NetlistFormatError(CircuitError):
    """
    Raised when the overall netlist structure is unexpected.

    Examples
    --------
    - `.subckt` not found when expected,
    - required sections absent,
    - invalid topology template layout.
    """


# =============================================================================
# Designer-related errors
# =============================================================================

class DesignerError(ValueError):
    """
    Base class for Designer-related failures.

    Intended for errors in netlist generation/patching/design-variable mapping.
    """


class DesignVarCountMismatch(DesignerError):
    """
    Raised when the number of design variables does not match expectation.

    Parameters
    ----------
    expected:
        Expected number of design variables.
    got:
        Observed number.

    Attributes
    ----------
    expected:
        Expected count.
    got:
        Observed count.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"Design variable count mismatch: expected={expected}, got={got}")
        self.expected = int(expected)
        self.got = int(got)


class TargetNetlistDesignVarCountMismatch(DesignerError):
    """
    Raised when parsing the target netlist inside marker bounds yields a count
    that does not match the expected design-variable count.

    Parameters
    ----------
    expected:
        Expected number of variables (usually from configuration).
    parsed:
        Parsed number of variables inside the marker region.

    Attributes
    ----------
    expected:
        Expected count.
    parsed:
        Parsed count.
    """

    def __init__(self, expected: int, parsed: int) -> None:
        super().__init__(
            "Target netlist design-var count mismatch inside markers: "
            f"expected={expected}, parsed={parsed}"
        )
        self.expected = int(expected)
        self.parsed = int(parsed)


class NetlistPatchError(DesignerError):
    """Raised when patching a netlist fails for any reason."""


class NetlistWriteError(DesignerError):
    """Raised when writing a patched/generated netlist fails."""


# =============================================================================
# Patcher errors (marker-based patch operations)
# =============================================================================

class PatchError(ValueError):
    """
    Base class for marker-based patcher errors.

    These errors occur during textual patching/replacement operations based on
    sentinel markers (start/end boundaries).
    """


class MarkerNotFound(PatchError):
    """
    Raised when start/end markers are not found for a patch operation.

    Parameters
    ----------
    marker:
        Marker pair containing start/end sentinel strings.
    required:
        Whether the marker region was required. Error message indicates required
        vs optional.

    Attributes
    ----------
    marker:
        Marker object.
    required:
        Boolean indicating whether the marker was required.
    """

    def __init__(self, marker: Marker, *, required: bool) -> None:
        super().__init__(
            f"{'Required' if required else 'Optional'} markers not found: "
            f"start={marker.start!r}, end={marker.end!r}"
        )
        self.marker = marker
        self.required = bool(required)


class MarkerMalformed(PatchError):
    """
    Raised when markers are present but malformed.

    Typical cases
    -------------
    - start marker found but end marker missing,
    - end appears before start,
    - nested/duplicated marker blocks where not supported.
    """

    def __init__(self, marker: Marker) -> None:
        super().__init__(f"Markers not found or malformed: start={marker.start!r}, end={marker.end!r}")
        self.marker = marker


# =============================================================================
# Reader facade errors (retry policy, factory creation, etc.)
# =============================================================================

class ReaderError(RuntimeError):
    """
    Base error for Reader facade failures.

    This category is useful if you want a single catch point for analysis-reader
    orchestration logic.
    """


class ReaderRetryableError(ReaderError):
    """
    Errors considered retryable by the Reader facade.

    Examples
    --------
    - file not ready yet (created but empty),
    - transient filesystem race (writer not finished),
    - temporary parse failure due to partial writes.
    """


class ReaderNonRetryableError(ReaderError):
    """
    Errors considered non-retryable by the Reader facade.

    Examples
    --------
    - invalid arguments,
    - unsupported schema,
    - unknown analysis type,
    - deterministic parse failure not expected to change on retry.
    """


class ReaderFailedAfterRetries(ReaderRetryableError):
    """
    Raised when a retryable operation fails repeatedly and the retry budget is exhausted.

    Parameters
    ----------
    analysis_type:
        Analysis type name used to resolve the reader implementation.
    result_path:
        Path to the result artifact being read.
    attempts:
        Number of attempts made.
    last_error:
        The last exception encountered.

    Attributes
    ----------
    analysis_type:
        Analysis type name.
    result_path:
        Target path.
    attempts:
        Attempt count.
    last_error:
        Last exception instance.
    """

    def __init__(
        self,
        *,
        analysis_type: str,
        result_path: str,
        attempts: int,
        last_error: Exception,
    ) -> None:
        super().__init__(
            f"Reader({analysis_type!r}) failed after {attempts} attempts "
            f"on {result_path!r}. last_error={type(last_error).__name__}: {last_error}"
        )
        self.analysis_type = analysis_type
        self.result_path = result_path
        self.attempts = int(attempts)
        self.last_error = last_error


class ReaderImplCreationError(ReaderNonRetryableError):
    """
    Raised when constructing a concrete reader implementation fails.

    Parameters
    ----------
    analysis_type:
        Analysis type key requested.
    cause:
        Original exception raised during instantiation.

    Attributes
    ----------
    analysis_type:
        Analysis type key.
    cause:
        Original exception.
    """

    def __init__(self, analysis_type: str, cause: Exception) -> None:
        super().__init__(f"Failed to create reader impl for analysis_type={analysis_type!r}: {cause}")
        self.analysis_type = analysis_type
        self.cause = cause


class ReaderFileNotReady(ReaderRetryableError):
    """
    File missing/empty during transient window.

    Use this for conditions that are expected to resolve shortly, e.g., when a
    producer is still writing the file.
    """


class ReaderParseFailed(ReaderRetryableError):
    """
    Parsing failed repeatedly.

    Use this when the file exists but parse fails in a way you want to treat as
    retryable (e.g., partial write). If it is truly deterministic corruption,
    consider using a non-retryable error instead.
    """


# =============================================================================
# Registry errors (analysis-reader registry)
# =============================================================================

class RegistryError(ValueError):
    """
    Base error for analysis-reader registry failures.

    Registry manages mapping: analysis_type -> reader factory/implementation.
    """


class UnknownAnalysisType(RegistryError):
    """
    Raised when an analysis_type is not registered/supported.

    Parameters
    ----------
    analysis_type:
        The requested analysis type.
    supported:
        List of supported analysis types.

    Attributes
    ----------
    analysis_type:
        Requested analysis type.
    supported:
        Supported analysis types at the time of error.
    """

    def __init__(self, analysis_type: str, supported: list[str]) -> None:
        super().__init__(f"Unknown analysis_type: {analysis_type!r}. supported={supported}")
        self.analysis_type = analysis_type
        self.supported = supported


class DuplicateAnalysisType(RegistryError):
    """
    Raised when an analysis_type is registered more than once.

    Parameters
    ----------
    analysis_type:
        Duplicate key.
    """

    def __init__(self, analysis_type: str) -> None:
        super().__init__(f"Duplicate registration for analysis_type: {analysis_type!r}")
        self.analysis_type = analysis_type


class ReaderFactoryError(RegistryError):
    """
    Raised when a registry factory fails to create a reader instance.

    Parameters
    ----------
    analysis_type:
        The analysis type requested.
    cause:
        Original exception.

    Attributes
    ----------
    analysis_type:
        Analysis type key.
    cause:
        Original exception instance.
    """

    def __init__(self, analysis_type: str, cause: Exception) -> None:
        super().__init__(f"Failed to create reader for analysis_type={analysis_type!r}: {cause}")
        self.analysis_type = analysis_type
        self.cause = cause
