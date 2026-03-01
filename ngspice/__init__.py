"""
ngspice
=======

A lightweight Python toolkit for running ngspice in batch mode and parsing
its text/RAW-ascii outputs into structured objects (NumPy/Pandas).

This package provides:
- A subprocess wrapper (`Kernel`) that runs ngspice with robust diagnostics.
- A small error hierarchy that cleanly separates execution failures from
  parsing/schema failures.
- Utilities for safe text I/O (read/atomic write).
- Parsers for ngspice RAW-ascii outputs and helper regex/markers.

The `ngspice` top-level namespace intentionally exposes a *stable public API*.
Lower-level modules (e.g., regex patterns) remain available under submodules,
but should be treated as implementation details unless explicitly exported here.

Quickstart
----------
Run ngspice:

>>> from ngspice import Kernel
>>> kernel = Kernel(ngspice_bin="ngspice", timeout_sec=30.0)
>>> res = kernel.run(circuit, expected_outputs=["sparams.csv"])

Read a RAW-ascii table:

>>> from ngspice import read_raw_table
>>> df = read_raw_table("ac.raw.txt")

Public API
----------
Execution
- Kernel

Parsing
- read_raw_table

Core types
- NgspiceRunResult, KernelContext, RawAscii, Marker

Errors
- NgspiceError, KernelError, NgspiceTimeout, NgspiceNonZeroExit, ...

Notes
-----
- The package is designed to be strict about schema mismatches to prevent
  silent wrong results (e.g., missing frequency columns, malformed RAW blocks).
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------

__all__: list[str]  # populated at end


try:
    # If you manage versioning elsewhere (setuptools_scm, hatch, etc.)
    from ._version import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


# -----------------------------------------------------------------------------
# Public imports (stable API surface)
# -----------------------------------------------------------------------------
# Keep these imports shallow and explicit. Avoid importing heavy deps here if
# you want fast import times.

# Execution
from .kernel import Kernel

# Parsers / I/O
from .ascii import read_raw_table
from .textio import read_text, atomic_write

# Types
from .types import (
    Marker,
    RawAscii,
    KernelContext,
    NgspiceRunResult,
    PathLike,
    KV,
)

# Errors (curated subset; export what users should catch)
from .errors import (
    # Parse / schema
    RawParseError,
    DataSchemaError,
    MissingColumnError,

    # Execution (kernel)
    NgspiceError,
    KernelError,
    KernelNetlistNotFound,
    NgspiceExecutableNotFound,
    NgspiceTimeout,
    NgspiceMissingOutputs,
    NgspiceNonZeroExit,

    # Text I/O
    TextIOError,
    TextFileNotFound,
    TextReadError,
    TextWriteError,

    # Reader/registry (if you want them public)
    ReaderError,
    ReaderRetryableError,
    ReaderNonRetryableError,
    ReaderFailedAfterRetries,
    ReaderImplCreationError,
    RegistryError,
    UnknownAnalysisType,
    DuplicateAnalysisType,
    ReaderFactoryError,

    # Circuit-level (optional public)
    CircuitError,
    CircuitNetlistNotFound,
    NetlistMarkerError,
    NetlistFormatError,

    # Patcher-level (optional public)
    PatchError,
    MarkerNotFound,
    MarkerMalformed,
)

__all__ = [
    # Version
    "__version__",

    # Execution
    "Kernel",

    # Parsers / I/O
    "read_raw_table",
    "read_text",
    "atomic_write",

    # Types
    "Marker",
    "RawAscii",
    "KernelContext",
    "NgspiceRunResult",
    "PathLike",
    "KV",

    # Errors (stable API)
    "RawParseError",
    "DataSchemaError",
    "MissingColumnError",

    "NgspiceError",
    "KernelError",
    "KernelNetlistNotFound",
    "NgspiceExecutableNotFound",
    "NgspiceTimeout",
    "NgspiceMissingOutputs",
    "NgspiceNonZeroExit",

    "TextIOError",
    "TextFileNotFound",
    "TextReadError",
    "TextWriteError",

    "ReaderError",
    "ReaderRetryableError",
    "ReaderNonRetryableError",
    "ReaderFailedAfterRetries",
    "ReaderImplCreationError",

    "RegistryError",
    "UnknownAnalysisType",
    "DuplicateAnalysisType",
    "ReaderFactoryError",

    "CircuitError",
    "CircuitNetlistNotFound",
    "NetlistMarkerError",
    "NetlistFormatError",

    "PatchError",
    "MarkerNotFound",
    "MarkerMalformed",
]
