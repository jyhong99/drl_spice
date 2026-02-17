"""
ngspice.analysis
================

Analysis-level reader interfaces and registry utilities.

This subpackage provides a thin abstraction layer for parsing analysis results
produced by ngspice (or ngspice-adjacent tooling) into structured
`pandas.DataFrame` objects.

Design goals
------------
- Decouple *analysis semantics* (S-parameters, noise, linearity, etc.)
  from *execution* (Kernel) and *netlist manipulation*.
- Support multiple analysis types via a registry/factory pattern.
- Provide a backward-compatible facade (`Reader`) with retry logic for
  asynchronously written output files.

Public API
----------
Classes
- Reader
    Backward-compatible facade that resolves a concrete analysis reader
    via the registry and optionally retries parsing on transient failures.

Functions
- create_reader
    Create a concrete analysis reader implementation by analysis type.
- supported
    Return the list of registered analysis types.

Notes
-----
- Concrete analysis readers are typically registered at import time via
  decorators in `analysis.readers.*`. Importing `Reader` ensures these
  side effects have occurred.
- Only the symbols re-exported here are considered part of the stable,
  public API. Lower-level registry and reader base classes should be
  treated as implementation details unless explicitly imported.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Public re-exports
# -----------------------------------------------------------------------------

from .registry import create_reader, supported
from .reader import Reader

__all__ = [
    "Reader",
    "create_reader",
    "supported",
]
