"""
ngspice.analysis.readers
=======================

Concrete analysis reader implementations.

This subpackage contains concrete implementations of the
:class:`~ngspice.analysis.base.AnalysisReader` protocol for different
ngspice analysis types, such as:

- S-parameter analysis
- Noise analysis
- Linearity / IIP3 analysis
- Stability factor analysis
- Raw table passthrough analyses (Transient, AC, DC operating point)

Design notes
------------
- Readers are **registered via side effects** using the
  :func:`ngspice.analysis.registry.register` decorator.
- Importing this package is sufficient to populate the analysis
  reader registry.
- No symbols are re-exported on purpose; users should rely on
  the registry/facade API instead:

    >>> from ngspice.analysis import Reader
    >>> df = Reader("Noise Analysis").read("noise.csv")

Contents
--------
The following reader modules are imported for their registration
side effects:

- linearity
- noise
- sparams
- stability
- table
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Import reader modules for registration side effects
# -----------------------------------------------------------------------------

from .linearity import LinearityReader  # noqa: F401
from .noise import NoiseReader          # noqa: F401
from .sparams import SParameterReader  # noqa: F401
from .stability import StabilityReader  # noqa: F401
from .table import RawTableReader   # noqa: F401

# Intentionally no __all__:
# This package is side-effect driven (registry population).
