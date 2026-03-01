"""
ngspice.netlist
===============

Netlist-level utilities for ngspice-based circuit design and modification.

This subpackage provides high-level abstractions for working with ngspice
netlists, including:

- Parsing rendered netlists into structured circuit representations.
- Patching marker-delimited regions (e.g., design variables, knobs).
- Applying design-time modifications in a controlled and auditable manner.

The intent of this module is to separate **netlist semantics** from
**execution (Kernel)** and **result parsing (Reader)**, allowing clean
composition in optimization, reinforcement learning, or automated design flows.

Public API
----------
Classes
- Circuit
    Parse and analyze a rendered netlist, exposing design-variable and
    device-to-variable mappings.
- Designer
    Patch a netlist in-place by injecting design variables and optional knobs.

Functions
- patch_block
    Replace the content of a marker-delimited block with generated lines.
- has_marker
    Lightweight check for the existence of a well-formed marker region.

Notes
-----
- Marker semantics are defined by the `Marker` type and sentinel strings
  in `ngspice.patterns`.
- All public symbols are re-exported here for convenience; internal helpers
  (e.g., split functions, regex patterns) remain in their respective modules
  and are not part of the stable API.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Public re-exports
# -----------------------------------------------------------------------------

from .circuit import Circuit
from .designer import Designer
from .patcher import has_marker, patch_block

__all__ = [
    "Circuit",
    "Designer",
    "patch_block",
    "has_marker",
]
