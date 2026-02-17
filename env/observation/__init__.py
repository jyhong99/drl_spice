"""
observation
===========

Observation construction utilities for the environment.

This package defines how raw environment outputs (simulation results,
design variables, auxiliary signals) are transformed into a flat observation
vector suitable for reinforcement learning agents.

Design goals
------------
- **Explicit observation contract**: the layout and dimensionality of the
  observation vector are deterministic and documented.
- **Pipeline-friendly**: observations are constructed directly from the
  evaluation/simulation pipeline outputs.
- **RL compatibility**: outputs are flat ``float32`` NumPy arrays.

Main Components
---------------
ObservationModel
    Builds a single observation vector from:
    - performance metrics ``f``
    - current normalized design parameters ``x``
    - auxiliary variables ``u``

    The canonical observation layout is::

        obs = [f, x, u]

Typical Usage
-------------
>>> from observation import ObservationModel
>>>
>>> obs_model = ObservationModel()
>>> obs_model.reset()
>>>
>>> obs = obs_model.build_from_pipeline(
...     x=x,
...     eval_out={"f": f, "u": u},
...     action=action,
... )

Notes
-----
- This package does **not** define the environment dynamics; it only handles
  observation assembly.
- Previous-step information (e.g., last performance or action) may be cached
  internally for optional downstream use, but is not required by default.

Public API
----------
Only the following symbols are intended to be imported by users of this package:

>>> from observation import ObservationModel
"""

from __future__ import annotations

# Public re-export (stable API surface)
from .observation import ObservationModel

__all__ = ("ObservationModel",)
