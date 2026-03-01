"""
action
==========

Action and parameter decoding utilities.

This package contains small, composable components that define:

1) **ActionModel**
   Maps a policy action vector (typically in [-1, 1]) into the environment's
   normalized parameter space update (e.g., delta update with step size `eta`).

2) **SpecDecoder**
   Maps the environment's normalized parameter vector `x` into *physical*
   circuit parameter values (e.g., device widths, inductances, capacitances)
   based on a circuit-specific parameter domain.

Typical usage
-------------
Create the action model and decoder once (environment construction time),
then use them at every step:

>>> action_model = ActionModel(m_param=n_params, eta=2.0 / max_steps)
>>> decoder = SpecDecoder(circuit_type="CS", max_param=1.0, clip_x=True)
>>>
>>> x_next = action_model.update(x=x, action=action)
>>> phys = decoder.decode_x_to_physical(x_next)
>>> cfg = decoder.make_design_config(x_next)

Public API
----------
This module re-exports the public objects so callers can do:

>>> from env.action import ActionModel, SpecDecoder, action_space_bounds

Notes
-----
- Public callables expose NumPy-style docstrings with shape, dtype, and
  validation contracts.
- Hot-path methods (`ActionModel.update`, `SpecDecoder.decode_x_to_physical`)
  are optimized for per-step execution in RL loops.
"""

from __future__ import annotations

# Re-export public API from submodules.
from .action import ActionModel, action_space_bounds
from .decoder import SpecDecoder

__all__ = [
    "ActionModel",
    "action_space_bounds",
    "SpecDecoder",
]
