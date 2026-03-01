"""
termination
===========

Episode termination and reset-decision utilities.

This package defines the logic that governs:

1) **When an episode should terminate**
2) **What reset action should be taken after termination**

The design explicitly separates *termination conditions* from *reset policies*
to keep environment control logic modular, testable, and extensible.

Main Components
---------------
TerminationModel
    Stateless termination checker that determines whether an episode ends
    based on step count and simulation validity (e.g., non-convergence).

ResetRule
    Post-termination reset policy that decides whether to:
    - continue from the current state, or
    - reset to a new random initial state

    The decision may depend on the termination reason and stochastic rules.

Typical Usage
-------------
>>> from env.termination import TerminationModel, ResetRule
>>>
>>> term = TerminationModel(max_steps=200)
>>> reset_rule = ResetRule(p_reset=0.05)
>>>
>>> done, reason = term.check(step=t, non_convergent=nonconv)
>>> reset_mode = reset_rule.decide(done_reason=reason)

Notes
-----
- Termination reasons are always returned explicitly as a ``DoneReason`` value.
  This allows downstream components (reset logic, logging, curriculum learning)
  to react differently depending on *why* an episode ended.
- Non-convergent termination is treated as a hard failure and always triggers
  a random reset by default.

Public API
----------
>>> from env.termination import TerminationModel, ResetRule
"""

from __future__ import annotations

# Public re-exports (stable API surface)
from .termination import TerminationModel, ResetRule

__all__ = (
    "TerminationModel",
    "ResetRule",
)
