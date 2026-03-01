"""
reward
======

Reward and objective utilities for the environment.

This package defines the full reward stack used in the environment:

1) **ConstraintModel**
   Converts raw performance metrics into:
   - normalized satisfaction vector ``f``
   - per-metric violations ``viol``
   - aggregate violation cost ``J_viol`` (e.g., weighted p-norm)
   - feasibility/invalidity flags (e.g., non-convergence, instability)

2) **ObjectiveModel**
   Computes scalar objective costs from performances, typically based on a
   figure-of-merit (FoM). The common convention is:

   - FoM (higher is better)
   - Objective cost ``J_fom = -FoM`` (lower is better)

3) **RewardModel**
   Combines costs into a scalar RL reward, optionally including
   Potential-Based Reward Shaping (PBRS)-style temporal difference terms.

4) **RewardPipeline**
   Orchestrates the end-to-end evaluation flow:

       performances/aux
           -> constraints (ConstraintModel)
           -> objective   (ObjectiveModel)
           -> reward      (RewardModel)
           -> merged dict output (RewardPipeline.evaluate)

Typical Usage
-------------
>>> from env.reward import ConstraintModel, ObjectiveModel, RewardModel, RewardPipeline
>>>
>>> constraints = ConstraintModel(circuit_type="CS", enable_linearity=False)
>>> objective = ObjectiveModel()
>>> reward = RewardModel(gamma=0.99, beta=0.5)
>>> pipe = RewardPipeline(constraints=constraints, objective=objective, reward=reward)
>>>
>>> pipe.reset()
>>> out = pipe.evaluate(performances, aux={"K_min": 1.2})
>>> r = out["reward"]  # depends on your patterns.REWARD_KEY

Notes
-----
- Only the symbols re-exported in ``__all__`` are considered public API.
- The returned evaluation dictionary is designed for direct logging (e.g., CSV).

Public API
----------
>>> from env.reward import ConstraintModel, ObjectiveModel, RewardModel, RewardPipeline
"""

from __future__ import annotations

# Public re-exports (stable API surface)
from .objective import ConstraintModel, ObjectiveModel
from .reward import RewardModel, RewardPipeline

__all__ = (
    "ConstraintModel",
    "ObjectiveModel",
    "RewardModel",
    "RewardPipeline",
)
