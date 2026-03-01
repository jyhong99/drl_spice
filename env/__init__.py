"""
env
===

Modular RF LNA optimization environment with pluggable simulators.

This package implements a reinforcement-learning environment for automated
analog/RF circuit optimization (e.g., CS / CGCS LNA topologies). The core
design principles are:

- **Modularity**: action/observation/reward/simulation/termination are isolated
  components with explicit contracts.
- **Reproducibility**: deterministic registries (specs), stable output schemas
  (patterns), and structured event logging.
- **Practicality**: simulation integration with template-based netlists and
  per-run isolated output directories.

Key Modules
-----------
types
    Shared type aliases and dataclasses (e.g., ParamSpec, PerfSpec, SimResult).
specs
    Registries for circuit domains/specs and metric ordering utilities.
patterns
    Constants and regular expressions defining stable keys and CSV schema.
simulator
    Simulation backend (injected) that produces `SimResult`.
modular
    Gymnasium/Gym-compatible environment implementation `LNAEnvModular`.
factory
    Convenience builder `build_lna_env` wiring all components together.

Public API
----------
End users typically only need:

>>> from env import build_lna_env, LNAEnvModular
>>> env = build_lna_env(circuit_type="CS", max_steps=2000, simulator=simulator)

or for advanced usage:

>>> from env import RewardPipeline, ConstraintModel

Notes
-----
- The environment is designed to be compatible with both `gymnasium` and `gym`.
- Templates must contain a replaceable Sky130 `.lib ... sky130.lib.spice <corner>`
  line if you want library paths to be injected programmatically.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

# Core types and registries (optional but useful for advanced users)
from .types import (
    ParamDomain,
    ParamSpec,
    PerfSpec,
    SpecItem,
    LoggerSettings,
    EnvConfig,
    SimResult,
    SimStatus,
)

from .specs import (
    perf_metric_order,
    get_perf_spec,
    get_param_domain,
    get_circuit_names,
    get_analysis_knobs,
    get_let_knobs,
)

if TYPE_CHECKING:
    from .factory import build_lna_env
    from .modular import LNAEnvModular
    from .reward.objective import ConstraintModel, ObjectiveModel
    from .reward.reward import RewardModel, RewardPipeline

__all__ = [
    # factory / env
    "build_lna_env",
    "LNAEnvModular",
    "EnvConfig",
    # reward pipeline
    "ConstraintModel",
    "ObjectiveModel",
    "RewardModel",
    "RewardPipeline",
    # types
    "ParamSpec",
    "ParamDomain",
    "PerfSpec",
    "SpecItem",
    "SimResult",
    "SimStatus",
    "LoggerSettings",
    # specs helpers
    "perf_metric_order",
    "get_perf_spec",
    "get_param_domain",
    "get_circuit_names",
    "get_analysis_knobs",
    "get_let_knobs",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy exports to avoid package import cycles."""
    if name == "build_lna_env":
        from .factory import build_lna_env as _build_lna_env
        return _build_lna_env
    if name == "LNAEnvModular":
        from .modular import LNAEnvModular as _LNAEnvModular
        return _LNAEnvModular
    if name == "ConstraintModel":
        from .reward.objective import ConstraintModel as _ConstraintModel
        return _ConstraintModel
    if name == "ObjectiveModel":
        from .reward.objective import ObjectiveModel as _ObjectiveModel
        return _ObjectiveModel
    if name == "RewardModel":
        from .reward.reward import RewardModel as _RewardModel
        return _RewardModel
    if name == "RewardPipeline":
        from .reward.reward import RewardPipeline as _RewardPipeline
        return _RewardPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
