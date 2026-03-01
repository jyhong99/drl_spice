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
from importlib import import_module

# Core types and registries (optional but useful for advanced users)
from .types import (
    ParamDomain,
    ParamSpec,
    PerfSpec,
    SpecItem,
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
    # specs helpers
    "perf_metric_order",
    "get_perf_spec",
    "get_param_domain",
    "get_circuit_names",
    "get_analysis_knobs",
    "get_let_knobs",
]

_LAZY_EXPORTS = {
    "build_lna_env": ("env.factory", "build_lna_env"),
    "LNAEnvModular": ("env.modular", "LNAEnvModular"),
    "ConstraintModel": ("env.reward.objective", "ConstraintModel"),
    "ObjectiveModel": ("env.reward.objective", "ObjectiveModel"),
    "RewardModel": ("env.reward.reward", "RewardModel"),
    "RewardPipeline": ("env.reward.reward", "RewardPipeline"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily resolve selected package exports.

    Parameters
    ----------
    name : str
        Attribute name requested from the :mod:`env` package namespace.

    Returns
    -------
    Any
        The resolved exported object (class or function) corresponding to
        ``name`` when present in the lazy-export table.

    Raises
    ------
    AttributeError
        If ``name`` is not a known lazy export.

    Notes
    -----
    Lazy resolution prevents import cycles and reduces import-time overhead by
    deferring heavier module imports until first attribute access.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    return getattr(import_module(module_name), attr_name)
