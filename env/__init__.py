"""
env
===

Modular RF LNA optimization environment backed by ngspice.

This package implements a reinforcement-learning environment for automated
analog/RF circuit optimization (e.g., CS / CGCS LNA topologies). The core
design principles are:

- **Modularity**: action/observation/reward/simulation/termination are isolated
  components with explicit contracts.
- **Reproducibility**: deterministic registries (specs), stable output schemas
  (patterns), and structured event logging.
- **Practicality**: ngspice integration with template-based netlists and
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
    ngspice execution wrapper that produces `SimResult`.
modular
    Gymnasium/Gym-compatible environment implementation `LNAEnvModular`.
factory
    Convenience builder `build_lna_env` wiring all components together.

Public API
----------
End users typically only need:

>>> from env import build_lna_env, LNAEnvModular
>>> env = build_lna_env(circuit_type="CS", work_root="./runs", max_steps=2000)

or for advanced usage:

>>> from env import NgspiceSimulator, RewardPipeline, ConstraintModel

Notes
-----
- The environment is designed to be compatible with both `gymnasium` and `gym`.
- ngspice templates must contain a replaceable Sky130 `.lib ... sky130.lib.spice <corner>`
  line if you want `sky130_lib_path` to be injected programmatically.
"""

from __future__ import annotations

# High-level builder / environment
from .factory import build_lna_env
from .modular import LNAEnvModular

# Simulation core
from .simulator import NgspiceSimulator

# Reward pipeline
from .reward.objective import ConstraintModel, ObjectiveModel
from .reward.reward import RewardModel, RewardPipeline

# Core types and registries (optional but useful for advanced users)
from .types import (
    ParamSpec,
    ParamDomain,
    PerfSpec,
    SpecItem,
    SimResult,
    SimStatus,
    LoggerSettings,
    EnvConfig,
)

from .specs import (
    perf_metric_order,
    get_perf_spec,
    get_param_domain,
    get_circuit_names,
    get_analysis_knobs,
    get_let_knobs,
)

__all__ = [
    # factory / env
    "build_lna_env",
    "LNAEnvModular",
    "EnvConfig",
    # simulator
    "NgspiceSimulator",
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
