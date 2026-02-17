"""
utils
=====

Shared utility functions for the RL–SPICE framework.

This package collects small, focused helper modules used across:
- action / observation pipelines
- reward & constraint evaluation
- simulation & netlist handling
- logging and serialization
- general numeric validation

Design principles
-----------------
- Utilities are **stateless** and **side-effect free** where possible.
- Internal helpers are prefixed with `_` and not re-exported.
- Public helpers are re-exported here for convenience and stable imports.

Recommended usage
-----------------
>>> from utils import to_flat_np, redirect_write_paths, atomic_write_text
"""

# -----------------------------------------------------------------------------
# Numeric / array helpers
# -----------------------------------------------------------------------------
from .common_utils import (
    _to_flat_np,
    _require_positive_int,
    _require_len,
    _require_finite,
    _require_finite_positive,
    _require_in_01
)

# -----------------------------------------------------------------------------
# Action utilities
# -----------------------------------------------------------------------------
from .action_utils import (
    _clip01_scalar,
    _interp_linear,
    _interp_log10,
    _round_sig_scalar,
)

# -----------------------------------------------------------------------------
# Observation utilities
# -----------------------------------------------------------------------------
from .observation_utils import (
    _require_key,
)

# -----------------------------------------------------------------------------
# Reward / constraint numeric helpers
# -----------------------------------------------------------------------------
from .reward_utils import (
    _as_1d_f64,
    _safe_denom,
    _is_non_convergent,
    _u_vec,
)

# -----------------------------------------------------------------------------
# Simulation / netlist helpers
# -----------------------------------------------------------------------------
from .simulation_utils import (
    _redirect_write_paths,
    _ensure_noise_writes_frequency,
    _set_sky130_lib,
    _coerce_float_knob,
    _sorted_str_keys,
    _merge_and_sort,
    _normalize_knobs,
)

# -----------------------------------------------------------------------------
# Factory / template helpers
# -----------------------------------------------------------------------------
from .factory_utils import (
    _resolve_path,
    _prepare_spice_scripts_env,
    _default_netlist_templates,
    _normalize_analyses,
    _resolve_templates,
)

# -----------------------------------------------------------------------------
# Logger / serialization helpers
# -----------------------------------------------------------------------------
from .logger_utils import (
    _atomic_write_text,
    _to_jsonable,
    _now_ymd_hms,
    _empty_row,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # common
    "_to_flat_np",
    "_require_positive_int",
    "_require_len",
    "_require_finite",
    "_require_finite_positive",
    "_require_in_01",

    # action
    "_clip01_scalar",
    "_interp_linear",
    "_interp_log10",
    "_round_sig_scalar",

    # observation
    "_require_key",

    # reward / constraint
    "_as_1d_f64",
    "_safe_denom",
    "_is_non_convergent",
    "_u_vec",

    # simulation
    "_redirect_write_paths",
    "_ensure_noise_writes_frequency",
    "_set_sky130_lib",
    "_coerce_float_knob",
    "_sorted_str_keys",
    "_merge_and_sort",
    "_normalize_knobs",

    # factory
    "_resolve_path",
    "_prepare_spice_scripts_env",
    "_default_netlist_templates",
    "_normalize_analyses",
    "_resolve_templates",

    # logger
    "_atomic_write_text",
    "_to_jsonable",
    "_now_ymd_hms",
    "_empty_row",
]
