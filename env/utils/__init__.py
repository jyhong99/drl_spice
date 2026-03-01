"""
env.utils
=========

Lazy export surface for utility helpers.

This module intentionally avoids eager importing of utility submodules to
prevent circular imports during package initialization.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "_to_flat_np": ("env.utils.common_utils", "_to_flat_np"),
    "_require_positive_int": ("env.utils.common_utils", "_require_positive_int"),
    "_require_len": ("env.utils.common_utils", "_require_len"),
    "_require_finite": ("env.utils.common_utils", "_require_finite"),
    "_require_finite_positive": ("env.utils.common_utils", "_require_finite_positive"),
    "_require_in_01": ("env.utils.common_utils", "_require_in_01"),
    "_clip01_scalar": ("env.utils.action_utils", "_clip01_scalar"),
    "_interp_linear": ("env.utils.action_utils", "_interp_linear"),
    "_interp_log10": ("env.utils.action_utils", "_interp_log10"),
    "_round_sig_scalar": ("env.utils.action_utils", "_round_sig_scalar"),
    "_require_key": ("env.utils.observation_utils", "_require_key"),
    "_as_1d_f64": ("env.utils.reward_utils", "_as_1d_f64"),
    "_safe_denom": ("env.utils.reward_utils", "_safe_denom"),
    "_is_non_convergent": ("env.utils.reward_utils", "_is_non_convergent"),
    "_u_vec": ("env.utils.reward_utils", "_u_vec"),
    "_redirect_write_paths": ("env.utils.simulation_utils", "_redirect_write_paths"),
    "_ensure_noise_writes_frequency": ("env.utils.simulation_utils", "_ensure_noise_writes_frequency"),
    "_set_sky130_lib": ("env.utils.simulation_utils", "_set_sky130_lib"),
    "_coerce_float_knob": ("env.utils.simulation_utils", "_coerce_float_knob"),
    "_sorted_str_keys": ("env.utils.simulation_utils", "_sorted_str_keys"),
    "_merge_and_sort": ("env.utils.simulation_utils", "_merge_and_sort"),
    "_normalize_knobs": ("env.utils.simulation_utils", "_normalize_knobs"),
    "_resolve_path": ("env.utils.factory_utils", "_resolve_path"),
    "_prepare_spice_scripts_env": ("env.utils.factory_utils", "_prepare_spice_scripts_env"),
    "_default_netlist_templates": ("env.utils.factory_utils", "_default_netlist_templates"),
    "_normalize_analyses": ("env.utils.factory_utils", "_normalize_analyses"),
    "_resolve_templates": ("env.utils.factory_utils", "_resolve_templates"),
    "_atomic_write_text": ("env.utils.logger_utils", "_atomic_write_text"),
    "_to_jsonable": ("env.utils.logger_utils", "_to_jsonable"),
    "_now_ymd_hms": ("env.utils.logger_utils", "_now_ymd_hms"),
    "_empty_row": ("env.utils.logger_utils", "_empty_row"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
