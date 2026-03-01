from __future__ import annotations

"""
Environment specifications.

This module keeps environment-specific performance specs and re-exports shared
simulation registries from `ngspice.specs` to avoid duplicated definitions.
"""

from typing import Dict, Final, Tuple

from ngspice.specs import (
    COMMON_FIXED_VALUES_DEFAULT,
    COMMON_ANALYSIS_KNOBS,
    COMMON_LET_KNOBS,
    CIRCUIT_NAMES_REGISTRY,
    ANALYSIS_KNOB_REGISTRY,
    LET_KNOB_REGISTRY,
    PARAM_DOMAIN_REGISTRY,
    get_param_domain,
    get_circuit_names,
    get_analysis_knobs,
    get_let_knobs,
)

from .types import (
    CircuitType,
    ParamDomain,
    AnalysisKnobSet,
    CircuitNames,
    PerfSpec,
    SpecItem,
)


# =============================================================================
# Performance metric ordering
# =============================================================================

PERF_METRIC_ORDER_BASE: Final[Tuple[str, ...]] = ("S11_dB", "S21_dB", "S22_dB", "NF_dB", "PD_mW")
PERF_METRIC_ORDER_LINEARITY: Final[Tuple[str, ...]] = (*PERF_METRIC_ORDER_BASE, "IIP3_dBm")


# =============================================================================
# Performance specs (constraints) + references
# =============================================================================

CS_PERF_SPEC: Final[PerfSpec] = PerfSpec(
    circuit_type="CS",
    items=(
        SpecItem("S11_dB", "lte", -10.0),
        SpecItem("S21_dB", "gte", 20.0),
        SpecItem("S22_dB", "lte", -10.0),
        SpecItem("NF_dB", "lte", 2.0),
        SpecItem("PD_mW", "lte", 5.0),
    ),
    references={
        "S11_dB": 0.0,
        "S21_dB": 10.0,
        "S22_dB": 0.0,
        "NF_dB": 4.0,
        "PD_mW": 10.0,
    },
)

CS_PERF_SPEC_LINEARITY: Final[PerfSpec] = PerfSpec(
    circuit_type="CS",
    items=(*CS_PERF_SPEC.items, SpecItem("IIP3_dBm", "gte", -5.0)),
    references={**CS_PERF_SPEC.references, "IIP3_dBm": -15.0},
)

CGCS_PERF_SPEC: Final[PerfSpec] = PerfSpec(
    circuit_type="CGCS",
    items=(
        SpecItem("S11_dB", "lte", -10.0),
        SpecItem("S21_dB", "gte", 15.0),
        SpecItem("S22_dB", "lte", -10.0),
        SpecItem("NF_dB", "lte", 5.0),
        SpecItem("PD_mW", "lte", 7.0),
    ),
    references={
        "S11_dB": 0.0,
        "S21_dB": 5.0,
        "S22_dB": 0.0,
        "NF_dB": 7.0,
        "PD_mW": 12.0,
    },
)

CGCS_PERF_SPEC_LINEARITY: Final[PerfSpec] = PerfSpec(
    circuit_type="CGCS",
    items=(*CGCS_PERF_SPEC.items, SpecItem("IIP3_dBm", "gte", 0.0)),
    references={**CGCS_PERF_SPEC.references, "IIP3_dBm": -10.0},
)

PERF_SPEC_REGISTRY: Final[Dict[CircuitType, PerfSpec]] = {
    "CS": CS_PERF_SPEC,
    "CGCS": CGCS_PERF_SPEC,
}

PERF_SPEC_LINEARITY_REGISTRY: Final[Dict[CircuitType, PerfSpec]] = {
    "CS": CS_PERF_SPEC_LINEARITY,
    "CGCS": CGCS_PERF_SPEC_LINEARITY,
}


# =============================================================================
# Accessors
# =============================================================================

def perf_metric_order(*, enable_linearity: bool) -> Tuple[str, ...]:
    """
    Return ordered performance-metric keys used by the environment.

    Parameters
    ----------
    enable_linearity : bool
        If ``True``, include the linearity metric (currently ``"IIP3_dBm"``)
        as the last entry in the returned order.

    Returns
    -------
    tuple[str, ...]
        Deterministic metric order used to map simulation vectors to named
        metrics in constraint/objective/reward code paths.

    Notes
    -----
    The returned tuple defines the canonical index contract across:
    - simulator output vector layout
    - constraint normalization
    - objective FoM computation
    """
    return PERF_METRIC_ORDER_LINEARITY if enable_linearity else PERF_METRIC_ORDER_BASE


def get_perf_spec(*, circuit_type: str, enable_linearity: bool) -> PerfSpec:
    """
    Resolve performance specification for a circuit type and mode.

    Parameters
    ----------
    circuit_type : str
        Circuit identifier, e.g. ``"CS"`` or ``"CGCS"``.
    enable_linearity : bool
        If ``True``, resolve from the linearity-augmented registry; otherwise
        resolve from the base registry.

    Returns
    -------
    PerfSpec
        Immutable performance specification object containing:
        - ordered constraint items
        - per-metric reference values for normalization

    Raises
    ------
    ValueError
        If no matching spec is found for the given configuration.

    Notes
    -----
    Lookup is case-sensitive after string coercion to preserve explicit
    registry keys and avoid ambiguous implicit normalization.
    """
    ct = str(circuit_type)
    reg = PERF_SPEC_LINEARITY_REGISTRY if enable_linearity else PERF_SPEC_REGISTRY
    spec = reg.get(ct)
    if spec is not None:
        return spec
    raise ValueError(
        f"PerfSpec not found for circuit_type={ct!r}, enable_linearity={enable_linearity}. "
        f"Available (base): {tuple(PERF_SPEC_REGISTRY.keys())}, "
        f"(linearity): {tuple(PERF_SPEC_LINEARITY_REGISTRY.keys())}"
    )


__all__ = [
    "PERF_METRIC_ORDER_BASE",
    "PERF_METRIC_ORDER_LINEARITY",
    "PERF_SPEC_REGISTRY",
    "PERF_SPEC_LINEARITY_REGISTRY",
    "perf_metric_order",
    "get_perf_spec",
    "COMMON_FIXED_VALUES_DEFAULT",
    "COMMON_ANALYSIS_KNOBS",
    "COMMON_LET_KNOBS",
    "CIRCUIT_NAMES_REGISTRY",
    "ANALYSIS_KNOB_REGISTRY",
    "LET_KNOB_REGISTRY",
    "PARAM_DOMAIN_REGISTRY",
    "get_param_domain",
    "get_circuit_names",
    "get_analysis_knobs",
    "get_let_knobs",
    "ParamDomain",
    "AnalysisKnobSet",
    "CircuitNames",
]
