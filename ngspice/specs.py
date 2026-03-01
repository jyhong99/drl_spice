from __future__ import annotations

"""
Simulation-facing registries and accessors.

This module owns the circuit registries required by ngspice simulation and
exposes validated accessors so orchestration code can depend on `ngspice.*`
imports without reaching into `env.*`.
"""

from typing import Dict, Final, Tuple

from ngspice.types import (
    AnalysisKnobSet,
    AnalysisKnobSpec,
    CircuitNames,
    CircuitType,
    ParamDomain,
    ParamKeys,
    ParamSpec,
)


# =============================================================================
# Common fixed values (shared across domains)
# =============================================================================

COMMON_FIXED_VALUES_DEFAULT: Final[Dict[str, float]] = {
    "v_dd": 1.8,
    "r_b": 1e4,
    "c_1": 1e-11,
    "l_m": 0.15,
}


# =============================================================================
# Circuit naming registry
# =============================================================================

CIRCUIT_NAMES_REGISTRY: Final[Dict[CircuitType, CircuitNames]] = {
    "CS": CircuitNames(
        designvar_names=(
            "v_dd", "r_b", "c_1", "l_m",
            "v_b",
            "r_d",
            "l_d", "l_g", "l_s",
            "c_d", "c_ex",
            "w_m1", "w_m2",
        ),
        device_names=(
            "V_DD", "R_b", "C_1",
            "V_b",
            "R_D",
            "L_D", "L_G", "L_S",
            "C_D", "C_ex",
            "XM1", "XM2",
        ),
    ),
    "CGCS": CircuitNames(
        designvar_names=(
            "v_dd", "r_b", "c_1", "l_m",
            "v_b1", "v_b2", "v_b3", "v_b4",
            "r_d1", "r_d4", "r_s5",
            "c_d1", "c_d4", "c_s3", "c_s4",
            "w_m1", "w_m2", "w_m3", "w_m4", "w_m5",
        ),
        device_names=(
            "V_DD", "R_b", "C_1",
            "V_b1", "V_b2", "V_b3", "V_b4",
            "R_D1", "R_D4", "R_S5",
            "C_D1", "C_D4", "C_S3", "C_S4",
            "XM1", "XM2", "XM3", "XM4", "XM5",
        ),
    ),
}


# =============================================================================
# Analysis knob definitions
# =============================================================================

COMMON_ANALYSIS_KNOBS: Final[Tuple[AnalysisKnobSpec, ...]] = (
    AnalysisKnobSpec("reltol", 1e-3, "Relative tolerance for Newton iterations"),
    AnalysisKnobSpec("abstol", 1e-12, "Absolute current tolerance"),
    AnalysisKnobSpec("gmin", 1e-12, "Minimum conductance added to all PN junctions"),
    AnalysisKnobSpec("itl1", 500, "DC iteration limit"),
    AnalysisKnobSpec("temp", 300.15, "Simulation temperature (K)"),
    AnalysisKnobSpec("tran_tstep", "20p", "Transient print step"),
    AnalysisKnobSpec("tran_tstop", "10u", "Transient stop time"),
    AnalysisKnobSpec("tran_tstart", "2u", "Transient start saving time"),
    AnalysisKnobSpec("tran_tmax", "10p", "Transient max internal timestep"),
    AnalysisKnobSpec("Vin_amp", 5e-3, "Two-tone amplitude (V)"),
    AnalysisKnobSpec("f1", 2.399e9, "Tone #1 frequency (Hz)"),
    AnalysisKnobSpec("f2", 2.401e9, "Tone #2 frequency (Hz)"),
)

COMMON_LET_KNOBS: Final[Tuple[AnalysisKnobSpec, ...]] = (
    AnalysisKnobSpec("target_frequency", 2.4e9, "Nearest-point selection and Q-factor computations (Hz)"),
    AnalysisKnobSpec("Q_factor", 20, "Inductor Q used for series loss emulation (dimensionless)"),
    AnalysisKnobSpec("T_kelvin", 300.15, "Noise temperature (K)"),
    AnalysisKnobSpec("Rin_ohm", 50.0, "Input reference resistance for NF calc (ohm)"),
    AnalysisKnobSpec("noise_pts", 500, "Noise sweep points/decade"),
    AnalysisKnobSpec("noise_fstart", "1M", "Noise sweep start frequency"),
    AnalysisKnobSpec("noise_fstop", "8G", "Noise sweep stop frequency"),
    AnalysisKnobSpec("sp_pts", 500, "S-parameter sweep points/decade"),
    AnalysisKnobSpec("sp_fstart", "1M", "S-parameter sweep start frequency"),
    AnalysisKnobSpec("sp_fstop", "8G", "S-parameter sweep stop frequency"),
)

ANALYSIS_KNOB_REGISTRY: Final[Dict[CircuitType, AnalysisKnobSet]] = {
    "CS": AnalysisKnobSet("CS", COMMON_ANALYSIS_KNOBS),
    "CGCS": AnalysisKnobSet("CGCS", COMMON_ANALYSIS_KNOBS),
}

LET_KNOB_REGISTRY: Final[Dict[CircuitType, AnalysisKnobSet]] = {
    "CS": AnalysisKnobSet("CS", COMMON_LET_KNOBS),
    "CGCS": AnalysisKnobSet("CGCS", COMMON_LET_KNOBS),
}


# =============================================================================
# Parameter domains (bounds)
# =============================================================================

CS_PARAM_KEYS: Final[ParamKeys] = ParamKeys(
    (
        "v_b",
        "r_d",
        "l_d", "l_g", "l_s",
        "c_d", "c_ex",
        "w_m1", "w_m2",
    )
)

CS_DOMAIN: Final[ParamDomain] = ParamDomain(
    circuit_type="CS",
    keys=CS_PARAM_KEYS,
    specs=(
        ParamSpec("v_b", 0.7, 1.0, scale="linear", round_sig_k=4),
        ParamSpec("r_d", 10.0, 1000.0, scale="log", round_sig_k=4),
        ParamSpec("l_d", 1e-10, 2e-8, scale="log", round_sig_k=4),
        ParamSpec("l_g", 1e-10, 2e-8, scale="log", round_sig_k=4),
        ParamSpec("l_s", 1e-11, 2e-9, scale="log", round_sig_k=4),
        ParamSpec("c_d", 5e-14, 5e-12, scale="log", round_sig_k=4),
        ParamSpec("c_ex", 5e-15, 5e-13, scale="log", round_sig_k=4),
        ParamSpec("w_m1", 1.0, 100.0, scale="log", round_sig_k=4),
        ParamSpec("w_m2", 1.0, 100.0, scale="log", round_sig_k=4),
    ),
    fixed_values_default=COMMON_FIXED_VALUES_DEFAULT,
)

CGCS_PARAM_KEYS: Final[ParamKeys] = ParamKeys(
    (
        "v_b1", "v_b2", "v_b3", "v_b4",
        "r_d1", "r_d4", "r_s5",
        "c_d1", "c_d4", "c_s3", "c_s4",
        "w_m1", "w_m2", "w_m3", "w_m4", "w_m5",
    )
)

CGCS_DOMAIN: Final[ParamDomain] = ParamDomain(
    circuit_type="CGCS",
    keys=CGCS_PARAM_KEYS,
    specs=(
        ParamSpec("v_b1", 0.6, 1.2, scale="linear", round_sig_k=4),
        ParamSpec("v_b2", 0.6, 1.2, scale="linear", round_sig_k=4),
        ParamSpec("v_b3", 0.6, 1.2, scale="linear", round_sig_k=4),
        ParamSpec("v_b4", 0.6, 1.2, scale="linear", round_sig_k=4),
        ParamSpec("r_d1", 10.0, 1000.0, scale="log", round_sig_k=4),
        ParamSpec("r_d4", 10.0, 1000.0, scale="log", round_sig_k=4),
        ParamSpec("r_s5", 10.0, 1000.0, scale="log", round_sig_k=4),
        ParamSpec("c_d1", 1e-13, 1e-11, scale="log", round_sig_k=4),
        ParamSpec("c_d4", 1e-13, 1e-11, scale="log", round_sig_k=4),
        ParamSpec("c_s3", 1e-13, 1e-11, scale="log", round_sig_k=4),
        ParamSpec("c_s4", 1e-13, 1e-11, scale="log", round_sig_k=4),
        ParamSpec("w_m1", 1.0, 100.0, scale="log", round_sig_k=4),
        ParamSpec("w_m2", 1.0, 100.0, scale="log", round_sig_k=4),
        ParamSpec("w_m3", 1.0, 100.0, scale="log", round_sig_k=4),
        ParamSpec("w_m4", 1.0, 100.0, scale="log", round_sig_k=4),
        ParamSpec("w_m5", 1.0, 100.0, scale="log", round_sig_k=4),
    ),
    fixed_values_default=COMMON_FIXED_VALUES_DEFAULT,
)

PARAM_DOMAIN_REGISTRY: Final[Dict[CircuitType, ParamDomain]] = {
    "CS": CS_DOMAIN,
    "CGCS": CGCS_DOMAIN,
}


# =============================================================================
# Accessors
# =============================================================================

def get_param_domain(*, circuit_type: str) -> ParamDomain:
    """
    Lookup the `ParamDomain` for a circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit type key (e.g., "CS", "CGCS").

    Returns
    -------
    ParamDomain
        Design-parameter domain (keys + bounds + scaling).

    Raises
    ------
    ValueError
        If the circuit type is not registered.
    """
    ct = str(circuit_type)
    dom = PARAM_DOMAIN_REGISTRY.get(ct)
    if dom is not None:
        return dom
    raise ValueError(
        f"ParamDomain not found for circuit_type={ct!r}. Available: {tuple(PARAM_DOMAIN_REGISTRY.keys())}"
    )


def get_circuit_names(*, circuit_type: str) -> CircuitNames:
    """
    Lookup the `CircuitNames` contract for a circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit type key (e.g., "CS", "CGCS").

    Returns
    -------
    CircuitNames
        Name contract containing:
        - ordered design-variable names
        - ordered device instance names

    Raises
    ------
    ValueError
        If the circuit type is not registered.
    """
    ct = str(circuit_type)
    names = CIRCUIT_NAMES_REGISTRY.get(ct)
    if names is not None:
        return names
    raise ValueError(
        f"CircuitNames not found for circuit_type={ct!r}. Available: {tuple(CIRCUIT_NAMES_REGISTRY.keys())}"
    )


def get_analysis_knobs(*, circuit_type: str) -> AnalysisKnobSet:
    """
    Lookup the default analysis knob set for a circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit type key.

    Returns
    -------
    AnalysisKnobSet
        Analysis knob definitions for that circuit type.

    Raises
    ------
    ValueError
        If the circuit type is not registered.
    """
    ct = str(circuit_type)
    knobs = ANALYSIS_KNOB_REGISTRY.get(ct)
    if knobs is not None:
        return knobs
    raise ValueError(
        f"AnalysisKnobSet not found for circuit_type={ct!r}. Available: {tuple(ANALYSIS_KNOB_REGISTRY.keys())}"
    )


def get_let_knobs(*, circuit_type: str) -> AnalysisKnobSet:
    """
    Lookup the LET (auxiliary) knob set for a circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit type key.

    Returns
    -------
    AnalysisKnobSet
        LET knob definitions for that circuit type.

    Raises
    ------
    ValueError
        If the circuit type is not registered.
    """
    ct = str(circuit_type)
    knobs = LET_KNOB_REGISTRY.get(ct)
    if knobs is not None:
        return knobs
    raise ValueError(
        f"LET KnobSet not found for circuit_type={ct!r}. Available: {tuple(LET_KNOB_REGISTRY.keys())}"
    )


__all__ = [
    "COMMON_FIXED_VALUES_DEFAULT",
    "CIRCUIT_NAMES_REGISTRY",
    "ANALYSIS_KNOB_REGISTRY",
    "LET_KNOB_REGISTRY",
    "PARAM_DOMAIN_REGISTRY",
    "get_param_domain",
    "get_circuit_names",
    "get_analysis_knobs",
    "get_let_knobs",
]
