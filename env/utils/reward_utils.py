from __future__ import annotations

from typing import Any

import numpy as np

from ..patterns import U_DIM


# =============================================================================
# Array coercion / shape helpers
# =============================================================================

def _as_1d_f64(x: Any) -> np.ndarray:
    """
    Convert an input object into a flattened float64 NumPy array.

    Parameters
    ----------
    x : Any
        Array-like input. Common types include scalars, lists/tuples, NumPy arrays,
        and objects accepted by `np.asarray`.

    Returns
    -------
    np.ndarray
        1D NumPy array of dtype float64 with shape ``(D,)``.

    Notes
    -----
    - This is a low-level helper used in constraint and reward computations
      where consistent dtype/shape simplifies numeric logic.
    - Scalars become shape ``(1,)`` after `reshape(-1)`.
    """
    return np.asarray(x, dtype=np.float64).reshape(-1)


# =============================================================================
# ConstraintModel: configuration validation helpers
# =============================================================================


def _require_finite_nonneg(name: str, v: float) -> None:
    """
    Require a finite non-negative scalar (constraint config context).

    Parameters
    ----------
    name : str
        Field name used in error messages.
    v : float
        Scalar value to validate.

    Raises
    ------
    ValueError
        If `v` is not finite or is negative.
    """
    v = float(v)
    if (not np.isfinite(v)) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {v}")


def _require_spec_ref_separation(
    *,
    ctx: str,
    key: str,
    spec_v: float,
    ref_v: float,
    eps_denom: float,
) -> None:
    """
    Require sufficient separation between a spec value and reference value.

    This guards against degenerate normalization where `(spec - ref)` becomes
    too small, leading to unstable scaling in satisfaction computation.

    Parameters
    ----------
    ctx : str
        Context label (e.g., circuit type) used in error messages.
    key : str
        Metric key name (e.g., "S21_dB").
    spec_v : float
        Specification/target value for the metric.
    ref_v : float
        Reference value for normalization.
    eps_denom : float
        Minimum allowed absolute separation between spec and reference.

    Raises
    ------
    ValueError
        If `abs(spec_v - ref_v) <= eps_denom`.
    """
    spec_v = float(spec_v)
    ref_v = float(ref_v)
    eps_denom = float(eps_denom)
    if abs(spec_v - ref_v) <= eps_denom:
        raise ValueError(
            f"[{ctx}] spec-ref too small for {key}: spec={spec_v}, ref={ref_v}"
        )


# =============================================================================
# ConstraintModel: numeric stability helpers
# =============================================================================

def _safe_denom(spec: np.ndarray, ref: np.ndarray, *, eps_denom: float) -> np.ndarray:
    """
    Compute a safe denominator for normalization.

    Given spec and reference vectors, the raw denominator is:

        denom = spec - ref

    For entries where `abs(denom) <= eps_denom`, this function substitutes `1.0`
    to avoid division by near-zero values.

    Parameters
    ----------
    spec : np.ndarray
        Specification vector (shape `(n,)`).
    ref : np.ndarray
        Reference vector (shape `(n,)`).
    eps_denom : float
        Threshold below which denominators are treated as unstable.

    Returns
    -------
    np.ndarray
        Safe denominator vector (shape `(n,)`).

    Notes
    -----
    Substituting `1.0` is a pragmatic choice that keeps scaling finite. In your
    config validation, `_require_spec_ref_separation` already rejects cases where
    spec and ref are too close; `_safe_denom` is a second defensive layer.
    """
    eps_denom = float(eps_denom)
    denom = spec - ref
    return np.where(np.abs(denom) > eps_denom, denom, 1.0)


def _is_non_convergent(o: np.ndarray) -> bool:
    """
    Detect non-convergent or invalid simulation output.

    Parameters
    ----------
    o : np.ndarray
        Performance vector.

    Returns
    -------
    bool
        True if any element of `o` is non-finite (NaN or +/-Inf).
    """
    return bool(np.any(~np.isfinite(o)))


def _u_vec(*, non_convergent: bool, unstable: bool) -> np.ndarray:
    """
    Build the auxiliary invalidity indicator vector `u`.

    Parameters
    ----------
    non_convergent : bool
        If True, set the non-convergence flag.
    unstable : bool
        If True, set the instability flag (e.g., K-factor below threshold).

    Returns
    -------
    np.ndarray
        Vector of shape `(U_DIM,)` with float64 dtype.

    Notes
    -----
    Current convention (must match the rest of the codebase):
    - u[0] == 1.0  => non-convergent
    - u[1] == 1.0  => unstable
    """
    u = np.zeros(U_DIM, dtype=np.float64)
    u[0] = 1.0 if non_convergent else 0.0
    u[1] = 1.0 if unstable else 0.0
    return u