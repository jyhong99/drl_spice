from __future__ import annotations

import numpy as np

from ..types import ParamSpec


# =============================================================================
# ActionModel: configuration / input validation helpers
# =============================================================================


def _require_x_in_bounds(
    x: np.ndarray,
    *,
    clip_low: float,
    clip_high: float,
    x_tol: float,
) -> None:
    """
    Require that `x` lies within [clip_low, clip_high] up to a tolerance.

    This is used to validate that the state/parameter vector `x` stays within
    its normalized/clipped range before applying an action update.

    Parameters
    ----------
    x : np.ndarray
        Vector to validate. Assumed to be 1D or already flattened.
    clip_low : float
        Allowed minimum.
    clip_high : float
        Allowed maximum.
    x_tol : float
        Tolerance margin. Values below (clip_low - x_tol) or above (clip_high + x_tol)
        are rejected.

    Raises
    ------
    ValueError
        If `x` is outside the allowed range beyond tolerance.
    """
    clip_low = float(clip_low)
    clip_high = float(clip_high)
    x_tol = float(x_tol)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if (x_min < clip_low - x_tol) or (x_max > clip_high + x_tol):
        raise ValueError(
            f"x is out of bounds [{clip_low},{clip_high}]. min={x_min}, max={x_max}"
        )


def _clip_or_validate_action(
    a: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    clip_action: bool,
    eps: float,
) -> np.ndarray:
    """
    Clip or validate an action vector against bounds.

    Parameters
    ----------
    a : np.ndarray
        Action vector (assumed 1D or already flattened).
    action_low : float
        Lower bound for each action coordinate.
    action_high : float
        Upper bound for each action coordinate.
    clip_action : bool
        If True, return `np.clip(a, action_low, action_high)`.
        If False, validate bounds and raise on violation.
    eps : float
        Numerical tolerance for bound validation when `clip_action` is False.

    Returns
    -------
    np.ndarray
        Action vector that is either clipped (if `clip_action=True`) or the original
        action vector (if within bounds).

    Raises
    ------
    ValueError
        If `clip_action` is False and any value violates bounds beyond `eps`.
    """
    action_low = float(action_low)
    action_high = float(action_high)
    eps = float(eps)

    if clip_action:
        return np.clip(a, action_low, action_high)

    a_min = float(np.min(a))
    a_max = float(np.max(a))
    if (a_min < action_low - eps) or (a_max > action_high + eps):
        raise ValueError(
            f"action out of bounds [{action_low},{action_high}]: min={a_min}, max={a_max}"
        )
    return a


# =============================================================================
# SpecDecoder: configuration / input validation helpers
# =============================================================================


def _require_log_positive_bounds(ctx: str, s: ParamSpec) -> None:
    """
    Require strictly positive bounds for log-scale parameters.

    Parameters
    ----------
    ctx : str
        Context label used in error messages.
    s : ParamSpec
        Parameter specification. If `s.scale == "log"`, bounds must be > 0.

    Raises
    ------
    ValueError
        If log-scale is requested with non-positive bounds.
    """
    if s.scale == "log" and (float(s.p_min) <= 0.0 or float(s.p_max) <= 0.0):
        raise ValueError(
            f"[{ctx}] log-scale requires positive bounds for {s.name}: {s.p_min}, {s.p_max}"
        )


def _require_step_positive(ctx: str, s: ParamSpec) -> None:
    """
    Require positive step size if step snapping is enabled.

    Parameters
    ----------
    ctx : str
        Context label used in error messages.
    s : ParamSpec
        Parameter specification.

    Raises
    ------
    ValueError
        If `s.step` is provided but not strictly positive.
    """
    if s.step is not None and float(s.step) <= 0.0:
        raise ValueError(f"[{ctx}] step must be > 0 for {s.name}, got {s.step}")


# =============================================================================
# Scalar utilities (interpolation / clipping / quantization)
# =============================================================================


def _clip01_scalar(t: float) -> float:
    """
    Clip a scalar into the [0, 1] interval.

    Parameters
    ----------
    t : float
        Input scalar.

    Returns
    -------
    float
        `t` clipped to [0, 1].
    """
    t = float(t)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t


def _snap_step_scalar(v: float, *, step: float, p_min: float, p_max: float) -> float:
    """
    Quantize a scalar to the nearest multiple of `step`, then clamp to bounds.

    Parameters
    ----------
    v : float
        Input scalar to quantize.
    step : float
        Step size for quantization (must be > 0).
    p_min : float
        Lower bound for clamping.
    p_max : float
        Upper bound for clamping.

    Returns
    -------
    float
        Quantized and clamped scalar value.

    Notes
    -----
    The value is snapped using:

        snapped = round(v / step) * step

    then clamped into [p_min, p_max].
    """
    v = float(v)
    step = float(step)
    p_min = float(p_min)
    p_max = float(p_max)

    snapped = round(v / step) * step
    if snapped < p_min:
        snapped = p_min
    elif snapped > p_max:
        snapped = p_max
    return float(snapped)


def _interp_linear(t: float, *, p_min: float, p_max: float) -> float:
    """
    Linear interpolation between bounds.

    Parameters
    ----------
    t : float
        Interpolation coefficient in [0, 1].
    p_min : float
        Lower bound.
    p_max : float
        Upper bound.

    Returns
    -------
    float
        Interpolated value: p_min + t * (p_max - p_min)
    """
    t = float(t)
    p_min = float(p_min)
    p_max = float(p_max)
    return float(p_min + t * (p_max - p_min))


def _interp_log10(t: float, *, p_min: float, p_max: float) -> float:
    """
    Log10-space interpolation between positive bounds.

    Parameters
    ----------
    t : float
        Interpolation coefficient in [0, 1].
    p_min : float
        Lower bound (must be > 0).
    p_max : float
        Upper bound (must be > 0).

    Returns
    -------
    float
        Interpolated value in original scale. Interpolation is performed in
        log10 space:

            log_min = log10(p_min)
            log_max = log10(p_max)
            out = 10 ** (log_min + t * (log_max - log_min))
    """
    t = float(t)
    p_min = float(p_min)
    p_max = float(p_max)

    log_min = float(np.log10(p_min))
    log_max = float(np.log10(p_max))
    return float(10.0 ** (log_min + t * (log_max - log_min)))


def _round_sig_scalar(v: float, k: int) -> float:
    """
    Round a scalar to `k` significant digits.

    This is useful when you want a value-independent rounding rule (e.g., for
    parameter quantization) that preserves a fixed number of significant digits
    rather than a fixed number of decimal places.

    Parameters
    ----------
    v : float
        Input scalar.
    k : int
        Number of significant digits. If `k <= 0`, `v` is returned unchanged.

    Returns
    -------
    float
        Rounded scalar value.

    Notes
    -----
    The rounding is implemented by converting "significant digits" into a
    decimal rounding position.

    Let:
    - ``m = floor(log10(|v|))`` be the order of magnitude of `v`.
    - We want `k` significant digits, so the decimal rounding exponent is:

      ``d = k - 1 - m``

    Then we compute ``np.round(v, d)``.

    Edge cases
    ----------
    - If `v == 0`, returns `0.0` (log10 undefined).
    - If `k <= 0`, returns `v` unchanged.

    Examples
    --------
    >>> _round_sig_scalar(1234.5, 2)
    1200.0

    >>> _round_sig_scalar(0.012345, 2)
    0.012
    """
    v = float(v)
    if int(k) <= 0:
        return v
    if v == 0.0:
        return 0.0

    d = int(k) - 1 - int(np.floor(np.log10(abs(v))))
    return float(np.round(v, d))
