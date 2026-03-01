from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..utils.common_utils import (
    _to_flat_np,
    _require_len,
    _require_ordered_bounds,
    _require_positive_int,
    _require_finite_positive,
)
from ..utils.action_utils import _clip_or_validate_action, _require_x_in_bounds
from ..types import ActionMode


class ActionModelConfigError(ValueError):
    """
    Configuration error raised during `ActionModel` initialization.

    This is used for invalid static settings (e.g., unsupported mode) rather
    than per-step runtime input errors.
    """


@dataclass
class ActionModel:
    """
    Action-to-parameter update model for continuous control.

    This class defines how an action vector produced by an RL policy
    is mapped to a parameter update in the normalized design space.

    Currently, only **delta-based updates** are supported:

        x_{t+1} = clip(x_t + eta * a_t)

    where:
    - x is the current parameter vector
    - a is the action vector (typically in [-1, 1])
    - eta is the step size (learning rate)
    - clip enforces parameter bounds

    Parameters
    ----------
    m_param : int
        Dimensionality of the parameter/action vector.
    eta : float
        Step size for delta updates.
    mode : ActionMode, default="delta"
        Action interpretation mode.
        Only `"delta"` is currently supported.
    clip_low : float, default=0.0
        Lower bound for parameters after update.
    clip_high : float, default=1.0
        Upper bound for parameters after update.
    action_low : float, default=-1.0
        Minimum allowed action value.
    action_high : float, default=1.0
        Maximum allowed action value.
    clip_action : bool, default=True
        Whether to clip actions into [action_low, action_high]
        or raise an error when violated.
    eps : float, default=1e-12
        Numerical tolerance used in action validation.
    x_tol : float, default=1e-6
        Tolerance when validating whether x lies within bounds.

    Notes
    -----
    - This model assumes **normalized parameter space**.
    - Clipping is applied *after* the update to guarantee feasibility.
    - All inputs are flattened to 1D arrays internally.
    """

    m_param: int
    eta: float

    mode: ActionMode = "delta"

    clip_low: float = 0.0
    clip_high: float = 1.0

    action_low: float = -1.0
    action_high: float = 1.0
    clip_action: bool = True

    eps: float = 1e-12
    x_tol: float = 1e-6

    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.

        Raises
        ------
        ActionModelConfigError
            If an unsupported action mode is specified.
        ValueError
            If any numeric/bound configuration is invalid.
        """
        _require_positive_int("m_param", self.m_param)
        _require_finite_positive("eta", float(self.eta))

        _require_ordered_bounds(
            "clip", float(self.clip_low), float(self.clip_high)
        )
        _require_ordered_bounds(
            "action", float(self.action_low), float(self.action_high)
        )

        if self.mode != "delta":
            raise ActionModelConfigError(
                f"Unsupported mode={self.mode!r}. "
                "Only 'delta' action mode is supported."
            )

    def update(self, *, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Apply an action to the current parameter vector.

        This performs a delta update followed by clipping:

            x_next = clip(x + eta * action)

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
            Must have length `m_param` and lie within [clip_low, clip_high].
        action : np.ndarray
            Action vector.
            Must have length `m_param` and lie within
            [action_low, action_high] (or will be clipped).

        Returns
        -------
        np.ndarray
            Updated parameter vector with dtype float32.

        Raises
        ------
        ValueError
            If input shapes or bounds are invalid.
        """
        # Flatten inputs (defensive against accidental shape pollution)
        x_f = _to_flat_np(x)
        a_f = _to_flat_np(action)

        # Shape validation
        _require_len("x", x_f, self.m_param)
        _require_len("action", a_f, self.m_param)
        if not np.all(np.isfinite(x_f)):
            raise ValueError("x must contain only finite values")
        if not np.all(np.isfinite(a_f)):
            raise ValueError("action must contain only finite values")

        # Validate current parameter feasibility
        _require_x_in_bounds(
            x_f,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            x_tol=self.x_tol,
        )

        # Validate or clip action
        a_f = _clip_or_validate_action(
            a_f,
            action_low=self.action_low,
            action_high=self.action_high,
            clip_action=self.clip_action,
            eps=self.eps,
        )

        # Delta update + projection
        x_next = x_f + float(self.eta) * a_f
        x_next = np.clip(x_next, self.clip_low, self.clip_high)

        return x_next.astype(np.float32, copy=False)

    def compute_eta_from_horizon(self, *, max_steps: int) -> float:
        """
        Compute a step size based on a fixed episode horizon.

        This heuristic ensures that a full-range traversal
        is possible within `max_steps` updates:

            eta = 2 / max_steps

        Parameters
        ----------
        max_steps : int
            Maximum number of steps in an episode.

        Returns
        -------
        float
            Recommended step size.

        Notes
        -----
        This is useful for curriculum design or fixed-horizon optimization.
        """
        _require_positive_int("max_steps", max_steps)
        return 2.0 / float(max_steps)


def action_space_bounds(
    m_param: int,
    low: float = -1.0,
    high: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct symmetric action space bounds.

    Parameters
    ----------
    m_param : int
        Action dimensionality.
    low : float, default=-1.0
        Lower bound for each action dimension.
    high : float, default=1.0
        Upper bound for each action dimension.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple of (low, high) bounds, each of shape (m_param,).

    Raises
    ------
    ValueError
        If bounds are invalid/non-finite or m_param is not positive.
    """
    _require_positive_int("m_param", m_param)
    if (not np.isfinite(float(low))) or (not np.isfinite(float(high))):
        raise ValueError(f"low/high must be finite, got low={low}, high={high}")
    _require_ordered_bounds("low/high", float(low), float(high))

    lo = np.full((m_param,), float(low), dtype=np.float32)
    hi = np.full((m_param,), float(high), dtype=np.float32)

    return lo, hi
