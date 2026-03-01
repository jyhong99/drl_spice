from __future__ import annotations

from dataclasses import dataclass, field
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
    """Raised when `ActionModel` receives an invalid static configuration."""


@dataclass
class ActionModel:
    """
    Deterministic action-to-state transition model in normalized design space.

    The model consumes a normalized state vector ``x`` and an action vector ``a``
    and returns the projected next state ``x_next``. The current implementation
    supports a single update mode:

    .. math::

        x_{t+1} = \mathrm{clip}(x_t + \eta a_t,\; \text{clip\_low},\; \text{clip\_high})

    Parameters
    ----------
    m_param : int
        Number of controllable parameters. Both ``x`` and ``action`` must have
        this length after flattening.
    eta : float
        Positive step size that scales the action contribution.
    mode : {"delta"}, default="delta"
        Update mode. Only ``"delta"`` is currently supported.
    clip_low : float, default=0.0
        Lower bound of the normalized state domain.
    clip_high : float, default=1.0
        Upper bound of the normalized state domain.
    action_low : float, default=-1.0
        Lower bound accepted for action coordinates.
    action_high : float, default=1.0
        Upper bound accepted for action coordinates.
    clip_action : bool, default=True
        If True, out-of-range actions are clipped into
        ``[action_low, action_high]``. If False, out-of-range values raise.
    eps : float, default=1e-12
        Validation tolerance used when ``clip_action=False``.
    x_tol : float, default=1e-6
        Tolerance used to validate that input ``x`` lies within
        ``[clip_low, clip_high]``.

    Attributes
    ----------
    m_param : int
        Validated parameter dimension.
    eta : float
        Positive step size.

    Notes
    -----
    - Inputs are flattened to 1D arrays and converted to ``float32`` for stable
      interaction with RL frameworks.
    - Configuration scalars are cached in private fields to reduce repeated
      casting in the hot ``update`` path.
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

    _eta: float = field(init=False, repr=False)
    _clip_low: float = field(init=False, repr=False)
    _clip_high: float = field(init=False, repr=False)
    _action_low: float = field(init=False, repr=False)
    _action_high: float = field(init=False, repr=False)
    _eps: float = field(init=False, repr=False)
    _x_tol: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate and normalize configuration values.

        Raises
        ------
        ActionModelConfigError
            If ``mode`` is unsupported.
        ValueError
            If dimensions/bounds/scalars are invalid.
        """
        self.m_param = _require_positive_int("m_param", self.m_param)
        self.eta = _require_finite_positive("eta", float(self.eta))

        self.clip_low = float(self.clip_low)
        self.clip_high = float(self.clip_high)
        self.action_low = float(self.action_low)
        self.action_high = float(self.action_high)
        self.eps = float(self.eps)
        self.x_tol = float(self.x_tol)

        _require_ordered_bounds("clip", self.clip_low, self.clip_high)
        _require_ordered_bounds("action", self.action_low, self.action_high)

        if self.mode != "delta":
            raise ActionModelConfigError(
                f"Unsupported mode={self.mode!r}. Only 'delta' action mode is supported."
            )

        # Cache scalars for the hot update path.
        self._eta = self.eta
        self._clip_low = self.clip_low
        self._clip_high = self.clip_high
        self._action_low = self.action_low
        self._action_high = self.action_high
        self._eps = self.eps
        self._x_tol = self.x_tol

    def update(self, *, x: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute the next normalized state from the current state and action.

        Parameters
        ----------
        x : array_like
            Current normalized state. After flattening, shape must be
            ``(m_param,)``.
        action : array_like
            Action vector. After flattening, shape must be ``(m_param,)``.

        Returns
        -------
        numpy.ndarray
            Next state ``x_next`` with shape ``(m_param,)`` and dtype
            ``float32``.

        Raises
        ------
        ValueError
            If inputs have wrong size, contain non-finite values, or violate
            configured bounds (when not clipping action).

        Notes
        -----
        The method performs the following sequence:

        1. Flatten and cast inputs to 1D ``float32``.
        2. Validate shape and finiteness.
        3. Validate current state bounds with tolerance ``x_tol``.
        4. Clip/validate action bounds.
        5. Apply delta update and project back into state bounds.
        """
        x_f = _to_flat_np(x)
        a_f = _to_flat_np(action)

        _require_len("x", x_f, self.m_param)
        _require_len("action", a_f, self.m_param)

        if not np.all(np.isfinite(x_f)):
            raise ValueError("x must contain only finite values")
        if not np.all(np.isfinite(a_f)):
            raise ValueError("action must contain only finite values")

        _require_x_in_bounds(
            x_f,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            x_tol=self._x_tol,
        )

        a_f = _clip_or_validate_action(
            a_f,
            action_low=self._action_low,
            action_high=self._action_high,
            clip_action=self.clip_action,
            eps=self._eps,
        )

        x_next = np.clip(x_f + self._eta * a_f, self._clip_low, self._clip_high)
        return x_next.astype(np.float32, copy=False)

    def compute_eta_from_horizon(self, *, max_steps: int) -> float:
        """
        Compute a heuristic step size from an episode horizon.

        Parameters
        ----------
        max_steps : int
            Positive number of environment steps in one episode.

        Returns
        -------
        float
            Heuristic step size:

            .. math::

                \eta = \frac{2}{\text{max\_steps}}

        Raises
        ------
        ValueError
            If ``max_steps`` is not a positive integer.
        """
        _require_positive_int("max_steps", max_steps)
        return 2.0 / float(max_steps)


def action_space_bounds(
    m_param: int,
    low: float = -1.0,
    high: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build vectorized action-space bounds for continuous control.

    Parameters
    ----------
    m_param : int
        Action dimension.
    low : float, default=-1.0
        Lower bound for each dimension.
    high : float, default=1.0
        Upper bound for each dimension.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Two ``float32`` arrays ``(lo, hi)`` each with shape ``(m_param,)``.

    Raises
    ------
    ValueError
        If ``m_param`` is not positive, bounds are non-finite, or
        ``low >= high``.

    Examples
    --------
    >>> lo, hi = action_space_bounds(3, low=-1.0, high=1.0)
    >>> lo.shape, hi.shape
    ((3,), (3,))
    """
    _require_positive_int("m_param", m_param)

    low_f = float(low)
    high_f = float(high)
    if (not np.isfinite(low_f)) or (not np.isfinite(high_f)):
        raise ValueError(f"low/high must be finite, got low={low}, high={high}")
    _require_ordered_bounds("low/high", low_f, high_f)

    lo = np.full((m_param,), low_f, dtype=np.float32)
    hi = np.full((m_param,), high_f, dtype=np.float32)
    return lo, hi
