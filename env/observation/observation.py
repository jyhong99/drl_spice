from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..utils.common_utils import _to_flat_np, _require_len, _require_positive_int
from ..utils.observation_utils import _require_key, _require_mapping
from ..patterns import F_KEY, U_KEY, U_DIM


@dataclass
class ObservationModel:
    """
    Assemble RL observations from pipeline outputs and normalized state.

    The canonical observation layout is:

    .. math::

        \\mathrm{obs} = [f, x, u]

    where:

    - ``f`` is the satisfaction/performance vector from reward/constraint logic.
    - ``x`` is the current normalized design vector.
    - ``u`` is a fixed-size auxiliary indicator vector with length ``U_DIM``.

    Attributes
    ----------
    _prev_f : np.ndarray or None
        Cached copy of the latest ``f`` vector.
    _prev_action : np.ndarray or None
        Cached copy of the latest action passed to :meth:`build_from_pipeline`.

    Notes
    -----
    - The class does not evolve environment dynamics; it only formats vectors.
    - Outputs are always returned as contiguous ``float32`` arrays.
    - A preallocated output buffer is used per call (via ``np.empty`` + slice
      assignment) to reduce temporary allocations versus `np.concatenate`.
    """

    _prev_f: Optional[np.ndarray] = None
    _prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        """
        Clear cached previous-step vectors.

        Notes
        -----
        Call this at episode reset boundaries to avoid accidental reuse of
        stale diagnostic vectors.
        """
        self._prev_f = None
        self._prev_action = None

    @staticmethod
    def dim(*, n_perf: int, m_param: int) -> int:
        """
        Compute observation dimensionality for ``obs = [f, x, u]``.

        Parameters
        ----------
        n_perf : int
            Length of performance vector ``f``.
        m_param : int
            Length of normalized state vector ``x``.

        Returns
        -------
        int
            Observation length:

            .. math::

                n_{\\mathrm{perf}} + m_{\\mathrm{param}} + U_{\\mathrm{DIM}}

        Notes
        -----
        ``U_DIM`` is the fixed auxiliary vector size defined in
        :mod:`env.patterns`.
        """
        _require_positive_int("n_perf", int(n_perf))
        _require_positive_int("m_param", int(m_param))
        return int(n_perf + m_param + U_DIM)

    def build_from_pipeline(
        self,
        *,
        x: np.ndarray,
        eval_out: Dict[str, Any],
        action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build one observation vector from state and pipeline outputs.

        Parameters
        ----------
        x : array_like
            Current normalized state vector. Flattened internally.
        eval_out : dict[str, Any]
            Mapping produced by the evaluation pipeline. Must contain:

            - ``F_KEY``: array-like vector ``f``.
            - ``U_KEY``: array-like vector ``u`` with length ``U_DIM``.
        action : array_like or None, default=None
            Action applied at the current step. If provided, a validated copy is
            cached for diagnostics.

        Returns
        -------
        np.ndarray
            Flat observation array with dtype ``float32`` and shape
            ``(len(f) + len(x) + U_DIM,)``.

        Raises
        ------
        KeyError
            If ``eval_out`` is missing ``F_KEY`` or ``U_KEY``.
        ValueError
            If ``eval_out`` is not a mapping.
        ValueError
            If ``u`` has invalid length, or if any numeric input contains
            non-finite values.

        Notes
        -----
        Construction order is strictly ``[f, x, u]``.

        Examples
        --------
        >>> m = ObservationModel()
        >>> x = np.array([0.1, 0.2], dtype=np.float32)
        >>> eval_out = {\"f\": np.array([1.0, 0.0]), \"u\": np.array([0.0, 1.0])}
        >>> obs = m.build_from_pipeline(x=x, eval_out=eval_out)
        >>> obs.shape
        (6,)
        """
        eval_map = _require_mapping("eval_out", eval_out)

        # Validate required pipeline outputs
        _require_key(eval_map, F_KEY)
        _require_key(eval_map, U_KEY)

        # Flatten inputs defensively
        f = _to_flat_np(eval_map[F_KEY])
        u = _to_flat_np(eval_map[U_KEY])
        x_f = _to_flat_np(x)

        # Validate auxiliary vector dimensionality
        _require_len("u", u, U_DIM)
        if not np.all(np.isfinite(f)):
            raise ValueError("eval_out[f] must contain only finite values")
        if not np.all(np.isfinite(u)):
            raise ValueError("eval_out[u] must contain only finite values")
        if not np.all(np.isfinite(x_f)):
            raise ValueError("x must contain only finite values")

        # Construct observation with one allocation.
        n_f = int(f.size)
        n_x = int(x_f.size)
        obs = np.empty((n_f + n_x + U_DIM,), dtype=np.float32)
        obs[:n_f] = f
        obs[n_f:n_f + n_x] = x_f
        obs[n_f + n_x:] = u

        # Cache previous-step information
        self._prev_f = f.copy()
        if action is not None:
            a_f = _to_flat_np(action)
            if not np.all(np.isfinite(a_f)):
                raise ValueError("action must contain only finite values when provided")
            self._prev_action = a_f.copy()
        else:
            self._prev_action = None

        return obs

    def prev_f(self) -> Optional[np.ndarray]:
        """
        Get cached previous performance vector.

        Returns
        -------
        np.ndarray or None
            Defensive copy of cached ``f`` if present; otherwise ``None``.
        """
        return None if self._prev_f is None else self._prev_f.copy()

    def prev_action(self) -> Optional[np.ndarray]:
        """
        Get cached previous action vector.

        Returns
        -------
        np.ndarray or None
            Defensive copy of cached action if present; otherwise ``None``.
        """
        return None if self._prev_action is None else self._prev_action.copy()
