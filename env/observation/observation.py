from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..utils.common_utils import _to_flat_np, _require_len
from ..utils.observation_utils import _require_key
from ..patterns import F_KEY, U_KEY, U_DIM


@dataclass
class ObservationModel:
    """
    Observation construction model for the environment.

    This class builds a flat observation vector from multiple sources produced
    by the environment pipeline, typically consisting of:

    - performance metrics (``f``)
    - current normalized design parameters (``x``)
    - auxiliary variables (``u``)

    The resulting observation has the form::

        obs = [f, x, u]

    This model also keeps track of the previous performance vector and the
    previous action for optional downstream use (e.g., diagnostics, shaping,
    or extended observation variants).

    Attributes
    ----------
    _prev_f : np.ndarray or None
        Cached performance vector from the previous step.
    _prev_action : np.ndarray or None
        Cached action vector from the previous step.

    Notes
    -----
    - This class itself is *stateless with respect to the environment dynamics*;
      it only caches previous values for optional external access.
    - All outputs are returned as ``float32`` arrays for RL framework compatibility.
    """

    _prev_f: Optional[np.ndarray] = None
    _prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        """
        Reset cached previous-step information.

        This should be called at environment reset to avoid leaking information
        across episode boundaries.
        """
        self._prev_f = None
        self._prev_action = None

    @staticmethod
    def dim(*, n_perf: int, m_param: int) -> int:
        """
        Compute the observation dimensionality.

        Parameters
        ----------
        n_perf : int
            Number of performance metrics (length of ``f``).
        m_param : int
            Number of design parameters (length of ``x``).

        Returns
        -------
        int
            Total observation dimension::

                n_perf + m_param + U_DIM

        Notes
        -----
        ``U_DIM`` is a fixed constant describing the dimensionality of the
        auxiliary vector ``u``.
        """
        return int(n_perf + m_param + U_DIM)

    def build_from_pipeline(
        self,
        *,
        x: np.ndarray,
        eval_out: Dict[str, Any],
        action: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build an observation vector from pipeline outputs.

        Parameters
        ----------
        x : np.ndarray
            Current normalized design parameter vector.
        eval_out : dict[str, Any]
            Output dictionary produced by the evaluation/simulation pipeline.
            Must contain:
            - ``F_KEY`` : performance vector ``f``
            - ``U_KEY`` : auxiliary vector ``u``
        action : np.ndarray or None, default=None
            Action applied at the current step. If provided, it is cached as
            the previous action.

        Returns
        -------
        np.ndarray
            Flat observation vector with dtype ``float32``.

        Raises
        ------
        KeyError
            If required keys (``F_KEY``, ``U_KEY``) are missing from ``eval_out``.
        ValueError
            If the auxiliary vector ``u`` does not have length ``U_DIM``.

        Notes
        -----
        - All inputs are flattened defensively to avoid shape-related bugs.
        - The observation layout is strictly ordered as ``[f, x, u]``.
        """
        # Validate required pipeline outputs
        _require_key(eval_out, F_KEY)
        _require_key(eval_out, U_KEY)

        # Flatten inputs defensively
        f = _to_flat_np(eval_out[F_KEY])
        u = _to_flat_np(eval_out[U_KEY])
        x_f = _to_flat_np(x)

        # Validate auxiliary vector dimensionality
        _require_len("u", u, U_DIM)

        # Construct observation
        obs = np.concatenate([f, x_f, u], axis=0).astype(np.float32, copy=False)

        # Cache previous-step information
        self._prev_f = f.copy()
        if action is not None:
            self._prev_action = _to_flat_np(action).copy()

        return obs
