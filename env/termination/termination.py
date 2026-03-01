from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..patterns import (
    DONE_NON_CONV,
    DONE_RUNNING,
    DONE_TIME_LIMIT,
    RESET_CONTINUE,
    RESET_RANDOM,
)
from ..types import DoneReason, ResetMode
from ..utils.common_utils import _require_positive_int, _require_in_01


@dataclass
class TerminationModel:
    """
    Episode termination condition model.

    This model determines whether an episode should terminate based on:
    - the current step index
    - simulation non-convergence status

    Termination reasons are returned explicitly to allow downstream logic
    (e.g., reset rules, logging, curriculum control) to react differently
    depending on *why* an episode ended.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps allowed in a single episode. If the current
        step index reaches or exceeds this value, the episode terminates with
        reason ``DONE_TIME_LIMIT``.

    Notes
    -----
    - Step counting is assumed to be monotonically increasing within an episode.
    - This model is stateless; `reset()` exists only for interface symmetry.
    """

    max_steps: int
    _max_steps_i: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration.

        Raises
        ------
        ValueError
            If `max_steps` is not a positive integer.
        """
        _require_positive_int("max_steps", self.max_steps)
        self._max_steps_i = int(self.max_steps)

    def reset(self) -> None:
        """
        Reset internal state.

        Notes
        -----
        This model does not maintain internal state, so this method is a no-op.
        It exists to keep a consistent interface with other environment models.
        """
        return None

    def check(self, *, step: int, non_convergent: bool) -> Tuple[bool, DoneReason]:
        """
        Check whether the episode should terminate.

        Parameters
        ----------
        step : int
            Current step index. Expected to be monotonic within an episode.
            Must be non-negative.
        non_convergent : bool
            Simulation convergence flag for the current transition. When
            ``True``, termination is immediate.

        Returns
        -------
        done : bool
            True if the episode should terminate.
        reason : DoneReason
            Categorical termination reason:
            - ``DONE_TIME_LIMIT`` if step >= max_steps
            - ``DONE_NON_CONV`` if non-convergent
            - ``DONE_RUNNING`` otherwise

        Raises
        ------
        ValueError
            If ``step < 0``.

        Notes
        -----
        Evaluation order is deterministic and intentionally prioritized:

        1. Non-convergence check
        2. Time-limit check
        3. Running state fallback

        Non-convergence is prioritized over time-limit because it is a hard
        simulator failure that often requires stronger recovery logic.
        """
        step_i = int(step)
        if step_i < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        if bool(non_convergent):
            return True, DONE_NON_CONV

        if step_i >= self._max_steps_i:
            return True, DONE_TIME_LIMIT

        return False, DONE_RUNNING


@dataclass
class ResetRule:
    """
    Reset decision policy following episode termination.

    This model decides whether the environment should:
    - continue from the current state
    - reset to a new random initial state

    based on the termination reason and a stochastic reset probability.

    Parameters
    ----------
    p_reset : float, default=0.0
        Probability of performing a random reset after a time-limit termination.
        Must lie in [0, 1].
    rng : np.random.Generator or None, default=None
        Random number generator used for stochastic reset decisions.
        If None, a default NumPy generator is created.

    Notes
    -----
    - Non-convergent termination always triggers a random reset.
    - Time-limit termination triggers a random reset with probability `p_reset`.
    - Other termination reasons default to continuing from the current state.

    This design allows:
    - aggressive recovery from invalid states (non-convergence)
    - gentle exploration encouragement via occasional random resets
    """

    p_reset: float = 0.0
    rng: Optional[np.random.Generator] = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _p_reset_f: float = field(init=False, repr=False)
    _VALID_DONE_REASONS = {DONE_RUNNING, DONE_TIME_LIMIT, DONE_NON_CONV}

    def __post_init__(self) -> None:
        """
        Validate configuration and initialize RNG.

        Raises
        ------
        ValueError
            If `p_reset` is not in [0, 1].
        """
        self._p_reset_f = float(self.p_reset)
        _require_in_01("p_reset", self._p_reset_f)
        if self.rng is not None:
            if not hasattr(self.rng, "random") or not callable(getattr(self.rng, "random")):
                raise ValueError("rng must provide a callable random() method")
            self._rng = self.rng
        else:
            self._rng = np.random.default_rng()

    def decide(self, *, done_reason: DoneReason) -> ResetMode:
        """
        Decide how to reset (or not) after episode termination.

        Parameters
        ----------
        done_reason : DoneReason
            Reason why the episode terminated. Must be one of:
            ``DONE_RUNNING``, ``DONE_TIME_LIMIT``, ``DONE_NON_CONV``.

        Returns
        -------
        ResetMode
            Reset decision:
            - ``RESET_RANDOM``   : reset to a new random initial state
            - ``RESET_CONTINUE`` : continue from the current state

        Raises
        ------
        ValueError
            If ``done_reason`` is not recognized.

        Notes
        -----
        Policy logic:

        - ``DONE_NON_CONV``:
          Always returns ``RESET_RANDOM`` (hard-failure recovery).
        - ``DONE_TIME_LIMIT``:
          Draws ``u ~ Uniform[0, 1)`` and returns ``RESET_RANDOM`` when
          ``u < p_reset``; otherwise ``RESET_CONTINUE``.
        - ``DONE_RUNNING``:
          Returns ``RESET_CONTINUE``.
        """
        reason = str(done_reason)
        if reason not in self._VALID_DONE_REASONS:
            raise ValueError(
                f"Unknown done_reason={done_reason!r}. valid={tuple(sorted(self._VALID_DONE_REASONS))}"
            )

        if reason == DONE_NON_CONV:
            return RESET_RANDOM

        if reason == DONE_TIME_LIMIT:
            return RESET_RANDOM if (self._rng.random() < self._p_reset_f) else RESET_CONTINUE

        return RESET_CONTINUE
