from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .objective import ConstraintModel, ObjectiveModel
from ..patterns import (
    F_KEY,
    J_FOM_KEY,
    J_VIOL_KEY,
    REWARD_KEY,
    R_FOM_KEY,
    R_VAR_KEY,
    R_VIOL_KEY,
    TJ_FOM_KEY,
    TJ_VIOL_KEY,
    D_TJ_FOM_PBRS_KEY,
    D_TJ_VIOL_PBRS_KEY,
    FEASIBLE_KEY,
    INVALID_KEY,
    FOM_DB_KEY,
    VAR_F_KEY,
    VIOL_TERM_KEY,
)
from ..utils.common_utils import _require_in_01
from ..utils.reward_utils import _as_1d_f64, _require_finite_nonneg


@dataclass
class RewardModel:
    """
    Scalar reward model with penalty shaping and PBRS-style temporal difference terms.

    This model combines multiple reward components into a single scalar reward:

    - violation term (constraint penalty)
    - FoM term (objective / performance)
    - variance term (stability / smoothness proxy)

    It also supports a Potential-Based Reward Shaping (PBRS)-like mechanism via
    a temporal difference of a shaped cost `tJ`:

        d_tJ = -gamma * tJ_t + tJ_{t-1}

    The final per-component rewards are blended between:
    - immediate shaped cost (negative)
    - PBRS temporal difference term

        r_component = -beta * tJ + (1 - beta) * d_tJ

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor used in the PBRS temporal difference term.
        Must lie in [0, 1].
    beta : float, default=0.5
        Blend factor between immediate cost and PBRS temporal difference term.
        Must lie in [0, 1].

    lambda_viol : float, default=12.0
        Weight applied to the violation reward component.
    lambda_fom : float, default=0.4
        Weight applied to the FoM reward component.
    lambda_var : float, default=0.1
        Weight applied to the variance reward component.

    kappa_viol : float, default=2.0
        Additive penalty applied to `J_viol` when invalid is True.
    kappa_fom : float, default=20.0
        Additive penalty applied to `J_fom` when invalid is True.
    bad_cost : float, default=1e6
        Replacement cost used when incoming cost terms are non-finite.

    Notes
    -----
    - This model treats `J_viol` and `J_fom` as *costs* (lower is better).
    - The returned reward is higher when costs are lower (due to negative signs).
    - The variance term uses `r_var = -var_f`, so lower variance is rewarded.
    """

    gamma: float = 0.99
    beta: float = 0.5

    lambda_viol: float = 12.0
    lambda_fom: float = 0.4
    lambda_var: float = 0.1

    kappa_viol: float = 2.0
    kappa_fom: float = 20.0
    bad_cost: float = 1e6

    _prev_tJ_viol: float = field(init=False, repr=False, default=0.0)
    _prev_tJ_fom: float = field(init=False, repr=False, default=0.0)
    _has_prev: bool = field(init=False, repr=False, default=False)
    _gamma: float = field(init=False, repr=False)
    _beta: float = field(init=False, repr=False)
    _lambda_viol: float = field(init=False, repr=False)
    _lambda_fom: float = field(init=False, repr=False)
    _lambda_var: float = field(init=False, repr=False)
    _kappa_viol: float = field(init=False, repr=False)
    _kappa_fom: float = field(init=False, repr=False)
    _bad_cost: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If gamma/beta are not in [0, 1] or if coefficients are invalid.
        """
        _require_in_01("gamma", float(self.gamma))
        _require_in_01("beta", float(self.beta))

        _require_finite_nonneg("lambda_viol", float(self.lambda_viol))
        _require_finite_nonneg("lambda_fom", float(self.lambda_fom))
        _require_finite_nonneg("lambda_var", float(self.lambda_var))

        _require_finite_nonneg("kappa_viol", float(self.kappa_viol))
        _require_finite_nonneg("kappa_fom", float(self.kappa_fom))
        _require_finite_nonneg("bad_cost", float(self.bad_cost))

        self._gamma = float(self.gamma)
        self._beta = float(self.beta)
        self._lambda_viol = float(self.lambda_viol)
        self._lambda_fom = float(self.lambda_fom)
        self._lambda_var = float(self.lambda_var)
        self._kappa_viol = float(self.kappa_viol)
        self._kappa_fom = float(self.kappa_fom)
        self._bad_cost = float(self.bad_cost)

    def reset(self) -> None:
        """
        Reset internal PBRS memory.

        This should be called at episode reset to prevent cross-episode leakage.
        """
        self._prev_tJ_viol = 0.0
        self._prev_tJ_fom = 0.0
        self._has_prev = False

    @staticmethod
    def _penalize(J: float, *, invalid: bool, kappa: float) -> float:
        """
        Apply an additive penalty to a cost when invalid.

        Parameters
        ----------
        J : float
            Base cost value.
        invalid : bool
            If True, apply penalty.
        kappa : float
            Additive penalty amount.

        Returns
        -------
        float
            Penalized cost.
        """
        return J + kappa if invalid else J

    def compute(
        self,
        *,
        J_viol: float,
        J_fom: float,
        var_f: float,
        invalid: bool,
    ) -> Dict[str, float]:
        """
        Compute per-step reward components and the final scalar reward.

        Parameters
        ----------
        J_viol : float
            Constraint-violation cost (lower is better), typically the output
            of ``ConstraintModel`` under ``VIOL_TERM_KEY``.
        J_fom : float
            Objective cost (lower is better), typically ``-FoM``.
        var_f : float
            Variance cost derived from the satisfaction vector ``f``.
            Non-finite values are replaced with ``bad_cost``; negatives are
            clamped to zero.
        invalid : bool
            Invalid-transition flag (e.g., non-convergent simulation or unstable
            operating point). When true, additive penalties are injected into
            violation/FoM costs before PBRS blending.

        Returns
        -------
        dict[str, float]
            Reward decomposition dictionary:

            - ``REWARD_KEY`` : float
              Final weighted scalar reward.
            - ``R_VIOL_KEY`` : float
              Violation reward component after PBRS/immediate blending.
            - ``R_FOM_KEY`` : float
              FoM reward component after PBRS/immediate blending.
            - ``R_VAR_KEY`` : float
              Variance reward component (``-var_f``).
            - ``TJ_VIOL_KEY`` : float
              Penalized violation cost used for PBRS dynamics.
            - ``TJ_FOM_KEY`` : float
              Penalized FoM cost used for PBRS dynamics.
            - ``D_TJ_VIOL_PBRS_KEY`` : float
              PBRS delta for violation cost, ``-gamma*tJ_viol + prev_tJ_viol``.
            - ``D_TJ_FOM_PBRS_KEY`` : float
              PBRS delta for FoM cost, ``-gamma*tJ_fom + prev_tJ_fom``.

        Raises
        ------
        None
            The function is intentionally non-raising for numeric anomalies.
            Non-finite costs are mapped to ``bad_cost`` to keep reward outputs
            finite and predictable.

        Notes
        -----
        Let ``tJ`` be the invalidity-penalized cost and ``d_tJ`` the PBRS delta.
        For each shaped component:

        ``r_component = -beta * tJ + (1 - beta) * d_tJ``.

        The total reward is:

        ``reward = lambda_viol*r_viol + lambda_fom*r_fom + lambda_var*r_var``.

        On the first call after ``reset()``, PBRS deltas are defined as zero
        because no previous potential exists.
        """
        J_viol_f = float(J_viol)
        J_fom_f = float(J_fom)
        var_f_f = float(var_f)
        bad_cost = self._bad_cost

        if not np.isfinite(J_viol_f):
            J_viol_f = bad_cost
        if not np.isfinite(J_fom_f):
            J_fom_f = bad_cost
        if not np.isfinite(var_f_f):
            var_f_f = bad_cost
        var_f_f = max(0.0, var_f_f)

        invalid_b = bool(invalid)
        tJ_viol = self._penalize(J_viol_f, invalid=invalid_b, kappa=self._kappa_viol)
        tJ_fom = self._penalize(J_fom_f, invalid=invalid_b, kappa=self._kappa_fom)

        if not self._has_prev:
            d_tJ_viol_pbrs = 0.0
            d_tJ_fom_pbrs = 0.0
            self._has_prev = True
        else:
            g = self._gamma
            d_tJ_viol_pbrs = -g * tJ_viol + float(self._prev_tJ_viol)
            d_tJ_fom_pbrs = -g * tJ_fom + float(self._prev_tJ_fom)

        b = self._beta
        r_viol = -b * tJ_viol + (1.0 - b) * d_tJ_viol_pbrs
        r_fom = -b * tJ_fom + (1.0 - b) * d_tJ_fom_pbrs
        r_var = -var_f_f

        r_total = (
            self._lambda_viol * r_viol
            + self._lambda_fom * r_fom
            + self._lambda_var * r_var
        )

        # Update memory for next PBRS delta computation
        self._prev_tJ_viol = float(tJ_viol)
        self._prev_tJ_fom = float(tJ_fom)

        return {
            REWARD_KEY: float(r_total),
            R_VIOL_KEY: float(r_viol),
            R_FOM_KEY: float(r_fom),
            R_VAR_KEY: float(r_var),
            TJ_VIOL_KEY: float(tJ_viol),
            TJ_FOM_KEY: float(tJ_fom),
            D_TJ_VIOL_PBRS_KEY: float(d_tJ_viol_pbrs),
            D_TJ_FOM_PBRS_KEY: float(d_tJ_fom_pbrs),
        }


@dataclass
class RewardPipeline:
    """
    End-to-end pipeline: constraints -> objective -> reward.

    This component orchestrates evaluation of:
    1) constraint satisfaction/violation (ConstraintModel)
    2) objective cost (ObjectiveModel)
    3) reward shaping and scalar reward (RewardModel)

    Parameters
    ----------
    constraints : ConstraintModel
        Constraint evaluator producing satisfaction vector, violations, feasibility flags, etc.
    objective : ObjectiveModel
        Objective evaluator producing FoM-based cost(s).
    reward : RewardModel
        Reward composer that merges costs into a scalar reward.

    Notes
    -----
    - The pipeline returns a merged dictionary containing all intermediate
      components and final reward values. This is useful for logging and
      debugging reward shaping.
    """

    constraints: ConstraintModel
    objective: ObjectiveModel
    reward: RewardModel

    def reset(self) -> None:
        """
        Reset pipeline state (currently only the reward model's PBRS memory).
        """
        self.reward.reset()

    def evaluate(
        self,
        performances: np.ndarray,
        aux: Optional[Mapping[str, float]] = None,
        *,
        non_convergent: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate constraints/objective and compute the shaped reward.

        Parameters
        ----------
        performances : np.ndarray
            Performance vector aligned with the constraint model's metric ordering.
        aux : dict[str, float] or None, default=None
            Auxiliary scalar metrics (e.g., K-min, bandwidth, IIP3, etc.).
        non_convergent : bool or None, default=None
            Overrides automatic non-convergence detection inside constraints.
            If None, constraints determine it from performances.

        Returns
        -------
        dict[str, Any]
            Merged output dictionary containing:
            - constraint outputs (see ConstraintModel.compute)
            - objective scalars:
                - J_VIOL_KEY : float (constraint violation cost)
                - J_FOM_KEY  : float (objective cost = -FoM)
                - FOM_DB_KEY : float (= -J_fom)
                - VAR_F_KEY  : float (variance of satisfaction vector)
            - reward outputs (see RewardModel.compute)
            - FEASIBLE_KEY : bool (redundantly included for convenience)

        Notes
        -----
        This function is intended to be called once per environment step after
        simulation/evaluation completes.
        """
        c = self.constraints.compute(performances, aux, non_convergent=non_convergent)

        f = _as_1d_f64(c[F_KEY])
        J_viol = float(c[VIOL_TERM_KEY])
        J_fom = float(self.objective.j_fom(performances))
        var_f = float(self.objective.var_from_f(f))

        invalid = bool(c[INVALID_KEY])
        feasible = bool(c[FEASIBLE_KEY])

        r = self.reward.compute(J_viol=J_viol, J_fom=J_fom, var_f=var_f, invalid=invalid)

        out: Dict[str, Any] = {}
        out.update(c)
        out.update(
            {
                J_VIOL_KEY: J_viol,
                J_FOM_KEY: J_fom,
                FOM_DB_KEY: -J_fom,
                VAR_F_KEY: var_f,
                FEASIBLE_KEY: feasible,
            }
        )
        out.update(r)
        return out
