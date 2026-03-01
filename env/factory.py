from __future__ import annotations

"""
Environment factory utilities.

This module provides a high-level builder for `LNAEnvModular` that wires
up action/observation/reward/termination components around an injected
simulator backend.
"""

from typing import Optional, Tuple

import numpy as np

from .action.action import ActionModel
from .action.decoder import SpecDecoder
from .modular import LNAEnvModular
from .logger import Logger
from .observation.observation import ObservationModel
from .reward.objective import ConstraintModel, ObjectiveModel
from .reward.reward import RewardModel, RewardPipeline
from .termination.termination import ResetRule, TerminationModel
from .patterns import NOISE_KEY, SPARAM_KEY
from .types import EnvConfig, SimulatorProtocol
from .utils.factory_utils import _normalize_analyses


def build_lna_env(
    *,
    circuit_type: str,
    max_steps: int,
    simulator: SimulatorProtocol,
    p_reset: float = 0.01,
    analyses: Tuple[str, ...] = (SPARAM_KEY, NOISE_KEY),
    require_k: bool = False,
    enable_linearity: bool = False,
    seed: Optional[int] = None,
    eta: Optional[float] = None,
    action_mode: str = "delta",
    clip_low: float = 0.0,
    clip_high: float = 1.0,
    action_low: float = -1.0,
    action_high: float = 1.0,
    clip_action: bool = True,
    action_eps: float = 1e-12,
    x_tol: float = 1e-6,
    max_param: float = 1.0,
    decoder_clip_x: bool = True,
    constraint_p: float = 2.0,
    constraint_w: Optional[np.ndarray] = None,
    use_k: bool = True,
    k_min: float = 1.0,
    eps_denom: float = 1e-12,
    eps_viol0: float = 1e-12,
    eps_wsum: float = 1e-12,
    objective_eps: float = 1e-9,
    gamma: float = 0.99,
    beta: float = 0.5,
    lambda_viol: float = 12.0,
    lambda_fom: float = 0.4,
    lambda_var: float = 0.1,
    kappa_viol: float = 2.0,
    kappa_fom: float = 20.0,
    log_enabled: bool = True,
    log_store_physical_params: bool = True,
    log_store_eval: bool = True,
    log_store_obs: bool = False,
    logger: Optional[Logger] = None,
    env_id: Optional[str] = None,
) -> LNAEnvModular:
    """
    Build a fully configured `LNAEnvModular` instance.

    Parameters
    ----------
    circuit_type : str
        Circuit identifier. Must be supported by your spec registries
        (e.g., "CS", "CGCS").
    max_steps : int
        Maximum number of environment steps per episode (time-limit termination).
    simulator : object
        Pre-built simulator instance providing a `.simulate(...)` method.
    p_reset : float, default=0.0
        Probability of random reset when the episode ends due to time limit.
        Non-convergence always triggers a random reset (per `ResetRule`).
    analyses : tuple[str, ...], default=(SPARAM_KEY, NOISE_KEY)
        Requested analyses. If `enable_linearity=True`, linearity is forced to be included.
    require_k : bool, default=False
        If True, treat missing `K_min` (stability metric) as non-convergent.
        This is checked after simulator output parsing.
    enable_linearity : bool, default=False
        If True, add linearity analysis and extend performance dimension accordingly.
    seed : int or None, default=None
        Seed used for `ResetRule` RNG and the environment RNG (if passed to env).
    eta : float or None, default=None
        Step size for the `ActionModel`. If None, uses `2.0 / max_steps`.
    action_mode : str, default="delta"
        Action model mode (e.g., "delta" update rule).
    clip_low, clip_high : float
        State bounds for normalized state `x` (typically [0, 1]).
    action_low, action_high : float
        Action bounds for action vector `a` (typically [-1, 1]).
    clip_action : bool, default=True
        If True, clip actions to bounds. If False, raise on out-of-bounds actions.
    action_eps : float, default=1e-12
        Numerical tolerance for action bound validation when `clip_action=False`.
    x_tol : float, default=1e-6
        Numerical tolerance for validating `x` stays within `[clip_low, clip_high]`.
    max_param : float, default=1.0
        Decoder parameter scaling/normalization factor (implementation-specific).
    decoder_clip_x : bool, default=True
        If True, decoder clips input `x` into range before decoding.
    constraint_p : float, default=2.0
        Constraint aggregation exponent (e.g., p-norm style aggregation).
    constraint_w : np.ndarray or None
        Optional per-metric weights for constraint aggregation.
    use_k : bool, default=True
        If True, include K-factor constraints/flags in the constraint model.
    k_min : float, default=1.0
        Minimum allowed K-factor (stability threshold).
    eps_denom : float, default=1e-12
        Denominator epsilon for constraint normalization safety.
    eps_viol0 : float, default=1e-12
        Baseline epsilon for violation terms (avoid exact zeros / degeneracy).
    eps_wsum : float, default=1e-12
        Weight-sum epsilon for stable normalization when weights are used.
    objective_eps : float, default=1e-9
        Objective model epsilon for numerical stability.
    gamma : float, default=0.99
        Discount factor used in PBRS delta computation inside `RewardModel`.
    beta : float, default=0.5
        Mixture coefficient between immediate penalty and PBRS temporal delta terms.
    lambda_viol, lambda_fom, lambda_var : float
        Reward composition weights for violation, FOM, and variance terms.
    kappa_viol, kappa_fom : float
        Invalid-sample penalty offsets applied to `J_viol`/`J_fom` before reward shaping.
    log_enabled : bool, default=True
        Enable logging inside the environment (requires a Logger attached in env).
    log_store_physical_params : bool, default=True
        Store decoded physical parameters in `sim_info` if True.
    log_store_eval : bool, default=True
        Store pipeline eval dictionary in log payloads if True.
    log_store_obs : bool, default=False
        Store observations in log payloads if True.
    logger : Logger or None, default=None
        Optional run logger. If provided and enabled by `log_enabled`, the env
        emits episode and step events.
    env_id : str or None, default=None
        Optional environment id to include in log payloads.

    Returns
    -------
    env : LNAEnvModular
        Fully constructed environment ready for RL training.

    Notes
    -----
    Construction order:
    1) Normalize analysis kinds (inject linearity if enabled).
    2) Build `SpecDecoder` and infer `m_param`.
    3) Build `ActionModel` (defaults `eta=2/max_steps`).
    4) Create constraint/objective/reward models and pipeline.
    5) Create observation model, termination model, and reset rule.
    6) Create `EnvConfig` and instantiate `LNAEnvModular`.
    """
    # ------------------------------------------------------------------
    # 0) Normalize analyses (force linearity if requested)
    # ------------------------------------------------------------------
    analyses = _normalize_analyses(tuple(analyses), enable_linearity=bool(enable_linearity))
    if not hasattr(simulator, "simulate") or not callable(getattr(simulator, "simulate")):
        raise TypeError("simulator must provide a callable simulate(...) method")

    # ------------------------------------------------------------------
    # 1) Decoder (x -> physical params) and dimensionality
    # ------------------------------------------------------------------
    decoder = SpecDecoder(
        circuit_type=str(circuit_type),
        max_param=float(max_param),
        clip_x=bool(decoder_clip_x),
    )
    m_param = int(len(decoder.variable_keys()))

    # ------------------------------------------------------------------
    # 2) Action model (x transition)
    # ------------------------------------------------------------------
    if eta is None:
        eta = float(2.0 / float(max_steps))

    action_model = ActionModel(
        m_param=int(m_param),
        eta=float(eta),
        mode=str(action_mode),
        clip_low=float(clip_low),
        clip_high=float(clip_high),
        action_low=float(action_low),
        action_high=float(action_high),
        clip_action=bool(clip_action),
        eps=float(action_eps),
        x_tol=float(x_tol),
    )

    # ------------------------------------------------------------------
    # 3) Constraints / objective / reward pipeline
    # ------------------------------------------------------------------
    constraints = ConstraintModel(
        circuit_type=str(circuit_type),
        enable_linearity=bool(enable_linearity),
        p=float(constraint_p),
        w=constraint_w,
        use_k=bool(use_k),
        k_min=float(k_min),
        eps_denom=float(eps_denom),
        eps_viol0=float(eps_viol0),
        eps_wsum=float(eps_wsum),
    )

    objective = ObjectiveModel(eps=float(objective_eps))

    reward = RewardModel(
        gamma=float(gamma),
        beta=float(beta),
        lambda_viol=float(lambda_viol),
        lambda_fom=float(lambda_fom),
        lambda_var=float(lambda_var),
        kappa_viol=float(kappa_viol),
        kappa_fom=float(kappa_fom),
    )

    pipeline = RewardPipeline(constraints=constraints, objective=objective, reward=reward)

    # ------------------------------------------------------------------
    # 4) Observation + termination + reset rule
    # ------------------------------------------------------------------
    obs_builder = ObservationModel()
    termination = TerminationModel(max_steps=int(max_steps))
    reset_rule = ResetRule(p_reset=float(p_reset), rng=np.random.default_rng(seed))

    # ------------------------------------------------------------------
    # 5) Environment config and env instantiation
    # ------------------------------------------------------------------
    cfg = EnvConfig(
        analyses=tuple(analyses),
        require_k=bool(require_k),
        enable_linearity=bool(enable_linearity),
        log_enabled=bool(log_enabled),
        log_store_physical_params=bool(log_store_physical_params),
        log_store_eval=bool(log_store_eval),
        log_store_obs=bool(log_store_obs),
    )

    env = LNAEnvModular(
        domain_decoder=decoder,
        simulator=simulator,
        pipeline=pipeline,
        obs_builder=obs_builder,
        action_model=action_model,
        termination=termination,
        reset_rule=reset_rule,
        cfg=cfg,
        seed=seed,
        logger=logger,
        env_id=env_id,
    )
    return env
