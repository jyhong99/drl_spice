from __future__ import annotations

"""
Modular gym-compatible environment implementation.

This module defines `LNAEnvModular`, a composable RL environment that delegates
simulation to an injected backend and uses pluggable action/observation/reward
components.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    import gym
    from gym import spaces

from .action.action import ActionModel, action_space_bounds
from .action.decoder import SpecDecoder
from .observation.observation import ObservationModel
from .reward.reward import RewardPipeline
from .types import SimStatus, SimulatorProtocol
from .termination.termination import ResetRule, TerminationModel
from .patterns import (
    OBS_KEY,
    EVAL_KEY,
    SIM_KEY,
    X_PREV_KEY,
    X_NEXT_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    NEXT_RESET_MODE_KEY,
    RESET_CONTINUE,
    RESET_RANDOM,
    NONCONV_KEY,
    ENV_ID_KEY,
    X_KEY,
    K_MIN_KEY,
    EPISODE_KEY,
    STEP_KEY,
    ACTION_KEY,
)
from .types import EnvInfo, EvalOut, SimInfo, ResetOptions, EnvConfig


class LNAEnvModular(gym.Env):
    """
    Modular RL environment for simulator-driven LNA optimization.

    Parameters
    ----------
    domain_decoder : SpecDecoder
        Decoder that maps normalized state vectors `x` into physical design variable
        configurations (dict of parameter name -> float).
    simulator : SimulatorProtocol
        Simulation backend that runs requested analyses and returns a `SimResult`
        containing `performances` (vector) and auxiliary metrics.
    pipeline : RewardPipeline
        Pipeline that computes constraints, objectives, and reward. Must accept
        `(performances, aux, non_convergent)` and return a dict containing at least
        `REWARD_KEY`.
    obs_builder : ObservationModel
        Observation builder that assembles `obs = concat(f, x, u)` (or similar).
    action_model : ActionModel
        State transition model. Applies an action vector and returns the next
        normalized state `x_next`.
    termination : TerminationModel
        Termination policy. Typically ends episodes on time limit or non-convergence.
    reset_rule : ResetRule
        Rule that decides next reset behavior when the episode terminates.
    cfg : EnvConfig
        Environment configuration knobs (which analyses to run, logging flags, etc.).
    seed : int or None, default=None
        RNG seed for environment-level randomness (reset sampling).
    env_id : str or None, default=None
        Identifier to include in logs (useful when running vectorized envs).

    Attributes
    ----------
    action_space : gym.spaces.Box
        Continuous action space, typically `[-1, +1]^m`.
    observation_space : gym.spaces.Box
        Continuous observation space of dimension `obs_dim`. Bounds are unbounded
        (`[-inf, +inf]`).

    Notes
    -----
    - This environment keeps the internal state `x` in normalized coordinates.
      Physical parameters are derived via `SpecDecoder.make_design_config(x)`.
    - The simulator is treated as a black box; failures are encoded as
      `non_convergent=True`, which propagates into constraints/reward.
    """

    def __init__(
        self,
        *,
        domain_decoder: SpecDecoder,
        simulator: SimulatorProtocol,
        pipeline: RewardPipeline,
        obs_builder: ObservationModel,
        action_model: ActionModel,
        termination: TerminationModel,
        reset_rule: ResetRule,
        cfg: EnvConfig,
        seed: Optional[int] = None,
        env_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the modular LNA environment.

        Parameters
        ----------
        domain_decoder : SpecDecoder
            Decoder from normalized state vectors to physical design variable
            dictionaries.
        simulator : SimulatorProtocol
            Simulator backend implementing ``simulate(...)``.
        pipeline : RewardPipeline
            Constraint/objective/reward evaluation pipeline.
        obs_builder : ObservationModel
            Observation assembler.
        action_model : ActionModel
            State transition model applied on each ``step``.
        termination : TerminationModel
            Episode termination rule.
        reset_rule : ResetRule
            Post-termination reset decision rule.
        cfg : EnvConfig
            Immutable runtime configuration.
        seed : int or None, default=None
            Seed for environment RNG.
        env_id : str or None, default=None
            Identifier embedded in ``info`` payloads.

        Raises
        ------
        TypeError
            If ``simulator`` does not provide a callable ``simulate(...)``.
        """
        super().__init__()

        # Core components
        self.decoder = domain_decoder
        if not hasattr(simulator, "simulate") or not callable(getattr(simulator, "simulate")):
            raise TypeError("simulator must provide a callable simulate(...) method")
        self.simulator = simulator
        self.pipeline = pipeline
        self.obs_builder = obs_builder
        self.action_model = action_model
        self.termination = termination
        self.reset_rule = reset_rule
        self.cfg = cfg

        # RNG
        self.rng = np.random.default_rng(seed)

        self.env_id = str(env_id or "env")

        # Episode/step bookkeeping
        self._episode_idx: int = 0
        self._t_wall_reset: float = 0.0
        self._step: int = 0

        # Dimensions
        self.m_param = int(len(self.decoder.variable_keys()))
        self.n_perf = int(len(self.pipeline.constraints.perf_keys))
        self.obs_dim = int(self.obs_builder.dim(n_perf=self.n_perf, m_param=self.m_param))

        # Gym spaces
        a_lo, a_hi = action_space_bounds(self.m_param, low=-1.0, high=1.0)
        self.action_space = spaces.Box(low=a_lo, high=a_hi, dtype=np.float32)

        obs_lo = np.full((self.obs_dim,), -np.inf, dtype=np.float32)
        obs_hi = np.full((self.obs_dim,), +np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_lo, high=obs_hi, dtype=np.float32)

        # State and last evaluation snapshot
        self._x: np.ndarray = np.zeros((self.m_param,), dtype=np.float32)
        self._last_eval: EnvInfo = {}

    # -------------------------------------------------------------------------
    # Reset / step primitives
    # -------------------------------------------------------------------------

    def _sample_x_random(self) -> np.ndarray:
        """
        Sample a random normalized state vector in [0, 1]^m.

        Returns
        -------
        np.ndarray
            Random vector of shape `(m_param,)`, dtype float32.
        """
        return self.rng.random(self.m_param, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[ResetOptions] = None,
    ):
        """
        Reset the environment.

        Parameters
        ----------
        seed : int or None, default=None
            If provided, reseeds the environment RNG.
        options : dict or None, default=None
            Reset options. If `options["reset_mode"] == RESET_CONTINUE`, keep
            the current `x` (continue episode from last state). Otherwise sample
            a fresh random `x`.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        info : dict[str, Any]
            Info dict containing at least the current step, `x`, simulator info,
            and pipeline evaluation output.

        Notes
        -----
        ``reset_mode == RESET_CONTINUE`` keeps the current state vector ``x``.
        All other modes trigger random re-sampling in ``[0, 1]^m``.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._episode_idx += 1
        self._t_wall_reset = float(np.round(time.time(), 6))

        # Reset internal modules
        self._step = 0
        self.pipeline.reset()
        self.obs_builder.reset()
        self.termination.reset()

        # Determine reset mode
        reset_mode = None if options is None else options.get("reset_mode")
        if reset_mode != RESET_CONTINUE:
            self._x = self._sample_x_random()

        # Evaluate at initial x
        eval_out, sim_info = self._evaluate_at_x(self._x)
        obs = self.obs_builder.build_from_pipeline(x=self._x, eval_out=eval_out, action=None)

        # Assemble info
        resolved_reset_mode = reset_mode if reset_mode is not None else RESET_RANDOM
        info: EnvInfo = {
            EPISODE_KEY: int(self._episode_idx),
            STEP_KEY: int(self._step),
            ENV_ID_KEY: self.env_id,
            "reset_mode": resolved_reset_mode,
            NEXT_RESET_MODE_KEY: resolved_reset_mode,
            ACTION_KEY: None,
            X_PREV_KEY: None,
            X_NEXT_KEY: self._x.copy(),
            REWARD_KEY: float(eval_out.get(REWARD_KEY, 0.0)),
            TERMINATED_KEY: False,
            TRUNCATED_KEY: False,
            X_KEY: self._x.copy(),
            SIM_KEY: sim_info,
            EVAL_KEY: eval_out,
            OBS_KEY: obs.copy(),
        }
        self._last_eval = info

        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one environment step.

        Parameters
        ----------
        action : np.ndarray
            Action vector, expected shape `(m_param,)`. The `ActionModel` is
            responsible for validating/clipping and applying the action.

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward : float
            Scalar reward for the transition.
        terminated : bool
            True if the episode ended due to terminal condition.
        truncated : bool
            True if the episode ended due to truncation (time limit etc.).
            (Currently unused: the time limit is modeled as `terminated`.)
        info : dict[str, Any]
            Transition info dict including simulator and evaluation outputs.

        Raises
        ------
        ValueError
            Propagated from action/observation/reward components on invalid
            transition inputs or malformed simulator outputs.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        x_prev = self._x.copy()
        x_next = self.action_model.update(x=self._x, action=action)

        eval_out, sim_info = self._evaluate_at_x(x_next)
        obs = self.obs_builder.build_from_pipeline(x=x_next, eval_out=eval_out, action=action)

        reward = float(eval_out.get(REWARD_KEY, 0.0))
        next_step = int(self._step + 1)

        non_convergent = bool(eval_out.get(NONCONV_KEY, False))
        done, reason = self.termination.check(step=next_step, non_convergent=non_convergent)

        terminated = bool(done)
        truncated = False  # reserved for future separation of truncation semantics

        self._step = next_step
        self._x = x_next
        next_reset_mode = self.reset_rule.decide(done_reason=reason) if done else RESET_CONTINUE

        info: EnvInfo = {
            EPISODE_KEY: int(self._episode_idx),
            STEP_KEY: int(self._step),
            ENV_ID_KEY: self.env_id,
            "done_reason": reason,
            NEXT_RESET_MODE_KEY: next_reset_mode,
            ACTION_KEY: action.copy(),
            X_PREV_KEY: x_prev.copy(),
            X_NEXT_KEY: x_next.copy(),
            REWARD_KEY: float(reward),
            TERMINATED_KEY: bool(terminated),
            TRUNCATED_KEY: bool(truncated),
            X_KEY: self._x.copy(),
            SIM_KEY: sim_info,
            EVAL_KEY: eval_out,
            OBS_KEY: obs.copy(),
        }
        self._last_eval = info

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Simulator + pipeline evaluation
    # -------------------------------------------------------------------------

    def _evaluate_at_x(self, x: np.ndarray) -> Tuple[EvalOut, SimInfo]:
        """
        Run simulator and reward pipeline at a given normalized state `x`.

        Parameters
        ----------
        x : np.ndarray
            Normalized parameter vector of shape `(m_param,)`.

        Returns
        -------
        eval_out : dict[str, Any]
            Output of `RewardPipeline.evaluate(...)`. Must include constraint and
            reward keys used by the env (e.g., REWARD_KEY, NONCONV_KEY).
        sim_info : dict[str, Any]
            Simulation info dict with at least:
            - "status" : string (SimStatus)
            - "detail" : string message
            - "run_dir": string directory path
            Optional:
            - "params": physical parameter dict if enabled by cfg

        Notes
        -----
        Non-convergence is flagged when:
        - simulator status is not ``SimStatus.OK``
        - performance vector is missing or has unexpected length
        - ``cfg.require_k`` is true and ``K_min`` is absent from aux metrics
        """
        params = self.decoder.make_design_config(x)

        sim = self.simulator.simulate(
            design_variables_config=params,
            analyses=self.cfg.analyses,
            enable_linearity=bool(self.cfg.enable_linearity),
        )

        # Simulator status governs non-convergence behavior for the pipeline.
        non_convergent = bool(sim.status != SimStatus.OK)

        sim_info: SimInfo = {
            "status": sim.status.value,
            "detail": sim.detail,
            "run_dir": sim.run_dir,
        }

        # Always include physical parameter configuration in returned info.
        sim_info["params"] = {k: float(v) for k, v in params.items()}

        # Extract performance vector (or synthesize NaNs if missing)
        n_perf = self.n_perf
        if sim.performances is None:
            perf = np.full((n_perf,), np.nan, dtype=np.float64)
            aux: Dict[str, float] = {}
        else:
            perf = np.asarray(sim.performances, dtype=np.float64).reshape(-1)
            aux = dict(sim.aux or {})
            if perf.size != n_perf:
                non_convergent = True
                perf = np.full((n_perf,), np.nan, dtype=np.float64)

        # Enforce presence of K metric if required by config.
        if bool(self.cfg.require_k) and (K_MIN_KEY not in aux):
            non_convergent = True

        eval_out = self.pipeline.evaluate(perf, aux, non_convergent=non_convergent)
        return eval_out, sim_info
