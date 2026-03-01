from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..patterns import (
    TYPE_KEY,
    T_WALL_KEY,
    EPISODE_KEY,
    EVENT_EP_START,
    EVENT_EP_END,
    EVENT_STEP,
    EVENT_CUSTOM,
    OBS_KEY,
    ACTION_KEY,
    INFO_KEY,
    DONE_REASON_KEY,
    EP_STEPS_KEY,
    EP_RETURN_KEY,
    WALL_SEC_KEY,
    LAST_SIM_STATUS_KEY,
    LAST_RUN_DIR_KEY,
    STEP_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
)
from ..types import LoggerSettings
from .writer import CSVWriter


@dataclass
class Logger:
    """
    Episode/step event logger backed by a CSV event writer.

    This logger is intended for RL-style training/evaluation loops where you want
    to persist:
    - episode boundaries (start/end)
    - per-step events (reward, termination flags, optional arrays)
    - custom events (arbitrary metadata)

    The logger manages a unique run directory and delegates all persistence to
    `CSVEventWriter`. It also keeps simple episode aggregates (return, length)
    and writes them at episode end.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory under which a unique run directory is created.
    run_name : str, default="run"
        Prefix for the run directory name.
    settings : LoggerSettings, default=LoggerSettings()
        Logging settings passed to the underlying writer. In particular:
        - `flush_every_steps` controls periodic flush.
        - `store_arrays` controls if obs/action arrays are included inline.
    run_meta : dict[str, Any] or None, default=None
        Extra metadata about the run (e.g., git commit, seed, experiment tag).
        Written once at initialization.
    env_config : dict[str, Any] or None, default=None
        Environment configuration snapshot. Written once at initialization.

    Attributes
    ----------
    run_dir : pathlib.Path
        Unique directory created for this run. This is created in `__post_init__`.

    Notes
    -----
    - A unique run directory is created using timestamp + random suffix.
    - Episode index is 1-based in this implementation (first episode => 1).
    - Step index is reset at each episode start (also 1-based per episode).
    """

    root_dir: Union[str, Path]
    run_name: str = "run"
    settings: LoggerSettings = field(default_factory=LoggerSettings)
    run_meta: Optional[Dict[str, Any]] = None
    env_config: Optional[Dict[str, Any]] = None

    # NOTE: run_dir, _writer, and counters are initialized in __post_init__.

    def __post_init__(self) -> None:
        """
        Create the run directory, initialize writer, and persist run headers.

        This method:
        1) Ensures `root_dir` exists.
        2) Creates a unique `run_dir` using timestamp + UUID suffix.
        3) Initializes the CSV writer and internal counters.
        4) Writes run metadata and environment config, then flushes.

        Raises
        ------
        FileExistsError
            If the randomly generated run directory already exists (rare but possible).
        ValueError
            If `run_name` is empty after normalization.
        """
        root = Path(self.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        self.run_name = str(self.run_name).strip()
        if not self.run_name:
            raise ValueError("run_name must be a non-empty string")

        run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8]
        self.run_dir = root / f"{self.run_name}_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self._writer = CSVWriter(run_dir=self.run_dir, settings=self.settings)

        # Episode/step bookkeeping (episode index starts at 0 until first start_episode)
        self._ep_idx = 0
        self._step_idx = 0
        self._step_since_flush = 0

        # Episode aggregates
        self._ep_reward_sum = 0.0
        self._ep_len = 0
        self._ep_start_time = time.time()
        self._episode_open = False

        # Persist headers / configs
        self._writer.write_run_meta(
            run_meta=self.run_meta or {},
            settings_obj=self.settings,
            run_dir=self.run_dir,
        )
        self._writer.write_env_config(env_config=self.env_config or {})
        self._writer.flush()

    def start_episode(self, *, reset_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the beginning of a new episode and write an episode-start event.

        Parameters
        ----------
        reset_info : dict[str, Any] or None, default=None
            Extra information produced by environment reset. This is stored under:
            `INFO_KEY: {"reset_info": ...}`.

        Notes
        -----
        - Resets per-episode counters (step index, return sum, length).
        - Flushes immediately to ensure episode boundary is durable on disk.
        """
        self._ep_idx += 1
        self._step_idx = 0
        self._ep_reward_sum = 0.0
        self._ep_len = 0
        self._ep_start_time = time.time()
        self._episode_open = True
        self._step_since_flush = 0  # reset flush counter at episode boundary

        self._writer.write_event(
            {
                TYPE_KEY: EVENT_EP_START,
                EPISODE_KEY: self._ep_idx,
                T_WALL_KEY: time.time(),
                INFO_KEY: {"reset_info": reset_info or {}},
            }
        )
        self._writer.flush()

    def end_episode(
        self,
        *,
        final_info: Optional[Dict[str, Any]] = None,
        done_reason: str = "unknown",
    ) -> None:
        """
        Mark the end of the current episode and write an episode-end event.

        Parameters
        ----------
        final_info : dict[str, Any] or None, default=None
            Extra information produced at episode termination. If it contains
            `final_info["sim"]` as a dict, this logger attempts to extract:
            - `status` -> LAST_SIM_STATUS_KEY
            - `run_dir` -> LAST_RUN_DIR_KEY
        done_reason : str, default="unknown"
            Human-readable reason for termination (e.g., "terminated", "truncated",
            "timeout", "constraint_violation", etc.).

        Notes
        -----
        This writes aggregates:
        - EP_STEPS_KEY: episode length
        - EP_RETURN_KEY: sum of rewards
        - WALL_SEC_KEY: wall-clock duration of the episode

        Flushes immediately to make episode summary durable on disk.
        """
        if not self._episode_open:
            raise ValueError("end_episode called without an active episode; call start_episode() first")

        wall_sec = time.time() - self._ep_start_time

        last_sim_status = ""
        last_run_dir = ""
        if isinstance(final_info, dict):
            sim = final_info.get("sim", {})
            if isinstance(sim, dict):
                last_sim_status = str(sim.get("status", ""))
                last_run_dir = str(sim.get("run_dir", ""))

        self._writer.write_event(
            {
                TYPE_KEY: EVENT_EP_END,
                EPISODE_KEY: self._ep_idx,
                T_WALL_KEY: time.time(),
                DONE_REASON_KEY: done_reason,
                EP_STEPS_KEY: self._ep_len,
                EP_RETURN_KEY: self._ep_reward_sum,
                WALL_SEC_KEY: wall_sec,
                LAST_SIM_STATUS_KEY: last_sim_status,
                LAST_RUN_DIR_KEY: last_run_dir,
                INFO_KEY: {"final_info": final_info or {}},
            }
        )
        self._episode_open = False
        self._writer.flush()

    def log_step(
        self,
        *,
        obs: Optional[np.ndarray],
        action: Optional[np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single environment step event.

        Parameters
        ----------
        obs : np.ndarray or None
            Observation at this step. Stored only if `settings.store_arrays == "inline"`.
        action : np.ndarray or None
            Action taken at this step. Stored only if `settings.store_arrays == "inline"`.
        reward : float
            Reward obtained from this transition.
        terminated : bool
            True if the episode ended due to a terminal state (MDP termination).
        truncated : bool
            True if the episode ended due to a time-limit or external truncation.
        info : dict[str, Any] or None, default=None
            Additional metadata for this step.

        Notes
        -----
        - Increments step index and episode aggregates (length, return sum).
        - Flushes periodically based on `settings.flush_every_steps`.
        """
        if not self._episode_open:
            raise ValueError("log_step called without an active episode; call start_episode() first")

        self._step_idx += 1
        self._ep_len += 1
        self._ep_reward_sum += float(reward)

        event: Dict[str, Any] = {
            TYPE_KEY: EVENT_STEP,
            EPISODE_KEY: self._ep_idx,
            STEP_KEY: self._step_idx,
            T_WALL_KEY: time.time(),
            REWARD_KEY: float(reward),
            TERMINATED_KEY: bool(terminated),
            TRUNCATED_KEY: bool(truncated),
            INFO_KEY: info or {},
        }

        # Optional: store arrays inline (writer decides serialization format)
        if self.settings.store_arrays == "inline":
            if obs is not None:
                event[OBS_KEY] = np.asarray(obs)
            if action is not None:
                event[ACTION_KEY] = np.asarray(action)

        self._writer.write_event(event)

        # Periodic flushing for durability (reduces data loss on crash)
        self._step_since_flush += 1
        if self._step_since_flush >= int(self.settings.flush_every_steps):
            self._writer.flush()
            self._step_since_flush = 0

    def log_event(self, event: Dict[str, Any]) -> None:
        """
        Log a custom event and flush immediately.

        Parameters
        ----------
        event : dict[str, Any]
            Arbitrary event payload. If missing required keys, defaults are injected:
            - TYPE_KEY defaults to EVENT_CUSTOM
            - T_WALL_KEY defaults to current wall-clock time

        Notes
        -----
        This is useful for ad-hoc markers (e.g., evaluation start/end, checkpoints,
        parameter dumps, warnings).
        """
        e = dict(event)

        if T_WALL_KEY not in e:
            e[T_WALL_KEY] = time.time()
        if TYPE_KEY not in e:
            e[TYPE_KEY] = EVENT_CUSTOM

        self._writer.write_event(e)
        self._writer.flush()

    def flush(self) -> None:
        """
        Flush buffered writer state to disk.

        Notes
        -----
        This is a thin wrapper around the underlying writer's flush.
        """
        self._writer.flush()

    def close(self) -> None:
        """
        Close the underlying writer and release resources.

        Notes
        -----
        Call this at the end of a run to ensure file handles are closed cleanly.
        """
        self._writer.close()

    def __enter__(self) -> "Logger":
        """
        Enter context manager.

        Returns
        -------
        RunLogger
            The logger instance itself.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit context manager and close the logger.
        """
        self.close()
