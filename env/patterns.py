from __future__ import annotations

import re
from typing import Final

"""
patterns
============

Centralized constants and regular expressions used across the environment.

This module defines:

- **Vector / key conventions** for constraint, objective, reward, and logging pipelines.
- **Event type names** and **CSV schema** for the run logger.
- **Simulation / analysis identifiers** (e.g., SP/Noise/Linearity).
- **Regex patterns** used to rewrite ngspice templates (wrdata/write path redirection,
  noise writer patching, sky130 `.lib` line replacement).

Design goals
------------
- **Single source of truth** for magic strings (keys, event types, filenames).
- **Stable schemas** for CSV logging and downstream analysis.
- **Deterministic contracts** between simulator/reader/reward/logger components.

Notes
-----
- Most identifiers are defined as `typing.Final` constants.
- Regex patterns are compiled once at import time for performance and consistency.
"""

# =============================================================================
# Constraint / evaluation output contract
# =============================================================================

U_DIM: Final[int] = 2
"""Dimension of the auxiliary indicator vector `u`.

Current convention (must be consistent across the codebase):
- u[0] == 1.0  -> simulation non-convergent / invalid numeric output
- u[1] == 1.0  -> stability violation (e.g., K-factor below threshold)
"""

F_KEY: Final[str] = "f"
"""Key for normalized satisfaction vector `f` (shape: `(n_metrics,)`)."""

VIOL_KEY: Final[str] = "viol"
"""Key for per-metric violation vector (shape: `(n_metrics,)`)."""

VIOL_TERM_KEY: Final[str] = "viol_term"
"""Key for aggregated violation scalar (e.g., weighted Lp norm)."""

SAT_KEY: Final[str] = "satisfied_perf"
"""Key for boolean indicating whether all metric constraints are satisfied."""

FEASIBLE_KEY: Final[str] = "feasible"
"""Key for boolean indicating feasibility (satisfied constraints AND not invalid)."""

U_KEY: Final[str] = "u"
"""Key for auxiliary indicator vector `u` (shape: `(U_DIM,)`)."""

NONCONV_KEY: Final[str] = "non_convergent"
"""Key for boolean non-convergence flag (typically derived from u[0])."""

UNSTABLE_KEY: Final[str] = "unstable"
"""Key for boolean stability flag (typically derived from u[1])."""

INVALID_KEY: Final[str] = "invalid"
"""Key for boolean invalidity flag (e.g., non-convergent or unstable)."""

METRICS_KEY: Final[str] = "metrics"
"""Key for dictionary of raw metric values (e.g., S-params, NF, PD, K_min, ...)."""

SPEC_KEY: Final[str] = "spec"
"""Key for the spec/target vector used for normalization (shape: `(n_metrics,)`)."""

REF_KEY: Final[str] = "ref"
"""Key for the reference vector used for normalization (shape: `(n_metrics,)`)."""

PERF_KEYS_KEY: Final[str] = "perf_keys"
"""Key for the ordered tuple of performance metric names corresponding to vector entries."""

ENABLE_LIN_KEY: Final[str] = "enable_linearity"
"""Key for boolean flag indicating whether linearity metrics (e.g., IIP3) are enabled."""

K_MIN_KEY: Final[str] = "K_min"
"""Metric name used for stability thresholding (e.g., K-factor minimum)."""


# =============================================================================
# Objective / FOM constants
# =============================================================================

MIN_OBJ_LEN: Final[int] = 5
"""Minimum length required for objective computation (S11, S21, S22, NF, PD)."""

IIP3_INDEX: Final[int] = 5
"""Index of IIP3 in the performance vector when linearity is enabled."""

DEFAULT_BAD_FOM: Final[float] = -1e6
"""Fallback FOM value returned when the objective cannot be computed safely."""

PD_FLOOR: Final[float] = 1e-9
"""Floor for power dissipation in mW to avoid log(0) and numerical issues."""

NF_DB_MIN: Final[float] = 0.0
"""Minimum clamp for NF in dB (for numeric stability)."""

NF_DB_MAX: Final[float] = 200.0
"""Maximum clamp for NF in dB (for numeric stability)."""

J_VIOL_KEY: Final[str] = "J_viol"
"""Key for constraint violation objective value."""

J_FOM_KEY: Final[str] = "J_fom"
"""Key for FOM-based objective value (often negative of FOM in dB)."""

FOM_DB_KEY: Final[str] = "FOM_dB"
"""Key for the (positive) figure-of-merit value in dB."""

VAR_F_KEY: Final[str] = "var_f"
"""Key for variance of satisfaction vector `f` (used as a shaping/regularizer term)."""


# =============================================================================
# Reward outputs (shaped reward decomposition)
# =============================================================================

REWARD_KEY: Final[str] = "reward"
"""Key for total scalar reward."""

R_VIOL_KEY: Final[str] = "r_viol"
"""Key for the violation-related reward component."""

R_FOM_KEY: Final[str] = "r_fom"
"""Key for the FOM-related reward component."""

R_VAR_KEY: Final[str] = "r_var"
"""Key for the variance-related reward component."""

TJ_VIOL_KEY: Final[str] = "tJ_viol"
"""Key for transformed violation objective (after invalid penalty)."""

TJ_FOM_KEY: Final[str] = "tJ_fom"
"""Key for transformed FOM objective (after invalid penalty)."""

D_TJ_VIOL_PBRS_KEY: Final[str] = "d_tJ_viol_pbrs"
"""Key for PBRS potential difference term for violation objective."""

D_TJ_FOM_PBRS_KEY: Final[str] = "d_tJ_fom_pbrs"
"""Key for PBRS potential difference term for FOM objective."""


# =============================================================================
# Termination / reset conventions
# =============================================================================

DONE_RUNNING: Final[str] = "running"
"""Done reason indicating the episode is still running."""

DONE_TIME_LIMIT: Final[str] = "time_limit"
"""Done reason indicating episode ended due to max step limit."""

DONE_NON_CONV: Final[str] = "non_convergent"
"""Done reason indicating early termination due to non-convergent simulation."""

RESET_RANDOM: Final[str] = "random"
"""Reset mode indicating random reset (re-sample initial state)."""

RESET_CONTINUE: Final[str] = "continue_last"
"""Reset mode indicating continuation from last valid state."""


# =============================================================================
# Logger / storage modes
# =============================================================================

STORE_INLINE: Final[str] = "inline"
"""Store arrays inline (typically JSON-encoded into CSV cells)."""

STORE_OFF: Final[str] = "off"
"""Do not store arrays (omit large data to keep logs light)."""


# =============================================================================
# Event types for CSV logging
# =============================================================================

EVENT_RUN_META: Final[str] = "run_meta"
"""Event type for run-level metadata (seed, tags, git hash, etc.)."""

EVENT_ENV_CONFIG: Final[str] = "env_config"
"""Event type for environment configuration snapshot."""

EVENT_EP_START: Final[str] = "episode_start"
"""Event type for episode start marker."""

EVENT_STEP: Final[str] = "step"
"""Event type for per-step transition logging."""

EVENT_EP_END: Final[str] = "episode_end"
"""Event type for episode end summary."""

EVENT_CUSTOM: Final[str] = "custom"
"""Event type for user-defined custom events."""

CSV_NAME: Final[str] = "run.csv"
"""Default CSV filename written under the run directory."""


# =============================================================================
# CSV schema (column keys)
# =============================================================================

T_WALL_KEY: Final[str] = "t_wall"
"""Wall-clock timestamp string (typically `YYYY-MM-DD HH:MM:SS`)."""

TYPE_KEY: Final[str] = "type"
"""Event type key (one of EVENT_* constants)."""

EPISODE_KEY: Final[str] = "episode"
"""Episode index (integer)."""

STEP_KEY: Final[str] = "step"
"""Step index within the episode (integer)."""

TERMINATED_KEY: Final[str] = "terminated"
"""Boolean terminated flag (MDP terminal state)."""

TRUNCATED_KEY: Final[str] = "truncated"
"""Boolean truncated flag (time limit or external truncation)."""

DONE_REASON_KEY: Final[str] = "done_reason"
"""String done reason (DONE_* constants)."""

EP_STEPS_KEY: Final[str] = "ep_steps"
"""Number of steps in the episode (episode summary)."""

EP_RETURN_KEY: Final[str] = "ep_return"
"""Episode return (sum of rewards, episode summary)."""

WALL_SEC_KEY: Final[str] = "wall_sec"
"""Elapsed wall-clock seconds for an episode or step (optional)."""

LAST_SIM_STATUS_KEY: Final[str] = "last_sim_status"
"""Most recent simulation status string (SimStatus)."""

LAST_RUN_DIR_KEY: Final[str] = "last_run_dir"
"""Most recent simulation run directory path string."""

OBS_JSON_KEY: Final[str] = "obs_json"
"""JSON cell storing observation array (if enabled)."""

ACTION_JSON_KEY: Final[str] = "action_json"
"""JSON cell storing action array (if enabled)."""

INFO_JSON_KEY: Final[str] = "info_json"
"""JSON cell storing env `info` dict (if enabled)."""

PAYLOAD_JSON_KEY: Final[str] = "payload_json"
"""JSON cell storing arbitrary payload (metrics, configs, custom events)."""

OBS_KEY: Final[str] = "obs"
"""In-memory key for raw observation object (not a CSV column)."""

ACTION_KEY: Final[str] = "action"
"""In-memory key for raw action object (not a CSV column)."""

INFO_KEY: Final[str] = "info"
"""In-memory key for raw info object (not a CSV column)."""

CSV_COLUMNS: Final[list[str]] = [
    T_WALL_KEY,
    TYPE_KEY,
    EPISODE_KEY,
    STEP_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    DONE_REASON_KEY,
    EP_STEPS_KEY,
    EP_RETURN_KEY,
    WALL_SEC_KEY,
    LAST_SIM_STATUS_KEY,
    LAST_RUN_DIR_KEY,
    OBS_JSON_KEY,
    ACTION_JSON_KEY,
    INFO_JSON_KEY,
    PAYLOAD_JSON_KEY,
]
"""Ordered list of CSV columns. All written rows should include every column."""

USED_EVENT_KEYS: Final[set[str]] = {
    T_WALL_KEY,
    TYPE_KEY,
    EPISODE_KEY,
    STEP_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    DONE_REASON_KEY,
    EP_STEPS_KEY,
    EP_RETURN_KEY,
    WALL_SEC_KEY,
    LAST_SIM_STATUS_KEY,
    LAST_RUN_DIR_KEY,
    OBS_KEY,
    ACTION_KEY,
    INFO_KEY,
    PAYLOAD_JSON_KEY,
}
"""Keys recognized/used by logger writers when assembling event payloads."""


# =============================================================================
# Environment / simulator keys (nested dict conventions)
# =============================================================================

EVAL_KEY: Final[str] = "eval"
"""Key for evaluation dictionary (constraint/objective/reward pipeline output)."""

SIM_KEY: Final[str] = "sim"
"""Key for simulator result dictionary."""

STATUS_KEY: Final[str] = "status"
"""Key for status string (often from SimStatus)."""

DETAIL_KEY: Final[str] = "detail"
"""Key for human-readable detail message."""

RUN_DIR_KEY: Final[str] = "run_dir"
"""Key for run directory path."""

X_PREV_KEY: Final[str] = "x_prev"
"""Key for previous parameter/state vector."""

X_NEXT_KEY: Final[str] = "x_next"
"""Key for next parameter/state vector."""

ENV_ID_KEY: Final[str] = "env_id"
"""Key for environment identifier string."""

X_KEY: Final[str] = "x"
"""Key for current parameter/state vector."""

RESET_MODE_KEY: Final[str] = "reset_mode"
"""Key for current reset mode (RESET_* constants)."""

NEXT_RESET_MODE_KEY: Final[str] = "next_reset_mode"
"""Key for next reset mode decision (RESET_* constants)."""


# =============================================================================
# Analysis kind keys and environment variable names
# =============================================================================

SPARAM_KEY: Final[str] = "sp"
"""Analysis key for S-parameter simulation."""

NOISE_KEY: Final[str] = "noise"
"""Analysis key for noise figure simulation."""

LINEARITY_KEY: Final[str] = "linearity"
"""Analysis key for linearity / two-tone / FFT-based simulation."""

SPICE_SCRIPTS_KEY: Final[str] = "SPICE_SCRIPTS"
"""Environment variable name used by ngspice for locating helper scripts."""


# =============================================================================
# ngspice template rewriting regex patterns
# =============================================================================

_WRITE_RE = re.compile(r"(?im)^\s*(write|wrdata)\s+(\S+)\s+(.*)$")
"""Match ngspice write commands that produce output files.

Groups
------
1) command   : 'write' or 'wrdata'
2) path token: output path (no whitespace)
3) rhs       : remaining tokens (signals/expressions)
"""

_NOISE_WRITE_BARE_RE = re.compile(r"(?im)^\s*(write|wrdata)\s+(\S+)\s+NoiseFigure\s*$")
"""Match noise write lines that contain only `NoiseFigure` and omit frequency.

This is used by `_ensure_noise_writes_frequency(...)` to patch such lines into:
    <cmd> <path> frequency NoiseFigure
"""

_RE_LIB = re.compile(r"(?im)^\s*\.lib\s+(\S+)\s+(\S+)\s*$")
"""Match `.lib <path> <corner>` lines in templates.

Groups
------
1) lib path : path token
2) corner   : corner token
"""
