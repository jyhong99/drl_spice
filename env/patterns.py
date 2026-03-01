from __future__ import annotations

"""Core environment constants used across env modules."""

# Constraint / evaluation keys
U_DIM = 2

F_KEY = "f"
VIOL_KEY = "viol"
VIOL_TERM_KEY = "viol_term"
SAT_KEY = "satisfied_perf"
FEASIBLE_KEY = "feasible"
U_KEY = "u"
NONCONV_KEY = "non_convergent"
UNSTABLE_KEY = "unstable"
INVALID_KEY = "invalid"
METRICS_KEY = "metrics"
SPEC_KEY = "spec"
REF_KEY = "ref"
PERF_KEYS_KEY = "perf_keys"
ENABLE_LIN_KEY = "enable_linearity"
K_MIN_KEY = "K_min"

# Objective constants
MIN_OBJ_LEN = 5
IIP3_INDEX = 5
DEFAULT_BAD_FOM = -1e6

J_VIOL_KEY = "J_viol"
J_FOM_KEY = "J_fom"
FOM_DB_KEY = "FOM_dB"
VAR_F_KEY = "var_f"

# Reward keys
REWARD_KEY = "reward"
R_VIOL_KEY = "r_viol"
R_FOM_KEY = "r_fom"
R_VAR_KEY = "r_var"
TJ_VIOL_KEY = "tJ_viol"
TJ_FOM_KEY = "tJ_fom"
D_TJ_VIOL_PBRS_KEY = "d_tJ_viol_pbrs"
D_TJ_FOM_PBRS_KEY = "d_tJ_fom_pbrs"

# Termination / reset
DONE_RUNNING = "running"
DONE_TIME_LIMIT = "time_limit"
DONE_NON_CONV = "non_convergent"
RESET_RANDOM = "random"
RESET_CONTINUE = "continue_last"

# Info keys
EVAL_KEY = "eval"
SIM_KEY = "sim"
X_PREV_KEY = "x_prev"
X_NEXT_KEY = "x_next"
ENV_ID_KEY = "env_id"
X_KEY = "x"
NEXT_RESET_MODE_KEY = "next_reset_mode"
EPISODE_KEY = "episode"
STEP_KEY = "step"
ACTION_KEY = "action"
OBS_KEY = "obs"
TERMINATED_KEY = "terminated"
TRUNCATED_KEY = "truncated"

# Analysis keys
SPARAM_KEY = "sp"
NOISE_KEY = "noise"
LINEARITY_KEY = "linearity"

__all__ = [k for k in list(globals()) if k.isupper()]
