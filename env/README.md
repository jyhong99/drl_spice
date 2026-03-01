# `env`: Modular RL Environment for LNA Design Optimization

This package provides a Gym/Gymnasium-compatible reinforcement learning environment that optimizes analog/RF LNA designs through an injected simulator backend (typically ngspice).

The environment is intentionally modular:
- action dynamics (`env/action`)
- normalized-to-physical parameter decoding (`env/action/decoder.py`)
- constraint/objective/reward pipeline (`env/reward`)
- observation assembly (`env/observation`)
- episode termination and reset policy (`env/termination`)
- high-level environment wiring (`env/factory.py`)

---

## 1. Core Concepts

The environment state is a normalized parameter vector:

- `x in [0, 1]^m` where `m = number of design variables`

Per step, the policy emits continuous action `a` (default bounds `[-1, 1]^m`), and the action model updates:

- `x_{t+1} = clip(x_t + eta * a_t, clip_low, clip_high)`

`x` is decoded into physical circuit parameters via `SpecDecoder`, then passed to a simulator:

- `params = decoder.make_design_config(x)`
- `simulator.simulate(design_variables_config=params, analyses=..., enable_linearity=...)`

Simulator outputs are evaluated through the reward pipeline:

- constraints (`f`, `viol`, `viol_term`, feasibility/invalid flags)
- objective (`J_fom = -FoM`)
- reward shaping and final scalar reward

Observation is assembled as:

- `obs = [f, x, u]`

where `u` is a 2D invalidity indicator (`U_DIM = 2`):

- `u[0]`: non-convergent
- `u[1]`: unstable

---

## 2. Quick Start

### 2.1 Minimal usage with a simulator stub

```python
import numpy as np
from env.factory import build_lna_env
from env.specs import perf_metric_order
from env.types import SimResult, SimStatus


class DummySimulator:
    def __init__(self, n_perf: int):
        self.n_perf = int(n_perf)

    def simulate(self, *, design_variables_config, analyses, enable_linearity):
        perf = np.zeros((self.n_perf,), dtype=float)
        aux = {"K_min": 2.0}
        return SimResult(
            performances=perf,
            aux=aux,
            status=SimStatus.OK,
            detail="",
            run_dir="/tmp",
        )


n_perf = len(perf_metric_order(enable_linearity=False))
env = build_lna_env(
    circuit_type="CS",
    max_steps=20,
    simulator=DummySimulator(n_perf=n_perf),
    seed=0,
)

obs, info = env.reset(seed=123)
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        # recommended: obey env-provided reset mode
        reset_mode = info.get("next_reset_mode", "random")
        obs, info = env.reset(options={"reset_mode": reset_mode})
```

### 2.2 Build from project entrypoint (already in `main.py`)

Project root includes smoke tests:

```bash
python main.py --quick-test-env --circuit-type CS
python main.py --quick-test-ngspice --circuit-type CS
```

---

## 3. Public API

Primary entry points:

- `env.build_lna_env(...)`
- `env.LNAEnvModular`

Common direct imports:

- `from env.reward import ConstraintModel, ObjectiveModel, RewardModel, RewardPipeline`
- `from env.action import ActionModel, SpecDecoder`
- `from env.observation import ObservationModel`
- `from env.termination import TerminationModel, ResetRule`

Lazy exports are defined in `env/__init__.py`.

---

## 4. Environment Construction (`build_lna_env`)

`build_lna_env(...)` wires all subcomponents in this order:

1. Normalize analyses (force linearity analysis if enabled).
2. Build `SpecDecoder` and infer `m_param`.
3. Build `ActionModel` (`eta` defaults to `2.0 / max_steps` when omitted).
4. Build `ConstraintModel`, `ObjectiveModel`, `RewardModel`, `RewardPipeline`.
5. Build `ObservationModel`, `TerminationModel`, `ResetRule`.
6. Assemble `EnvConfig` and construct `LNAEnvModular`.

Key constructor arguments:

- Environment/runtime:
  - `circuit_type` (`"CS"` or `"CGCS"`)
  - `max_steps`
  - `simulator` (must expose callable `simulate(...)`)
  - `seed`, `env_id`
- Action update:
  - `eta`, `action_mode="delta"`
  - `clip_low/high` for state, `action_low/high` for action
  - `clip_action`, `action_eps`, `x_tol`
- Decoder:
  - `max_param` (normalized upper bound)
  - `decoder_clip_x`
- Constraints/objective/reward shaping:
  - `constraint_p`, `constraint_w`, `use_k`, `k_min`
  - `eps_denom`, `eps_viol0`, `eps_wsum`, `objective_eps`
  - `gamma`, `beta`
  - `lambda_viol`, `lambda_fom`, `lambda_var`
  - `kappa_viol`, `kappa_fom`
- Reset behavior:
  - `p_reset` (random reset probability after time limit)

---

## 5. Step/Reset Semantics

### 5.1 `reset(seed=None, options=None)`

- Increments episode counter.
- Resets internal state for reward/observation/termination modules.
- `options["reset_mode"] == "continue_last"` keeps previous `x`.
- Any other reset mode samples new random `x ~ Uniform([0,1]^m)`.
- Evaluates simulator/pipeline at initial `x` and returns initial observation.

### 5.2 `step(action)`

- Applies action model to get `x_next`.
- Runs simulator and pipeline at `x_next`.
- Computes `reward = eval_out["reward"]`.
- Checks done reason:
  - non-convergent first
  - then time limit (`step >= max_steps`)
- Returns Gymnasium tuple:
  - `(obs, reward, terminated, truncated, info)`

Current behavior:

- `terminated` is used for done conditions.
- `truncated` is always `False` (reserved for future split semantics).

---

## 6. Observation Contract

`ObservationModel` outputs flat `float32` vector:

- `obs = concat(f, x, u)`
- `len(obs) = n_perf + m_param + U_DIM`
- `U_DIM = 2`

Where:

- `f`: normalized satisfaction vector from constraints
- `x`: normalized design state
- `u`: `[non_convergent, unstable]`

`ObservationModel.dim(n_perf, m_param)` returns exact observation size.

---

## 7. Constraint / Objective / Reward Details

### 7.1 Constraint model (`ConstraintModel.compute`)

Given performance vector `o`, spec vector `spec`, reference vector `ref`:

- `f = (o - ref) / safe_denom(spec - ref)`
- `viol = max(0, 1 - f)` (element-wise)

Aggregate violation term:

- `viol_term = ((sum_i w_i * viol_i^p) / (sum_i w_i + eps_wsum))^(1/p)`

Flags:

- `satisfied_perf`: all `viol <= eps_viol0`
- `unstable`: true if `use_k=True`, `K_min` exists, and `K_min < k_min` (or non-finite)
- `non_convergent`: provided externally or inferred from non-finite performances
- `invalid`: non-convergent or unstable
- `feasible`: satisfied_perf and not invalid

Also returns `metrics`, `spec`, `ref`, `perf_keys`, linearity flag, and indicator `u`.

### 7.2 Objective model (`ObjectiveModel`)

- Computes FoM in dB from base ordered metrics:
  - `[S11_dB, S21_dB, S22_dB, NF_dB, PD_mW]`
- Optional IIP3 bonus if linearity metric is present.
- Returns objective cost as `J_fom = -FoM`.
- Returns `DEFAULT_BAD_FOM` for malformed/non-finite required inputs.

### 7.3 Reward model (`RewardModel`)

Inputs:

- `J_viol`, `J_fom`, `var_f`, `invalid`

Invalid transitions receive additive penalties:

- `tJ_viol = J_viol + kappa_viol`
- `tJ_fom = J_fom + kappa_fom`

PBRS-like deltas:

- `d_tJ = -gamma * tJ_t + tJ_{t-1}`

Per-component rewards:

- `r_component = -beta * tJ + (1 - beta) * d_tJ`
- `r_var = -var_f`

Final reward:

- `reward = lambda_viol * r_viol + lambda_fom * r_fom + lambda_var * r_var`

`RewardPipeline.evaluate(...)` returns a merged dictionary containing all intermediate and final values.

---

## 8. Performance Specs and Metric Ordering

Defined in `env/specs.py`.

Base order:

- `("S11_dB", "S21_dB", "S22_dB", "NF_dB", "PD_mW")`

Linearity-enabled order:

- base order + `"IIP3_dBm"`

Supported circuit specs:

- `CS`
- `CGCS`

Fetch helpers:

- `perf_metric_order(enable_linearity=...)`
- `get_perf_spec(circuit_type=..., enable_linearity=...)`

---

## 9. Simulator Interface Contract

The environment depends on `SimulatorProtocol`:

```python
simulate(
    *,
    design_variables_config: dict[str, float],
    analyses: tuple[str, ...],
    enable_linearity: bool,
) -> SimResult
```

Expected `SimResult` fields:

- `performances`: array-like or `None`
- `aux`: `dict[str, float]`
- `status`: `SimStatus` enum (`ok`, `timeout`, `parse_fail`, ...)
- `detail`: str
- `run_dir`: str

Non-convergence in the environment is triggered when any of the following is true:

- `sim.status != SimStatus.OK`
- `performances is None`
- `performances` length mismatches expected metric count
- `cfg.require_k=True` and `"K_min"` missing from aux

---

## 10. Info Dictionary Schema

`reset` and `step` return an `info` dict with structured keys from `env/patterns.py`.

Common keys:

- `episode`, `step`, `env_id`
- `action`, `x_prev`, `x_next`, `x`
- `reward`, `terminated`, `truncated`
- `obs`
- `sim`: simulator metadata (`status`, `detail`, `run_dir`, `params`)
- `eval`: full merged reward-pipeline output
- `next_reset_mode`: one of `"random"` or `"continue_last"`
- `done_reason` (on step): `"running"`, `"time_limit"`, `"non_convergent"`

Recommended episode loop pattern:

- when `terminated or truncated`:
  - read `next_reset_mode` from `info`
  - call `reset(options={"reset_mode": next_reset_mode})`

---

## 11. Action and Parameter Decoding

### 11.1 Action model

`ActionModel` validates shapes and finiteness, optionally clips actions, validates current `x` bounds, then applies delta update with projection back to `[clip_low, clip_high]`.

### 11.2 Spec decoder

`SpecDecoder` maps normalized coordinates to physical parameter values using per-parameter scale:

- linear interpolation for `scale="linear"`
- log10 interpolation for `scale="log"` (positive bounds required)

Optional post-processing per parameter:

- significant-digit rounding (`round_sig_k`)
- step snapping (`step`)

Parameter domains are loaded from `PARAM_DOMAIN_REGISTRY`.

---

## 12. Termination and Reset Policy

### 12.1 Termination (`TerminationModel.check`)

Priority order:

1. if `non_convergent`: done with reason `"non_convergent"`
2. else if `step >= max_steps`: done with reason `"time_limit"`
3. else running

### 12.2 Reset rule (`ResetRule.decide`)

- `non_convergent` -> always `"random"`
- `time_limit` -> `"random"` with probability `p_reset`, else `"continue_last"`
- `running` -> `"continue_last"`

---

## 13. Gym/Gymnasium Compatibility

`env/modular.py` tries `gymnasium` first and falls back to `gym`.

- action space: `Box(low=-1, high=1, shape=(m_param,), dtype=float32)`
- observation space: unbounded `Box(shape=(obs_dim,), dtype=float32)`

The API follows Gymnasium-style return signatures.

---

## 14. Integration with ngspice

The default simulator in this repository is `ngspice.simulator.NgspiceSimulator`.

Helpful utilities in `env/utils/factory_utils.py`:

- `_resolve_templates(...)` for netlist template resolution/validation
- `_prepare_spice_scripts_env(...)` for `SPICE_SCRIPTS` env wiring
- `_normalize_analyses(...)` for analysis list normalization

Default template mappings exist for:

- `CS`: `ngspice/templates/cs_lna/*.spice`
- `CGCS`: `ngspice/templates/cgcs_lna/*.spice`

---

## 15. Common Failure Modes and Checks

- `TypeError: simulator must provide a callable simulate(...) method`
  - Ensure simulator object exposes `simulate(...)` with expected keyword args.
- `performances` shape mismatch
  - Ensure simulator returns vector aligned with `perf_metric_order(...)`.
- missing linearity template when linearity is enabled
  - Provide template for `linearity` analysis.
- unstable/non-convergent episodes dominating training
  - Inspect `info["eval"]` keys: `non_convergent`, `unstable`, `invalid`, `viol_term`.

---

## 16. File Map

- `env/factory.py`: builder API for full env wiring
- `env/modular.py`: `LNAEnvModular` implementation
- `env/action/action.py`: action transition model
- `env/action/decoder.py`: normalized-to-physical parameter decoder
- `env/observation/observation.py`: observation assembly
- `env/reward/objective.py`: constraints + FoM objective
- `env/reward/reward.py`: reward model + pipeline orchestrator
- `env/termination/termination.py`: done checks + reset rules
- `env/specs.py`: metric ordering and per-circuit performance specs
- `env/patterns.py`: stable keys/constants
- `env/types.py`: shared dataclasses/protocols/aliases

---

## 17. Extension Guidelines

To extend this environment safely:

- Add a new circuit type:
  - register parameter domain in `ngspice.specs`
  - add performance spec/reference in `env/specs.py`
  - ensure simulator returns performance vector in the exact metric order
- Add new metrics:
  - update metric order constants and spec registries
  - verify `ObjectiveModel` and downstream logging assumptions
- Customize reward shaping:
  - keep `RewardModel.compute(...)` outputs backward-compatible with existing info/logging keys

