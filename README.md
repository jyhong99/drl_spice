# drl_spice

`drl_spice` is a modular analog/RF circuit optimization codebase centered on two core packages:

- `ngspice`: simulation, netlist patching, execution, and result parsing
- `env`: Gym/Gymnasium-compatible RL environment that wraps a simulator backend

This README intentionally excludes documentation for:

- `rllib/`
- `main.py`

If you need those later, they can be documented separately.

---

## What This Repository Provides

At a high level, the workflow is:

1. Build a normalized state/action RL environment (`env`).
2. Decode normalized design vectors into physical parameters.
3. Simulate those parameters with ngspice (`ngspice`).
4. Parse simulation outputs into canonical performance metrics.
5. Compute constraints/objective/reward and return RL observations.

The design emphasizes:

- strict schema checks (fail fast on malformed outputs)
- explicit error taxonomy
- marker-based netlist patching (predictable templating)
- modularity (you can swap simulator implementations if they satisfy protocol)

---

## Repository Structure (Documented Scope)

- `env/`
  - Modular environment stack (`action`, `observation`, `reward`, `termination`)
  - Environment wiring (`factory.py`, `modular.py`)
  - Environment-level specs and shared keys/constants
- `ngspice/`
  - `NgspiceSimulator` orchestration
  - subprocess kernel wrapper
  - netlist patcher/designer/circuit parser
  - analysis reader registry + built-in readers
  - RAW-ascii parser and utilities
  - specs/types/patterns/errors

Other paths are intentionally omitted from this README.

---

## Quick Start (Without `main.py`)

### 1) Create an ngspice simulator instance

```python
from pathlib import Path
from ngspice.simulator import NgspiceSimulator

sim = NgspiceSimulator(
    circuit_type="CS",
    netlist_templates={
        "sp": Path("ngspice/templates/cs_lna/cs_lna_sparam.spice"),
        "noise": Path("ngspice/templates/cs_lna/cs_lna_nf.spice"),
        "linearity": Path("ngspice/templates/cs_lna/cs_lna_fft.spice"),
    },
    work_root=Path("./work_ngspice"),
    sky130_lib_path=Path("/path/to/sky130.lib.spice"),
    corner="tt",
    ngspice_bin="ngspice",
    timeout_sec=30.0,
)
```

### 2) Build environment around that simulator

```python
from env.factory import build_lna_env

env = build_lna_env(
    circuit_type="CS",
    max_steps=200,
    simulator=sim,
    seed=0,
    enable_linearity=False,
)
```

### 3) Run one rollout loop

```python
obs, info = env.reset(seed=123)
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        reset_mode = info.get("next_reset_mode", "random")
        obs, info = env.reset(options={"reset_mode": reset_mode})
```

---

## Core Contracts

### Simulator Protocol (used by `env`)

Any simulator can be injected into `env` if it implements:

```python
simulate(
    *,
    design_variables_config: dict[str, float],
    analyses: tuple[str, ...],
    enable_linearity: bool,
) -> SimResult
```

`NgspiceSimulator` already satisfies this protocol.

### Environment Interface

`LNAEnvModular` follows Gymnasium-style API:

- `reset(...) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

Observation layout is deterministic:

- `obs = [f, x, u]`
  - `f`: normalized satisfaction vector
  - `x`: normalized design state
  - `u`: invalidity indicators (`non_convergent`, `unstable`)

---

## `env` Package Summary

### Main modules

- `env/factory.py`
  - `build_lna_env(...)` high-level builder
- `env/modular.py`
  - `LNAEnvModular` environment implementation
- `env/action/`
  - `ActionModel`: state transition in normalized space
  - `SpecDecoder`: normalized-to-physical parameter decoding
- `env/reward/`
  - `ConstraintModel`, `ObjectiveModel`, `RewardModel`, `RewardPipeline`
- `env/observation/`
  - `ObservationModel` (`obs = [f, x, u]`)
- `env/termination/`
  - `TerminationModel`, `ResetRule`

### Metric/spec configuration

- `env/specs.py` defines performance metric ordering and per-circuit targets/references.
- Supported circuit families: `CS`, `CGCS`.

### Key behavior

- action update (default mode):
  - `x_next = clip(x + eta * action, clip_low, clip_high)`
- reward pipeline computes constraints + FoM-derived objective + shaped reward
- done condition prioritizes non-convergence, then time limit

For full detail see [env/README.md](/home/jyhong/projects/drl_spice/env/README.md).

---

## `ngspice` Package Summary

### Main modules

- `ngspice/simulator.py`
  - `NgspiceSimulator` orchestration class
- `ngspice/kernel.py`
  - `Kernel` subprocess wrapper (`ngspice -b ...`)
- `ngspice/netlist/`
  - marker-based patching and circuit parsing (`Designer`, `Circuit`)
- `ngspice/analysis/`
  - reader registry/factory + concrete readers
- `ngspice/ascii.py`
  - RAW-ascii parser to `DataFrame`

### Built-in readers

Registered analysis readers include:

- `S-Parameter Analysis`
- `Noise Analysis`
- `Linearity`
- `Stability Factor`
- passthrough table readers (`Transient`, `DC_Operating_Point`, `AC_simulation`)

### Simulation result contract

`SimResult` includes:

- `performances`
- `aux`
- `status` (`SimStatus`)
- `detail`
- `run_dir`

### Template/marker conventions

Templates are expected to contain marker blocks for design variables and knobs, plus a replaceable Sky130 `.lib` line. Output write paths are rewritten to per-run directories.

For full detail see [ngspice/README.md](/home/jyhong/projects/drl_spice/ngspice/README.md).

---

## Error Handling Model

Both packages use explicit, typed errors.

- ngspice execution/parsing errors are structured in `ngspice/errors.py`
  - execution (`NgspiceTimeout`, `NgspiceMissingOutputs`, ...)
  - parsing/schema (`RawParseError`, `MissingColumnError`, ...)
  - netlist patching/marker errors
- simulator-level orchestration generally maps failures into `SimStatus`
- environment-level logic interprets non-OK simulation outcomes as invalid/non-convergent transitions

This separation makes debugging reproducible and avoids silent failure modes.

---

## Supported Circuits and Analyses

### Circuit types

- `CS`
- `CGCS`

### Common analysis keys

- `"sp"`
- `"noise"`
- `"linearity"`

Linearity can be enabled/disabled at environment/simulator call time.

---

## Suggested Read Order

1. [ngspice/README.md](/home/jyhong/projects/drl_spice/ngspice/README.md)
2. [env/README.md](/home/jyhong/projects/drl_spice/env/README.md)
3. Source files in `env/factory.py`, `env/modular.py`, `ngspice/simulator.py`

This order helps because `env` depends on simulator contracts defined by `ngspice`.

---

## Notes on Scope

This root README intentionally does not describe:

- training algorithms and infrastructure under `rllib/`
- CLI/runtime orchestration in `main.py`

