# `ngspice`: Simulation and Parsing Toolkit for DRL LNA Optimization

This package is the simulation backend used by the DRL environment in this repository.
It provides:

- netlist template patching and rendering
- ngspice batch execution with robust error mapping
- RAW-ascii/table parsing into `pandas.DataFrame`
- analysis-specific reader registry (S-parameters, noise, linearity, stability, etc.)
- a high-level `NgspiceSimulator` that returns normalized `SimResult`

The code is built to be strict about schemas and explicit about failures so RL loops do not silently train on bad data.

---

## 1. Package Layout

- `ngspice/simulator.py`
  - `NgspiceSimulator` (main orchestration class)
- `ngspice/kernel.py`
  - `Kernel` subprocess wrapper for batch ngspice execution
- `ngspice/netlist/`
  - `Circuit`: parse rendered netlists and infer mappings
  - `Designer`: patch marker-delimited netlist blocks
  - `patcher.py`: marker-based text patch operations
- `ngspice/analysis/`
  - `registry.py`: analysis reader registration/factory
  - `reader.py`: retry-capable facade
  - `readers/`: concrete analysis parsers
- `ngspice/ascii.py`
  - RAW-ascii parser (`read_raw_table`)
- `ngspice/specs.py`
  - registries for circuit names, parameter domains, default knobs
- `ngspice/patterns.py`
  - regex primitives, marker strings, analysis keys
- `ngspice/utils.py`
  - schema helpers, frequency selection, netlist rewrite helpers, knob coercion
- `ngspice/errors.py`
  - structured exception hierarchy
- `ngspice/textio.py`
  - safe text read/write helpers (`read_text`, `atomic_write`)
- `ngspice/types.py`
  - shared dataclasses/enums/type aliases (`SimResult`, `SimStatus`, `ParamSpec`, ...)

---

## 2. End-to-End Execution Flow

`NgspiceSimulator.simulate(...)` runs this pipeline:

1. Validate circuit type against registries.
2. Merge design variables with circuit fixed values.
3. Create unique run directory under `work_root`.
4. Resolve target frequency from LET knobs.
5. For each requested analysis (`sp`, `noise`, optional `linearity`):
   - load template netlist
   - rewrite output paths into run directory
   - patch `.lib` line for Sky130 library path/corner
   - (noise only) enforce writing `frequency NoiseFigure`
   - patch design variables/knobs into marker blocks (`Designer`)
   - run ngspice via `Kernel.run(...)`
   - parse outputs via registered analysis readers
6. Merge parsed metrics into canonical performance vector:
   - `[S11_dB, S21_dB, S22_dB, NF_dB, PD_mW, (optional) IIP3_dBm]`
7. Return `SimResult(status=ok, performances=..., aux=..., run_dir=...)`.
8. Prune old run directories according to `max_run_dirs`.

Any failure is mapped to `SimStatus` instead of propagating raw exceptions in normal simulator usage.

---

## 3. High-Level API (`NgspiceSimulator`)

### 3.1 Constructor

```python
NgspiceSimulator(
    *,
    circuit_type: str,
    netlist_templates: dict[str, PathLike],
    work_root: PathLike,
    sky130_lib_path: PathLike,
    corner: str,
    ngspice_bin: str = "ngspice",
    timeout_sec: float = 30.0,
    kernel_env: Optional[dict[str, str]] = None,
    max_run_dirs: int = 10,
)
```

Important notes:

- `circuit_type` must exist in all registries:
  - `CIRCUIT_NAMES_REGISTRY`
  - `ANALYSIS_KNOB_REGISTRY`
  - `LET_KNOB_REGISTRY`
  - `PARAM_DOMAIN_REGISTRY`
- templates must include analysis keys used at runtime (`sp`, `noise`, optionally `linearity`).
- `work_root` is auto-created.
- `kernel_env` is merged into subprocess env (useful for `SPICE_SCRIPTS`, PDK paths).

### 3.2 `simulate(...)`

```python
simulate(
    *,
    design_variables_config: dict[str, float],
    analyses: tuple[str, ...] = ("sp", "noise"),
    enable_linearity: bool = False,
) -> SimResult
```

Returns `SimResult` with:

- `performances`: `np.ndarray` or `None`
- `aux`: `dict[str, float]` (currently often empty)
- `status`: `SimStatus`
- `detail`: short reason string
- `run_dir`: path string

Failure mapping inside simulator:

- timeout -> `SimStatus.TIMEOUT`
- ngspice executable missing or non-zero exit -> `SimStatus.NGSPICE_FAIL`
- expected outputs missing / files missing -> `SimStatus.NO_OUTPUT`
- validation issues (e.g., bad analysis key, invalid `target_frequency`) -> `SimStatus.VALIDATION`
- unknown exception -> `SimStatus.UNKNOWN`

---

## 4. Type Contracts

From `ngspice/types.py`:

- `SimStatus` enum values:
  - `ok`, `timeout`, `ngspice_fail`, `no_output`, `parse_fail`, `nan`, `validation`, `unknown`
- `SimResult`:
  - `performances: Optional[Any]`
  - `aux: Dict[str, float]`
  - `status: SimStatus`
  - `detail: str`
  - `run_dir: str`
- `NgspiceRunResult` (from `Kernel.run`):
  - `returncode`, `stdout`, `stderr`, `elapsed_sec`, `cmd`, `cwd`
- `ParamDomain`:
  - `circuit_type`, ordered `keys`, `specs`, `fixed_values_default`
- `AnalysisKnobSet`:
  - circuit-specific defaults for simulation knobs

---

## 5. Registries in `specs.py`

### 5.1 Circuit names registry

`CIRCUIT_NAMES_REGISTRY` defines:

- `designvar_names`: names inserted into netlist blocks
- `device_names`: instance names used for circuit mapping

Supported circuits in this repo:

- `CS`
- `CGCS`

### 5.2 Parameter domains

`PARAM_DOMAIN_REGISTRY` defines per-circuit parameter bounds and scales:

- each parameter has `p_min`, `p_max`, `scale` (`linear`/`log`), optional rounding and step
- fixed defaults shared via `COMMON_FIXED_VALUES_DEFAULT`, including:
  - `v_dd`, `r_b`, `c_1`, `l_m`

### 5.3 Knob registries

- `ANALYSIS_KNOB_REGISTRY`: analysis runtime knobs (`reltol`, `abstol`, transient settings, two-tone setup)
- `LET_KNOB_REGISTRY`: helper knobs (`target_frequency`, sweep controls, `Q_factor`, etc.)

Accessors:

- `get_param_domain(circuit_type=...)`
- `get_circuit_names(circuit_type=...)`
- `get_analysis_knobs(circuit_type=...)`
- `get_let_knobs(circuit_type=...)`

---

## 6. Template and Marker Conventions

Marker strings (from `patterns.py`) must exist in templates for patching:

- `** design_variables_start`
- `** design_variables_end`
- `** analysis_knobs_start`
- `** analysis_knobs_end`
- `** let_knobs_start`
- `** let_knobs_end`

Device parsing in `Circuit` uses boundary:

- starts at `.subckt` line
- ends at `**** begin user architecture code`

`.lib` replacement requirement:

- templates must contain a Sky130 line like `.lib ... sky130.lib.spice tt`
- `_set_sky130_lib(...)` replaces it with configured path/corner

Output path rewrite:

- `_redirect_write_paths(...)` rewrites `write`/`wrdata` destinations into `run_dir`

Noise command normalization:

- `_ensure_noise_writes_frequency(...)` rewrites bare `NoiseFigure` output to include frequency column:
  - `write <path> NoiseFigure`
  - -> `write <path> frequency NoiseFigure`

---

## 7. Kernel Execution (`kernel.py`)

`Kernel.run(...)` invokes ngspice batch mode:

```python
cmd = (ngspice_bin, "-b", netlist_path)
```

Features:

- timeout enforcement
- captured stdout/stderr
- optional log files (`ngspice_stdout.txt`, `ngspice_stderr.txt`)
- expected output validation
- domain-specific exceptions with `KernelContext`

Exception mapping in kernel layer:

- missing netlist -> `KernelNetlistNotFound`
- executable missing -> `NgspiceExecutableNotFound`
- timeout -> `NgspiceTimeout`
- expected files missing -> `NgspiceMissingOutputs`
- non-zero exit with `check=True` -> `NgspiceNonZeroExit`

`KernelContext` carries debug data (`cmd`, `cwd`, netlist, stdout/stderr, elapsed) and has `tail(limit)` helper for concise diagnostics.

---

## 8. Analysis Reader Registry and Factory

### 8.1 Registry behavior

`analysis/registry.py` stores:

- `_REGISTRY: Dict[str, Callable[[], AnalysisReader]]`

Use decorator:

```python
@register("Noise Analysis")
def _factory():
    return NoiseReader()
```

Lookup:

- `create_reader(analysis_type)`
- `supported()`

Errors:

- duplicate key -> `DuplicateAnalysisType`
- unknown key -> `UnknownAnalysisType`
- factory creation failure -> `ReaderFactoryError`

### 8.2 Facade with retry

`analysis/reader.py` exposes `Reader(analysis_type)` wrapper with retry policy:

- default retryable exceptions include:
  - `FileNotFoundError`
  - `pandas.errors.EmptyDataError`
  - `RawParseError`
  - `IndexError`
- configurable per call via:
  - `max_retries`
  - `sleep_sec`
  - `retry_on`

On exhaustion:

- raises `ReaderFailedAfterRetries`

---

## 9. Built-In Analysis Readers

### 9.1 `SParameterReader` (`"S-Parameter Analysis"`)

- Input: RAW-ascii table
- Produces canonical dB columns:
  - `S11_dB`, `S12_dB`, `S21_dB`, `S22_dB`
- Requires at least:
  - `S11_dB`, `S21_dB`, `S22_dB`
- Supports kwargs:
  - `target_frequency` (default `2.4e9`)
  - `return_full` (default `False`)
- Selection:
  - nearest frequency bin via `nearest_index(...)`

### 9.2 `NoiseReader` (`"Noise Analysis"`)

- Input: RAW-ascii table
- Normalizes/infers `NoiseFigure` column
- Strict about frequency and schema
- Supports kwargs:
  - `target_frequency`
  - `return_full`

### 9.3 `LinearityReader` (`"Linearity"`)

- Parses `Values:` section directly from text
- Expects alternating lines:
  - frequency line
  - complex output line
- Computes two-tone metrics:
  - `Pin_dBm`
  - `Pfund_dBm_avg`
  - `Pim3_dBm_avg`
  - `IIP3_dBm = Pin + (Pfund_avg - Pim3_avg)/2`
- Supported kwargs:
  - `Vin_amp`, `f1`, `f2`, `R_load`

### 9.4 `StabilityReader` (`"Stability Factor"`)

- Reads S-parameters and computes:
  - `K_min`
  - `mu_min`
  - `mup_min`
  - `mu_mup_min`
  - `bandwidth_hz` (3 dB from |S21|)

### 9.5 `RawTableReader` (generic)

Registered as:

- `"Transient"`
- `"DC_Operating_Point"`
- `"AC_simulation"`

Returns parsed table unchanged (strictly rejects unsupported kwargs).

---

## 10. RAW-ASCII Parsing (`ascii.py`)

`read_raw_table(path)` steps:

1. Read file with `read_text(...)`.
2. Parse headers:
   - `No. Variables`
   - `No. Points`
3. Parse variable block between `Variables` and `Values`.
4. Parse value lines with support for:
   - indexed rows
   - non-indexed fallback
   - real and complex values
5. Convert to arrays (`float64` when imag ~ 0, else `complex128`).
6. Return `DataFrame` with consistency check (`n_pts == len(df)`).

Main parser error:

- `RawParseError`

---

## 11. Netlist Components

### 11.1 `Circuit`

`Circuit` loads a rendered netlist and builds:

- `dsgnvar_to_val`: design var -> numeric value
- `dvc_to_dsgnvar`: device attribute -> variable name
- `dvc_to_val`: device attribute -> numeric value

Parsing logic relies on marker boundaries and regex patterns from `patterns.py`.

Useful methods:

- `refresh()` (recompute maps from in-memory text)
- `reload()` (re-read from disk + recompute)
- `get_designvar_to_val()`

### 11.2 `Designer`

`Designer.design_circuit(...)` patches marker blocks in-place:

- design variables -> `.param k = v`
- analysis knobs -> `.param k = v`
- let knobs -> `let k = v`

Uses `atomic_write(...)` for safe writeback and refreshes target circuit mappings.

Optional strict count checks:

- `DesignVarCountMismatch`
- `TargetNetlistDesignVarCountMismatch`

### 11.3 `patcher.patch_block(...)`

General block replacement utility:

- locate marker start/end
- render lines with user-supplied line builder
- replace first matching block
- deterministic key sort option

Errors:

- `MarkerNotFound`
- `MarkerMalformed`

---

## 12. Utility Highlights (`utils.py`)

DataFrame/schema helpers:

- `pick_col(df, *names)`
- `find_freq_col(df, candidates=...)`
- `nearest_index(x, target)`
- `attach_nearest_meta(...)`
- `single_row_with_meta(...)`

Netlist rewriting:

- `_redirect_write_paths(netlist_text, run_dir)`
- `_ensure_noise_writes_frequency(netlist_text)`
- `_set_sky130_lib(netlist_text, lib_path, corner)`

Knob normalization/coercion:

- `_coerce_float_knob(...)`
  - supports SPICE suffixes: `G`, `M`, `meg`, `k`, `m`, `u`, `n`, `p`, etc.
- `_normalize_knobs(...)`
  - synchronizes `Q_factor` and `q_factor` aliases

---

## 13. Error Taxonomy (`errors.py`)

Major categories:

- RAW/schema/parsing:
  - `RawParseError`, `DataSchemaError`, `MissingColumnError`
- execution/kernel:
  - `NgspiceError`, `KernelError`, `NgspiceTimeout`, `NgspiceMissingOutputs`, ...
- text I/O:
  - `TextIOError`, `TextFileNotFound`, `TextReadError`, `TextWriteError`
- netlist/circuit:
  - `CircuitError`, `NetlistMarkerError`, `NetlistFormatError`
- designer/patching:
  - `DesignerError`, `NetlistPatchError`, `MarkerMalformed`, ...
- reader/registry:
  - `ReaderError`, `ReaderFailedAfterRetries`, `UnknownAnalysisType`, ...

The hierarchy is designed so callers can catch broad categories or specific failure modes.

---

## 14. Analysis Keys and Defaults

From `patterns.py`:

- `SPARAM_KEY = "sp"`
- `NOISE_KEY = "noise"`
- `LINEARITY_KEY = "linearity"`

`NgspiceSimulator.simulate(...)` defaults to:

- analyses: `(sp, noise)`
- linearity disabled unless `enable_linearity=True`

When linearity is enabled and missing from `analyses`, it is auto-appended.

---

## 15. Output Artifact Expectations

Collectors expect these files in each run directory:

- S-parameter collector (`_collect_sp`):
  - `S_Param_Bandwidth.csv`
  - optional `DC_OP.csv` (for `PD_mW` estimate)
- noise collector (`_collect_noise`):
  - `NoiseFigure.csv`
- linearity collector (`_collect_linearity`):
  - default `FFT.csv` (overridable via knob `linearity_result_file`)

Linearity expected-output checking is explicit in simulator for this result file.

---

## 16. Quick Usage Examples

### 16.1 Build and run simulator directly

```python
from pathlib import Path
from ngspice.simulator import NgspiceSimulator
from ngspice.patterns import SPARAM_KEY, NOISE_KEY

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

res = sim.simulate(
    design_variables_config={
        "v_b": 0.85,
        "r_d": 120.0,
        "l_d": 2e-9,
        "l_g": 2e-9,
        "l_s": 1e-10,
        "c_d": 8e-13,
        "c_ex": 3e-14,
        "w_m1": 20.0,
        "w_m2": 25.0,
    },
    analyses=(SPARAM_KEY, NOISE_KEY),
    enable_linearity=False,
)

print(res.status.value, res.detail)
print(res.run_dir)
print(res.performances)
```

### 16.2 Use analysis facade with retry

```python
from ngspice.analysis import Reader

r = Reader("Noise Analysis")
df = r.read(
    "work/run_xxx/NoiseFigure.csv",
    target_frequency=2.4e9,
    return_full=False,
    max_retries=5,
    sleep_sec=0.5,
)
print(df)
```

---

## 17. Integration with `env` Package

`env` expects simulator objects implementing:

```python
simulate(
    *,
    design_variables_config: dict[str, float],
    analyses: tuple[str, ...],
    enable_linearity: bool,
) -> SimResult
```

`NgspiceSimulator` satisfies this protocol.

Environment-side non-convergence logic often treats these as invalid:

- `status != SimStatus.OK`
- `performances is None`
- performance length mismatch
- missing stability info when configured to require `K_min`

---

## 18. Common Troubleshooting

- `NgspiceExecutableNotFound`
  - verify `ngspice` binary installation/path.
- `No sky130 .lib line found`
  - ensure each template contains a replaceable Sky130 `.lib ... sky130.lib.spice <corner>` line.
- `NgspiceMissingOutputs`
  - check `wrdata/write` commands and rewritten output paths in generated netlist under run directory.
- `RawParseError` or missing required columns
  - inspect output file format; ensure expected variable names and frequency columns exist.
- frequent `unknown` status
  - inspect `res.detail`, plus `ngspice_stdout.txt` and `ngspice_stderr.txt` in `run_dir`.

---

## 19. Extending the Package

### 19.1 Add a new analysis reader

1. Create `ngspice/analysis/readers/my_reader.py`.
2. Implement class with `read(result_path, **kwargs) -> DataFrame`.
3. Register with decorator:

```python
from ngspice.analysis.registry import register

@register("My Analysis")
def _factory():
    return MyReader()
```

4. Import module in `ngspice/analysis/readers/__init__.py` for registration side effects.

### 19.2 Add a new circuit topology

1. Add entries in `ngspice/specs.py`:
   - `CIRCUIT_NAMES_REGISTRY`
   - `PARAM_DOMAIN_REGISTRY`
   - knob registries
2. Provide corresponding templates under `ngspice/templates/...`.
3. Ensure output files match collector expectations or extend `NgspiceSimulator` collectors.

---

## 20. Public Namespace (`ngspice/__init__.py`)

Top-level stable exports include:

- execution: `Kernel`
- parsing/I/O: `read_raw_table`, `read_text`, `atomic_write`
- types: `Marker`, `RawAscii`, `KernelContext`, `NgspiceRunResult`, `PathLike`, `KV`
- curated error classes for catching/reporting

Use submodules directly for advanced orchestration (`NgspiceSimulator`, readers, netlist patching).

