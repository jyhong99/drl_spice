from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ngspice.analysis.reader import create_reader
from ngspice.kernel import Kernel
from ngspice.netlist.circuit import Circuit
from ngspice.netlist.designer import Designer

from ngspice.specs import (
    ANALYSIS_KNOB_REGISTRY,
    CIRCUIT_NAMES_REGISTRY,
    LET_KNOB_REGISTRY,
    PARAM_DOMAIN_REGISTRY,
)
from ngspice.patterns import SPARAM_KEY, NOISE_KEY, LINEARITY_KEY
from ngspice.types import KnobValue, SimResult, SimStatus, PathLike

from .utils import (
    _coerce_float_knob,
    _ensure_noise_writes_frequency,
    _merge_and_sort,
    _normalize_knobs,
    _redirect_write_paths,
    _set_sky130_lib,
)


class NgspiceSimulator:
    """
    Run ngspice simulations for a given LNA topology (circuit type) using
    template-based netlists and a structured results contract.

    This class orchestrates:
    - Selecting/validating circuit registries (names, domains, knob defaults)
    - Rendering netlist templates (path rewrites, sky130 library patching)
    - Invoking ngspice through `Kernel`
    - Reading outputs via `create_reader(...)` analysis readers
    - Producing a `SimResult` containing a performance vector and auxiliary metrics

    Parameters
    ----------
    circuit_type : str
        Circuit type key used to select registries (e.g., "CS", "CGCS").
        Must exist in:
        - `CIRCUIT_NAMES_REGISTRY`
        - `ANALYSIS_KNOB_REGISTRY`
        - `LET_KNOB_REGISTRY`
        - `PARAM_DOMAIN_REGISTRY`
    netlist_templates : dict[str, PathLike]
        Mapping from analysis kind key to template file path.
        Expected keys typically include:
        - `SPARAM_KEY`
        - `NOISE_KEY`
        - (optional) `LINEARITY_KEY`
    work_root : PathLike
        Root directory where per-run subdirectories will be created.
    sky130_lib_path : PathLike
        Path to the Sky130 ngspice library file (e.g., ".../sky130.lib.spice").
        Templates must contain a placeholder `.lib ... sky130.lib.spice <corner>` line
        so `_set_sky130_lib(...)` can replace it.
    corner : str
        Process corner string (e.g., "tt", "ff", "ss").
    ngspice_bin : str, default="ngspice"
        Path or command name for ngspice executable.
    timeout_sec : float, default=30.0
        Per-run timeout forwarded to `Kernel`.
    kernel_env : dict[str, str] or None, default=None
        Environment variables forwarded to `Kernel` execution.
    max_run_dirs : int, default=10
        Maximum number of run directories to keep under `work_root`. Older runs
        are deleted after each simulation.

    Notes
    -----
    - This implementation uses a "run directory" per simulation to isolate
      artifacts and logs.
    - Output file names used by collectors are currently hard-coded:
      - S-parameters: `S_Param_Bandwidth.csv`
      - DC op: `DC_OP.csv`
      - Noise: `NoiseFigure.csv`
      - Linearity: knob-driven (default: `FFT.csv`)
      Ensure your templates and ngspice writers follow this contract.
    """

    def __init__(
        self,
        *,
        circuit_type: str,
        netlist_templates: Dict[str, PathLike],
        work_root: PathLike,
        sky130_lib_path: PathLike,
        corner: str,
        ngspice_bin: str = "ngspice",
        timeout_sec: float = 30.0,
        kernel_env: Optional[Dict[str, str]] = None,
        max_run_dirs: int = 10,
    ) -> None:
        self.circuit_type = str(circuit_type)
        self._validate_registries(self.circuit_type)

        # Registry-derived contracts
        self.circuit_names = CIRCUIT_NAMES_REGISTRY[self.circuit_type]
        self.default_analysis_knobs: Dict[str, Any] = dict(
            ANALYSIS_KNOB_REGISTRY[self.circuit_type].defaults()
        )
        self.default_let_knobs: Dict[str, Any] = dict(
            LET_KNOB_REGISTRY[self.circuit_type].defaults()
        )
        self.default_knobs: Dict[str, Any] = _merge_and_sort(
            self.default_analysis_knobs, self.default_let_knobs
        )
        self.fixed_values: Dict[str, float] = dict(
            PARAM_DOMAIN_REGISTRY[self.circuit_type].fixed_values_default
        )

        # Template path normalization
        self.templates: Dict[str, Path] = {str(k): Path(v) for k, v in netlist_templates.items()}

        # Work directory root
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.max_run_dirs = int(max_run_dirs)

        # Kernel / designer setup
        self.kernel = Kernel(
            ngspice_bin=str(ngspice_bin),
            timeout_sec=float(timeout_sec),
            env=kernel_env,
        )
        self.designer = Designer(strict_count=False)

        self.sky130_lib_path = str(sky130_lib_path)
        self.corner = str(corner)

    @staticmethod
    def _validate_registries(circuit_type: str) -> None:
        """
        Validate that `circuit_type` exists in all required registries.

        Parameters
        ----------
        circuit_type : str
            Circuit type key.

        Raises
        ------
        ValueError
            If the circuit type is missing from any registry.
        """
        ct = str(circuit_type)
        if ct not in CIRCUIT_NAMES_REGISTRY:
            raise ValueError(f"Unknown circuit_type for CIRCUIT_NAMES_REGISTRY: {ct!r}")
        if ct not in ANALYSIS_KNOB_REGISTRY:
            raise ValueError(f"Unknown circuit_type for ANALYSIS_KNOB_REGISTRY: {ct!r}")
        if ct not in LET_KNOB_REGISTRY:
            raise ValueError(f"Unknown circuit_type for LET_KNOB_REGISTRY: {ct!r}")
        if ct not in PARAM_DOMAIN_REGISTRY:
            raise ValueError(f"Unknown circuit_type for PARAM_DOMAIN_REGISTRY: {ct!r}")

    def _merged_knobs(self) -> Dict[str, KnobValue]:
        """
        Return the merged (analysis + LET) knob dictionary with normalized aliases.

        Returns
        -------
        dict[str, KnobValue]
            Deterministically sorted knob mapping.
        """
        merged: Dict[str, KnobValue] = dict(self.default_knobs)
        merged = _normalize_knobs(merged)
        return dict(sorted(merged.items(), key=lambda kv: str(kv[0])))

    def simulate(
        self,
        *,
        design_variables_config: Dict[str, float],
        analyses: Tuple[str, ...] = (SPARAM_KEY, NOISE_KEY),
        enable_linearity: bool = False,
    ) -> SimResult:
        """
        Run one simulation episode and collect performance metrics.

        Parameters
        ----------
        design_variables_config : dict[str, float]
            Design-variable assignment (the controllable variables).
            These override any overlapping fixed values.
        analyses : tuple[str, ...], default=(SPARAM_KEY, NOISE_KEY)
            Sequence of analysis kinds to execute. Each kind must have a matching
            template in `self.templates`.
        enable_linearity : bool, default=False
            If True, force inclusion of `LINEARITY_KEY` analysis.

        Returns
        -------
        SimResult
            Result object containing:
            - `performances`: performance vector (or None on failure)
            - `aux`: auxiliary metrics (currently unused in this implementation)
            - `status`: SimStatus
            - `detail`: human-readable reason
            - `run_dir`: directory containing artifacts/logs for this run

        Notes
        -----
        - On failure, `performances` is returned as None and `status` indicates
          the failure class.
        - This function is intentionally defensive: many failures are converted
          to structured `SimResult` rather than bubbling exceptions.
        """
        params: Dict[str, float] = {**self.fixed_values, **design_variables_config}
        run_dir = self._make_run_dir()

        # Normalize knobs and keep deterministic ordering (useful for reproducibility)
        analysis_knobs = dict(
            sorted(_normalize_knobs(self.default_analysis_knobs).items(), key=lambda kv: str(kv[0]))
        )
        let_knobs = dict(
            sorted(_normalize_knobs(self.default_let_knobs).items(), key=lambda kv: str(kv[0]))
        )
        merged_knobs = _merge_and_sort(analysis_knobs, let_knobs)

        # IMPORTANT: `target_frequency` lives in LET knobs in your registries.
        tf = _coerce_float_knob(merged_knobs.get("target_frequency", 2.4e9))
        if tf is None:
            result = SimResult(
                None, {}, SimStatus.VALIDATION, "invalid target_frequency in knobs", str(run_dir)
            )
            self._prune_run_dirs(keep=self.max_run_dirs, keep_paths={run_dir})
            return result
        target_frequency_hz = float(tf)

        try:
            kinds = list(analyses)
            if enable_linearity and (LINEARITY_KEY not in kinds):
                kinds.append(LINEARITY_KEY)

            merged: Dict[str, float] = {}
            aux: Dict[str, float] = {}

            ordered_design = dict(sorted(params.items(), key=lambda kv: str(kv[0])))

            for kind in kinds:
                kind = str(kind)
                if kind not in self.templates:
                    return SimResult(
                        None,
                        {},
                        SimStatus.VALIDATION,
                        f"missing template for analysis={kind!r}",
                        str(run_dir),
                    )

                netlist_path = self._render_netlist(
                    kind=kind,
                    template_path=self.templates[kind],
                    run_dir=run_dir,
                )

                ckt = Circuit(netlist_path, names=self.circuit_names)

                # Patch circuit with both design variables and knob configs
                self.designer.design_circuit(
                    ckt,
                    design_variables_config=ordered_design,
                    analysis_knobs_config=analysis_knobs,
                    let_knobs_config=let_knobs,
                )

                expected_outputs = self._expected_outputs(kind=kind, knobs=analysis_knobs)
                self.kernel.run(
                    ckt,
                    cwd=run_dir,
                    save_logs=True,
                    expected_outputs=expected_outputs,
                )

                if kind == SPARAM_KEY:
                    merged.update(self._collect_sp(run_dir, params, target_frequency_hz))
                elif kind == NOISE_KEY:
                    merged.update(self._collect_noise(run_dir, target_frequency_hz))
                elif kind == LINEARITY_KEY:
                    merged.update(self._collect_linearity(run_dir, merged_knobs))
                else:
                    return SimResult(
                        None,
                        {},
                        SimStatus.VALIDATION,
                        f"unsupported analysis kind: {kind!r}",
                        str(run_dir),
                    )

            performances = self._build_performance_vector(
                merged=merged, enable_linearity=enable_linearity
            )
            result = SimResult(performances, aux, SimStatus.OK, "ok", str(run_dir))

        except RuntimeError as e:
            msg = str(e)
            if "timed out" in msg.lower():
                result = SimResult(None, {}, SimStatus.TIMEOUT, msg, str(run_dir))
            else:
                result = SimResult(None, {}, SimStatus.NGSPICE_FAIL, msg, str(run_dir))

        except FileNotFoundError as e:
            result = SimResult(None, {}, SimStatus.NO_OUTPUT, f"{type(e).__name__}: {e}", str(run_dir))

        except Exception as e:
            result = SimResult(None, {}, SimStatus.UNKNOWN, f"{type(e).__name__}: {e}", str(run_dir))

        finally:
            self._prune_run_dirs(keep=self.max_run_dirs, keep_paths={run_dir})

        return result

    def _render_netlist(self, *, kind: str, template_path: Path, run_dir: Path) -> Path:
        """
        Load a template netlist, apply standard rewrites, and write into `run_dir`.

        Parameters
        ----------
        kind : str
            Analysis kind key (e.g., SPARAM_KEY, NOISE_KEY, LINEARITY_KEY).
        template_path : pathlib.Path
            Source template path.
        run_dir : pathlib.Path
            Destination run directory.

        Returns
        -------
        pathlib.Path
            Written netlist path inside `run_dir`, named `<kind>.spice`.

        Notes
        -----
        Applied rewrites:
        - Redirect write/wrdata outputs to `run_dir` (`_redirect_write_paths`)
        - Replace sky130 `.lib` placeholder with the configured library path/corner
        - For noise analysis, ensure the noise writer includes frequency column
        """
        netlist_text = template_path.read_text(errors="replace")
        netlist_text = _redirect_write_paths(netlist_text, run_dir)
        netlist_text = _set_sky130_lib(netlist_text, lib_path=self.sky130_lib_path, corner=self.corner)

        if kind == NOISE_KEY:
            netlist_text = _ensure_noise_writes_frequency(netlist_text)

        out_path = run_dir / f"{kind}.spice"
        out_path.write_text(netlist_text)
        return out_path

    @staticmethod
    def _expected_outputs(*, kind: str, knobs: Mapping[str, Any]) -> list[str]:
        """
        Compute expected output file basenames for a given analysis kind.

        Parameters
        ----------
        kind : str
            Analysis kind key.
        knobs : Mapping[str, Any]
            Analysis knob mapping (used to read `linearity_result_file`).

        Returns
        -------
        list[str]
            List of expected output file basenames.

        Notes
        -----
        Only linearity currently uses explicit expected output checking.
        """
        if kind != LINEARITY_KEY:
            return []
        result_file = str(knobs.get("linearity_result_file", "FFT.csv"))
        return [Path(result_file).name]

    @staticmethod
    def _build_performance_vector(*, merged: Mapping[str, float], enable_linearity: bool) -> np.ndarray:
        """
        Build the canonical performance vector from collected metrics.

        Parameters
        ----------
        merged : Mapping[str, float]
            Metric dictionary accumulated from collectors.
        enable_linearity : bool
            If True, append `IIP3_dBm` at the end.

        Returns
        -------
        np.ndarray
            Performance vector of dtype float64.

        Notes
        -----
        Vector order must match `perf_metric_order(...)` used elsewhere:
        - [S11_dB, S21_dB, S22_dB, NF_dB, PD_mW, (optional) IIP3_dBm]
        """
        perf_list: list[float] = [
            float(merged.get("S11_dB", np.nan)),
            float(merged.get("S21_dB", np.nan)),
            float(merged.get("S22_dB", np.nan)),
            float(merged.get("NF_dB", np.nan)),
            float(merged.get("PD_mW", 0.0)),
        ]
        if enable_linearity:
            perf_list.append(float(merged.get("IIP3_dBm", np.nan)))
        return np.array(perf_list, dtype=np.float64)

    def _pick_vdd_current_col(self, cols: Sequence[str]) -> Optional[str]:
        """
        Heuristically choose a DC operating point column corresponding to VDD current.

        Parameters
        ----------
        cols : Sequence[str]
            Column names from the DC OP DataFrame.

        Returns
        -------
        str or None
            First matching column name if found; otherwise None.

        Notes
        -----
        This relies on naming heuristics:
        - Column contains "v_dd" or "vdd"
        - Column resembles a current measurement (e.g., "I(...)", "#branch")
        """
        candidates: list[str] = []
        for c in cols:
            cl = c.lower()
            if ("v_dd" in cl) or ("vdd" in cl):
                if cl.startswith("i(") or ("#branch" in cl) or ("i]" in cl):
                    candidates.append(c)
        return candidates[0] if candidates else None

    def _collect_sp(self, run_dir: Path, params: Mapping[str, float], target_frequency_hz: float) -> Dict[str, float]:
        """
        Collect S-parameter metrics and optional stability/power metrics.

        Parameters
        ----------
        run_dir : pathlib.Path
            Run directory containing output artifacts.
        params : Mapping[str, float]
            Full parameter mapping including fixed values. Used for VDD in PD calc.
        target_frequency_hz : float
            Target frequency used for nearest-point selection.

        Returns
        -------
        dict[str, float]
            Collected metrics including:
            - nearest_frequency_hz
            - S11_dB, S21_dB, S22_dB
            - (optional) K_min, mu_min, mup_min, mu_mup_min, bandwidth_hz
            - (optional) PD_mW (if DC OP current is available)

        Notes
        -----
        Expected output filenames:
        - `S_Param_Bandwidth.csv`
        - `DC_OP.csv` (optional)
        """
        sp_path = run_dir / "S_Param_Bandwidth.csv"

        sp_reader = create_reader("S-Parameter Analysis")
        df_near = sp_reader.read(str(sp_path), target_frequency=target_frequency_hz, return_full=False)

        out: Dict[str, float] = {
            "nearest_frequency_hz": float(df_near["nearest_frequency"].iloc[0]),
            "S11_dB": float(df_near["S11_dB"].iloc[0]),
            "S21_dB": float(df_near["S21_dB"].iloc[0]),
            "S22_dB": float(df_near["S22_dB"].iloc[0]),
        }

        stab_reader = create_reader("Stability Factor")
        df_stab = stab_reader.read(str(sp_path))
        for k in ("K_min", "mu_min", "mup_min", "mu_mup_min", "bandwidth_hz"):
            if k in df_stab.columns:
                out[k] = float(df_stab[k].iloc[0])

        # Optional DC power estimation: PD_mW = |VDD * I(VDD)| * 1e3
        dc_path = run_dir / "DC_OP.csv"
        if dc_path.exists():
            dc_reader = create_reader("DC_Operating_Point")
            df_dc = dc_reader.read(str(dc_path))
            col = self._pick_vdd_current_col(df_dc.columns)
            if col is not None:
                i_vdd = float(df_dc[col].iloc[0])
                vdd = float(params.get("v_dd", np.nan))
                if np.isfinite(vdd):
                    out["PD_mW"] = abs(vdd * i_vdd) * 1e3

        return out

    def _collect_noise(self, run_dir: Path, target_frequency_hz: float) -> Dict[str, float]:
        """
        Collect noise figure at the nearest frequency to `target_frequency_hz`.

        Parameters
        ----------
        run_dir : pathlib.Path
            Run directory containing output artifacts.
        target_frequency_hz : float
            Target frequency used for nearest-point selection.

        Returns
        -------
        dict[str, float]
            Collected metrics including:
            - NF_dB
            - (optional) nearest_frequency_hz

        Notes
        -----
        Expected output filename:
        - `NoiseFigure.csv`
        """
        nf_path = run_dir / "NoiseFigure.csv"

        nf_reader = create_reader("Noise Analysis")
        df_nf = nf_reader.read(str(nf_path), target_frequency=target_frequency_hz, return_full=False)

        out: Dict[str, float] = {}
        if "nearest_frequency" in df_nf.columns:
            out["nearest_frequency_hz"] = float(df_nf["nearest_frequency"].iloc[0])
        out["NF_dB"] = float(df_nf["NoiseFigure"].iloc[0])
        return out

    def _collect_linearity(self, run_dir: Path, knobs: Mapping[str, KnobValue]) -> Dict[str, float]:
        """
        Collect linearity metrics (e.g., IIP3) from FFT/two-tone results.

        Parameters
        ----------
        run_dir : pathlib.Path
            Run directory containing output artifacts.
        knobs : Mapping[str, KnobValue]
            Knob mapping used to supply linearity reader parameters:
            - Vin_amp
            - f1, f2
            - R_load
            - linearity_result_file (optional; default "FFT.csv")

        Returns
        -------
        dict[str, float]
            Collected linearity metrics (if present):
            - Pin_dBm
            - Pfund_dBm_avg
            - Pim3_dBm_avg
            - IIP3_dBm

        Raises
        ------
        ValueError
            If required numeric knobs cannot be coerced to floats.

        Notes
        -----
        The file path is resolved as:
            run_dir / Path(result_file).name
        so only the basename is used (safety + run isolation).
        """
        vin = _coerce_float_knob(knobs.get("Vin_amp", 5e-3))
        f1 = _coerce_float_knob(knobs.get("f1", 2.399e9))
        f2 = _coerce_float_knob(knobs.get("f2", 2.401e9))
        rld = _coerce_float_knob(knobs.get("R_load", 50.0))

        if vin is None or f1 is None or f2 is None or rld is None:
            raise ValueError("Invalid linearity knobs: Vin_amp/f1/f2/R_load must be numeric")

        result_file = str(knobs.get("linearity_result_file", "FFT.csv"))
        lin_path = run_dir / Path(result_file).name

        lin_reader = create_reader("Linearity")
        df = lin_reader.read(
            str(lin_path),
            Vin_amp=float(vin),
            f1=float(f1),
            f2=float(f2),
            R_load=float(rld),
        )

        out: Dict[str, float] = {}
        for k in ("Pin_dBm", "Pfund_dBm_avg", "Pim3_dBm_avg", "IIP3_dBm"):
            if k in df.columns:
                out[k] = float(df[k].iloc[0])
        return out

    def _make_run_dir(self) -> Path:
        """
        Create a unique run directory under `work_root`.

        Returns
        -------
        pathlib.Path
            Newly created run directory path.

        Notes
        -----
        Directory name format:
            run_YYYY-MM-DD_HH-MM-SS_<microsec_mod>
        """
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.work_root / f"run_{ts}_{int(time.time() * 1e6) % 1_000_000:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _prune_run_dirs(self, *, keep: int, keep_paths: Optional[set[Path]] = None) -> None:
        """
        Remove old run directories to cap storage usage.

        Parameters
        ----------
        keep : int
            Number of most-recent run directories to keep (including any in keep_paths).
            If keep <= 0, pruning is disabled.
        keep_paths : set[pathlib.Path] or None
            Explicit run directories to always preserve (e.g., the current run).
        """
        if keep <= 0:
            return
        keep_paths = {Path(p).resolve() for p in (keep_paths or set())}

        try:
            dirs = [p for p in self.work_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        except Exception:
            return

        # Sort by mtime ascending (oldest first). Directories can disappear
        # between listing and stat when multiple workers prune concurrently.
        dir_mtimes = []
        for p in dirs:
            try:
                dir_mtimes.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                continue
            except Exception:
                continue
        dir_mtimes.sort(key=lambda t: t[0])
        dirs = [p for _, p in dir_mtimes]

        # Exclude keep_paths from deletion.
        candidates = [p for p in dirs if p.resolve() not in keep_paths]
        # Determine how many we can keep besides keep_paths.
        remaining_keep = max(0, keep - len(keep_paths))
        to_delete = candidates[: max(0, len(candidates) - remaining_keep)]

        for p in to_delete:
            try:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
