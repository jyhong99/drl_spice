from __future__ import annotations

"""
Subprocess execution wrapper for ngspice.

This module provides a `Kernel` that runs ngspice in batch mode, captures
stdout/stderr, enforces timeouts, and validates expected output artifacts.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ngspice.errors import (
    KernelNetlistNotFound,
    NgspiceExecutableNotFound,
    NgspiceMissingOutputs,
    NgspiceNonZeroExit,
    NgspiceTimeout,
)
from ngspice.netlist.circuit import Circuit
from ngspice.types import PathLike, KernelContext, NgspiceRunResult


# =============================================================================
# Kernel
# =============================================================================

class Kernel:
    """
    Thin subprocess wrapper around ngspice (batch mode) with robust diagnostics.

    The kernel is responsible for:
    - invoking ngspice in batch mode (`-b`),
    - capturing stdout/stderr,
    - optionally writing log files to disk,
    - enforcing a wall-clock timeout,
    - validating expected output artifacts (wrdata outputs, etc.),
    - raising domain-specific exceptions with contextual debug info.

    Parameters
    ----------
    ngspice_bin:
        Executable name or absolute path to the ngspice binary. Defaults to
        `"ngspice"` (resolved via PATH).
    timeout_sec:
        Maximum wall-clock time per run. If exceeded, `NgspiceTimeout` is raised.
    env:
        Optional environment overrides to merge into the process environment.
        Useful for things like `SPICE_SCRIPTS`, PDK paths, etc.

    Notes
    -----
    - This class uses `subprocess.run(..., text=True)` to decode output as text.
    - On timeout, `TimeoutExpired` may contain partial stdout/stderr; those are
      captured when available and logged (if enabled).
    """

    def __init__(
        self,
        *,
        ngspice_bin: str = "ngspice",
        timeout_sec: float = 60.0,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.ngspice_bin = str(ngspice_bin).strip()
        if not self.ngspice_bin:
            raise ValueError("ngspice_bin must be a non-empty string")
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0.0:
            raise ValueError(f"timeout_sec must be > 0 (got {self.timeout_sec})")
        self.env = dict(env) if env else None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _ensure_dir(p: Path) -> Path:
        """
        Create a directory (including parents) if it does not exist.

        Parameters
        ----------
        p:
            Directory path.

        Returns
        -------
        pathlib.Path
            The same path `p` (for fluent usage).
        """
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _write_logs(
        run_cwd: Path,
        stdout_name: str,
        stderr_name: str,
        stdout: str,
        stderr: str,
    ) -> None:
        """
        Persist captured stdout/stderr into files under `run_cwd`.

        Parameters
        ----------
        run_cwd:
            Directory where log files are written.
        stdout_name:
            File name for stdout log.
        stderr_name:
            File name for stderr log.
        stdout:
            Captured stdout text.
        stderr:
            Captured stderr text.

        Notes
        -----
        - Logs are written with UTF-8 encoding and `errors="replace"` so
          undecodable characters do not crash diagnostics.
        """
        (run_cwd / stdout_name).write_text(stdout or "", encoding="utf-8", errors="replace")
        (run_cwd / stderr_name).write_text(stderr or "", encoding="utf-8", errors="replace")

    @staticmethod
    def _normalize_outputs(run_cwd: Path, expected_outputs: Sequence[PathLike]) -> list[Path]:
        """
        Normalize expected output paths to absolute paths under `run_cwd`.

        Parameters
        ----------
        run_cwd:
            Base directory for relative output paths.
        expected_outputs:
            Sequence of paths. Each path may be absolute or relative.

        Returns
        -------
        list[pathlib.Path]
            Absolute paths for output existence checks.

        Notes
        -----
        - Relative paths are interpreted as relative to `run_cwd`, not the current
          process working directory.
        """
        outs: list[Path] = []
        for outp in expected_outputs:
            p = Path(outp)
            if not str(p).strip():
                raise ValueError("expected_outputs contains an empty path")
            outs.append(p if p.is_absolute() else (run_cwd / p))
        return outs

    def _build_env(self) -> dict[str, str]:
        """
        Build the environment dict passed to the ngspice subprocess.

        Returns
        -------
        dict[str, str]
            A copy of the current process environment with optional overrides.

        Notes
        -----
        - `self.env` values override existing keys in `os.environ`.
        """
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        return env

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        circuit: Circuit,
        *,
        cwd: Optional[PathLike] = None,
        save_logs: bool = True,
        stdout_name: str = "ngspice_stdout.txt",
        stderr_name: str = "ngspice_stderr.txt",
        check: bool = True,
        expected_outputs: Optional[Sequence[PathLike]] = None,
        tail_limit_missing: int = 4000,
        tail_limit_fail: int = 2000,
    ) -> NgspiceRunResult:
        """
        Run ngspice for the given circuit netlist in batch mode.

        Parameters
        ----------
        circuit:
            Circuit object containing a `netlist_path` attribute that points to a
            rendered ngspice netlist file.
        cwd:
            Working directory for the ngspice process. If None, defaults to the
            netlist directory.
        save_logs:
            If True, writes stdout/stderr logs to files in `cwd`.
        stdout_name:
            File name to store stdout when `save_logs=True`.
        stderr_name:
            File name to store stderr when `save_logs=True`.
        check:
            If True, non-zero ngspice return codes raise `NgspiceNonZeroExit`.
            If False, return a result even on non-zero exit (caller decides).
        expected_outputs:
            Optional list of output files expected to be created by ngspice
            (e.g., `wrdata` outputs). If provided, missing outputs raise
            `NgspiceMissingOutputs` regardless of return code.
        tail_limit_missing:
            Tail length (characters) included in error messages when expected
            outputs are missing.
        tail_limit_fail:
            Tail length (characters) included in error messages for timeout or
            non-zero exit.

        Returns
        -------
        ngspice.types.NgspiceRunResult
            Captured execution result (return code, stdio, elapsed time, cmd, cwd).

        Raises
        ------
        KernelNetlistNotFound
            If `circuit.netlist_path` does not exist on disk.
        NgspiceExecutableNotFound
            If the `ngspice_bin` cannot be executed (typically not in PATH).
        NgspiceTimeout
            If execution exceeds `timeout_sec`.
        NgspiceMissingOutputs
            If `expected_outputs` is provided and one or more files are missing.
        NgspiceNonZeroExit
            If `check=True` and ngspice returns a non-zero exit code.

        Notes
        -----
        - Ordering of checks:
          1) timeout/executable errors are handled during `subprocess.run`.
          2) stdout/stderr are always captured and (optionally) logged.
          3) missing expected outputs are checked next (if configured).
          4) non-zero exit code is checked last (if `check=True`).
        - The exception messages include a compact tail of stderr+stdout to
          facilitate debugging without dumping entire logs.
        """
        netlist_path = Path(circuit.netlist_path).expanduser().resolve()
        if not netlist_path.exists() or not netlist_path.is_file():
            raise KernelNetlistNotFound(f"netlist not found: {netlist_path}", ctx=None)
        if tail_limit_missing < 0 or tail_limit_fail < 0:
            raise ValueError("tail_limit_missing and tail_limit_fail must be >= 0")

        run_cwd = self._ensure_dir((Path(cwd).expanduser().resolve()) if cwd is not None else netlist_path.parent)

        # ngspice batch mode: -b
        cmd: tuple[str, ...] = (self.ngspice_bin, "-b", str(netlist_path))

        t0 = time.time()
        ctx = KernelContext(cmd=cmd, cwd=str(run_cwd), netlist=str(netlist_path))

        try:
            cp = subprocess.run(
                cmd,
                cwd=ctx.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
                check=False,   # we do our own error mapping
                text=True,     # decode to str
                env=self._build_env(),
            )
        except FileNotFoundError as e:
            # Typically: executable not found (ngspice missing or wrong path)
            raise NgspiceExecutableNotFound(
                f"ngspice executable not found: {self.ngspice_bin}",
                ctx=ctx,
            ) from e

        except subprocess.TimeoutExpired as e:
            # TimeoutExpired may contain partial output; capture if present
            ctx.elapsed_sec = float(time.time() - t0)

            out = getattr(e, "stdout", None)
            err = getattr(e, "stderr", None)
            ctx.stdout = out if isinstance(out, str) else ""
            ctx.stderr = err if isinstance(err, str) else ""

            if save_logs:
                self._write_logs(run_cwd, stdout_name, stderr_name, ctx.stdout, ctx.stderr)

            raise NgspiceTimeout(
                f"ngspice timed out after {self.timeout_sec:.1f}s "
                f"(cwd={run_cwd}, netlist={netlist_path})\n"
                f"tail=\n{ctx.tail(tail_limit_fail)}",
                ctx=ctx,
            ) from e

        # Normal completion (may still be non-zero exit)
        ctx.elapsed_sec = float(time.time() - t0)
        ctx.stdout = cp.stdout or ""
        ctx.stderr = cp.stderr or ""

        res = NgspiceRunResult(
            returncode=int(cp.returncode),
            stdout=ctx.stdout,
            stderr=ctx.stderr,
            elapsed_sec=ctx.elapsed_sec,
            cmd=cmd,
            cwd=ctx.cwd,
        )

        if save_logs:
            self._write_logs(run_cwd, stdout_name, stderr_name, res.stdout, res.stderr)

        # Validate expected artifacts (wrdata outputs etc.)
        if expected_outputs:
            expected_paths = self._normalize_outputs(run_cwd, expected_outputs)
            missing_paths = [str(p) for p in expected_paths if not p.exists()]
            if missing_paths:
                raise NgspiceMissingOutputs(
                    "ngspice finished but expected output files were not created:\n"
                    + "\n".join(missing_paths)
                    + "\n\n--- tail ---\n"
                    + ctx.tail(tail_limit_missing),
                    ctx=ctx,
                    missing=missing_paths,
                )

        # Map non-zero exit to a domain-specific exception if requested
        if check and res.returncode != 0:
            raise NgspiceNonZeroExit(
                "ngspice failed.\n"
                f"cmd={cmd}\n"
                f"cwd={run_cwd}\n"
                f"returncode={res.returncode}\n"
                f"tail=\n{ctx.tail(tail_limit_fail)}",
                ctx=ctx,
                returncode=res.returncode,
            )

        return res
