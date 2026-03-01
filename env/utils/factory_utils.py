from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ..patterns import (
    LINEARITY_KEY,
    NOISE_KEY,
    SPARAM_KEY,
    SPICE_SCRIPTS_KEY,
)
from ..types import KernelEnv, TemplateMap, TemplatePath


def _resolve_path(p: Union[str, Path]) -> Path:
    """
    Resolve a filesystem path robustly.

    This helper expands user home markers (``~``) and attempts to return an
    absolute, normalized path. If full resolution fails (e.g., due to broken
    symlinks or permission constraints), it falls back to `absolute()`.

    Parameters
    ----------
    p : str or pathlib.Path
        Input path.

    Returns
    -------
    pathlib.Path
        Resolved path if possible; otherwise an absolute path.
    """
    q = Path(p).expanduser()
    try:
        return q.resolve()
    except Exception:
        return q.absolute()


def _prepare_spice_scripts_env(
    spice_scripts_path: Optional[str],
    *,
    repo_root: Optional[Path] = None,
    touch_file: bool = True,
) -> KernelEnv:
    """
    Prepare kernel environment variables for ngspice script directory usage.

    This constructs the environment mapping needed by the simulation kernel
    (typically `SpiceKernel`) so that ngspice can locate helper scripts via a
    stable environment variable (e.g., ``SPICE_SCRIPTS_KEY``).

    Behavior
    --------
    - If `spice_scripts_path` is None, return an empty environment dict.
    - Otherwise:
      1) Resolve and ensure the target directory exists.
      2) If creation fails with PermissionError, fall back to a directory under:
         - `repo_root` if provided, else
         - the user's home directory
         using the conventional subpath: ``ngspice/ngspice_scripts``.
      3) Optionally create a `.keep` placeholder file to ensure the directory
         is preserved in version control or created on disk.

    Parameters
    ----------
    spice_scripts_path : str or None
        User-provided path for the SPICE scripts directory. If None, no env var
        is provided.
    repo_root : pathlib.Path or None, default=None
        Repository root used for fallback when the requested directory cannot
        be created due to permission issues.
    touch_file : bool, default=True
        If True, create a `.keep` file in the scripts directory (if missing).

    Returns
    -------
    KernelEnv
        Environment mapping containing `{SPICE_SCRIPTS_KEY: <dir>}` or an empty
        dict if `spice_scripts_path` is None.

    Raises
    ------
    OSError
        If both the primary directory creation and fallback creation fail for
        reasons other than PermissionError in the primary path.
    """
    if spice_scripts_path is None:
        return {}

    p = _resolve_path(spice_scripts_path)

    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        base = _resolve_path(repo_root or Path.home())
        p = _resolve_path(base / "ngspice" / "ngspice_scripts")
        p.mkdir(parents=True, exist_ok=True)

    if touch_file:
        keep = p / ".keep"
        if not keep.exists():
            keep.write_text("# Placeholder for SPICE_SCRIPTS directory\n", encoding="utf-8")

    return {SPICE_SCRIPTS_KEY: str(p)}


def _default_netlist_templates(circuit_type: str, *, repo_root: Path) -> TemplateMap:
    """
    Return default netlist template paths for a given circuit type.

    Parameters
    ----------
    circuit_type : str
        Circuit identifier (case-insensitive). Supported values:
        - "CS"
        - "CGCS" (and a few aliases)
    repo_root : pathlib.Path
        Repository root that contains the `ngspice/templates/...` tree.

    Returns
    -------
    TemplateMap
        Mapping from analysis key to template path, e.g.:
        - SPARAM_KEY -> s-parameter template
        - NOISE_KEY  -> noise figure template
        - LINEARITY_KEY -> linearity/FFT template

    Raises
    ------
    ValueError
        If the circuit type is unknown.
    """
    ct = str(circuit_type).strip().lower()

    if ct == "cs":
        base = repo_root / "ngspice" / "templates" / "cs_lna"
        return {
            SPARAM_KEY: base / "cs_lna_sparam.spice",
            NOISE_KEY: base / "cs_lna_nf.spice",
            LINEARITY_KEY: base / "cs_lna_fft.spice",
        }

    if ct in ("cgcs", "cg_cs", "cg-cs", "cgcascode", "cgcs_lna"):
        base = repo_root / "ngspice" / "templates" / "cgcs_lna"
        return {
            SPARAM_KEY: base / "cgcs_lna_sparam.spice",
            NOISE_KEY: base / "cgcs_lna_nf.spice",
            LINEARITY_KEY: base / "cgcs_lna_fft.spice",
        }

    raise ValueError(f"Unknown circuit_type={circuit_type!r}. expected 'CS' or 'CGCS'")


def _normalize_analyses(analyses: Tuple[str, ...], *, enable_linearity: bool) -> Tuple[str, ...]:
    """
    Normalize analysis kinds and optionally force inclusion of linearity analysis.

    Parameters
    ----------
    analyses : tuple[str, ...]
        Requested analysis kinds (strings).
    enable_linearity : bool
        If True, ensure `LINEARITY_KEY` is included.

    Returns
    -------
    tuple[str, ...]
        Normalized analysis kinds, preserving order and appending `LINEARITY_KEY`
        if required and missing.

    Notes
    -----
    This function does not validate whether the analysis kinds are supported;
    it only normalizes and ensures required keys are present.
    """
    kinds = [str(a) for a in analyses]
    if enable_linearity and (LINEARITY_KEY not in kinds):
        kinds.append(LINEARITY_KEY)
    return tuple(kinds)


def _resolve_templates(
    circuit_type: str,
    *,
    repo_root: Path,
    netlist_templates: Optional[Dict[str, TemplatePath]],
    enable_linearity: bool,
) -> TemplateMap:
    """
    Resolve and validate netlist template paths (defaults + overrides).

    This function:
    1) Loads defaults based on `circuit_type`.
    2) Applies user overrides (`netlist_templates`), if provided.
    3) Normalizes all paths using `_resolve_path`.
    4) Validates required templates exist on disk.
    5) Enforces that if `enable_linearity=True`, a linearity template is present.

    Parameters
    ----------
    circuit_type : str
        Circuit type used to pick default templates.
    repo_root : pathlib.Path
        Repository root for default template discovery.
    netlist_templates : dict[str, TemplatePath] or None
        Optional mapping of template overrides, keyed by analysis type
        (e.g., SPARAM_KEY / NOISE_KEY / LINEARITY_KEY).
    enable_linearity : bool
        If True, require `LINEARITY_KEY` to be present in the resolved map.

    Returns
    -------
    TemplateMap
        Fully resolved template map: `dict[str, pathlib.Path]`.

    Raises
    ------
    ValueError
        If `enable_linearity=True` but a linearity template cannot be resolved.
    OSError
        If any resolved template paths do not exist on disk.
    """
    defaults = _default_netlist_templates(circuit_type, repo_root=repo_root)
    overrides = netlist_templates or {}

    # Merge defaults with user overrides
    merged: Dict[str, Path] = dict(defaults)
    for k, v in overrides.items():
        merged[str(k)] = _resolve_path(v)

    # Normalize all paths
    merged = {k: _resolve_path(v) for k, v in merged.items()}

    if enable_linearity and LINEARITY_KEY not in merged:
        raise ValueError("enable_linearity=True but no linearity template resolved")

    # Validate filesystem existence
    missing = {k: str(p) for k, p in merged.items() if not p.exists()}
    if missing:
        raise OSError(f"Missing netlist templates: {missing}")

    return merged
