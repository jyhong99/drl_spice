from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..patterns import _NOISE_WRITE_BARE_RE, _RE_LIB, _WRITE_RE
from ..types import KnobValue, PathLike


# =============================================================================
# Netlist text rewriting helpers
# =============================================================================

def _redirect_write_paths(netlist_text: str, run_dir: PathLike) -> str:
    """
    Redirect ngspice `write`/`wrdata` output paths into a run directory.

    Parameters
    ----------
    netlist_text : str
        Full netlist text.
    run_dir : PathLike
        Run directory where output artifacts should be written.

    Returns
    -------
    str
        Modified netlist text, terminated with a newline.
    """
    rd = Path(run_dir)
    out: list[str] = []

    for line in netlist_text.splitlines():
        m = _WRITE_RE.match(line)
        if not m:
            out.append(line)
            continue

        cmd = m.group(1).lower()
        path_tok = m.group(2).strip("\"'")
        rhs = m.group(3)

        base = Path(path_tok).name
        new_path = (rd / base).as_posix()
        out.append(f"{cmd} {new_path} {rhs}")

    return "\n".join(out) + "\n"


def _ensure_noise_writes_frequency(netlist_text: str) -> str:
    """
    Ensure noise-analysis write commands include the frequency column.

    Parameters
    ----------
    netlist_text : str
        Full netlist text.

    Returns
    -------
    str
        Modified netlist text, terminated with a newline.
    """
    out: list[str] = []

    for ln in netlist_text.splitlines():
        m = _NOISE_WRITE_BARE_RE.match(ln.strip())
        if m:
            cmd = m.group(1).lower()
            path = m.group(2)
            out.append(f"{cmd} {path} frequency NoiseFigure")
        else:
            out.append(ln)

    return "\n".join(out) + "\n"


def _set_sky130_lib(netlist_text: str, *, lib_path: PathLike, corner: str = "tt") -> str:
    """
    Replace the Sky130 `.lib ... sky130.lib.spice <corner>` line with a given path.

    Parameters
    ----------
    netlist_text : str
        Full netlist text.
    lib_path : PathLike
        Path to the Sky130 ngspice library file to include.
    corner : str, default="tt"
        Process corner string (e.g., "tt", "ff", "ss").

    Returns
    -------
    str
        Modified netlist text, terminated with a newline.
    """
    lib_path_s = str(Path(lib_path).expanduser())
    corner_s = str(corner).strip()

    out: list[str] = []
    replaced = False

    for ln in netlist_text.splitlines():
        m = _RE_LIB.match(ln)
        if not m:
            out.append(ln)
            continue

        old_path = m.group(1)
        if "sky130.lib.spice" in old_path:
            out.append(f".lib {lib_path_s} {corner_s}")
            replaced = True
        else:
            out.append(ln)

    if not replaced:
        raise ValueError(
            "No sky130 `.lib ... sky130.lib.spice <corner>` line found in template. "
            "Keep one `.lib ... sky130.lib.spice tt` line in each template."
        )

    return "\n".join(out) + "\n"


# =============================================================================
# Knob / config normalization helpers
# =============================================================================

def _coerce_float_knob(x: KnobValue) -> Optional[float]:
    """
    Coerce a knob value into a float if possible.

    Parameters
    ----------
    x : KnobValue
        Input knob value (commonly float/int/str).

    Returns
    -------
    float or None
        Parsed float value if conversion succeeds, else None.
    """
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
    except Exception:
        return None
    return None


def _sorted_str_keys(d: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict sorted by stringified keys.

    Parameters
    ----------
    d : Mapping[str, Any]
        Input mapping.

    Returns
    -------
    dict[str, Any]
        New dictionary sorted by `str(key)`.
    """
    return dict(sorted(d.items(), key=lambda kv: str(kv[0])))


def _merge_and_sort(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge two mappings and return a dict sorted by stringified keys.

    Parameters
    ----------
    a : Mapping[str, Any]
        Base mapping.
    b : Mapping[str, Any]
        Override mapping. Keys here overwrite keys in `a`.

    Returns
    -------
    dict[str, Any]
        Merged and deterministically ordered dictionary.
    """
    merged = dict(a)
    merged.update(b)
    return _sorted_str_keys(merged)


def _normalize_knobs(knobs: Mapping[str, KnobValue]) -> Dict[str, KnobValue]:
    """
    Normalize knob names by adding common aliases.

    Parameters
    ----------
    knobs : Mapping[str, KnobValue]
        Raw knob dictionary.

    Returns
    -------
    dict[str, KnobValue]
        Normalized knob dictionary.
    """
    out: Dict[str, KnobValue] = dict(knobs)

    if "Q_factor" in out and "q_factor" not in out:
        out["q_factor"] = out["Q_factor"]

    if "q_factor" in out and "Q_factor" not in out:
        out["Q_factor"] = out["q_factor"]

    return out
