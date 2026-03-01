from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..patterns import CSV_COLUMNS, STORE_INLINE, STORE_OFF


# =============================================================================
# Settings / validation helpers
# =============================================================================

def _require_store_arrays(v: str) -> str:
    """
    Validate the `store_arrays` logger setting.

    Parameters
    ----------
    v : str
        Candidate value for the `store_arrays` setting.

    Returns
    -------
    str
        Normalized string value (exactly one of `STORE_INLINE` or `STORE_OFF`).

    Raises
    ------
    ValueError
        If `v` is not one of the supported values.
    """
    s = str(v)
    if s not in (STORE_INLINE, STORE_OFF):
        raise ValueError(
            f"store_arrays must be {STORE_INLINE!r} or {STORE_OFF!r}, got {s!r}"
        )
    return s


# =============================================================================
# CSV row helpers
# =============================================================================

def _empty_row() -> Dict[str, str]:
    """
    Construct an empty CSV row with the fixed schema.

    Returns
    -------
    dict[str, str]
        Mapping from each column name in `CSV_COLUMNS` to an empty string.

    Notes
    -----
    This ensures every written CSV row always contains all expected columns,
    which makes downstream parsing stable.
    """
    return {k: "" for k in CSV_COLUMNS}


def _json_cell(obj: Any) -> str:
    """
    JSON-encode a value for safe placement into a CSV cell.

    Parameters
    ----------
    obj : Any
        Input object to serialize.

    Returns
    -------
    str
        JSON string produced by `json.dumps`, using `ensure_ascii=False`.

    Notes
    -----
    This uses `to_jsonable` to best-effort convert objects into JSON-friendly
    structures before dumping.
    """
    return json.dumps(_to_jsonable(obj), ensure_ascii=False)


# =============================================================================
# Time helpers
# =============================================================================

def _now_ymd_hms() -> str:
    """
    Return the current local time formatted as a string.

    Returns
    -------
    str
        Current local time formatted as::

            YYYY-MM-DD HH:MM:SS

    Notes
    -----
    This uses `time.strftime` with the process's local timezone settings.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Serialization helpers
# =============================================================================

def _to_jsonable(x: Any) -> Any:
    """
    Convert an object into a JSON-serializable representation (best effort).

    This function recursively transforms common non-JSON-friendly objects into
    equivalents that can be passed to `json.dumps`.

    Conversion rules
    ----------------
    The following conversions are applied (recursively when applicable):

    - ``None`` -> ``None``
    - ``(str, int, float, bool)`` -> unchanged
    - ``pathlib.Path`` -> ``str(path)``
    - dataclass instance -> ``dict`` (via `dataclasses.asdict`)
    - ``dict`` -> dict with ``str(key)`` and converted values
    - ``list`` / ``tuple`` -> list of converted elements
    - ``np.ndarray`` -> list (via ``tolist()``)
    - NumPy scalar (or scalar-like object with ``.item()``) -> Python scalar
    - fallback -> ``str(x)``

    Parameters
    ----------
    x : Any
        Input object.

    Returns
    -------
    Any
        JSON-friendly object composed of primitives, lists, dicts, and strings.

    Notes
    -----
    - JSON requires string keys; dict keys are converted via `str(k)`.
    - The fallback `str(x)` makes this function total over arbitrary inputs,
      which is useful for robust logging, but may lose structured information.
    """
    if x is None:
        return None

    # JSON primitives
    if isinstance(x, (str, int, float, bool)):
        return x

    # Filesystem paths
    if isinstance(x, Path):
        return str(x)

    # Dataclasses
    if is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in asdict(x).items()}

    # Mappings / sequences
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # NumPy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()

    # NumPy scalars (or scalar-like objects exposing `.item()`)
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return _to_jsonable(x.item())
        except Exception:
            # Fall through to string fallback
            pass

    # Fallback: string representation
    return str(x)


# =============================================================================
# File I/O helpers
# =============================================================================

def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomically write UTF-8 text to a file.

    This prevents readers from observing partially written files by writing
    into a temporary sibling file first and then replacing the target file
    using an atomic rename operation.

    Algorithm
    ---------
    1. Ensure the parent directory exists.
    2. Write contents to a temporary sibling file:

       ``<path>.tmp``  (implemented via suffix extension)

    3. Replace the final path with the temporary file via `os.replace`.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    text : str
        UTF-8 text content.

    Returns
    -------
    None

    Notes
    -----
    - `os.replace(src, dst)` is atomic on POSIX when both files are on the same
      filesystem. This is why the temporary file is created in the same directory.
    - If the process crashes before `os.replace`, the `.tmp` file may remain.
      If desired, cleanup can be implemented by the caller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")

    os.replace(tmp, path)
