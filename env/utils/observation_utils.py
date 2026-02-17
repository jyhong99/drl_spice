from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# Observation validation helpers
# =============================================================================

def _require_key(d: Dict[str, Any], key: str) -> None:
    """
    Require that a dictionary contains a specific key.

    This function is used to validate the schema of observation-related
    dictionaries (e.g., `eval_out`) before numeric processing begins.

    Parameters
    ----------
    d : dict[str, Any]
        Input dictionary to validate.
    key : str
        Required key name.

    Raises
    ------
    KeyError)
        If `key` is not present in `d`.

    Notes
    -----
    - This function only checks key existence, not the value type or shape.
    - Value validation should be performed separately after this check.
    """
    if key not in d:
        raise KeyError(f"eval_out must contain key {key!r}")
