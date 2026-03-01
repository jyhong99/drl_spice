from __future__ import annotations

from typing import Any, Mapping


# =============================================================================
# Observation validation helpers
# =============================================================================

def _require_key(d: Mapping[str, Any], key: str) -> None:
    """
    Require that a dictionary contains a specific key.

    This function is used to validate the schema of observation-related
    dictionaries (e.g., `eval_out`) before numeric processing begins.

    Parameters
    ----------
    d : Mapping[str, Any]
        Input mapping to validate.
    key : str
        Required key name.

    Raises
    ------
    KeyError
        If `key` is not present in `d`.

    Notes
    -----
    - This function only checks key existence, not the value type or shape.
    - Value validation should be performed separately after this check.
    """
    if key not in d:
        raise KeyError(f"eval_out must contain key {key!r}")


def _require_mapping(name: str, obj: Any) -> Mapping[str, Any]:
    """
    Require that an object is a mapping and return it.

    Parameters
    ----------
    name : str
        Argument name used in error messages.
    obj : Any
        Object expected to implement mapping semantics.

    Returns
    -------
    Mapping[str, Any]
        The validated mapping object.

    Raises
    ------
    ValueError
        If `obj` is not a mapping.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(obj).__name__}")
    return obj
