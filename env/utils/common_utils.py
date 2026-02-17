from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch as th
except Exception:  # pragma: no cover
    th = None


# =============================================================================
# Numeric helpers
# =============================================================================

def _to_flat_np(x: Any, *, dtype: Optional[np.dtype] = np.float32) -> np.ndarray:
    """
    Convert an input object into a flattened NumPy array.

    This helper function standardizes array-like inputs across the environment,
    action, observation, and reward pipelines. It is designed to be robust to
    common numeric containers and optional PyTorch tensors.

    Conversion rules
    ----------------
    The conversion follows these steps in order:

    1. **PyTorch tensor handling** (if torch is available):
       - Detach from the computation graph.
       - Move the tensor to CPU.
       - Convert to a NumPy array via ``.numpy()``.

    2. **Fallback conversion**:
       - Use ``np.asarray(x)`` for Python sequences or NumPy-compatible objects.

    3. **Flattening**:
       - The result is reshaped to one dimension using ``reshape(-1)``.

    4. **Optional dtype casting**:
       - If ``dtype`` is not ``None``, cast using
         ``astype(dtype, copy=False)``.

    Parameters
    ----------
    x : Any
        Input object to convert. Common supported types include:
        - Python scalar
        - list / tuple
        - ``np.ndarray``
        - ``torch.Tensor`` (if PyTorch is installed)
    dtype : np.dtype or None, default=np.float32
        Desired output dtype.
        - If provided, the output array is cast to this dtype.
        - If ``None``, the inferred dtype from conversion is preserved.

    Returns
    -------
    np.ndarray
        Flattened NumPy array with shape ``(D,)``.

    Notes
    -----
    - For scalar inputs, the output shape is ``(1,)``.
    - ``reshape(-1)`` guarantees a contiguous 1D view when possible.
    - ``copy=False`` is used during casting to avoid unnecessary memory copies.
    - Torch conversion uses ``detach().cpu().numpy()`` to ensure:
        - no autograd tracking
        - host-accessible memory

    Examples
    --------
    >>> to_flat_np([1, 2, 3]).shape
    (3,)

    >>> to_flat_np(np.array([[1, 2], [3, 4]])).shape
    (4,)

    >>> to_flat_np(5.0)
    array([5.], dtype=float32)

    >>> import torch
    >>> to_flat_np(torch.tensor([[1., 2.], [3., 4.]])).shape
    (4,)
    """
    # Torch tensor → NumPy (if available)
    if th is not None and hasattr(th, "is_tensor") and th.is_tensor(x):
        # detach: stop gradient tracking
        # cpu: ensure host memory
        # numpy: convert to NumPy array
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # Flatten to 1D
    arr = arr.reshape(-1)

    # Optional dtype cast
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def _require_positive_int(name: str, v: int) -> int:
    """
    Require a strictly positive integer.

    Parameters
    ----------
    name : str
        Field name used in the error message.
    v : int
        Value to validate.
    exc : type[Exception]
        Exception type to raise on failure.

    Returns
    -------
    int
        Validated integer value.

    Raises
    ------
    exc
        If `v <= 0`.
    """
    n = int(v)
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return n


def _require_len(name: str, arr: np.ndarray, n: int) -> None:
    """
    Require a 1D array to have expected length along axis 0.

    Parameters
    ----------
    name : str
        Argument name for messages.
    arr : np.ndarray
        Input array (assumed 1D or flattened).
    n : int
        Expected length.
    exc : type[Exception]
        Exception type to raise on failure.

    Raises
    ------
    exc
        If `arr.shape[0] != n`.
    """
    if int(arr.shape[0]) != int(n):
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")


def _require_finite_positive(name: str, v: float) -> float:
    """
    Require a finite strictly positive float.

    Raises `exc` if not finite or <= 0.

    Returns
    -------
    float
        Validated float value.
    """
    x = float(v)
    if (not np.isfinite(x)) or x <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {v}")
    return x


def _require_finite(name: str, v: float) -> float:
    """
    Require a finite float.

    Returns
    -------
    float
        Validated float value.
    """
    x = float(v)
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {v}")
    return x


def _require_in_01(name: str, v: float) -> float:
    """
    Require a finite probability in [0, 1].

    Returns
    -------
    float
        Validated float value in [0, 1].
    """
    x = float(v)
    if (not np.isfinite(x)) or (x < 0.0) or (x > 1.0):
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


def _require_ordered_bounds(label: str, low: float, high: float) -> None:
    """
    Require ordered bounds: low < high.

    Parameters
    ----------
    label : str
        Label for error message context (e.g., "clip", "action").
    low, high : float
        Bounds.
    exc : type[Exception]
        Exception type to raise on failure.
    """
    lo = float(low)
    hi = float(high)
    if lo >= hi:
        raise ValueError(f"{label}_low must be < {label}_high, got {lo} >= {hi}")