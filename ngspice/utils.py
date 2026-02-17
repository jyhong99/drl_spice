from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .errors import MissingColumnError


# =============================================================================
# Helpers
# =============================================================================

def pick_col(df: pd.DataFrame, *names: str) -> str:
    """
    Pick the first existing column name from a list of candidates.

    This is a strict schema helper: it searches `df.columns` in the order
    provided and returns the first match. If none are present, it raises
    `MissingColumnError` rather than silently falling back.

    Parameters
    ----------
    df:
        Input DataFrame to inspect.
    *names:
        Candidate column names, checked in order.

    Returns
    -------
    str
        The first column name that exists in `df.columns`.

    Raises
    ------
    MissingColumnError
        If none of `names` are present in `df.columns`.

    Notes
    -----
    - This function is intentionally strict to avoid downstream silent
      misinterpretation of data schemas.
    """
    cols = df.columns
    for n in names:
        if n in cols:
            return n
    raise MissingColumnError(names, cols)


def find_freq_col(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = ("frequency", "freq", "f", "v(freq)"),
) -> str:
    """
    Find a frequency-like column name in a DataFrame.

    Unlike permissive implementations that fall back to the first column
    (e.g., `df.columns[0]`), this helper is strict: if no frequency-like
    column is found, it raises. This prevents silently computing results
    using the wrong x-axis.

    Parameters
    ----------
    df:
        Input DataFrame to inspect.
    candidates:
        Ordered list/tuple of candidate names that may represent frequency.

    Returns
    -------
    str
        The selected frequency column name.

    Raises
    ------
    MissingColumnError
        If none of the `candidates` exist in `df.columns`.

    Examples
    --------
    >>> col = find_freq_col(df)
    >>> f = df[col].to_numpy()
    """
    # `pick_col` expects variadic strings; convert Sequence[str] -> tuple[str, ...]
    return pick_col(df, *tuple(candidates))


def nearest_index(x: np.ndarray, target: float) -> int:
    """
    Return the index of the element in `x` closest to `target`.

    The distance metric is absolute difference: `abs(x[i] - target)`.
    NaNs (or non-finite values) in `x` are ignored by assigning their
    distance as +inf, so they cannot win the argmin.

    Parameters
    ----------
    x:
        1D array-like of numeric values. Will be converted to `float`.
    target:
        Target value to locate the nearest neighbor for.

    Returns
    -------
    int
        Index `i` that minimizes `abs(x[i] - target)` among finite entries.

    Raises
    ------
    ValueError
        If `x` is empty, or if all entries yield non-finite distances
        (e.g., `x` is all-NaN/inf).

    Notes
    -----
    - If multiple indices tie for the same minimum distance, NumPy's
      `argmin` returns the first occurrence.
    - This function assumes `x` is 1D; higher-dimensional inputs are
      flattened by `np.asarray` only if provided already in that shape.
      If you require strict 1D behavior, validate `x.ndim == 1` upstream.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Empty array for nearest search")

    target_f = float(target)
    dist = np.abs(x - target_f)

    # Robustness: NaN/inf distances are set to +inf so they won't win argmin.
    dist = np.where(np.isfinite(dist), dist, np.inf)

    idx = int(np.argmin(dist))
    if not np.isfinite(dist[idx]):
        raise ValueError("All distances are non-finite (x may be all-NaN/inf)")
    return idx


def attach_nearest_meta(
    df: pd.DataFrame,
    *,
    target_frequency: float,
    idx: int,
    f_near: float,
) -> pd.DataFrame:
    """
    Attach nearest-frequency lookup metadata to `df.attrs` (in-place).

    This is useful for preserving provenance when you compute a nearest
    frequency row but return the full DataFrame (e.g., for later inspection
    or logging).

    Parameters
    ----------
    df:
        DataFrame to annotate. Modified in-place via `df.attrs`.
    target_frequency:
        The desired frequency (Hz) used as the lookup target.
    idx:
        The integer index (row position) of the nearest frequency.
    f_near:
        The actual frequency (Hz) at `idx` (nearest available).

    Returns
    -------
    pandas.DataFrame
        The same `df` object (for fluent/chained usage).

    Notes
    -----
    - `DataFrame.attrs` is intended for lightweight metadata.
    - Because this mutates `df` in-place, be mindful if you reuse `df`
      across different targets; metadata will be overwritten.
    """
    df.attrs["target_frequency"] = float(target_frequency)
    df.attrs["nearest_index"] = int(idx)
    df.attrs["nearest_frequency"] = float(f_near)
    return df


def single_row_with_meta(
    df: pd.DataFrame,
    *,
    target_frequency: float,
    idx: int,
    f_near: float,
) -> pd.DataFrame:
    """
    Return a one-row DataFrame (df.iloc[[idx]]) with explicit meta columns.

    This is convenient for writing a single selected row to CSV/logs while
    keeping the target/nearest lookup info directly in the tabular schema.

    Parameters
    ----------
    df:
        Source DataFrame.
    target_frequency:
        The desired frequency (Hz) used as the lookup target.
    idx:
        Row position (iloc index) of the selected nearest row.
    f_near:
        The actual nearest frequency value (Hz) at `idx`.

    Returns
    -------
    pandas.DataFrame
        A copy of `df.iloc[[idx]]` with three new columns prepended:
        - `target_frequency`
        - `nearest_frequency`
        - `nearest_index`

    Notes
    -----
    - This function copies the selected row (`.copy()`), so modifying the
      returned DataFrame will not affect the original `df`.
    - Columns are inserted at positions 0, 1, 2 to keep metadata visible.
    """
    idx_i = int(idx)
    out = df.iloc[[idx_i]].copy()
    out.insert(0, "target_frequency", float(target_frequency))
    out.insert(1, "nearest_frequency", float(f_near))
    out.insert(2, "nearest_index", idx_i)
    return out
