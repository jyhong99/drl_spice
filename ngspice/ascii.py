from __future__ import annotations

"""
RAW-ascii parsing utilities for ngspice outputs.

This module parses ngspice RAW-ascii-like tables into structured DataFrames
and intermediate `RawAscii` records.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ngspice.errors import RawParseError
from ngspice.patterns import (
    _RE_HEADER_INT_TPL,
    _RE_INT_TOKEN,
    _RE_NUM_TOKEN,
    _RE_SPLIT_WS,
    _RE_VARS_BLOCK,
    _RE_VALUES_HDR_EOL,
)
from ngspice.textio import read_text
from ngspice.types import PathLike, RawAscii


# =============================================================================
# Header helpers
# =============================================================================

def grab_int(text: str, key: str) -> int:
    """
    Extract an integer header field from ngspice RAW-ascii text.

    Parameters
    ----------
    text:
        Full RAW-ascii text blob produced by ngspice.
    key:
        Header field name to locate (e.g., "No. Variables", "No. Points").

    Returns
    -------
    int
        Parsed integer value for the requested header key.

    Raises
    ------
    RawParseError
        If the requested header key cannot be found.

    Notes
    -----
    - Uses a template regex `_RE_HEADER_INT_TPL` and `re.escape(key)` so keys
      containing punctuation are handled safely.
    - Matching is case-insensitive to tolerate formatting differences.
    """
    pat = re.compile(_RE_HEADER_INT_TPL.format(key=re.escape(key)), flags=re.I)
    m = pat.search(text)
    if not m:
        raise RawParseError(f"Missing header field: {key}")
    return int(m.group(1))


def parse_header_and_values(text: str) -> RawAscii:
    """
    Parse RAW-ascii header information and extract the raw 'Values' lines.

    This function:
    1) reads header fields (number of variables/points),
    2) extracts the Variables block to obtain variable names in order,
    3) locates the Values section and returns all non-empty value lines.

    Parameters
    ----------
    text:
        Full RAW-ascii text.

    Returns
    -------
    ngspice.types.RawAscii
        Parsed header and raw value lines:
        - `n_var`: number of variables
        - `n_pts`: number of points
        - `var_names`: list of variable names in declared order
        - `value_lines`: raw non-empty lines from the Values section

    Raises
    ------
    RawParseError
        If any required section is missing or inconsistent:
        - missing "No. Variables"/"No. Points"
        - missing Variables..Values boundary
        - parsed variable count mismatches header
        - missing/empty Values section

    Notes
    -----
    - The Variables block is expected to include lines like:
        "<index> <name> <type> ..."
      This parser keeps only the (index, name) portion.
    - This stage intentionally does not parse numeric values; it only collects
      the raw lines for the next stage.
    """
    n_var = grab_int(text, "No. Variables")
    n_pts = grab_int(text, "No. Points")
    if n_var < 1:
        raise RawParseError(f"Invalid No. Variables value: {n_var}")
    if n_pts < 0:
        raise RawParseError(f"Invalid No. Points value: {n_pts}")

    m = _RE_VARS_BLOCK.search(text)
    if not m:
        raise RawParseError("Cannot find 'Variables' .. 'Values:' block")

    var_block_lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

    var_names: List[str] = []
    for ln in var_block_lines:
        parts = [p for p in _RE_SPLIT_WS.split(ln) if p]
        # Typical format: "<int> <var_name> <var_type> ..."
        if len(parts) >= 2 and _RE_INT_TOKEN.fullmatch(parts[0]):
            var_names.append(parts[1])

    if len(var_names) != n_var:
        raise RawParseError(
            f"n_var mismatch: header={n_var}, parsed={len(var_names)}; head={var_names[:6]}"
        )

    vpos = _RE_VALUES_HDR_EOL.search(text)
    if not vpos:
        raise RawParseError("Missing 'Values:' section")

    values_text = text[vpos.end():]
    value_lines = [ln.strip() for ln in values_text.splitlines() if ln.strip()]
    if not value_lines:
        raise RawParseError("Values section is empty")

    return RawAscii(n_var=n_var, n_pts=n_pts, var_names=var_names, value_lines=value_lines)


# =============================================================================
# Value-line parsing
# =============================================================================

def parse_value_line(line: str) -> Tuple[Optional[int], List[float]]:
    """
    Parse one value line into an optional point index and a list of floats.

    The Values section often contains tokens such as:
    - an integer index as the first token,
    - one numeric token (real) or two numeric tokens (real, imag),
    - sometimes extra punctuation or separators (commas, units, parentheses).

    This parser:
    - normalizes commas to spaces,
    - splits by whitespace,
    - strips an optional integer leading index token,
    - extracts numeric tokens that match `_RE_NUM_TOKEN`.

    Parameters
    ----------
    line:
        A raw line from the Values section.

    Returns
    -------
    (idx, nums) : tuple[Optional[int], list[float]]
        idx:
            Integer index if present as the first token, otherwise None.
        nums:
            List of floats extracted from the remainder of the line.

    Notes
    -----
    - If a token does not match `_RE_NUM_TOKEN`, we attempt a conservative
      cleanup by removing non-numeric characters:
        keep only digits, e/E, '+', '-', '.'
      and try matching again. This can salvage cases like "1.23)" or "(4e-3".
    - This function does not enforce how many floats are expected. The caller
      decides whether 1 float means real-only and 2 floats mean complex.
    """
    return _parse_value_line(line, allow_leading_index=True)


def _parse_value_line(line: str, *, allow_leading_index: bool) -> Tuple[Optional[int], List[float]]:
    """
    Parse one value line with configurable leading-index handling.

    Parameters
    ----------
    line:
        Raw Values-section line.
    allow_leading_index:
        If True, a leading integer token is treated as a point index and removed
        from numeric payload parsing. If False, all tokens are treated as data.

    Returns
    -------
    tuple[Optional[int], list[float]]
        Optional integer index and parsed numeric payload.
    """
    s = line.strip().replace(",", " ")
    parts = [p for p in _RE_SPLIT_WS.split(s) if p]

    idx: Optional[int] = None
    if allow_leading_index and parts and _RE_INT_TOKEN.fullmatch(parts[0]):
        idx = int(parts[0])
        parts = parts[1:]

    nums: List[float] = []
    for p in parts:
        # Fast path: token is a clean numeric literal
        if _RE_NUM_TOKEN.fullmatch(p):
            nums.append(float(p))
            continue

        # Slow path: salvage numeric substring by stripping junk characters
        q = re.sub(r"[^\deE\+\-\.]", "", p)
        if q and _RE_NUM_TOKEN.fullmatch(q):
            nums.append(float(q))

    return idx, nums


def _looks_like_indexed_values(lines: List[str], *, n_var: int, n_pts: int) -> bool:
    """
    Heuristically detect whether RAW Values lines include leading point indices.

    Parameters
    ----------
    lines:
        Raw non-empty Values lines.
    n_var:
        Number of variables (lines per point block).
    n_pts:
        Number of expected points.

    Returns
    -------
    bool
        True when a consistent repeated-point-index layout is detected.
    """
    if n_var < 1 or n_pts < 1:
        return False

    # Probe only a prefix for performance while requiring enough evidence.
    probe_pts = min(n_pts, 5)
    need = probe_pts * n_var
    if len(lines) < need:
        return False

    for p in range(probe_pts):
        expect = p
        for j in range(n_var):
            tokens = [t for t in _RE_SPLIT_WS.split(lines[p * n_var + j].strip()) if t]
            if len(tokens) < 2:
                return False
            if not _RE_INT_TOKEN.fullmatch(tokens[0]):
                return False
            if int(tokens[0]) != expect:
                return False
    return True


# =============================================================================
# RAW-ascii -> arrays
# =============================================================================

def values_to_arrays(raw: RawAscii) -> Dict[str, np.ndarray]:
    """
    Convert parsed RAW-ascii value lines into per-variable NumPy arrays.

    ngspice RAW-ascii Values typically encodes each *point* as a block of
    `n_var` lines:
      - first line: value for variable 0 (often includes point index)
      - next (n_var-1) lines: values for variables 1..(n_var-1)

    This function walks the `value_lines`, finds point starts, and consumes
    blocks of `n_var` lines until `n_pts` points are parsed.

    Parameters
    ----------
    raw:
        Parsed RAW-ascii skeleton containing `n_var`, `n_pts`, `var_names`,
        and `value_lines`.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from variable name to a NumPy array of length `n_pts`.
        Arrays are:
        - float64 if all imaginary parts are (near) zero,
        - complex128 otherwise.

    Raises
    ------
    RawParseError
        If:
        - a point block ends unexpectedly (EOF in the middle of a block),
        - a line cannot be parsed into at least one numeric value,
        - the number of parsed points does not match header `n_pts`.

    Notes
    -----
    - If a line yields one float, it is treated as a real value.
      If it yields >=2 floats, the first two are treated as (real, imag).
    - Point start detection uses `_RE_INT_PREFIX` to identify lines that begin
      with an integer token.
    """
    n_var, n_pts = raw.n_var, raw.n_pts
    names, lines = raw.var_names, raw.value_lines

    # Accumulate as Python complex numbers first; convert at the end.
    out: Dict[str, List[complex]] = {name: [] for name in names}

    def push(name: str, nums: List[float], ctx_line: str) -> None:
        """
        Append one parsed value into `out[name]`, interpreting nums as real/complex.

        Raises
        ------
        RawParseError
            If `nums` is empty (cannot parse a numeric value).
        """
        if not nums:
            raise RawParseError(f"Could not parse value line: {ctx_line!r}")
        if len(nums) == 1:
            out[name].append(complex(nums[0], 0.0))
        else:
            out[name].append(complex(nums[0], nums[1]))

    has_indexed_rows = _looks_like_indexed_values(lines, n_var=n_var, n_pts=n_pts)
    i = 0
    pts = 0

    if has_indexed_rows:
        while i < len(lines) and pts < n_pts:
            # Skip non-point-start lines (defensive: some dumps include extra lines)
            tokens_i = [t for t in _RE_SPLIT_WS.split(lines[i].strip()) if t]
            if not tokens_i or not _RE_INT_TOKEN.fullmatch(tokens_i[0]):
                i += 1
                continue

            # First variable of this point block
            _, nums0 = _parse_value_line(lines[i], allow_leading_index=True)
            push(names[0], nums0, lines[i])
            i += 1

            # Remaining variables in this point block
            for j in range(1, n_var):
                if i >= len(lines):
                    raise RawParseError("Unexpected EOF while reading a point block")
                _, nums = _parse_value_line(lines[i], allow_leading_index=True)
                push(names[j], nums, lines[i])
                i += 1

            pts += 1
    else:
        # Fallback for raw dumps that omit explicit integer point indices.
        while i < len(lines) and pts < n_pts:
            for j in range(n_var):
                if i >= len(lines):
                    raise RawParseError("Unexpected EOF while reading an index-free point block")
                _, nums = _parse_value_line(lines[i], allow_leading_index=False)
                push(names[j], nums, lines[i])
                i += 1
            pts += 1

    if pts != n_pts:
        raise RawParseError(f"n_pts mismatch: header={n_pts}, parsed={pts}")

    # Convert to numpy arrays; downcast to real if imag is numerically zero.
    arrs: Dict[str, np.ndarray] = {}
    for k, vals in out.items():
        c = np.asarray(vals, dtype=np.complex128)
        if np.all(np.abs(c.imag) < 1e-30):
            arrs[k] = c.real.astype(np.float64)
        else:
            arrs[k] = c

    return arrs


# =============================================================================
# Public API
# =============================================================================

def read_raw_table(path: PathLike) -> pd.DataFrame:
    """
    Read an ngspice RAW-ascii output file and convert it into a DataFrame.

    Parameters
    ----------
    path:
        Path to the RAW-ascii text file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one column per variable and `n_pts` rows.

    Raises
    ------
    RawParseError
        If header/values parsing fails or if the resulting row count mismatches
        the header `No. Points`.

    Notes
    -----
    - This function performs a strict consistency check: `raw.n_pts` must match
      `len(df)`. If not, the file is treated as malformed/incomplete.
    """
    text = read_text(path)
    raw = parse_header_and_values(text)
    arrs = values_to_arrays(raw)
    df = pd.DataFrame(arrs)

    if raw.n_pts != len(df):
        raise RawParseError(f"n_pts mismatch: header={raw.n_pts}, parsed_rows={len(df)}")

    return df
