from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from ngspice.errors import MarkerMalformed, MarkerNotFound
from ngspice.types import Marker


# =============================================================================
# Marker utilities
# =============================================================================

def has_marker(text: str, marker: Marker) -> bool:
    """
    Check whether a marker region exists in the given text.

    A marker region is considered present only if:
    1) `marker.start` exists, and
    2) `marker.end` exists *after* the start marker.

    Parameters
    ----------
    text:
        Input text (e.g., a netlist template).
    marker:
        Marker object containing `start` and `end` sentinel strings.

    Returns
    -------
    bool
        True if both start and end markers exist in the correct order, else False.

    Notes
    -----
    - This performs a simple `str.find` search. It does not validate uniqueness
      (e.g., multiple marker regions).
    - `marker.end` is searched starting from the end of the start marker to
      enforce correct ordering.
    """
    if not marker.start or not marker.end:
        return False

    s = text.find(marker.start)
    if s < 0:
        return False
    e = text.find(marker.end, s + len(marker.start))
    return e >= 0


def split_around_marker(text: str, marker: Marker) -> tuple[str, str]:
    """
    Split text into (before, after) segments around a marker-delimited region.

    This function finds the first occurrence of `marker.start`, then the first
    occurrence of `marker.end` after it, and returns:
    - `before`: text preceding the start marker (with trailing whitespace trimmed)
    - `after`:  text following the end marker (with leading whitespace trimmed)

    Parameters
    ----------
    text:
        Input text containing marker boundaries.
    marker:
        Marker object with `start` and `end` sentinel strings.

    Returns
    -------
    (before, after) : tuple[str, str]
        before:
            `text[:start_index]` with `.rstrip()` applied, plus a single trailing newline.
        after:
            `text[end_index_after_marker:].lstrip()`.

    Raises
    ------
    MarkerMalformed
        If the start marker is missing, or the end marker does not occur after
        the start marker.

    Notes
    -----
    - The returned `before` is normalized to end with exactly one newline.
      This makes later concatenation deterministic.
    - The returned `after` is `.lstrip()`-ed so the patched block can be inserted
      cleanly with a blank line separator if desired.
    """
    s = text.find(marker.start)
    if s < 0:
        raise MarkerMalformed(marker)

    e = text.find(marker.end, s + len(marker.start))
    if e < 0:
        raise MarkerMalformed(marker)

    e2 = e + len(marker.end)
    before = text[:s].rstrip() + "\n"
    after = text[e2:].lstrip()
    return before, after


# =============================================================================
# Patching
# =============================================================================

def iter_items(params: Mapping[str, Any], *, sort_keys: bool) -> Iterable[tuple[str, Any]]:
    """
    Iterate mapping items optionally sorted by key (as string).

    Parameters
    ----------
    params:
        Mapping from parameter name to value.
    sort_keys:
        If True, items are sorted by `str(key)` for deterministic output.
        If False, preserve the mapping's iteration order.

    Returns
    -------
    Iterable[tuple[str, Any]]
        Iterable of (key, value) pairs.

    Notes
    -----
    - Sorting by `str(key)` is robust even if keys are not strictly `str`
      (though upstream contracts typically use string keys).
    """
    items = params.items()
    return sorted(items, key=lambda kv: str(kv[0])) if sort_keys else items


def patch_block(
    text: str,
    *,
    marker: Marker,
    params: Mapping[str, Any],
    line_builder: Callable[[str, Any], str],
    sort_keys: bool = True,
    required: bool = True,
) -> str:
    """
    Replace the content of a marker-delimited block with newly rendered lines.

    The patched block is rendered as:

    - marker.start
    - one line per `(k, v)` in `params`, using `line_builder(k, v)`
    - marker.end

    The returned string is a modified copy of `text`, where the *first* marker
    region encountered is replaced by the newly generated block.

    Parameters
    ----------
    text:
        Original text (e.g., a netlist template).
    marker:
        Marker defining the region to replace (`start` and `end` sentinels).
    params:
        Parameter mapping used to generate lines within the marker block.
    line_builder:
        Callable that renders a single line from `(key, value)`.
        Examples:
        - lambda k, v: f".param {k} = {v}"
        - lambda k, v: f"let {k} = {v}"
    sort_keys:
        If True, keys are emitted in deterministic sorted order.
    required:
        If True, missing markers raise an exception.
        If False, missing markers return `text` unchanged.

    Returns
    -------
    str
        Patched text.

    Raises
    ------
    MarkerNotFound
        If `required=True` and the start marker is missing.
    MarkerMalformed
        If the start marker exists but the end marker does not occur after it
        (and `required=True`).

    Notes
    -----
    - This function patches only the *first* marker region it finds.
    - Whitespace normalization policy:
      * `before` segment uses `.rstrip()` and appends one newline.
      * `after` segment uses `.lstrip()`.
      * the patched block is followed by two newlines before `after` to keep
        netlists readable and to avoid accidental line continuation issues.
    - If `required=False` and markers are malformed (start exists but end missing),
      the current policy is to return `text` unchanged (best-effort non-failing).
    """
    if not callable(line_builder):
        raise TypeError("line_builder must be callable")

    # Locate markers in a single pass.
    s = text.find(marker.start)
    if s < 0:
        if required:
            raise MarkerNotFound(marker, required=True)
        return text

    e = text.find(marker.end, s + len(marker.start))
    if e < 0:
        # Start exists but end missing -> malformed.
        if required:
            raise MarkerMalformed(marker)
        return text

    # Compute before/after slices around the *inclusive* marker region.
    e2 = e + len(marker.end)
    before = text[:s].rstrip() + "\n"
    after = text[e2:].lstrip()

    # Render patched block.
    lines: list[str] = [marker.start]
    for k, v in iter_items(params, sort_keys=sort_keys):
        k2 = str(k).strip()
        if not k2:
            continue
        lines.append(str(line_builder(k2, v)))
    lines.append(marker.end)

    return before + "\n".join(lines) + "\n\n" + after
