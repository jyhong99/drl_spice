from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ngspice.analysis.registry import register
from ngspice.ascii import read_raw_table
from ngspice.utils import (
    attach_nearest_meta,
    find_freq_col,
    nearest_index,
    single_row_with_meta,
)


@register("Noise Analysis")
def _factory() -> "NoiseReader":
    """
    Factory for registry-based reader construction.

    Returns
    -------
    NoiseReader
        A new reader instance.
    """
    return NoiseReader()


class NoiseReader:
    """
    Reader for ngspice noise analysis outputs (Noise Figure vs frequency).

    This reader loads a RAW-ascii-like table (via `read_raw_table`) into a
    DataFrame, identifies the frequency column, identifies/normalizes the
    noise-figure column to the canonical name ``"NoiseFigure"``, and then
    returns either:

    - the full table with nearest-frequency metadata attached (`return_full=True`), or
    - a single-row selection at the nearest frequency (`return_full=False`).

    Attributes
    ----------
    analysis_type:
        Registry key for this reader ("Noise Analysis").

    Notes
    -----
    - The reader is intentionally strict about schema: it will not silently guess
      the frequency column, and it will only infer the NoiseFigure column under
      a conservative heuristic (exactly two columns total).
    - Nearest-frequency lookup metadata is attached either as `df.attrs`
      (full-table mode) or as explicit prepended columns (single-row mode),
      depending on the helper used.
    """

    analysis_type = "Noise Analysis"

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Parse a noise analysis result table and select a target frequency.

        Parameters
        ----------
        result_path:
            Path to a RAW-ascii table file to parse (ngspice output).
        **kwargs:
            Reader options. Supported keys:

            target_frequency : float, optional
                Target frequency in Hz for nearest-bin selection.
                Default is 2.4e9.
            return_full : bool, optional
                If True, return the full DataFrame and attach metadata via
                `df.attrs`. If False, return a single-row DataFrame with explicit
                meta columns prepended. Default is False.

        Returns
        -------
        pandas.DataFrame
            If `return_full=False` (default), a single-row DataFrame at the
            nearest frequency to `target_frequency`, with columns:
              target_frequency, nearest_frequency, nearest_index, ...
            followed by the original data columns.

            If `return_full=True`, the full parsed DataFrame with lookup metadata
            stored in `df.attrs`:
              df.attrs["target_frequency"]
              df.attrs["nearest_frequency"]
              df.attrs["nearest_index"]

        Raises
        ------
        ValueError
            If unsupported kwargs are provided or required columns cannot be
            identified (frequency column or NoiseFigure column).
        Exception
            Parsing errors may propagate from `read_raw_table` and downstream
            helpers (e.g., `RawParseError`, `MissingColumnError`).

        Notes
        -----
        - The NoiseFigure column is normalized case-insensitively when an exact
          match to "noisefigure" is found (e.g., "noisefigure" -> "NoiseFigure").
        - If the NoiseFigure column is not present and the table has exactly two
          columns, the non-frequency column is assumed to be NoiseFigure.
          Otherwise, the reader raises to avoid silent schema mistakes.
        """
        target_frequency = float(kwargs.pop("target_frequency", 2.4e9))
        return_full = bool(kwargs.pop("return_full", False))

        if kwargs:
            raise ValueError(f"{self.analysis_type}: unsupported kwargs={sorted(kwargs)}")
        if (not math.isfinite(target_frequency)) or target_frequency < 0.0:
            raise ValueError(
                f"{self.analysis_type}: target_frequency must be finite and >= 0 (got {target_frequency})"
            )

        df = read_raw_table(result_path)

        # 1) Find frequency column first (avoid accidental renaming/heuristics).
        freq_col = find_freq_col(df)

        # 2) Normalize noise-figure column name to the canonical "NoiseFigure"
        #    using a strict case-insensitive exact match.
        if "NoiseFigure" not in df.columns:
            for c in list(df.columns):
                if str(c).strip().lower() == "noisefigure":
                    if c != "NoiseFigure":
                        df = df.rename(columns={c: "NoiseFigure"})
                    break

        # 3) If still missing, apply a conservative heuristic only when the table
        #    is exactly two columns: [freq_like, other] -> other is NoiseFigure.
        if "NoiseFigure" not in df.columns:
            if df.shape[1] == 2:
                other = df.columns[0] if df.columns[1] == freq_col else df.columns[1]
                if other == freq_col:
                    raise ValueError(
                        f"{self.analysis_type}: cannot infer NoiseFigure column (both columns look like frequency). "
                        f"columns={list(df.columns)} file={result_path!r}"
                    )
                df = df.rename(columns={other: "NoiseFigure"})
            else:
                raise ValueError(
                    f"{self.analysis_type}: NoiseFigure column not found. "
                    f"columns={list(df.columns)[:30]} file={result_path!r}"
                )

        # Nearest-frequency selection.
        freq = df[freq_col].to_numpy(copy=False)
        if np.iscomplexobj(freq):
            freq = np.real(freq)
        freq = np.asarray(freq, dtype=float)
        if not np.all(np.isfinite(freq)):
            raise ValueError(f"{self.analysis_type}: frequency column contains non-finite values ({freq_col!r})")

        idx = nearest_index(freq, target_frequency)
        f_near = float(freq[idx])

        if return_full:
            return attach_nearest_meta(df, target_frequency=target_frequency, idx=idx, f_near=f_near)

        return single_row_with_meta(df, target_frequency=target_frequency, idx=idx, f_near=f_near)
