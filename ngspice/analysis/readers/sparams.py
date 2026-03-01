from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from ngspice.analysis.registry import register
from ngspice.ascii import read_raw_table
from ngspice.utils import attach_nearest_meta, find_freq_col, nearest_index, single_row_with_meta


@register("S-Parameter Analysis")
def _factory() -> "SParameterReader":
    """
    Factory for registry-based reader construction.

    Returns
    -------
    SParameterReader
        A new reader instance.
    """
    return SParameterReader()


class SParameterReader:
    """
    Reader for ngspice S-parameter analysis outputs.

    This reader loads a RAW-ascii-like table (via `read_raw_table`) and produces
    canonical dB-magnitude columns:

    - S11_dB, S12_dB, S21_dB, S22_dB

    It supports typical ngspice/raw naming variations by mapping each canonical
    output column to a list of candidate source columns found in the raw table.

    After computing the canonical columns, the reader selects either:
    - the full DataFrame with nearest-frequency metadata attached (`return_full=True`), or
    - a single-row DataFrame at the nearest frequency (`return_full=False`).

    Attributes
    ----------
    analysis_type:
        Registry key for this reader ("S-Parameter Analysis").
    _S_MAP:
        Mapping: canonical output column name -> candidate source column names.

    Notes
    -----
    - Source columns are coerced to `np.complex128`. If the raw table stores
      complex numbers as Python complex or as (re, im) pairs serialized by
      your `read_raw_table`, coercion should succeed.
    - dB magnitude is computed as:
          20*log10(|Sxy|)
      and an epsilon clamp is applied to avoid log10(0).
    """

    analysis_type = "S-Parameter Analysis"

    # Canonical mapping: output column -> candidate source columns in raw table.
    _S_MAP: Dict[str, Tuple[str, ...]] = {
        "S11_dB": ("v(s_1_1)", "s_1_1", "S_1_1"),
        "S12_dB": ("v(s_1_2)", "s_1_2", "S_1_2"),
        "S21_dB": ("v(s_2_1)", "s_2_1", "S_2_1"),
        "S22_dB": ("v(s_2_2)", "s_2_2", "S_2_2"),
    }

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Parse an S-parameter result table and select a target frequency.

        Parameters
        ----------
        result_path:
            Path to the raw-ascii table file produced by ngspice.
        **kwargs:
            Reader options. Supported keys:

            target_frequency : float, optional
                Target frequency in Hz for nearest-bin selection.
                Default is 2.4e9.
            return_full : bool, optional
                If True, return the full DataFrame with nearest-frequency
                metadata in `df.attrs`. If False, return a single-row DataFrame
                with explicit meta columns prepended. Default is False.

        Returns
        -------
        pandas.DataFrame
            If `return_full=False` (default), a single-row DataFrame at the
            nearest frequency to `target_frequency`, with meta columns:
              target_frequency, nearest_frequency, nearest_index, ...
            followed by the original/raw columns and computed canonical Sxx_dB
            columns.

            If `return_full=True`, the full parsed DataFrame with lookup metadata
            stored in `df.attrs`.

        Raises
        ------
        ValueError
            If unsupported kwargs are provided; if required S-parameter columns
            cannot be produced; or if source columns cannot be coerced to complex.
        Exception
            Parsing errors may propagate from `read_raw_table` and downstream
            helpers (e.g., `RawParseError`, `MissingColumnError`).

        Notes
        -----
        - Required outputs are enforced to match downstream simulator/collector
          expectations. By default this reader requires:
            {"S11_dB", "S21_dB", "S22_dB"}.
        - dB conversion clamps |S| with a tiny epsilon to avoid -inf when |S|=0.
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

        # Frequency axis and nearest-bin selection.
        freq_col = find_freq_col(df)
        freq = df[freq_col].to_numpy(copy=False)
        if np.iscomplexobj(freq):
            freq = np.real(freq)
        freq = np.asarray(freq, dtype=float)
        if not np.all(np.isfinite(freq)):
            raise ValueError(f"{self.analysis_type}: frequency column contains non-finite values ({freq_col!r})")

        idx = nearest_index(freq, target_frequency)
        f_near = float(freq[idx])

        # Compute canonical Sxx_dB columns when corresponding source columns exist.
        produced: set[str] = set()

        for dst, srcs in self._S_MAP.items():
            src_found: str | None = None
            for src in srcs:
                if src in df.columns:
                    src_found = src
                    break
            if src_found is None:
                continue

            try:
                x = df[src_found].to_numpy(dtype=np.complex128, copy=False)
            except Exception as e:
                raise ValueError(
                    f"{self.analysis_type}: cannot coerce source column {src_found!r} to complex128 "
                    f"(file={result_path!r})"
                ) from e

            # 20*log10(|S|) with clamp to avoid log10(0).
            df[dst] = 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))
            produced.add(dst)

        # Enforce required outputs (align with your simulator collector expectations).
        required = {"S11_dB", "S21_dB", "S22_dB"}
        missing = sorted(required - produced)
        if missing:
            raise ValueError(
                f"{self.analysis_type}: missing required S-parameter columns {missing}. "
                f"available_columns={list(df.columns)[:30]} file={result_path!r}"
            )

        if return_full:
            return attach_nearest_meta(df, target_frequency=target_frequency, idx=idx, f_near=f_near)

        return single_row_with_meta(df, target_frequency=target_frequency, idx=idx, f_near=f_near)
