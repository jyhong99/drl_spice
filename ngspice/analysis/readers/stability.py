from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ngspice.analysis.registry import register
from ngspice.ascii import read_raw_table
from ngspice.utils import find_freq_col, pick_col


@register("Stability Factor")
def _factory() -> "StabilityReader":
    """
    Factory for registry-based reader construction.

    Returns
    -------
    StabilityReader
        A new reader instance.
    """
    return StabilityReader()


class StabilityReader:
    """
    Reader that computes RF stability metrics from S-parameters.

    This reader loads an S-parameter table (via `read_raw_table`) and computes:
    - Rollett stability factor K
    - Stability measures μ and μ' (mu-prime)
    - A simple 3 dB bandwidth estimate based on |S21| (gain magnitude)

    The returned DataFrame is a single-row summary intended for optimization
    loops and RL environments.

    Attributes
    ----------
    analysis_type:
        Registry key for this reader ("Stability Factor").

    Notes
    -----
    - The reader expects S-parameters (S11, S12, S21, S22) to be present as
      complex-valued columns in the table. Column naming variations are
      supported via `pick_col` candidate lists.
    - Division denominators are clamped with a small epsilon to avoid numerical
      blow-ups when S12*S21 is close to zero (e.g., at frequencies with poor
      numerical conditioning).
    - The optional frequency filter (>= 1e6 Hz) is a heuristic to ignore very
      low-frequency points that may be irrelevant or numerically odd for RF
      stability analysis. Adjust or remove it if needed.
    """

    analysis_type = "Stability Factor"

    # Heuristic: ignore extremely low frequencies for stability calculations.
    _FREQ_MIN_HZ = 1e6

    # Small clamp to avoid division-by-zero / NaNs.
    _EPS = 1e-18

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Parse an S-parameter table and compute stability summary metrics.

        Parameters
        ----------
        result_path:
            Path to the raw-ascii S-parameter table file.
        **kwargs:
            No keyword arguments are supported. Passing any kwargs raises
            `ValueError` to prevent silent misconfiguration.

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame containing:
            - K_min : float
                Minimum Rollett stability factor K over the frequency sweep.
            - mu_min : float
                Minimum μ over the frequency sweep.
            - mup_min : float
                Minimum μ' (mu-prime) over the frequency sweep.
            - mu_mup_min : float
                min(mu_min, mup_min), a conservative combined stability metric.
            - bandwidth_hz : float
                Estimated 3 dB bandwidth (Hz) based on |S21|, computed as the
                width of the frequency interval where gain is within 3 dB of peak.

        Raises
        ------
        ValueError
            If unsupported kwargs are provided, required columns are missing,
            coercion to complex fails, or the frequency filter removes all data.
        Exception
            Parse errors may propagate from `read_raw_table` (e.g., `RawParseError`).

        Notes
        -----
        - K is computed as:
              Δ = S11*S22 - S12*S21
              K = (1 - |S11|^2 - |S22|^2 + |Δ|^2) / (2*|S12*S21|)
        - μ and μ' are computed as:
              μ  = (1 - |S11|^2) / (|S22 - Δ*conj(S11)| + |S12*S21|)
              μ' = (1 - |S22|^2) / (|S11 - Δ*conj(S22)| + |S12*S21|)
        - 3 dB bandwidth is estimated from gain magnitude (|S21| in dB):
              threshold = peak(|S21|_dB) - 3
              bandwidth = f_high - f_low over all points above threshold.
          This is a coarse estimate (no interpolation); refine if needed.
        """
        # Protocol compatibility + fail-fast for silent misconfiguration.
        if kwargs:
            raise ValueError(f"{self.analysis_type}: unsupported kwargs={sorted(kwargs)}")

        df = read_raw_table(result_path)

        # Frequency axis.
        freq_col = find_freq_col(df)
        freq = df[freq_col].to_numpy(copy=False)
        if np.iscomplexobj(freq):
            freq = np.real(freq)
        freq = np.asarray(freq, dtype=float)
        if not np.all(np.isfinite(freq)):
            raise ValueError(f"{self.analysis_type}: frequency column contains non-finite values ({freq_col!r})")

        def _get_s(col_candidates: tuple[str, ...]) -> np.ndarray:
            """
            Resolve and coerce an S-parameter column to complex128.

            Parameters
            ----------
            col_candidates:
                Candidate column names to try in order.

            Returns
            -------
            numpy.ndarray
                Complex128 array of S-parameter values.

            Raises
            ------
            ValueError
                If coercion fails.
            """
            c = pick_col(df, *col_candidates)
            try:
                return df[c].to_numpy(dtype=np.complex128, copy=False)
            except Exception as e:
                raise ValueError(
                    f"{self.analysis_type}: cannot coerce column {c!r} to complex128 "
                    f"(file={result_path!r})"
                ) from e

        # Required S-parameters.
        s11 = _get_s(("v(s_1_1)", "s_1_1", "S_1_1"))
        s12 = _get_s(("v(s_1_2)", "s_1_2", "S_1_2"))
        s21 = _get_s(("v(s_2_1)", "s_2_1", "S_2_1"))
        s22 = _get_s(("v(s_2_2)", "s_2_2", "S_2_2"))
        for name, arr in (("s11", s11), ("s12", s12), ("s21", s21), ("s22", s22)):
            if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
                raise ValueError(f"{self.analysis_type}: non-finite values detected in {name}")

        # Optional heuristic: filter out very low frequencies.
        # If the frequency axis appears to be in a smaller unit (e.g., GHz),
        # disable the filter to avoid dropping all samples.
        freq_max = float(np.max(freq)) if freq.size else 0.0
        if freq_max < float(self._FREQ_MIN_HZ):
            mask = np.ones_like(freq, dtype=bool)
        else:
            mask = freq >= float(self._FREQ_MIN_HZ)
        freq, s11, s12, s21, s22 = freq[mask], s11[mask], s12[mask], s21[mask], s22[mask]

        if freq.size == 0:
            raise ValueError(
                f"{self.analysis_type}: no samples after frequency filter "
                f"(freq>={self._FREQ_MIN_HZ:g} Hz). file={result_path!r}"
            )

        eps = float(self._EPS)

        # Stability metrics.
        Delta = s11 * s22 - s12 * s21

        num_K = 1.0 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(Delta) ** 2
        den_K = 2.0 * np.abs(s12 * s21)
        K = num_K / np.maximum(den_K, eps)

        num_mu = 1.0 - np.abs(s11) ** 2
        den_mu = np.abs(s22 - Delta * np.conj(s11)) + np.abs(s12 * s21)
        mu = num_mu / np.maximum(den_mu, eps)

        num_mup = 1.0 - np.abs(s22) ** 2
        den_mup = np.abs(s11 - Delta * np.conj(s22)) + np.abs(s12 * s21)
        mup = num_mup / np.maximum(den_mup, eps)

        # -----------------------------------------------------------------
        # 3 dB bandwidth estimate based on |S21|
        # -----------------------------------------------------------------
        mag_s21 = np.maximum(np.abs(s21), 1e-300)
        mag_s21_db = 20.0 * np.log10(mag_s21)

        peak_db = float(np.max(mag_s21_db))
        thr_db = peak_db - 3.0
        above = mag_s21_db >= thr_db

        if np.any(above):
            idxs = np.where(above)[0]
            f_low = float(freq[idxs[0]])
            f_high = float(freq[idxs[-1]])
            bandwidth_hz = max(0.0, f_high - f_low)
        else:
            bandwidth_hz = 0.0

        # Summary statistics.
        k_min = float(np.min(K))
        mu_min = float(np.min(mu))
        mup_min = float(np.min(mup))

        return pd.DataFrame(
            {
                "K_min": [k_min],
                "mu_min": [mu_min],
                "mup_min": [mup_min],
                "mu_mup_min": [float(min(mu_min, mup_min))],
                "bandwidth_hz": [float(bandwidth_hz)],
            }
        )
