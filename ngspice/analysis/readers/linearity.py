from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ngspice.analysis.registry import register
from ngspice.errors import RawParseError
from ngspice.patterns import _RE_NUM_TOKEN, _RE_VALUES_HDR
from ngspice.textio import read_text
from ngspice.utils import nearest_index


@register("Linearity")
def _factory() -> "LinearityReader":
    """
    Factory for registry-based reader construction.

    Returns
    -------
    LinearityReader
        A new reader instance.
    """
    return LinearityReader()


class LinearityReader:
    """
    Reader for two-tone linearity results (IIP3 estimation).

    This reader parses a *RAW-ascii-like* output that contains a ``Values:``
    section. The expected structure after the ``Values:`` header is an
    alternating pair format:

    - Frequency line (contains a numeric frequency token somewhere in the line)
    - Complex value line (contains at least two numeric tokens: real imag)

    Repeated for each frequency bin:
        freq_line_0
        value_line_0
        freq_line_1
        value_line_1
        ...

    The reader:
    1) extracts the frequency bins and complex output voltage samples,
    2) computes magnitudes |Vout(f)|,
    3) finds nearest bins to f1, f2 and IM3 products (2*f1-f2, 2*f2-f1),
    4) converts peak voltage magnitudes to average power (dBm) using R_load,
    5) estimates IIP3 by:
         IIP3 = Pin + (Pfund_avg - Pim3_avg) / 2

    Attributes
    ----------
    analysis_type:
        Registry key for this reader ("Linearity").

    Notes
    -----
    - Voltage magnitude is treated as *peak* amplitude. Average power is
      computed as:
          P_avg = V_peak^2 / (2 * R)
      This is consistent with a sinusoid v(t)=V_peak*sin(ωt) across a resistor.
    - Nearest-bin selection is sensitive to FFT/grid resolution. Ensure your
      frequency bin spacing is fine enough around f1, f2, and IM3 products.
    """

    analysis_type = "Linearity"
    _MIN_LOG_W = 1e-30

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Parse a linearity result artifact and compute IIP3 summary metrics.

        Parameters
        ----------
        result_path:
            Path to the linearity result file (text) to parse.
        **kwargs:
            Reader options (Protocol/Facade compatible). Supported keys:

            Vin_amp : float, optional
                Input tone peak amplitude (Volts). Default is 5e-3.
            f1 : float, optional
                Fundamental tone #1 frequency (Hz). Default is 2.399e9.
            f2 : float, optional
                Fundamental tone #2 frequency (Hz). Default is 2.401e9.
            R_load : float, optional
                Resistive load used for power conversion (Ohms). Default is 50.0.

        Returns
        -------
        pandas.DataFrame
            Single-row summary table with columns:
            - Pin_dBm
            - Pfund_dBm_avg
            - Pim3_dBm_avg
            - IIP3_dBm

        Raises
        ------
        ValueError
            If unsupported kwargs are provided, or if parameters are invalid
            (e.g., R_load <= 0, Vin_amp < 0).
        RawParseError
            If required sections are missing or the expected alternating data
            format cannot be parsed.

        Notes
        -----
        - This returns a compact summary (one row). If you need the full spectrum
          (freqs, complex vout, magnitudes), consider adding a `return_full`
          option and returning additional tables or attaching arrays via `df.attrs`.
        """
        # Unified kwargs interface (Protocol/Facade compatible)
        Vin_amp = float(kwargs.pop("Vin_amp", 5e-3))
        f1 = float(kwargs.pop("f1", 2.399e9))
        f2 = float(kwargs.pop("f2", 2.401e9))
        R_load = float(kwargs.pop("R_load", 50.0))

        if kwargs:
            raise ValueError(f"{self.analysis_type}: unsupported kwargs={sorted(kwargs)}")

        if R_load <= 0:
            raise ValueError(f"{self.analysis_type}: R_load must be > 0 (got {R_load})")
        if Vin_amp < 0:
            raise ValueError(f"{self.analysis_type}: Vin_amp must be >= 0 (got {Vin_amp})")
        if (not math.isfinite(Vin_amp)) or (not math.isfinite(f1)) or (not math.isfinite(f2)) or (not math.isfinite(R_load)):
            raise ValueError(f"{self.analysis_type}: Vin_amp/f1/f2/R_load must be finite")
        if f1 <= 0.0 or f2 <= 0.0:
            raise ValueError(f"{self.analysis_type}: f1 and f2 must be > 0 (got f1={f1}, f2={f2})")

        text = read_text(result_path)
        lines = text.splitlines()

        # Locate the "Values:" header (line index right after the header).
        start: int | None = None
        for i, line in enumerate(lines):
            if _RE_VALUES_HDR.match(line):
                start = i + 1
                break
        if start is None:
            raise RawParseError(
                f"{self.analysis_type}: Missing 'Values:' in linearity result: {result_path!r}"
            )

        data_lines = [ln.strip() for ln in lines[start:] if ln.strip()]
        if len(data_lines) < 2:
            raise RawParseError(f"{self.analysis_type}: Not enough data lines for linearity")

        # Alternating format: freq_line, value_line, freq_line, value_line, ...
        freq_lines = data_lines[0::2]
        val_lines = data_lines[1::2]
        if len(freq_lines) != len(val_lines):
            raise RawParseError(f"{self.analysis_type}: freq/value lines count mismatch")

        n = len(freq_lines)
        freqs = np.empty(n, dtype=float)
        reals = np.empty(n, dtype=float)
        imags = np.empty(n, dtype=float)

        for k, (f_line, v_line) in enumerate(zip(freq_lines, val_lines)):
            # Frequency line often starts with a point index. Use the last numeric token.
            f_tokens = _RE_NUM_TOKEN.findall(f_line)
            if not f_tokens:
                raise RawParseError(f"{self.analysis_type}: Cannot parse frequency from line: {f_line!r}")
            freqs[k] = float(f_tokens[-1])

            # Value lines may include a leading point index; use the last two
            # numeric tokens as real/imag.
            nums = [float(t) for t in _RE_NUM_TOKEN.findall(v_line)]
            if len(nums) < 2:
                raise RawParseError(f"{self.analysis_type}: Cannot parse complex vout from line: {v_line!r}")
            reals[k], imags[k] = nums[-2], nums[-1]

        if not np.all(np.isfinite(freqs)):
            raise RawParseError(f"{self.analysis_type}: parsed non-finite frequency bins from {result_path!r}")

        vout = reals + 1j * imags
        mag = np.abs(vout)

        # IM3 frequencies for a standard two-tone test.
        im3_1 = 2.0 * f1 - f2
        im3_2 = 2.0 * f2 - f1

        def nearest_mag(target_hz: float) -> float:
            """
            Return |Vout| at the nearest frequency bin to `target_hz`.

            Parameters
            ----------
            target_hz:
                Target frequency in Hz.

            Returns
            -------
            float
                Magnitude at the closest available bin.
            """
            idx = nearest_index(freqs, target_hz)
            return float(mag[idx])

        v_fund1 = nearest_mag(f1)
        v_fund2 = nearest_mag(f2)
        v_im3_1 = nearest_mag(im3_1)
        v_im3_2 = nearest_mag(im3_2)

        def p_dbm_from_vpeak(v_peak: float, r_ohm: float) -> float:
            """
            Convert peak sinusoidal voltage amplitude to average power in dBm.

            Parameters
            ----------
            v_peak:
                Peak amplitude of a sinusoid (Volts).
            r_ohm:
                Load resistance (Ohms). Must be positive.

            Returns
            -------
            float
                Average power in dBm.

            Notes
            -----
            - Uses average power for a sinusoid:
                  P_avg = V_peak^2 / (2R)
            - Adds a small epsilon inside log10 to avoid -inf for zero power.
            """
            p_w = (v_peak**2) / (2.0 * r_ohm)
            return float(10.0 * np.log10(max(p_w * 1e3, self._MIN_LOG_W)))

        pf1 = p_dbm_from_vpeak(v_fund1, R_load)
        pf2 = p_dbm_from_vpeak(v_fund2, R_load)
        pim31 = p_dbm_from_vpeak(v_im3_1, R_load)
        pim32 = p_dbm_from_vpeak(v_im3_2, R_load)
        Pfund_dBm_avg = 0.5 * (pf1 + pf2)
        Pim3_dBm_avg = 0.5 * (pim31 + pim32)

        # Input power based on Vin_amp (peak) across the same reference resistance.
        Pin_dBm = p_dbm_from_vpeak(Vin_amp, R_load)

        # Standard two-tone IIP3 estimate (dBm):
        # IIP3 = Pin + (Pfund_avg - Pim3_avg) / 2
        IIP3_dBm = Pin_dBm + (Pfund_dBm_avg - Pim3_dBm_avg) / 2.0

        return pd.DataFrame(
            {
                "Pin_dBm": [Pin_dBm],
                "Pfund_dBm_avg": [Pfund_dBm_avg],
                "Pim3_dBm_avg": [Pim3_dBm_avg],
                "IIP3_dBm": [IIP3_dBm],
            }
        )
