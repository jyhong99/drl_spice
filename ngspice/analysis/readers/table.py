from __future__ import annotations

from typing import Any

import pandas as pd

from ngspice.analysis.registry import register
from ngspice.ascii import read_raw_table


@register("Transient")
def _transient_factory() -> "RawTableReader":
    """
    Factory for Transient analysis reader.

    Returns
    -------
    RawTableReader
        Reader instance configured for "Transient" analysis.
    """
    return RawTableReader("Transient")


@register("DC_Operating_Point")
def _dc_factory() -> "RawTableReader":
    """
    Factory for DC operating-point analysis reader.

    Returns
    -------
    RawTableReader
        Reader instance configured for "DC_Operating_Point" analysis.
    """
    return RawTableReader("DC_Operating_Point")


@register("AC_simulation")
def _ac_factory() -> "RawTableReader":
    """
    Factory for AC simulation analysis reader.

    Returns
    -------
    RawTableReader
        Reader instance configured for "AC_simulation" analysis.
    """
    return RawTableReader("AC_simulation")


class RawTableReader:
    """
    Generic passthrough reader for raw ngspice table outputs.

    This reader performs **no analysis-specific parsing or aggregation**.
    It simply loads a raw-ascii table (via `read_raw_table`) and returns
    the resulting DataFrame unchanged.

    The primary purpose of this reader is:
    - to support analyses where downstream code wants direct access to the
      full ngspice output table (e.g., Transient waveforms, AC sweeps),
    - to provide a consistent entry point compatible with the unified
      `AnalysisReader` protocol and registry/facade infrastructure.

    Attributes
    ----------
    analysis_type:
        Registry key identifying the analysis (e.g., "Transient",
        "DC_Operating_Point", "AC_simulation").

    Notes
    -----
    - Although this reader accepts ``**kwargs`` for protocol compatibility,
      **all keyword arguments are rejected** to avoid silent misconfiguration.
    - Any parsing, filtering, resampling, or feature extraction should be
      implemented in higher-level analysis-specific readers or post-processors.
    """

    def __init__(self, analysis_type: str) -> None:
        """
        Initialize a raw table reader.

        Parameters
        ----------
        analysis_type:
            Human-readable analysis identifier used in error messages and
            registry bookkeeping.
        """
        self.analysis_type = str(analysis_type)

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Load and return a raw ngspice output table.

        Parameters
        ----------
        result_path:
            Path to the ngspice output file containing a raw-ascii table.
        **kwargs:
            No keyword arguments are supported. Any provided kwargs will
            raise a ValueError to prevent silent configuration errors.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parsed raw table exactly as produced
            by ngspice.

        Raises
        ------
        ValueError
            If any unsupported keyword arguments are provided.
        Exception
            Parsing errors may propagate from `read_raw_table`, such as
            `RawParseError`, `MissingColumnError`, or I/O-related exceptions.

        Notes
        -----
        - This method does **not** attempt to interpret column semantics
          (e.g., time, voltage, current). It is intentionally minimal.
        - For reinforcement-learning environments, this reader is typically
          used when the observation builder or reward function performs its
          own feature extraction on the full waveform/table.
        """
        if kwargs:
            raise ValueError(f"{self.analysis_type}: unsupported kwargs={sorted(kwargs)}")

        return read_raw_table(result_path)
