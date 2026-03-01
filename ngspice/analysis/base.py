from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AnalysisReader(Protocol):
    """
    Protocol for analysis-specific result parsers.

    An `AnalysisReader` converts an analysis result artifact (typically a file
    produced by ngspice or an intermediate tool) into a `pandas.DataFrame`
    with a predictable schema.

    Implementations should treat schema mismatches as errors rather than
    silently producing wrong outputs.

    Attributes
    ----------
    analysis_type:
        String key identifying the analysis type handled by this reader
        (e.g., "SParam", "Noise", "Linearity", "Op", ...). This is typically
        used by a registry/factory to select the appropriate reader.

    Methods
    -------
    read(result_path, **kwargs) -> pandas.DataFrame
        Parse the artifact at `result_path` and return a DataFrame.

    Notes
    -----
    - Concrete readers may accept additional keyword arguments, for example:
        * target_frequency: float
        * return_full: bool
        * Vin_amp: float
        * f1, f2: float
        * R_load: float
      These should be documented by each implementation.
    - Recommended behavior: unknown kwargs should raise `ValueError` to avoid
      silent misconfiguration.
    - If parsing is retryable (file not ready yet), prefer raising a dedicated
      retryable error type (e.g., `ReaderRetryableError`) in higher layers.
    """

    analysis_type: str

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Parse an analysis output artifact into a DataFrame.

        Parameters
        ----------
        result_path:
            Path to the analysis result artifact (file) to parse.
            This is commonly a text output, CSV, or RAW-ascii dump.
        **kwargs:
            Analysis-specific optional keyword arguments.
            Each concrete reader defines which keys are supported.

        Returns
        -------
        pandas.DataFrame
            Parsed analysis results.

        Raises
        ------
        ValueError
            Recommended for unsupported/unknown keyword arguments or invalid
            argument values.
        Exception
            Implementations may raise domain-specific errors (e.g., RawParseError,
            DataSchemaError) when the result format is malformed or missing
            expected columns.

        Notes
        -----
        - Readers should be strict about required columns and units. If the
          frequency column is missing, for example, raising a schema error is
          preferred to guessing.
        - A reader may attach additional metadata (e.g., nearest_frequency) via
          `df.attrs` or explicit columns, depending on your project conventions.
        """
        ...
