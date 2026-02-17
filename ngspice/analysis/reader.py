from __future__ import annotations

import time
from typing import Any, Optional, Tuple, Type

import pandas as pd

from ngspice.errors import RawParseError
from ngspice.errors import (
    ReaderFailedAfterRetries,
    ReaderImplCreationError,
    ReaderNonRetryableError,
)
from .registry import create_reader
from .readers import __init__ as _autoload  # noqa: F401
# NOTE:
# - Importing `analysis.readers` triggers registration side-effects (decorators /
#   module-level registration). This line intentionally ensures that all reader
#   implementations are registered before `create_reader()` is called.


# =============================================================================
# Retry policy defaults
# =============================================================================

_DEFAULT_RETRYABLE: Tuple[Type[BaseException], ...] = (
    FileNotFoundError,        # output not created yet
    pd.errors.EmptyDataError, # file exists but empty
    RawParseError,            # partial write / format not stabilized yet
    IndexError,               # parser expects rows/blocks not present yet
    ValueError,               # e.g., float conversion fails during partial write
)
"""
Default set of exceptions considered retryable by the `Reader` facade.

Notes
-----
- This list is intentionally conservative and aims to cover common "file not ready"
  and "partial write" transient states.
- Be cautious with broad exceptions such as `ValueError`: if an implementation uses
  ValueError for *non-transient* argument validation, you may end up retrying a
  deterministic error. In that case, override `retry_on` per call.
"""


# =============================================================================
# Reader facade
# =============================================================================

class Reader:
    """
    Backward-compatible facade for analysis result readers.

    This class resolves a concrete reader implementation via the analysis
    registry and provides a small retry loop that is helpful when reading
    files produced by asynchronous processes (e.g., ngspice writing outputs).

    Parameters
    ----------
    analysis_type:
        Analysis type key used to resolve the concrete reader implementation
        (e.g., "SParam", "Noise", "Linearity", ...).

    Attributes
    ----------
    type:
        Analysis type key (string).
    _impl:
        Concrete reader implementation instance created by `create_reader()`.

    Raises
    ------
    ReaderImplCreationError
        If the registry cannot create a reader implementation for `analysis_type`.

    Examples
    --------
    >>> r = Reader("S-Parameter Analysis")
    >>> df = r.read("sparams.csv", target_frequency=2.4e9)

    Notes
    -----
    - Registration is typically performed via side effects when importing reader
      modules. This file ensures such imports have occurred via `_autoload`.
    - The facade consumes a few special kwargs (`max_retries`, `sleep_sec`,
      `retry_on`) and forwards remaining kwargs to the implementation.
    """

    def __init__(self, analysis_type: str):
        self.type = str(analysis_type)
        try:
            self._impl = create_reader(self.type)
        except Exception as e:
            raise ReaderImplCreationError(self.type, e) from e

    def read(self, result_path: str, **kwargs: Any) -> pd.DataFrame:
        """
        Read an analysis result file into a DataFrame with retry support.

        Parameters
        ----------
        result_path:
            Path to the analysis output artifact to parse.
        **kwargs:
            Reader-specific keyword arguments forwarded to the concrete reader,
            except for the following facade control kwargs:

            max_retries : int, optional
                Maximum number of attempts. Defaults to 5.
            sleep_sec : float, optional
                Sleep interval between retries. Defaults to 1.0 seconds.
            retry_on : tuple[type[BaseException], ...], optional
                Tuple of exception classes considered retryable. Defaults to
                `_DEFAULT_RETRYABLE`.

        Returns
        -------
        pandas.DataFrame
            Parsed results.

        Raises
        ------
        ReaderNonRetryableError
            If `max_retries < 1`.
        ReaderFailedAfterRetries
            If all retries are exhausted and the last encountered exception is
            considered retryable (or was caught by `retry_on`).

        Notes
        -----
        - This facade assumes that many parse failures are transient when a
          producer is still writing files (partial writes). If your readers
          distinguish transient vs deterministic parse errors, you can tighten
          `retry_on` accordingly.
        - Unknown kwargs should ideally be rejected by the concrete reader
          implementation (recommended contract).
        """
        max_retries = int(kwargs.pop("max_retries", 5))
        sleep_sec = float(kwargs.pop("sleep_sec", 1.0))
        retry_on = kwargs.pop("retry_on", _DEFAULT_RETRYABLE)

        if max_retries < 1:
            raise ReaderNonRetryableError(f"max_retries must be >= 1 (got {max_retries})")

        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                # Forward remaining kwargs to the concrete implementation.
                return self._impl.read(result_path, **kwargs)  # type: ignore[misc]
            except retry_on as e:  # type: ignore[misc]
                last_err = e
                if attempt < max_retries:
                    time.sleep(sleep_sec)
                continue

        assert last_err is not None  # for type checkers (loop guarantees this)

        # Final failure after exhausting retries.
        # (Optional classification hooks could be inserted here if you want to map
        # certain last_err types to more specific terminal exceptions.)
        raise ReaderFailedAfterRetries(
            analysis_type=self.type,
            result_path=result_path,
            attempts=max_retries,
            last_error=last_err,
        ) from last_err
