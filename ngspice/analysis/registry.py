from __future__ import annotations

"""
Registry for analysis readers.

Provides a mapping from analysis type strings to reader factories, with
validation and helpful errors for unknown or duplicate registrations.
"""

from typing import Callable, Dict

from ngspice.errors import DuplicateAnalysisType, ReaderFactoryError, UnknownAnalysisType

from .base import AnalysisReader


# =============================================================================
# Registry storage
# =============================================================================

_REGISTRY: Dict[str, Callable[[], AnalysisReader]] = {}
"""
Internal registry mapping analysis_type -> reader factory.

Notes
-----
- Values are *factories* (callables) instead of instances to keep construction
  lazy and to allow stateful readers to be created per call if desired.
- Keys are exact string matches. If you want case-insensitive behavior, normalize
  keys consistently (e.g., `.lower()`) in both registration and lookup.
"""


# =============================================================================
# Registry API
# =============================================================================

def register(analysis_type: str, *, allow_override: bool = False):
    """
    Decorator to register an analysis reader factory.

    Parameters
    ----------
    analysis_type:
        Registry key used by :func:`create_reader`. Typically a stable identifier
        such as "SParam", "Noise", "Linearity", etc.
    allow_override:
        If False (default), registering the same `analysis_type` twice raises
        :class:`ngspice.errors.DuplicateAnalysisType`.
        If True, the new factory replaces the existing factory for that key.

    Returns
    -------
    callable
        A decorator that takes a zero-argument factory and returns it unchanged
        after registration.

    Raises
    ------
    DuplicateAnalysisType
        If `allow_override=False` and `analysis_type` is already registered.

    Notes
    -----
    - Factories are expected to have signature ``() -> AnalysisReader``.
    - Registration is commonly performed at import-time for side effects.
    """
    analysis_type = str(analysis_type).strip()
    if not analysis_type:
        raise ValueError("analysis_type must be a non-empty string")

    def deco(factory: Callable[[], AnalysisReader]) -> Callable[[], AnalysisReader]:
        if not callable(factory):
            raise TypeError(f"factory must be callable, got {type(factory).__name__}")
        if (not allow_override) and (analysis_type in _REGISTRY):
            raise DuplicateAnalysisType(analysis_type)
        _REGISTRY[analysis_type] = factory
        return factory

    return deco


def create_reader(analysis_type: str) -> AnalysisReader:
    """
    Create a concrete analysis reader implementation for the given analysis type.

    Parameters
    ----------
    analysis_type:
        Registry key previously registered via :func:`register`.

    Returns
    -------
    AnalysisReader
        A new reader instance created by the registered factory.

    Raises
    ------
    UnknownAnalysisType
        If no factory is registered for `analysis_type`.
    ReaderFactoryError
        If a factory exists but raises an exception during instantiation.

    Notes
    -----
    - This function distinguishes two failure modes:
      1) unknown key (registry miss) -> `UnknownAnalysisType`
      2) factory failure -> `ReaderFactoryError`
    """
    analysis_type = str(analysis_type).strip()
    factory = _REGISTRY.get(analysis_type)
    if factory is None:
        raise UnknownAnalysisType(analysis_type, supported())

    try:
        reader = factory()
    except Exception as e:
        raise ReaderFactoryError(analysis_type, e) from e

    if not isinstance(reader, AnalysisReader):
        raise ReaderFactoryError(
            analysis_type,
            TypeError(
                f"factory returned {type(reader).__name__}, expected AnalysisReader-compatible object"
            ),
        )
    return reader


def supported() -> list[str]:
    """
    Return the list of registered analysis types.

    Returns
    -------
    list[str]
        Sorted list of analysis_type keys currently registered.

    Notes
    -----
    - This returns a snapshot of the registry keys at call time.
    """
    return sorted(_REGISTRY)
