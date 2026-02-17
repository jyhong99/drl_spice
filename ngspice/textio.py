from __future__ import annotations

from pathlib import Path

from .types import PathLike
from .errors import (
    TextFileNotFound,
    TextReadError,
    TextWriteError,
)


# =============================================================================
# Text I/O helpers
# =============================================================================

def read_text(path: PathLike) -> str:
    """
    Read a text file and return its contents as a string.

    This helper performs an explicit existence check and wraps any I/O-related
    exceptions into domain-specific errors to provide clearer diagnostics to
    higher layers.

    Parameters
    ----------
    path:
        Path to the text file. Can be a string or `pathlib.Path`.

    Returns
    -------
    str
        File contents decoded as text.

    Raises
    ------
    TextFileNotFound
        If the given path does not exist.
    TextReadError
        If the file exists but cannot be read for any reason
        (permission error, encoding issue, etc.).

    Notes
    -----
    - The file is read using `errors="replace"` to avoid hard failures on
      malformed encodings. Any undecodable bytes are replaced with the Unicode
      replacement character.
    - This function does not strip trailing newlines or whitespace.
    """
    p = Path(path)

    if not p.exists():
        raise TextFileNotFound(p)

    try:
        return p.read_text(errors="replace")
    except Exception as e:  # pragma: no cover - defensive wrapper
        raise TextReadError(p, e) from e


def atomic_write(path: PathLike, text: str) -> None:
    """
    Atomically write text to a file.

    The write is performed by first writing to a temporary file in the same
    directory and then replacing the target path. This ensures that readers
    never observe a partially-written file.

    Parameters
    ----------
    path:
        Destination file path. Can be a string or `pathlib.Path`.
    text:
        Text content to write.

    Raises
    ------
    TextWriteError
        If directory creation, temporary write, or final replace fails.

    Notes
    -----
    - On POSIX systems, `Path.replace()` is atomic.
    - On Windows, the operation is best-effort but still significantly safer
      than writing directly to the target file.
    - Parent directories are created automatically if they do not exist.
    - Text is written using UTF-8 encoding with `errors="replace"` to avoid
      encoding-related crashes.
    """
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        # Ensure parent directory exists
        p.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first
        tmp.write_text(text, encoding="utf-8", errors="replace")

        # Replace target with temp file (atomic on POSIX)
        tmp.replace(p)
    except Exception as e:  # pragma: no cover - defensive wrapper
        raise TextWriteError(p, e) from e
