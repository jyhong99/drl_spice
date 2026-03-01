"""
logger
=======

Structured experiment logging utilities.

This package implements a lightweight logging stack for reinforcement learning
(RL) and simulation-driven optimization workflows. The design goal is to
provide:

- **Stable event schema**: events are stored in a fixed-column CSV format
  for consistent downstream analysis.
- **Low operational complexity**: append-only logging with periodic flushing.
- **Debug-friendly artifacts**: run metadata, environment configuration,
  episode summaries, and per-step transitions.

The logging backend is CSV-based for simplicity, robustness, and easy
post-processing (e.g., with pandas).

Classes
-------
Logger
    High-level interface used by training/evaluation loops. Manages:
    - run directory creation
    - episode boundaries (start/end)
    - per-step transition logging
    - aggregation of episode return/length

CSVWriter
    Low-level append-only CSV event writer. Serializes structured event dicts
    into a fixed schema (column set) and supports JSON-encoding for nested
    payloads and optional inline array storage.

Examples
--------
Typical usage in a training loop:

>>> from env.logger import Logger
>>>
>>> logger = Logger(
...     root_dir="./runs",
...     run_name="exp",
...     run_meta={"seed": 42},
...     env_config={"env_id": "Pendulum-v1"},
... )
>>>
>>> logger.start_episode()
>>> logger.log_step(
...     obs=obs,
...     action=action,
...     reward=reward,
...     terminated=terminated,
...     truncated=truncated,
...     info=info,
... )
>>> logger.end_episode(done_reason="terminated")
>>> logger.close()

Notes
-----
- Only :class:`~env.logger.Logger` and :class:`~env.logger.CSVWriter` are considered
  part of the public API. Other modules and symbols may change without notice.
- Events are written incrementally (streaming) to reduce memory usage and to
  improve crash resilience.

See Also
--------
env.logger.logger.Logger
    High-level run/episode/step logging interface.
env.logger.writer.CSVWriter
    Low-level CSV event writer.
"""

from __future__ import annotations

# Public re-exports (stable API surface)
from .logger import Logger
from .writer import CSVWriter

__all__ = ("Logger", "CSVWriter")
