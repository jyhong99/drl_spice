from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from ..patterns import (
    ACTION_JSON_KEY,
    ACTION_KEY,
    CSV_COLUMNS,
    CSV_NAME,
    DONE_REASON_KEY,
    EPISODE_KEY,
    EP_RETURN_KEY,
    EP_STEPS_KEY,
    EVENT_CUSTOM,
    EVENT_ENV_CONFIG,
    EVENT_RUN_META,
    INFO_JSON_KEY,
    INFO_KEY,
    LAST_RUN_DIR_KEY,
    LAST_SIM_STATUS_KEY,
    OBS_JSON_KEY,
    OBS_KEY,
    PAYLOAD_JSON_KEY,
    REWARD_KEY,
    STEP_KEY,
    STORE_INLINE,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    T_WALL_KEY,
    TYPE_KEY,
    USED_EVENT_KEYS,
    WALL_SEC_KEY
)
from ..types import LoggerSettings
from ..utils.logger_utils import _now_ymd_hms, _to_jsonable, _json_cell, _empty_row


class CSVWriter:
    """
    Append-only CSV event writer for run logging.

    This writer serializes structured events into a single CSV file with a fixed
    schema (`CSV_COLUMNS`). It is designed to support:
    - episode boundary events (run meta, env config, episode start/end)
    - per-step events (reward, termination flags, optional obs/action arrays)
    - custom events with arbitrary payload fields

    Events are written row-by-row in an append-only manner to support streaming
    and robustness against process crashes.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory where the CSV log file is created/appended.
    settings : LoggerSettings
        Logger settings controlling behaviors like inline array storage.

    Attributes
    ----------
    run_dir : pathlib.Path
        Run directory path (created if missing).
    csv_path : pathlib.Path
        Full path to the CSV log file (e.g., `<run_dir>/<CSV_NAME>`).

    Raises
    ------
    OSError
        If the CSV file cannot be opened for appending.

    Notes
    -----
    - The CSV header is written automatically if the file is empty.
    - Arrays and nested objects are stored as JSON strings in dedicated columns.
    - Unknown or extra keys in an event are stored under `PAYLOAD_JSON_KEY`.
    """

    def __init__(self, *, run_dir: Path, settings: LoggerSettings) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.settings = settings
        self.csv_path = self.run_dir / CSV_NAME

        try:
            self._f = open(self.csv_path, "a", encoding="utf-8", newline="")
        except OSError as e:
            raise OSError(str(e)) from e

        self._w = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """
        Ensure the CSV header exists.

        If the file is new or empty, write the header row. This makes the log
        self-describing and robust to runs that start/stop multiple times.
        """
        try:
            if self.csv_path.stat().st_size == 0:
                self._w.writeheader()
                self._f.flush()
        except FileNotFoundError:
            # If stat fails due to a race/FS issue, still write header defensively.
            self._w.writeheader()
            self._f.flush()

    def write_event(self, event: Dict[str, Any]) -> None:
        """
        Serialize an event dict into one CSV row.

        The output row has a fixed schema (`CSV_COLUMNS`). Known keys are mapped
        into dedicated columns. Arrays and nested objects are JSON-encoded for
        safe CSV storage.

        Parameters
        ----------
        event : dict[str, Any]
            Event payload. Common keys include:
            - TYPE_KEY, T_WALL_KEY, EPISODE_KEY, STEP_KEY
            - REWARD_KEY, TERMINATED_KEY, TRUNCATED_KEY
            - INFO_KEY (dict-like)
            - OBS_KEY, ACTION_KEY (arrays) if inline storage enabled

        Notes
        -----
        - If `T_WALL_KEY` is missing, current `time.time()` is used.
        - If `TYPE_KEY` is missing, `EVENT_CUSTOM` is used.
        - Keys not in `USED_EVENT_KEYS` are packed into `PAYLOAD_JSON_KEY`.
        """
        row = _empty_row()

        # Required-ish keys with defaults
        row[T_WALL_KEY] = str(event.get(T_WALL_KEY, time.time()))
        row[TYPE_KEY] = str(event.get(TYPE_KEY, EVENT_CUSTOM))

        # Episode/step indices (optional)
        if EPISODE_KEY in event:
            row[EPISODE_KEY] = str(event[EPISODE_KEY])
        if STEP_KEY in event:
            row[STEP_KEY] = str(event[STEP_KEY])

        # Common per-step scalar fields
        for k in (REWARD_KEY, TERMINATED_KEY, TRUNCATED_KEY):
            if k in event:
                row[k] = str(event[k])

        # Common episode-summary / termination fields
        for k in (
            DONE_REASON_KEY,
            EP_STEPS_KEY,
            EP_RETURN_KEY,
            WALL_SEC_KEY,
            LAST_SIM_STATUS_KEY,
            LAST_RUN_DIR_KEY,
        ):
            if k in event:
                row[k] = str(event[k])

        # Optional: store arrays inline as JSON cells
        if self.settings.store_arrays == STORE_INLINE:
            if OBS_KEY in event:
                row[OBS_JSON_KEY] = _json_cell(event[OBS_KEY])
            if ACTION_KEY in event:
                row[ACTION_JSON_KEY] = _json_cell(event[ACTION_KEY])

        # Info dict stored as JSON (even if not inline arrays)
        if INFO_KEY in event:
            row[INFO_JSON_KEY] = _json_cell(event[INFO_KEY])

        # Pack any extra keys not part of the canonical schema into payload
        payload = {k: v for k, v in event.items() if k not in USED_EVENT_KEYS}
        if payload:
            row[PAYLOAD_JSON_KEY] = _json_cell(payload)
        else:
            # Special-case: caller may directly provide PAYLOAD_JSON_KEY already serialized
            if PAYLOAD_JSON_KEY in event and row[PAYLOAD_JSON_KEY] == "":
                row[PAYLOAD_JSON_KEY] = str(event[PAYLOAD_JSON_KEY])

        self._w.writerow(row)

    def write_run_meta(
        self,
        *,
        run_meta: Dict[str, Any],
        settings_obj: Any,
        run_dir: Path,
    ) -> None:
        """
        Write a run metadata event.

        This is typically written once at run start, and includes:
        - creation timestamp
        - run directory
        - serialized settings
        - arbitrary run metadata supplied by the caller

        Parameters
        ----------
        run_meta : dict[str, Any]
            User-provided metadata (e.g., experiment name, git hash, seed).
        settings_obj : Any
            Settings object to serialize (typically LoggerSettings).
        run_dir : pathlib.Path
            Run directory to record into the metadata payload.

        Notes
        -----
        The payload is stored in `PAYLOAD_JSON_KEY` as a JSON string.
        """
        payload = {
            "created_at": _now_ymd_hms(),
            "run_dir": str(run_dir),
            "settings": _to_jsonable(settings_obj),
            "run_meta": _to_jsonable(run_meta),
        }
        self.write_event(
            {
                TYPE_KEY: EVENT_RUN_META,
                PAYLOAD_JSON_KEY: json.dumps(payload, ensure_ascii=False),
            }
        )

    def write_env_config(self, *, env_config: Dict[str, Any]) -> None:
        """
        Write an environment configuration event.

        Parameters
        ----------
        env_config : dict[str, Any]
            Environment configuration snapshot to persist.

        Notes
        -----
        The payload is stored in `PAYLOAD_JSON_KEY` as a JSON string.
        """
        self.write_event(
            {
                TYPE_KEY: EVENT_ENV_CONFIG,
                PAYLOAD_JSON_KEY: json.dumps(_to_jsonable(env_config), ensure_ascii=False),
            }
        )

    def flush(self) -> None:
        """
        Flush buffered data to disk.

        Notes
        -----
        Uses `os.fsync` to force OS-level flush for durability. Any exception is
        swallowed to avoid crashing the training loop due to transient I/O issues.
        """
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
        except Exception:
            # Intentionally best-effort: logging should not crash training.
            pass

    def close(self) -> None:
        """
        Flush and close the underlying file handle.

        Notes
        -----
        Safe to call multiple times. Exceptions are swallowed to avoid masking
        upstream errors during teardown.
        """
        try:
            self.flush()
        finally:
            try:
                self._f.close()
            except Exception:
                pass
