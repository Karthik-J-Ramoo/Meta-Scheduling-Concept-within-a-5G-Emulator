"""
logger.py
=========
Centralised logging and CSV metric-recording utilities.

Design rationale
----------------
* A single ``setup_logger`` call wires both console and file handlers so that
  every module gets consistent formatting by using ``logging.getLogger(__name__)``.
* ``MetricCSVWriter`` streams per-timestep rows to a CSV file, keeping memory
  usage O(1) regardless of simulation length.
* Both components read defaults from ``config.simulation_config.LOGGING`` but
  accept overrides for test / notebook usage.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from config.simulation_config import LOGGING


# ── helpers ───────────────────────────────────────────────────────────────
_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s"
)
_DATE_FORMAT = "%H:%M:%S"

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _resolve_level(name: str) -> int:
    """Convert a human-readable level string to a ``logging`` constant."""
    return _LEVEL_MAP.get(name.upper(), logging.INFO)


# ── public API ────────────────────────────────────────────────────────────

def setup_logger(
    log_dir: str | None = None,
    log_filename: str | None = None,
    console_level: str | None = None,
    file_level: str | None = None,
) -> logging.Logger:
    """
    Configure the root logger with a console handler and a file handler.

    Parameters
    ----------
    log_dir : str, optional
        Directory for the log file.  Defaults to ``LOGGING["log_dir"]``.
    log_filename : str, optional
        Name of the log file.  Defaults to ``LOGGING["log_filename"]``.
    console_level : str, optional
        Minimum severity shown on the console.
    file_level : str, optional
        Minimum severity written to the log file.

    Returns
    -------
    logging.Logger
        The root logger (callers can also just use ``logging.getLogger(name)``).
    """
    log_dir = log_dir or LOGGING["log_dir"]
    log_filename = log_filename or LOGGING["log_filename"]
    console_level = console_level or LOGGING["console_level"]
    file_level = file_level or LOGGING["file_level"]

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # allow handlers to filter independently

    # Remove any pre-existing handlers (important when re-running in notebooks)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_resolve_level(console_level))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(_resolve_level(file_level))
    fh.setFormatter(formatter)
    root.addHandler(fh)

    root.info("Logger initialised – console=%s  file=%s (%s)",
              console_level, file_level, log_path)
    return root


class MetricCSVWriter:
    """
    Streams per-timestep metrics to a CSV file.

    Usage
    -----
    >>> writer = MetricCSVWriter(fieldnames=["t", "slice", "throughput"])
    >>> writer.write_row({"t": 0, "slice": "URLLC", "throughput": 42.0})
    >>> writer.close()

    The writer is also a context manager::

        with MetricCSVWriter(fieldnames=[...]) as w:
            w.write_row({...})
    """

    def __init__(
        self,
        fieldnames: Sequence[str],
        log_dir: str | None = None,
        csv_filename: str | None = None,
    ) -> None:
        log_dir = log_dir or LOGGING["log_dir"]
        csv_filename = csv_filename or LOGGING["csv_filename"]
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._path = os.path.join(log_dir, csv_filename)
        self._file = open(self._path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.debug("CSV writer opened → %s", self._path)

    # ── row writing ───────────────────────────────────────────────────
    def write_row(self, row: dict[str, Any]) -> None:
        """Append a single row (dict keyed by fieldnames)."""
        self._writer.writerow(row)

    def write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        """Append multiple rows in one call."""
        for row in rows:
            self._writer.writerow(row)

    # ── lifecycle ─────────────────────────────────────────────────────
    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        self._logger.debug("CSV writer closed → %s", self._path)

    def __enter__(self) -> "MetricCSVWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
