"""Additive v1 run records; unversioned archives are v0 (see monitoring_format.md)."""

from __future__ import annotations

import json
import time
from pathlib import Path

SCHEMA_VERSION = 1
EXTERNAL_MONITOR_ENV = "MYFLOWS_EXTERNAL_MONITOR"


def schema_version(row: dict) -> int:
    return row.get("schema_version", 0)


def record(row: dict, *, clocks: bool = False) -> dict:
    """Stamp a newly produced record without mutating the caller's dictionary."""
    base = {"monotonic_s": time.perf_counter(), "unix_s": time.time()} if clocks else {}
    return {**base, **row, "schema_version": SCHEMA_VERSION}


class JsonlCursor:
    """Consume complete lines once, without retaining the history in training memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.identity = None

    def poll(self) -> list[dict]:
        try:
            with self.path.open("rb") as handle:
                import os

                stat = os.fstat(handle.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if stat.st_size < self.offset or identity != self.identity:
                    self.offset, self.pending, self.identity = 0, b"", identity
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
