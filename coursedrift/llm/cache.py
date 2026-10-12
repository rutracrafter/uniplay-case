"""Memoizing validated responses so reruns are free and reproducible.

The cache stores the *raw* successful response rather than the parsed value,
so a cache hit is replayed through exactly the same extraction and validation
path as a live call. A cached result can therefore never be more trusted than
a fresh one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol


class Cache(Protocol):
    def key(self, client_name: str, prompt: str) -> str: ...
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, raw: str) -> None: ...
    def drop(self, key: str) -> None: ...


def _digest(client_name: str, prompt: str) -> str:
    return hashlib.sha256(f"{client_name}\x00{prompt}".encode()).hexdigest()


class NullCache:
    """No memoization. Used in tests that must observe every call."""

    def key(self, client_name: str, prompt: str) -> str:
        return _digest(client_name, prompt)

    def get(self, key: str) -> str | None:
        return None

    def put(self, key: str, raw: str) -> None:
        return None

    def drop(self, key: str) -> None:
        return None


class FileCache:
    """One JSON file per call, keyed by sha256(client name + prompt)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def key(self, client_name: str, prompt: str) -> str:
        return _digest(client_name, prompt)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload["raw"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A corrupt entry is treated as a miss, not as a crash.
            return None
        return raw if isinstance(raw, str) else None

    def put(self, key: str, raw: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps({"raw": raw}, indent=2), encoding="utf-8")

    def drop(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
