"""Shared fixtures: building real snapshots on disk from inline documents."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from coursedrift.corpus import load_snapshot
from coursedrift.models import Snapshot

SnapshotBuilder = Callable[[str, dict[str, str]], Snapshot]


@pytest.fixture
def build(tmp_path: Path) -> SnapshotBuilder:
    """Write documents to a named directory and load them as a Snapshot.

    Goes through the real loader rather than constructing models by hand, so
    segmentation and hashing are exercised by every diff test too.
    """

    def _build(name: str, docs: dict[str, str]) -> Snapshot:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        for doc_id, body in docs.items():
            path = root / doc_id
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        return load_snapshot(root)

    return _build
