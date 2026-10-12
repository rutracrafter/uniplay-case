"""Phase 1 — snapshot loading and document-level classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coursedrift.corpus import diff_manifests, load_course, load_snapshot
from coursedrift.models import Course


def _write(root: Path, name: str, body: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    old, new = tmp_path / "v1", tmp_path / "v2"
    old.mkdir()
    new.mkdir()

    # unchanged
    for root in (old, new):
        _write(root, "stable.md", "# Stable\n\nNothing moves here.\n")
    # modified
    _write(old, "policy.md", "# Policy\n\nNotice is 30 days.\n")
    _write(new, "policy.md", "# Policy\n\nNotice is 45 days.\n")
    # deleted
    _write(old, "onboarding.md", "# Onboarding\n\nOld material.\n")
    # added
    _write(new, "vendors.md", "# Vendors\n\nNew material.\n")
    # nested path, unchanged
    for root in (old, new):
        _write(root, "sub/nested.md", "# Nested\n\nDeep content.\n")
    return old, new


def test_doc_ids_are_posix_paths_relative_to_root(pair: tuple[Path, Path]) -> None:
    old, _ = pair
    snap = load_snapshot(old)
    assert set(snap.docs) == {"stable.md", "policy.md", "onboarding.md", "sub/nested.md"}


def test_manifest_classifies_all_four_states(pair: tuple[Path, Path]) -> None:
    old, new = pair
    diff = diff_manifests(load_snapshot(old), load_snapshot(new))
    assert diff.unchanged == {"stable.md", "sub/nested.md"}
    assert diff.modified == {"policy.md"}
    assert diff.added == {"vendors.md"}
    assert diff.deleted == {"onboarding.md"}


def test_touched_excludes_unchanged(pair: tuple[Path, Path]) -> None:
    old, new = pair
    diff = diff_manifests(load_snapshot(old), load_snapshot(new))
    assert diff.touched == {"policy.md", "vendors.md", "onboarding.md"}
    assert not diff.touched & diff.unchanged


def test_whitespace_reflow_is_modified_but_spans_still_match(tmp_path: Path) -> None:
    """A reflow is honestly a changed file; it just produces no span changes."""
    old, new = tmp_path / "a", tmp_path / "b"
    old.mkdir()
    new.mkdir()
    _write(old, "d.md", "# D\n\nOne sentence here. And another.\n")
    _write(new, "d.md", "# D\n\nOne sentence here.\nAnd another.\n")
    so, sn = load_snapshot(old), load_snapshot(new)
    assert diff_manifests(so, sn).modified == {"d.md"}
    assert {s.hash for s in so.all_spans()} == {s.hash for s in sn.all_spans()}


def test_by_hash_is_corpus_wide(tmp_path: Path) -> None:
    """The identity index must span documents, not sit inside one."""
    root = tmp_path / "c"
    root.mkdir()
    shared = "All employees must comply."
    _write(root, "a.md", f"# A\n\n{shared}\n")
    _write(root, "b.md", f"# B\n\n{shared}\n")
    index = load_snapshot(root).by_hash()
    duplicated = [spans for spans in index.values() if len(spans) == 2]
    assert len(duplicated) == 1
    assert {s.doc_id for s in duplicated[0]} == {"a.md", "b.md"}


def test_span_by_id_round_trips(pair: tuple[Path, Path]) -> None:
    snap = load_snapshot(pair[0])
    span = next(iter(snap.all_spans()))
    assert snap.span_by_id(span.span_id) == span
    assert snap.span_by_id("nope.md#p0.s0") is None


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        load_snapshot(tmp_path / "absent")


def test_load_course_validates(tmp_path: Path) -> None:
    path = tmp_path / "course.json"
    path.write_text(
        json.dumps(
            {
                "course_id": "c1",
                "title": "Handbook",
                "items": [
                    {
                        "item_id": "q1",
                        "level_id": "L1",
                        "stem": "How many days?",
                        "options": ["30", "45"],
                        "correct_index": 0,
                        "provenance": [],
                    }
                ],
            }
        )
    )
    course = load_course(path)
    assert isinstance(course, Course)
    assert course.items[0].correct_answer == "30"
    assert course.levels() == ["L1"]
