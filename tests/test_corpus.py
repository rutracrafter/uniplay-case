"""Phase 1 — snapshot loading and document-level classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from coursedrift.corpus import diff_manifests, load_course, load_snapshot
from coursedrift.models import Course, CourseItem


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


# -- course validation -----------------------------------------------------
#
# These guard an invariant every later stage relies on: an item's
# correct_index addresses an option that exists. Without it, a bad course
# validates cleanly and then raises IndexError the first time correct_answer
# is read — deep inside adjudication or provenance inference, as an uncaught
# crash rather than a routed finding.


def _course(**item_overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "item_id": "q1",
        "level_id": "L1",
        "stem": "How many days?",
        "options": ["30 days", "45 days"],
        "correct_index": 0,
        "provenance": [],
    }
    item.update(item_overrides)
    return {"course_id": "c", "title": "T", "items": [item]}


def _write_course(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "course.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_well_formed_course_loads(tmp_path: Path) -> None:
    course = load_course(_write_course(tmp_path, _course()))
    assert course.items[0].correct_answer == "30 days"


def test_correct_index_past_the_end_is_rejected_at_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="correct_index 7 is outside the 2 option"):
        load_course(_write_course(tmp_path, _course(correct_index=7)))


def test_negative_correct_index_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="correct_index -1 is outside"):
        load_course(_write_course(tmp_path, _course(correct_index=-1)))


def test_a_question_with_no_options_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="needs at least one option"):
        load_course(_write_course(tmp_path, _course(options=[])))


def test_the_last_option_is_addressable(tmp_path: Path) -> None:
    """An off-by-one here would reject every correctly-keyed final option."""
    course = load_course(_write_course(tmp_path, _course(correct_index=1)))
    assert course.items[0].correct_answer == "45 days"


def test_the_error_names_the_offending_item(tmp_path: Path) -> None:
    """A 100-item course needs to say which one is wrong."""
    payload = _course()
    items = payload["items"]
    assert isinstance(items, list)
    items.append(
        {
            "item_id": "q2-broken",
            "level_id": "L1",
            "stem": "Q?",
            "options": ["a", "b"],
            "correct_index": 9,
        }
    )
    with pytest.raises(ValidationError, match="q2-broken"):
        load_course(_write_course(tmp_path, payload))


def test_the_rule_holds_outside_the_loader_too() -> None:
    """On the model, not in load_course, so construction is guarded as well."""
    with pytest.raises(ValidationError, match="outside the 2 option"):
        CourseItem(
            item_id="direct", level_id="L1", stem="Q?", options=["a", "b"], correct_index=5
        )


def test_the_bundled_fixture_course_is_valid() -> None:
    fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "course.json"
    course = load_course(fixtures)
    assert len(course.items) == 14
    for item in course.items:
        assert item.correct_answer in item.options
