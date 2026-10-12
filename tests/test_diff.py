"""Phase 2 — matcher behaviour beyond the reorder invariant."""

from __future__ import annotations

from coursedrift.config import Config
from coursedrift.corpus import diff_manifests
from coursedrift.diff import (
    ContentOp,
    PositionOp,
    _phase_a_identity,
    inline_diff,
    is_cosmetic,
    jaccard,
    match_corpus,
    shingles,
)
from tests.conftest import SnapshotBuilder

# -- helpers ---------------------------------------------------------------


def test_shingles_of_short_text_is_the_whole_phrase() -> None:
    assert shingles("two words", 3) == {("two", "words")}


def test_shingles_slides_a_window() -> None:
    assert shingles("a b c d", 3) == {("a", "b", "c"), ("b", "c", "d")}


def test_jaccard_basics() -> None:
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0
    assert jaccard(frozenset("ab"), frozenset("bc")) == 1 / 3
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_is_cosmetic_detects_punctuation_and_casing() -> None:
    assert is_cosmetic("Before leaving, submit the form.", "Before leaving submit the form.")
    assert is_cosmetic("Notice is required.", "notice is required.")


def test_is_cosmetic_rejects_real_edits_and_no_ops() -> None:
    assert not is_cosmetic("Notice is 30 days.", "Notice is 45 days.")
    assert not is_cosmetic("Same text.", "Same text.")


def test_inline_diff_marks_both_sides() -> None:
    assert inline_diff("Notice is 30 days", "Notice is 45 days") == (
        "Notice is [-30-]{+45+} days"
    )
    assert inline_diff("Keep this", "Keep this too") == "Keep this {+too+}"


# -- Phase A ---------------------------------------------------------------


def test_whitespace_edit_never_reaches_phase_b(build: SnapshotBuilder) -> None:
    """The free path must stay free: Phase A leaves no residue at all."""
    old = build("v1", {"d.md": "# D\n\nNotice is 30 days.   Managers approve.\n"})
    new = build("v2", {"d.md": "# D\n\nNotice is 30 days.\nManagers approve.\n"})
    pairs, residue_old, residue_new = _phase_a_identity(old, new)
    assert len(pairs) == 2
    assert residue_old == []
    assert residue_new == []


def test_duplicate_boilerplate_matches_by_count(build: SnapshotBuilder) -> None:
    """Five copies old against three new: three match, two are removed."""
    line = "All employees must comply.\n\n"
    old = build("v1", {"b.md": "# B\n\n" + line * 5 + "Unique closing remark.\n"})
    new = build("v2", {"b.md": "# B\n\n" + line * 3 + "Unique closing remark.\n"})
    result = match_corpus(old, new, Config())

    identical = [c for c in result.changes if c.content is ContentOp.IDENTICAL]
    removed = [c for c in result.changes if c.content is ContentOp.REMOVED]
    boiler_identical = [c for c in identical if c.new is not None and "comply" in c.new.text]
    assert len(boiler_identical) == 3
    assert len(removed) == 2
    assert all(c.old is not None and "comply" in c.old.text for c in removed)


def test_cross_document_move_matches_with_the_flag(build: SnapshotBuilder) -> None:
    """Content folded from a deleted document into another must not read as drift."""
    moved = "Parental leave runs for twelve weeks."
    old = build(
        "v1",
        {
            "onboarding.md": f"# Onboarding\n\n{moved}\n",
            "leave.md": "# Leave\n\nNotice is 30 days.\n",
        },
    )
    new = build("v2", {"leave.md": f"# Leave\n\nNotice is 30 days.\n\n{moved}\n"})
    manifest = diff_manifests(old, new)
    result = match_corpus(old, new, Config(), manifest)

    crossing = [c for c in result.changes if c.across_docs]
    assert len(crossing) == 1
    assert crossing[0].content is ContentOp.IDENTICAL
    assert crossing[0].position is PositionOp.MOVED
    assert crossing[0].old is not None and crossing[0].old.doc_id == "onboarding.md"
    assert crossing[0].new is not None and crossing[0].new.doc_id == "leave.md"
    assert not [c for c in result.changes if c.content is ContentOp.REMOVED]


# -- Phase B ---------------------------------------------------------------


def test_reword_is_one_modified_not_a_delete_plus_add(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice must be given within 30 days.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice must be given within 45 days.\n"})
    result = match_corpus(old, new, Config())

    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.content is ContentOp.MODIFIED
    assert change.similarity is not None and change.similarity >= 0.6
    assert change.inline_diff == "Notice must be given within [-30-]{+45+} days."


def test_comma_only_edit_is_cosmetic(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nBefore leaving, submit the expense form.\n"})
    new = build("v2", {"p.md": "# P\n\nBefore leaving submit the expense form.\n"})
    result = match_corpus(old, new, Config())
    assert [c.content for c in result.changes] == [ContentOp.COSMETIC]


def test_casing_only_edit_is_cosmetic(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nManagers Must Approve All Requests.\n"})
    new = build("v2", {"p.md": "# P\n\nManagers must approve all requests.\n"})
    result = match_corpus(old, new, Config())
    assert [c.content for c in result.changes] == [ContentOp.COSMETIC]


def test_unrelated_sentences_do_not_match(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice must be given within 30 days.\n"})
    new = build("v2", {"p.md": "# P\n\nBicycles may be stored in the basement.\n"})
    result = match_corpus(old, new, Config())
    ops = sorted(c.content.value for c in result.changes)
    assert ops == ["ADDED", "REMOVED"]


def test_deleted_section_yields_removed(build: SnapshotBuilder) -> None:
    old = build(
        "v1",
        {"e.md": "# Expenses\n\nSubmit receipts.\n\n## Per-diem\n\nThe daily rate is fixed.\n"},
    )
    new = build("v2", {"e.md": "# Expenses\n\nSubmit receipts.\n"})
    result = match_corpus(old, new, Config())
    removed = [c for c in result.changes if c.content is ContentOp.REMOVED]
    assert len(removed) == 1
    assert removed[0].old is not None and "daily rate" in removed[0].old.text
    assert removed[0].position is PositionOp.NONE


def test_added_content_yields_added(build: SnapshotBuilder) -> None:
    old = build("v1", {"e.md": "# Expenses\n\nSubmit receipts.\n"})
    new = build("v2", {"e.md": "# Expenses\n\nSubmit receipts.\n\nUse the new portal.\n"})
    result = match_corpus(old, new, Config())
    added = [c for c in result.changes if c.content is ContentOp.ADDED]
    assert len(added) == 1
    assert added[0].old is None


def test_unchanged_documents_are_excluded(build: SnapshotBuilder) -> None:
    body = "# Stable\n\nNothing changes here. Truly nothing.\n"
    old = build("v1", {"stable.md": body, "p.md": "# P\n\nNotice is 30 days.\n"})
    new = build("v2", {"stable.md": body, "p.md": "# P\n\nNotice is 45 days.\n"})
    manifest = diff_manifests(old, new)
    result = match_corpus(old, new, Config(), manifest)
    assert {(c.new or c.old).doc_id for c in result.changes} == {"p.md"}  # type: ignore[union-attr]


def test_changes_are_deterministically_ordered(build: SnapshotBuilder) -> None:
    docs_old = {"a.md": "# A\n\nOne. Two. Three.\n", "b.md": "# B\n\nFour. Five.\n"}
    docs_new = {
        "a.md": "# A\n\nOne. Two changed. Three.\n",
        "b.md": "# B\n\nFour. Five more.\n",
    }
    first = match_corpus(build("v1", docs_old), build("v2", docs_new), Config())
    second = match_corpus(build("v3", docs_old), build("v4", docs_new), Config())
    assert [c.describe() for c in first.changes] == [c.describe() for c in second.changes]
