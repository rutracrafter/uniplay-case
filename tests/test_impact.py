"""Phase 4 — impact mapping and the deterministic rules.

`requires_adjudication` is the two-axis table from OVERVIEW.md, and it is
tested as a truth table because every model call the system makes or skips
hangs off it.
"""

from __future__ import annotations

import pytest

from coursedrift.anchor import AnchorReport, AnchorStatus, ResolvedAnchor, make_ref
from coursedrift.config import Config
from coursedrift.corpus import diff_manifests
from coursedrift.diff import ContentOp, PositionOp, SpanChange, match_corpus
from coursedrift.impact import build_candidates, requires_adjudication
from coursedrift.models import (
    Course,
    CourseItem,
    ProvenanceRef,
    Route,
    Snapshot,
    Span,
)
from tests.conftest import SnapshotBuilder


def _span(span_id: str = "d.md#p0.s0") -> Span:
    return Span(
        span_id=span_id,
        doc_id="d.md",
        section_path=("A",),
        para_index=0,
        sent_index=0,
        order=0,
        text="Notice is 30 days.",
        norm_text="Notice is 30 days.",
        hash="deadbeefdeadbeef",
    )


# -- the truth table -------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "position", "expected"),
    [
        (ContentOp.IDENTICAL, PositionOp.STABLE, False),
        (ContentOp.IDENTICAL, PositionOp.MOVED, False),
        (ContentOp.IDENTICAL, PositionOp.REPARENTED, True),
        (ContentOp.COSMETIC, PositionOp.STABLE, False),
        (ContentOp.COSMETIC, PositionOp.MOVED, False),
        (ContentOp.COSMETIC, PositionOp.REPARENTED, False),
        (ContentOp.MODIFIED, PositionOp.STABLE, True),
        (ContentOp.MODIFIED, PositionOp.MOVED, True),
        (ContentOp.MODIFIED, PositionOp.REPARENTED, True),
        (ContentOp.REMOVED, PositionOp.NONE, False),
        (ContentOp.ADDED, PositionOp.NONE, False),
    ],
)
def test_requires_adjudication_truth_table(
    content: ContentOp, position: PositionOp, expected: bool
) -> None:
    change = SpanChange(content=content, position=position, old=_span(), new=_span())
    assert requires_adjudication(change) is expected


# -- end to end over real snapshots ---------------------------------------


def _course(snap: Snapshot, needle: str, item_id: str = "q1") -> Course:
    for span in snap.all_spans():
        if needle in span.text:
            return Course(
                course_id="c",
                title="T",
                items=[
                    CourseItem(
                        item_id=item_id,
                        level_id="L1",
                        stem="How many days of notice?",
                        options=["30", "45"],
                        correct_index=0,
                        provenance=[make_ref(span)],
                    )
                ],
            )
    raise AssertionError(needle)


def _run(
    build: SnapshotBuilder, old_docs: dict[str, str], new_docs: dict[str, str], needle: str
) -> tuple[list, object]:
    from coursedrift.anchor import resolve_course

    old = build("v1", old_docs)
    new = build("v2", new_docs)
    course = _course(old, needle)
    manifest = diff_manifests(old, new)
    matches = match_corpus(old, new, Config(), manifest)
    anchors = resolve_course(course, old, new, Config())
    return build_candidates(course, matches, anchors, Config())


def test_untouched_item_never_reaches_the_model(build: SnapshotBuilder) -> None:
    body = "# P\n\nNotice is 30 days.\n"
    candidates, stats = _run(build, {"p.md": body}, {"p.md": body}, "Notice is 30")
    assert candidates[0].needs_model is False
    assert candidates[0].preset_route is Route.OK
    assert stats.untouched == 1
    assert stats.sent_to_model == 0


def test_modified_source_is_queued_once(build: SnapshotBuilder) -> None:
    candidates, stats = _run(
        build,
        {"p.md": "# P\n\nNotice is 30 days. Managers approve. Receipts required.\n"},
        {"p.md": "# P\n\nNotice is 45 days. Managers approve. Receipts required.\n"},
        "Notice is 30",
    )
    candidate = candidates[0]
    assert candidate.needs_model is True
    assert candidate.preset_route is None
    assert len(candidate.adjudicable()) == 1
    assert candidate.adjudicable()[0].content is ContentOp.MODIFIED
    assert stats.sent_to_model == 1


def test_cosmetic_change_is_skipped(build: SnapshotBuilder) -> None:
    candidates, stats = _run(
        build,
        {"p.md": "# P\n\nBefore leaving, submit the form.\n"},
        {"p.md": "# P\n\nBefore leaving submit the form.\n"},
        "Before leaving",
    )
    assert candidates[0].needs_model is False
    assert candidates[0].preset_route is Route.OK
    assert stats.skipped_no_impact == 1
    assert stats.cosmetic_spans == 1


def test_moved_but_identical_is_skipped(build: SnapshotBuilder) -> None:
    a = "# A\n\nNotice is 30 days. Managers approve.\n\n"
    b = "# B\n\nReceipts are required. Mileage is monthly.\n\n"
    candidates, stats = _run(build, {"p.md": a + b}, {"p.md": b + a}, "Notice is 30")
    assert candidates[0].needs_model is False
    assert candidates[0].preset_route is Route.OK
    assert stats.moved_spans == 1


def test_reparented_span_is_queued(build: SnapshotBuilder) -> None:
    moved = "The per-diem rate applies to overnight travel."
    before = f"# Mandatory\n\nTraining is required. {moved}\n\n# Deprecated\n\nLegacy codes.\n"
    after = f"# Mandatory\n\nTraining is required.\n\n# Deprecated\n\nLegacy codes. {moved}\n"
    candidates, _ = _run(build, {"e.md": before}, {"e.md": after}, "per-diem")
    assert candidates[0].needs_model is True
    assert candidates[0].has_reparenting is True


def test_orphaned_source_presets_retire_without_a_model(build: SnapshotBuilder) -> None:
    candidates, stats = _run(
        build,
        {"e.md": "# E\n\nSubmit receipts.\n\n## Per-diem\n\nThe daily rate is fixed.\n"},
        {"e.md": "# E\n\nSubmit receipts.\n"},
        "daily rate",
    )
    assert candidates[0].needs_model is False
    assert candidates[0].preset_route is Route.RETIRE_SUGGESTED
    assert stats.decided_deterministically == 1
    assert "removed with no relocation" in candidates[0].reasons[0]


def test_ambiguous_anchor_presets_review(build: SnapshotBuilder) -> None:
    shared = "All employees must comply."
    candidates, stats = _run(
        build,
        {"d.md": f"# Alpha\n\n{shared}\n"},
        {"d.md": f"# Alpha\n\n{shared}\n\nFiller sentence.\n\n{shared}\n"},
        shared,
    )
    assert candidates[0].preset_route is Route.NEEDS_REVIEW
    assert candidates[0].needs_model is False
    assert stats.decided_deterministically == 1


def test_item_with_no_provenance_at_all_presets_review() -> None:
    """Nothing recorded, nothing inferable: admit it rather than judge blind."""
    from coursedrift.diff import MatchResult

    item = CourseItem(
        item_id="q9", level_id="L1", stem="What?", options=["a", "b"], correct_index=0
    )
    course = Course(course_id="c", title="T", items=[item])
    candidates, stats = build_candidates(
        course, MatchResult(), AnchorReport(anchors={"q9": []}), Config()
    )
    assert candidates[0].preset_route is Route.NEEDS_REVIEW
    assert candidates[0].reasons == ["no provenance available for this item"]
    assert stats.decided_deterministically == 1


def test_partial_source_loss_with_no_other_change_presets_review() -> None:
    """Losing one of two sources is a real event, even with nothing else changed."""
    from coursedrift.diff import MatchResult

    kept = ProvenanceRef(
        doc_id="d.md", span_id="d.md#p0.s0", snapshot_text="Kept.", snapshot_hash="aaaa"
    )
    lost = ProvenanceRef(
        doc_id="d.md", span_id="d.md#p1.s0", snapshot_text="Lost.", snapshot_hash="bbbb"
    )
    item = CourseItem(
        item_id="q8", level_id="L1", stem="What?", options=["a", "b"], correct_index=0
    )
    anchors = AnchorReport(
        anchors={
            "q8": [
                ResolvedAnchor(ref=kept, status=AnchorStatus.RESOLVED, new_span=_span()),
                ResolvedAnchor(ref=lost, status=AnchorStatus.ORPHANED),
            ]
        }
    )
    candidates, _ = build_candidates(
        Course(course_id="c", title="T", items=[item]), MatchResult(), anchors, Config()
    )
    assert candidates[0].preset_route is Route.NEEDS_REVIEW
    assert "1 of 2 source span(s) removed" in candidates[0].reasons[0]


def test_inferred_provenance_is_flagged_on_the_candidate(build: SnapshotBuilder) -> None:
    from coursedrift.anchor import resolve_course

    old = build("v1", {"p.md": "# P\n\nNotice is 30 days. Receipts required.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice is 45 days. Receipts required.\n"})
    bare = CourseItem(
        item_id="q7",
        level_id="L1",
        stem="How many days of notice are required?",
        options=["30", "45"],
        correct_index=0,
    )
    course = Course(course_id="c", title="T", items=[bare])
    matches = match_corpus(old, new, Config(), diff_manifests(old, new))
    anchors = resolve_course(course, old, new, Config())
    candidates, _ = build_candidates(course, matches, anchors, Config())
    assert candidates[0].inferred_provenance is True
    assert "provenance was inferred, not recorded" in candidates[0].reasons


# -- accounting ------------------------------------------------------------


def test_skip_stats_account_for_every_item(build: SnapshotBuilder) -> None:
    from coursedrift.anchor import resolve_course

    old = build(
        "v1",
        {
            "p.md": "# P\n\nNotice is 30 days.\n\nBefore leaving, submit the form.\n",
            "s.md": "# S\n\nNothing changes here at all.\n",
        },
    )
    new = build(
        "v2",
        {
            "p.md": "# P\n\nNotice is 45 days.\n\nBefore leaving submit the form.\n",
            "s.md": "# S\n\nNothing changes here at all.\n",
        },
    )
    spans = {s.text[:12]: s for s in old.all_spans()}
    items = [
        CourseItem(
            item_id=f"q{i}",
            level_id="L1",
            stem="Q?",
            options=["a", "b"],
            correct_index=0,
            provenance=[make_ref(span)],
        )
        for i, span in enumerate(spans.values())
    ]
    course = Course(course_id="c", title="T", items=items)
    matches = match_corpus(old, new, Config(), diff_manifests(old, new))
    anchors = resolve_course(course, old, new, Config())
    candidates, stats = build_candidates(course, matches, anchors, Config())

    assert stats.total_items == len(items)
    assert stats.accounted == stats.total_items
    assert len(candidates) == len(items)
