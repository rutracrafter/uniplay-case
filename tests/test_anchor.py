"""Phase 3 — relocating recorded provenance in the new corpus."""

from __future__ import annotations

from coursedrift.anchor import AnchorStatus, make_ref, resolve_anchor, resolve_course
from coursedrift.config import Config
from coursedrift.models import Course, CourseItem, ProvenanceRef, Snapshot
from coursedrift.segment import normalize, span_hash
from tests.conftest import SnapshotBuilder


def _ref(snap: Snapshot, needle: str) -> ProvenanceRef:
    """Record provenance for the span containing `needle`, as a generator would."""
    for span in snap.all_spans():
        if needle in span.text:
            return make_ref(span)
    raise AssertionError(f"no span containing {needle!r}")


def _item(ref: ProvenanceRef, *, stem: str = "How many days?") -> CourseItem:
    return CourseItem(
        item_id="q1",
        level_id="L1",
        stem=stem,
        options=["30", "45"],
        correct_index=0,
        provenance=[ref],
    )


# -- the happy paths -------------------------------------------------------


def test_hash_hit_survives_unrelated_edits_elsewhere(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice is 30 days.\n\nReceipts are required.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice is 30 days.\n\nReceipts are now optional.\n"})
    anchor = resolve_anchor(_ref(old, "Notice is 30"), new, Config())
    assert anchor.status is AnchorStatus.RESOLVED
    assert anchor.new_span is not None and "Notice is 30" in anchor.new_span.text
    assert anchor.similarity == 1.0


def test_span_moved_to_another_section_is_moved(build: SnapshotBuilder) -> None:
    moved = "The per-diem rate applies to overnight travel."
    old = build("v1", {"e.md": f"# Mandatory\n\n{moved}\n\n# Deprecated\n\nLegacy codes.\n"})
    new = build("v2", {"e.md": f"# Mandatory\n\nOther text.\n\n# Deprecated\n\n{moved}\n"})
    ref = _ref(old, "per-diem")
    anchor = resolve_anchor(ref, new, Config(), old_span=old.span_by_id(ref.span_id))
    assert anchor.status is AnchorStatus.MOVED
    assert anchor.new_span is not None
    assert anchor.new_span.section_path == ("Deprecated",)


def test_span_moved_to_another_document_is_moved(build: SnapshotBuilder) -> None:
    moved = "Parental leave runs for twelve weeks."
    old = build("v1", {"onboarding.md": f"# Onboarding\n\n{moved}\n"})
    new = build("v2", {"leave.md": f"# Leave\n\nNotice is 30 days.\n\n{moved}\n"})
    anchor = resolve_anchor(_ref(old, "Parental leave"), new, Config())
    assert anchor.status is AnchorStatus.MOVED
    assert anchor.new_span is not None and anchor.new_span.doc_id == "leave.md"


def test_lightly_reworded_span_resolves_by_fuzzy_match(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice must be given within 30 days.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice must be given within 45 days.\n"})
    anchor = resolve_anchor(_ref(old, "30 days"), new, Config())
    assert anchor.status is AnchorStatus.RESOLVED
    assert anchor.similarity is not None and 0.85 <= anchor.similarity < 1.0
    assert anchor.new_span is not None and "45 days" in anchor.new_span.text


def test_deleted_span_is_orphaned(build: SnapshotBuilder) -> None:
    old = build("v1", {"e.md": "# E\n\nSubmit receipts.\n\nThe daily rate is fixed.\n"})
    new = build("v2", {"e.md": "# E\n\nSubmit receipts.\n"})
    anchor = resolve_anchor(_ref(old, "daily rate"), new, Config())
    assert anchor.status is AnchorStatus.ORPHANED
    assert anchor.new_span is None
    assert not anchor.located


# -- ambiguity -------------------------------------------------------------


def test_section_tiebreak_picks_the_right_duplicate(build: SnapshotBuilder) -> None:
    """Identical text in two sections resolves by matching the original one."""
    shared = "All employees must comply."
    old = build("v1", {"d.md": f"# Alpha\n\n{shared}\n\n# Beta\n\nOther text here.\n"})
    new = build("v2", {"d.md": f"# Alpha\n\n{shared}\n\n# Beta\n\n{shared}\n"})
    ref = _ref(old, shared)
    anchor = resolve_anchor(ref, new, Config(), old_span=old.span_by_id(ref.span_id))
    assert anchor.status is AnchorStatus.RESOLVED
    assert anchor.new_span is not None
    assert anchor.new_span.section_path == ("Alpha",)


def test_duplicates_in_the_same_section_are_ambiguous(build: SnapshotBuilder) -> None:
    """No tiebreak can separate these, so escalate rather than guess."""
    shared = "All employees must comply."
    old = build("v1", {"d.md": f"# Alpha\n\n{shared}\n"})
    new = build("v2", {"d.md": f"# Alpha\n\n{shared}\n\nFiller sentence.\n\n{shared}\n"})
    ref = _ref(old, shared)
    anchor = resolve_anchor(ref, new, Config(), old_span=old.span_by_id(ref.span_id))
    assert anchor.status is AnchorStatus.AMBIGUOUS
    assert len(anchor.candidates) == 2
    assert anchor.new_span is None


def test_tiebreak_never_turns_a_hit_into_an_orphan(build: SnapshotBuilder) -> None:
    """A filter that would eliminate everything is discarded, not applied."""
    text = "Mileage is reimbursed monthly."
    old = build("v1", {"a.md": f"# A\n\n{text}\n"})
    new = build("v2", {"b.md": f"# Different\n\n{text}\n"})
    ref = _ref(old, text)
    anchor = resolve_anchor(ref, new, Config(), old_span=old.span_by_id(ref.span_id))
    assert anchor.located
    assert anchor.new_span is not None and anchor.new_span.doc_id == "b.md"


def test_unknown_hash_and_unrelated_corpus_is_orphaned(build: SnapshotBuilder) -> None:
    new = build("v2", {"x.md": "# X\n\nCompletely unrelated material about bicycles.\n"})
    ref = ProvenanceRef(
        doc_id="gone.md",
        span_id="gone.md#p0.s0",
        snapshot_text="The notice period is thirty days.",
        snapshot_hash=span_hash(normalize("The notice period is thirty days.")),
    )
    assert resolve_anchor(ref, new, Config()).status is AnchorStatus.ORPHANED


# -- course level ----------------------------------------------------------


def test_resolve_course_resolves_recorded_provenance(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice is 30 days.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice is 45 days.\n"})
    course = Course(course_id="c", title="T", items=[_item(_ref(old, "30 days"))])
    report = resolve_course(course, old, new, Config())
    assert report.inferred_items == set()
    assert report.for_item("q1")[0].located


def test_resolve_course_infers_when_provenance_is_missing(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice is 30 days. Receipts are required.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice is 45 days. Receipts are required.\n"})
    bare = CourseItem(
        item_id="q2",
        level_id="L1",
        stem="How many days of notice are required?",
        options=["30", "45"],
        correct_index=0,
    )
    course = Course(course_id="c", title="T", items=[bare])
    report = resolve_course(course, old, new, Config())
    assert report.inferred_items == {"q2"}
    anchors = report.for_item("q2")
    assert anchors
    assert all(a.inferred for a in anchors)


def test_resolve_course_can_be_told_not_to_infer(build: SnapshotBuilder) -> None:
    old = build("v1", {"p.md": "# P\n\nNotice is 30 days.\n"})
    new = build("v2", {"p.md": "# P\n\nNotice is 45 days.\n"})
    bare = CourseItem(
        item_id="q3", level_id="L1", stem="How many?", options=["30", "45"], correct_index=0
    )
    course = Course(course_id="c", title="T", items=[bare])
    report = resolve_course(course, old, new, Config(), infer=False)
    assert report.for_item("q3") == []
    assert report.inferred_items == set()
