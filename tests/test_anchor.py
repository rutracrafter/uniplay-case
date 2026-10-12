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


# -- agreement with the matcher -------------------------------------------


def test_substantial_expansion_resolves_via_the_match_result(build: SnapshotBuilder) -> None:
    """The regression for anchoring and the matcher disagreeing.

    An independent fuzz.ratio lookup falls under the floor when a sentence
    grows this much, which would orphan the item and propose retiring a
    question because its source material got longer.
    """
    from coursedrift.config import Config as Cfg
    from coursedrift.corpus import diff_manifests
    from coursedrift.diff import match_corpus

    old = build("v1", {"p.md": "# P\n\nNotice must be given within 30 days.\n"})
    new = build(
        "v2",
        {
            "p.md": "# P\n\nNotice must typically be given within 30 days, though this "
            "may vary by region.\n"
        },
    )
    ref = _ref(old, "Notice must be given")

    bare = resolve_anchor(ref, new, Cfg())
    assert bare.status is AnchorStatus.ORPHANED, "the independent lookup gives up here"

    matches = match_corpus(old, new, Cfg(), diff_manifests(old, new))
    informed = resolve_anchor(ref, new, Cfg(), matches=matches)
    assert informed.status is AnchorStatus.RESOLVED
    assert informed.new_span is not None and "may vary by region" in informed.new_span.text


def test_matcher_removal_verdict_orphans_the_anchor(build: SnapshotBuilder) -> None:
    from coursedrift.config import Config as Cfg
    from coursedrift.corpus import diff_manifests
    from coursedrift.diff import match_corpus

    old = build("v1", {"e.md": "# E\n\nSubmit receipts.\n\nThe daily rate is fixed.\n"})
    new = build("v2", {"e.md": "# E\n\nSubmit receipts.\n"})
    ref = _ref(old, "daily rate")
    matches = match_corpus(old, new, Cfg(), diff_manifests(old, new))
    assert resolve_anchor(ref, new, Cfg(), matches=matches).status is AnchorStatus.ORPHANED


def test_identical_pairings_still_escalate_ambiguity(build: SnapshotBuilder) -> None:
    """The matcher's choice inside a duplicate group is arbitrary, not knowledge."""
    from coursedrift.config import Config as Cfg
    from coursedrift.corpus import diff_manifests
    from coursedrift.diff import match_corpus

    shared = "All employees must comply."
    old = build("v1", {"d.md": f"# Alpha\n\n{shared}\n"})
    new = build("v2", {"d.md": f"# Alpha\n\n{shared}\n\nFiller sentence.\n\n{shared}\n"})
    ref = _ref(old, shared)
    matches = match_corpus(old, new, Cfg(), diff_manifests(old, new))
    anchor = resolve_anchor(
        ref, new, Cfg(), old_span=old.span_by_id(ref.span_id), matches=matches
    )
    assert anchor.status is AnchorStatus.AMBIGUOUS
    assert len(anchor.candidates) == 2


def test_spans_the_matcher_never_saw_fall_through(build: SnapshotBuilder) -> None:
    """Unchanged documents are excluded from matching and resolve by hash."""
    from coursedrift.config import Config as Cfg
    from coursedrift.corpus import diff_manifests
    from coursedrift.diff import match_corpus

    body = "# S\n\nNothing changes here at all.\n"
    old = build("v1", {"s.md": body, "p.md": "# P\n\nNotice is 30 days.\n"})
    new = build("v2", {"s.md": body, "p.md": "# P\n\nNotice is 45 days.\n"})
    ref = _ref(old, "Nothing changes")
    matches = match_corpus(old, new, Cfg(), diff_manifests(old, new))
    assert "s.md" not in {(c.new or c.old).doc_id for c in matches.changes}  # type: ignore[union-attr]
    assert resolve_anchor(ref, new, Cfg(), matches=matches).status is AnchorStatus.RESOLVED


def test_a_stale_span_id_does_not_borrow_another_spans_fate(build: SnapshotBuilder) -> None:
    """span_id is positional, so it cannot be the basis of identity.

    Insert a paragraph into the old snapshot and every later span_id shifts.
    A ref recorded before that edit then names a span holding different
    content, and keying the change table on the id alone would judge a
    question against material it never came from. The recorded hash is what
    settles it.
    """
    from coursedrift.config import Config as Cfg
    from coursedrift.corpus import diff_manifests
    from coursedrift.diff import match_corpus

    before = build("v1", {"d.md": "# D\n\nNotice is 30 days.\n\nBadges must be worn.\n"})
    ref = _ref(before, "Badges must be worn")
    assert ref.span_id == "d.md#p1.s0"

    # A paragraph appears above it, so p1 now holds something else entirely.
    after_edit = build(
        "v1b", {"d.md": "# D\n\nNotice is 30 days.\n\nInserted line.\n\nBadges must be worn.\n"}
    )
    new = build(
        "v2", {"d.md": "# D\n\nNotice is 30 days.\n\nInserted line.\n\nBadges must be worn.\n"}
    )
    shifted = next(s for s in after_edit.all_spans() if "Badges" in s.text)
    assert shifted.span_id == "d.md#p2.s0", "the id really did move"

    matches = match_corpus(after_edit, new, Cfg(), diff_manifests(after_edit, new))
    anchor = resolve_anchor(ref, new, Cfg(), matches=matches)

    assert anchor.located
    assert anchor.new_span is not None
    assert "Badges" in anchor.new_span.text, "must resolve by content, not by stale id"
