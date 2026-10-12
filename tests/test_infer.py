"""Phase 3 — bounded provenance inference.

The inference path is the fuzziest component in the system. These tests pin
the one property that makes its immaturity safe: everything it produces is
marked `inferred`, which triage treats as a hard cap.
"""

from __future__ import annotations

from coursedrift.anchor import infer_provenance
from coursedrift.config import Config
from coursedrift.models import CourseItem
from tests.conftest import SnapshotBuilder

HANDBOOK = """# Leave Policy

Employees must give notice of thirty days before taking extended leave.
Managers approve all leave requests within one week.

# Expenses

Mileage is reimbursed monthly at the published rate.
Receipts are required for any expense over fifty euros.
"""


def _item(stem: str, answer: str) -> CourseItem:
    return CourseItem(
        item_id="q1",
        level_id="L1",
        stem=stem,
        options=[answer, "something else entirely"],
        correct_index=0,
    )


def test_ranks_the_source_span_first(build: SnapshotBuilder) -> None:
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(
        _item("How much notice is required before extended leave?", "thirty days"),
        old,
        Config(),
    )
    assert refs
    assert "notice of thirty days" in refs[0].snapshot_text


def test_ranks_the_right_span_for_a_different_topic(build: SnapshotBuilder) -> None:
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(
        _item("Above what amount are receipts required?", "fifty euros"), old, Config()
    )
    assert refs
    assert "Receipts are required" in refs[0].snapshot_text


def test_every_inferred_ref_is_marked(build: SnapshotBuilder) -> None:
    """The guard for the triage rule: inference can never look recorded."""
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(_item("How much notice?", "thirty days"), old, Config())
    assert refs
    assert all(ref.inferred for ref in refs)
    assert all(ref.infer_score is not None and ref.infer_score > 0 for ref in refs)


def test_respects_top_k(build: SnapshotBuilder) -> None:
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(
        _item("What about leave and expenses and receipts and mileage?", "thirty days"),
        old,
        Config(infer_top_k=2),
    )
    assert len(refs) <= 2


def test_no_term_overlap_infers_nothing(build: SnapshotBuilder) -> None:
    """Inferring noise is worse than inferring nothing."""
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(
        _item("Which spacecraft reached Jupiter first?", "Pioneer 10"), old, Config()
    )
    assert refs == []


def test_empty_corpus_infers_nothing(build: SnapshotBuilder) -> None:
    old = build("v1", {"empty.md": "# Nothing\n"})
    assert infer_provenance(_item("How much notice?", "thirty days"), old, Config()) == []


def test_inference_is_deterministic(build: SnapshotBuilder) -> None:
    item = _item("How much notice is required before extended leave?", "thirty days")
    first = infer_provenance(item, build("v1", {"h.md": HANDBOOK}), Config())
    second = infer_provenance(item, build("v2", {"h.md": HANDBOOK}), Config())
    assert [r.span_id for r in first] == [r.span_id for r in second]
    assert [r.infer_score for r in first] == [r.infer_score for r in second]


def test_prefers_spans_beside_the_best_hit(build: SnapshotBuilder) -> None:
    """Section agreement is the tiebreak among comparable scores."""
    old = build("v1", {"h.md": HANDBOOK})
    refs = infer_provenance(
        _item("What is the notice period for leave?", "thirty days"), old, Config()
    )
    assert refs
    best_section = refs[0].span_id.split("#")[0]
    assert all(r.doc_id == best_section for r in refs)
