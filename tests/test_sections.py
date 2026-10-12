"""Phase 2 — section matching, and the rename/reparent distinction.

The subtle case here is why `_apply_reparenting` runs after the section map
exists: a span under a *renamed* heading has not been reparented, so the
rename has to be resolved away before section paths are compared.
"""

from __future__ import annotations

from coursedrift.config import Config
from coursedrift.diff import ContentOp, PositionOp, SectionOp, match_corpus, match_sections
from tests.conftest import SnapshotBuilder


def _ops(links: list, op: SectionOp) -> list:
    return [link for link in links if link.op is op]


# -- split and merge -------------------------------------------------------


def test_split_is_detected(build: SnapshotBuilder) -> None:
    old = build(
        "v1",
        {
            "p.md": (
                "# Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    new = build(
        "v2",
        {
            "p.md": (
                "# Leave Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n\n"
                "# Expense Policies\n\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    sections = match_sections(old, new, Config())
    splits = _ops(sections.links, SectionOp.SPLIT)
    assert len(splits) == 1
    assert splits[0].old[0].heading == "Policies"
    assert {s.heading for s in splits[0].new} == {"Leave Policies", "Expense Policies"}


def test_merge_is_detected(build: SnapshotBuilder) -> None:
    old = build(
        "v1",
        {
            "p.md": (
                "# Leave Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n\n"
                "# Expense Policies\n\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    new = build(
        "v2",
        {
            "p.md": (
                "# Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    sections = match_sections(old, new, Config())
    merges = _ops(sections.links, SectionOp.MERGED)
    assert len(merges) == 1
    assert {s.heading for s in merges[0].old} == {"Leave Policies", "Expense Policies"}
    assert merges[0].new[0].heading == "Policies"


def test_split_sections_do_not_reparent_their_spans(build: SnapshotBuilder) -> None:
    """A span may legitimately land in any child of a section that split."""
    old = build(
        "v1",
        {
            "p.md": (
                "# Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    new = build(
        "v2",
        {
            "p.md": (
                "# Leave Policies\n\n"
                "Leave requires thirty days notice. Managers approve leave.\n\n"
                "# Expense Policies\n\n"
                "Expenses need receipts. Mileage is reimbursed monthly.\n"
            )
        },
    )
    result = match_corpus(old, new, Config())
    assert not [c for c in result.changes if c.position is PositionOp.REPARENTED]


# -- rename vs reparent ----------------------------------------------------


def test_renamed_section_with_unmoved_spans_does_not_reparent(
    build: SnapshotBuilder,
) -> None:
    """The heading changed around the span; the span did not move."""
    body = "Mileage is reimbursed monthly. Receipts are required.\n"
    old = build("v1", {"e.md": f"# Travel Costs\n\n{body}"})
    new = build("v2", {"e.md": f"# Travel Expenses\n\n{body}"})

    sections = match_sections(old, new, Config())
    assert len(_ops(sections.links, SectionOp.RENAMED)) == 1

    result = match_corpus(old, new, Config())
    assert {c.content for c in result.changes} == {ContentOp.IDENTICAL}
    assert not [c for c in result.changes if c.position is PositionOp.REPARENTED]


def test_span_moved_between_sections_is_reparented(build: SnapshotBuilder) -> None:
    """Byte-identical text, radically different context."""
    moved = "The per-diem rate applies to overnight travel."
    old = build(
        "v1",
        {
            "e.md": (
                "# Mandatory Policies\n\n"
                f"Employees must complete training. {moved}\n\n"
                "# Deprecated Practices\n\n"
                "Legacy expense codes remain documented.\n"
            )
        },
    )
    new = build(
        "v2",
        {
            "e.md": (
                "# Mandatory Policies\n\n"
                "Employees must complete training.\n\n"
                "# Deprecated Practices\n\n"
                f"Legacy expense codes remain documented. {moved}\n"
            )
        },
    )
    result = match_corpus(old, new, Config())
    reparented = [c for c in result.changes if c.position is PositionOp.REPARENTED]
    assert len(reparented) == 1
    change = reparented[0]
    assert change.content is ContentOp.IDENTICAL
    assert change.old is not None and change.old.section_path == ("Mandatory Policies",)
    assert change.new is not None and change.new.section_path == ("Deprecated Practices",)


# -- leftovers and counts --------------------------------------------------


def test_removed_and_added_sections(build: SnapshotBuilder) -> None:
    old = build("v1", {"d.md": "# Keep\n\nStays put.\n\n# Drop\n\nGoes away entirely.\n"})
    new = build("v2", {"d.md": "# Keep\n\nStays put.\n\n# Fresh\n\nBrand new material.\n"})
    sections = match_sections(old, new, Config())
    assert [link.old[0].heading for link in _ops(sections.links, SectionOp.REMOVED)] == ["Drop"]
    assert [link.new[0].heading for link in _ops(sections.links, SectionOp.ADDED)] == ["Fresh"]


def test_identical_corpus_has_only_same_sections(build: SnapshotBuilder) -> None:
    body = "# A\n\nOne. Two.\n\n## B\n\nThree.\n"
    sections = match_sections(
        build("v1", {"x.md": body}), build("v2", {"x.md": body}), Config()
    )
    assert set(sections.counts()) == {"SAME"}


def test_section_counts_summarise_restructuring(build: SnapshotBuilder) -> None:
    old = build("v1", {"d.md": "# Keep\n\nStays put.\n\n# Drop\n\nGoes away entirely.\n"})
    new = build("v2", {"d.md": "# Keep\n\nStays put.\n\n# Fresh\n\nBrand new material.\n"})
    counts = match_sections(old, new, Config()).counts()
    assert counts == {"SAME": 1, "REMOVED": 1, "ADDED": 1}


def test_expected_paths_defaults_to_identity(build: SnapshotBuilder) -> None:
    body = "# A\n\nOne. Two.\n"
    sections = match_sections(
        build("v1", {"x.md": body}), build("v2", {"x.md": body}), Config()
    )
    assert sections.expected_paths(("A",)) == {("A",)}
    assert sections.expected_paths(("Unseen",)) == {("Unseen",)}
