"""Phase 2 — the regression test for the core invariant.

    No stage may use positional order to establish identity.

A position-based differ reports every span of a reordered document as
REMOVED, orphans the questions anchored to them, and proposes retiring
content that is perfectly correct. If anything in this module fails, nothing
downstream can be trusted — do not work around it.
"""

from __future__ import annotations

from coursedrift.config import Config
from coursedrift.diff import ContentOp, PositionOp, match_corpus
from tests.conftest import SnapshotBuilder

ACCESS = "# Access Control\n\nAll accounts require MFA. Reviews happen quarterly.\n\n"
INCIDENT = (
    "# Incident Response\n\nReport incidents within one hour. The duty officer triages.\n\n"
)
RETENTION = "# Data Retention\n\nLogs are kept for 90 days. Backups are encrypted.\n\n"
VENDOR = "# Vendor Access\n\nVendors sign an NDA. Access expires after 30 days.\n\n"

ORDERED = ACCESS + INCIDENT + RETENTION + VENDOR
ROTATED = RETENTION + ACCESS + INCIDENT + VENDOR
REVERSED = VENDOR + RETENTION + INCIDENT + ACCESS


def test_reordering_produces_no_content_changes(build: SnapshotBuilder) -> None:
    """The headline assertion: a pure reorder is not drift."""
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": ROTATED})
    result = match_corpus(old, new, Config())

    content_ops = {c.content for c in result.changes}
    assert content_ops == {ContentOp.IDENTICAL}
    assert not [c for c in result.changes if c.content is ContentOp.MODIFIED]
    assert not [c for c in result.changes if c.content is ContentOp.REMOVED]
    assert not [c for c in result.changes if c.content is ContentOp.ADDED]


def test_reordering_marks_only_the_moved_block(build: SnapshotBuilder) -> None:
    """Rotation moves one section, not the whole document."""
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": ROTATED})
    result = match_corpus(old, new, Config())

    moved = [c for c in result.changes if c.position is PositionOp.MOVED]
    stable = [c for c in result.changes if c.position is PositionOp.STABLE]

    # Data Retention is the block that moved to the front: two spans.
    assert len(moved) == 2
    assert all(c.new is not None and c.new.section_path == ("Data Retention",) for c in moved)
    assert len(stable) == 6


def test_reordering_never_reparents(build: SnapshotBuilder) -> None:
    """Headings travel with their content, so no span changes section."""
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": REVERSED})
    result = match_corpus(old, new, Config())
    assert not [c for c in result.changes if c.position is PositionOp.REPARENTED]


def test_full_reversal_still_has_no_content_changes(build: SnapshotBuilder) -> None:
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": REVERSED})
    result = match_corpus(old, new, Config())
    assert {c.content for c in result.changes} == {ContentOp.IDENTICAL}


def test_reorder_plus_one_edit_yields_exactly_one_modified(build: SnapshotBuilder) -> None:
    """Reordering must not mask a real edit, or inflate it either."""
    edited = ROTATED.replace("kept for 90 days", "kept for 180 days")
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": edited})
    result = match_corpus(old, new, Config())

    modified = [c for c in result.changes if c.content is ContentOp.MODIFIED]
    assert len(modified) == 1
    assert modified[0].old is not None and "90 days" in modified[0].old.text
    assert modified[0].new is not None and "180 days" in modified[0].new.text
    assert not [c for c in result.changes if c.content in (ContentOp.REMOVED, ContentOp.ADDED)]


def test_reordering_is_reported_as_structure_not_findings(build: SnapshotBuilder) -> None:
    """Moved-but-identical spans are changes on the position axis only."""
    old = build("v1", {"security.md": ORDERED})
    new = build("v2", {"security.md": ROTATED})
    result = match_corpus(old, new, Config())

    real = result.real_changes()
    assert len(real) == 2
    assert all(c.content is ContentOp.IDENTICAL for c in real)
    assert all(c.position is PositionOp.MOVED for c in real)
