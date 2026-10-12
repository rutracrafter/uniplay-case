"""Deciding which items need a model, and which do not.

This is the narrowing stage, and it is the reason the run is cheap and the
report honest: an item touching nothing never reaches an LLM, and the number
of items skipped is reported rather than hidden.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from coursedrift.anchor import AnchorReport, AnchorStatus, ResolvedAnchor
from coursedrift.config import Config
from coursedrift.diff import ContentOp, MatchResult, PositionOp, SpanChange
from coursedrift.models import Course, CourseItem, Route


def requires_adjudication(change: SpanChange) -> bool:
    """Whether a span change needs a model's judgment.

    The two-axis table from OVERVIEW.md, as one function:

        IDENTICAL / STABLE      -> no, not a change at all
        IDENTICAL / MOVED       -> no, structural only
        IDENTICAL / REPARENTED  -> yes, same words in a different context
        COSMETIC  / any         -> no, deterministically trivial
        MODIFIED  / any         -> yes
        REMOVED                 -> no, deterministic retire candidate
    """
    if change.content is ContentOp.MODIFIED:
        return True
    if change.content is ContentOp.IDENTICAL:
        return change.position is PositionOp.REPARENTED
    return False


class ImpactCandidate(BaseModel):
    """One course item, with everything known about it before judgment."""

    item: CourseItem
    anchors: list[ResolvedAnchor] = Field(default_factory=list)
    changes: list[SpanChange] = Field(default_factory=list)
    needs_model: bool = False
    preset_route: Route | None = None
    reasons: list[str] = Field(default_factory=list)

    def adjudicable(self) -> list[SpanChange]:
        return [c for c in self.changes if requires_adjudication(c)]

    @property
    def inferred_provenance(self) -> bool:
        return any(a.inferred for a in self.anchors)

    @property
    def has_reparenting(self) -> bool:
        return any(c.position is PositionOp.REPARENTED for c in self.changes)

    @property
    def has_reordering(self) -> bool:
        return any(c.reordered for c in self.changes)


class SkipStats(BaseModel):
    """What the run did and did not spend a model call on.

    The first four counters are mutually exclusive and sum to `total_items`;
    the span counters below them are informational.
    """

    total_items: int = 0
    untouched: int = 0
    skipped_no_impact: int = 0
    decided_deterministically: int = 0
    sent_to_model: int = 0

    cosmetic_spans: int = 0
    moved_spans: int = 0

    @property
    def accounted(self) -> int:
        return (
            self.untouched
            + self.skipped_no_impact
            + self.decided_deterministically
            + self.sent_to_model
        )


def _classify(
    item: CourseItem,
    anchors: list[ResolvedAnchor],
    changes: list[SpanChange],
) -> tuple[bool, Route | None, list[str]]:
    """Decide whether this item needs a model, deterministically where possible.

    Precedence is deliberate and strongest-first: a question whose source is
    gone cannot be rescued by judging a different span, and an anchor nobody
    can pin down must not be judged as though it were pinned down.
    """
    reasons: list[str] = []
    adjudicable = [c for c in changes if requires_adjudication(c)]

    if not anchors:
        # Nothing was recorded and nothing could be inferred, so there is no
        # evidence to judge. Admitting that is the only honest outcome.
        return False, Route.NEEDS_REVIEW, ["no provenance available for this item"]

    orphaned = [a for a in anchors if a.status is AnchorStatus.ORPHANED]
    ambiguous = [a for a in anchors if a.status is AnchorStatus.AMBIGUOUS]

    if len(orphaned) == len(anchors):
        return (
            False,
            Route.RETIRE_SUGGESTED,
            [f"all {len(anchors)} source span(s) removed with no relocation found"],
        )

    if ambiguous:
        return (
            False,
            Route.NEEDS_REVIEW,
            [
                f"{len(ambiguous)} anchor(s) matched several candidates "
                "and cannot be pinned down"
            ],
        )

    if orphaned:
        reasons.append(f"{len(orphaned)} of {len(anchors)} source span(s) removed")
        if not adjudicable:
            return False, Route.NEEDS_REVIEW, reasons

    if not adjudicable:
        if not changes:
            reasons.append("no source span changed")
        else:
            reasons.append(
                "source changed only in ways that cannot affect the question "
                f"({', '.join(sorted({c.content.value for c in changes}))})"
            )
        return False, Route.OK, reasons

    reasons.append(f"{len(adjudicable)} source span(s) require judgment")
    if any(a.inferred for a in anchors):
        reasons.append("provenance was inferred, not recorded")
    return True, None, reasons


def build_candidates(
    course: Course,
    matches: MatchResult,
    anchors: AnchorReport,
    cfg: Config,
) -> tuple[list[ImpactCandidate], SkipStats]:
    """Join the course against the span-change table.

    A course item's provenance points at *old* spans, so the change table is
    keyed by old span id. Items in documents the manifest called unchanged
    have no entry there at all, which is exactly how they get skipped.
    """
    by_old = matches.by_old_span()
    candidates: list[ImpactCandidate] = []
    stats = SkipStats(total_items=len(course.items))

    for item in course.items:
        item_anchors = anchors.for_item(item.item_id)

        seen: set[str] = set()
        changes: list[SpanChange] = []
        for anchor in item_anchors:
            change = by_old.get(anchor.ref.span_id)
            if change is None:
                continue
            key = change.describe()
            if key in seen:
                continue
            seen.add(key)
            changes.append(change)

        needs_model, preset, reasons = _classify(item, item_anchors, changes)
        candidates.append(
            ImpactCandidate(
                item=item,
                anchors=item_anchors,
                changes=changes,
                needs_model=needs_model,
                preset_route=preset,
                reasons=reasons,
            )
        )

        stats.cosmetic_spans += sum(1 for c in changes if c.content is ContentOp.COSMETIC)
        stats.moved_spans += sum(
            1
            for c in changes
            if c.content is ContentOp.IDENTICAL and c.position is PositionOp.MOVED
        )
        if needs_model:
            stats.sent_to_model += 1
        elif preset in (Route.NEEDS_REVIEW, Route.RETIRE_SUGGESTED):
            stats.decided_deterministically += 1
        elif not changes:
            stats.untouched += 1
        else:
            stats.skipped_no_impact += 1

    return candidates, stats
