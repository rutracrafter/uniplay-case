"""Routing. Pure: no I/O, no client, no clock.

Everything the system could not decide lands in review. Nothing is silently
passed, and nothing is ever applied.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from coursedrift.adjudicate import BROKEN, AdjudicationResult, Verdict
from coursedrift.anchor import ResolvedAnchor
from coursedrift.config import Config
from coursedrift.diff import SpanChange
from coursedrift.impact import ImpactCandidate
from coursedrift.llm.base import CallOutcome
from coursedrift.models import Route
from coursedrift.remediate import Patch


class Finding(BaseModel):
    """One item's outcome, with everything a reviewer needs to check it."""

    item_id: str
    level_id: str
    route: Route
    reasons: list[str] = Field(default_factory=list)

    verdict: Verdict | None = None
    confidence: float | None = None
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    unanimous: bool = True

    patch: Patch | None = None
    rejected_patch: str | None = None

    changes: list[SpanChange] = Field(default_factory=list)
    anchors: list[ResolvedAnchor] = Field(default_factory=list)
    inferred_provenance: bool = False

    model_calls: int = 0
    cache_hits: int = 0

    @property
    def severity(self) -> int:
        return self.route.severity

    @property
    def doc_ids(self) -> list[str]:
        out: list[str] = []
        for change in self.changes:
            span = change.new or change.old
            if span is not None and span.doc_id not in out:
                out.append(span.doc_id)
        return out


def _dedupe(reasons: list[str]) -> list[str]:
    """Keep reason order but drop repeats.

    Phase 4 and phase 9 can both have something to say about the same fact —
    inferred provenance, for instance — and a finding that states it twice
    reads like a bug.
    """
    seen: set[str] = set()
    out: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            out.append(reason)
    return out


def _base(candidate: ImpactCandidate, route: Route, reasons: list[str]) -> Finding:
    return Finding(
        item_id=candidate.item.item_id,
        level_id=candidate.item.level_id,
        route=route,
        reasons=_dedupe(reasons),
        changes=candidate.changes,
        anchors=candidate.anchors,
        inferred_provenance=candidate.inferred_provenance,
    )


def route_candidate(
    candidate: ImpactCandidate,
    adjudication: AdjudicationResult | None,
    patch_outcome: CallOutcome[Patch] | None,
    cfg: Config,
) -> Finding:
    """Decide one item's route.

    Precedence, highest first. Each rule is a reason the system cannot or
    should not auto-approve, and the first that fires wins:

      1. a deterministic preset from phase 4
      2. no usable adjudication (failure, timeout, exhausted retries)
      3. the model abstained
      4. confidence below the floor
      5. votes disagreed
      6. provenance was inferred rather than recorded
      7. a span was reparented
      8. a span was rearranged into the same words in a different order
      9. the model says nothing is wrong
     10. no patch survived validation
     11. a validated patch, ready for a human to approve

    Two orderings here differ from my first draft of the plan, and both
    changes make the system fail closed rather than open:

    * The confidence floor (4) sits *above* the UNAFFECTED check (8). A
      model reporting "unaffected, confidence 0.3" is telling us it does not
      know, and a system whose whole posture is fail-closed must not read
      that as "fine".
    * The inferred-provenance cap (6) also sits above UNAFFECTED. Reaching
      OK on inferred provenance would mean we guessed which span the
      question came from, that guessed span changed, and we then reported
      nothing — a silent pass built on a guess. Items with inferred
      provenance whose spans did not change never reach triage at all;
      phase 4 presets them to OK.
    """
    if candidate.preset_route is not None:
        return _base(candidate, candidate.preset_route, list(candidate.reasons))

    reasons = list(candidate.reasons)
    finding = _base(candidate, Route.NEEDS_REVIEW, reasons)

    if adjudication is not None:
        finding = finding.model_copy(
            update={
                "verdict": adjudication.verdict,
                "confidence": adjudication.confidence if adjudication.ok else None,
                "rationale": adjudication.rationale,
                "evidence": adjudication.evidence,
                "unanimous": adjudication.unanimous,
                "model_calls": adjudication.calls,
                "cache_hits": adjudication.cache_hits,
            }
        )

    if patch_outcome is not None:
        finding = finding.model_copy(
            update={
                "model_calls": finding.model_calls + patch_outcome.attempts,
                "cache_hits": finding.cache_hits + (1 if patch_outcome.cached else 0),
            }
        )

    def review(reason: str) -> Finding:
        return finding.model_copy(
            update={"route": Route.NEEDS_REVIEW, "reasons": _dedupe([*reasons, reason])}
        )

    # 2 — nothing usable came back.
    if adjudication is None or not adjudication.ok:
        detail = (
            adjudication.reason if adjudication is not None else "no adjudication attempted"
        )
        return review(f"could not obtain a usable verdict ({detail})")

    # 3 — the model declined, which it is allowed to do.
    if adjudication.verdict is Verdict.UNCERTAIN:
        return review("the model could not determine the impact and abstained")

    # 4 — not confident enough to act on.
    if adjudication.confidence < cfg.confidence_floor:
        return review(
            f"confidence {adjudication.confidence:.2f} is below the "
            f"{cfg.confidence_floor:.2f} floor"
        )

    # 5 — self-consistency failed.
    if not adjudication.unanimous:
        return review(f"{cfg.votes} votes did not agree")

    # 6 — the fuzziest input in the system cannot auto-approve anything.
    if candidate.inferred_provenance:
        return review("provenance was inferred, not recorded")

    # 7 — same words, different context: the weakest case for text-only judgment.
    if candidate.has_reparenting:
        return review(
            "a source span moved to a different section while its text stayed the same"
        )

    # 8 — same words, different order. The same species of weakness as
    # reparenting, and the more dangerous one: swapping "always" and "never"
    # reverses an instruction while leaving every word in place, so a model
    # asked "did this change?" can quite reasonably answer no.
    if candidate.has_reordering:
        return review(
            "a source span was rewritten with the same words in a different order, "
            "which can reverse its meaning"
        )

    # 9 — nothing to do.
    if adjudication.verdict is Verdict.UNAFFECTED:
        return finding.model_copy(
            update={
                "route": Route.OK,
                "reasons": _dedupe([*reasons, "the change does not affect this question"]),
            }
        )

    assert adjudication.verdict in BROKEN

    # 10 — a fix was needed but none survived validation.
    if patch_outcome is None or not patch_outcome.ok or patch_outcome.value is None:
        detail = patch_outcome.reason if patch_outcome is not None else "no fix attempted"
        attempted = patch_outcome.raw[-1] if patch_outcome and patch_outcome.raw else None
        return finding.model_copy(
            update={
                "route": Route.NEEDS_REVIEW,
                "reasons": _dedupe([*reasons, f"no valid fix could be produced ({detail})"]),
                "rejected_patch": attempted,
            }
        )

    # 11 — ready for a human to approve. Still not applied.
    return finding.model_copy(
        update={
            "route": Route.AUTO_FIX_PROPOSED,
            "reasons": _dedupe(
                [
                    *reasons,
                    f"verdict {adjudication.verdict.value} at confidence "
                    f"{adjudication.confidence:.2f}, and the proposed fix passed every "
                    "validator",
                ]
            ),
            "patch": patch_outcome.value,
        }
    )
