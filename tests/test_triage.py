"""Phase 6 — the routing table, walked as a truth table.

Each test fires exactly one rule, and the precedence tests check that a
higher rule wins over a lower one that would also have fired.
"""

from __future__ import annotations

from coursedrift.adjudicate import Adjudication, AdjudicationResult, Verdict
from coursedrift.anchor import AnchorStatus, ResolvedAnchor
from coursedrift.config import Config
from coursedrift.diff import ContentOp, PositionOp, SpanChange
from coursedrift.impact import ImpactCandidate
from coursedrift.llm.base import CallOutcome, FailureKind
from coursedrift.models import CourseItem, ProvenanceRef, Route, Span
from coursedrift.remediate import Patch
from coursedrift.triage import route_candidate

CFG = Config()


def _span(section: tuple[str, ...] = ("A",)) -> Span:
    return Span(
        span_id="d.md#p0.s0",
        doc_id="d.md",
        section_path=section,
        para_index=0,
        sent_index=0,
        order=0,
        text="Notice is 30 days.",
        norm_text="Notice is 30 days.",
        hash="abcd1234abcd1234",
    )


def _candidate(
    *,
    preset: Route | None = None,
    inferred: bool = False,
    reparented: bool = False,
) -> ImpactCandidate:
    change = SpanChange(
        content=ContentOp.IDENTICAL if reparented else ContentOp.MODIFIED,
        position=PositionOp.REPARENTED if reparented else PositionOp.STABLE,
        old=_span(),
        new=_span(("B",)) if reparented else _span(),
        inline_diff="Notice is [-30-]{+45+} days.",
    )
    ref = ProvenanceRef(
        doc_id="d.md",
        span_id="d.md#p0.s0",
        snapshot_text="Notice is 30 days.",
        snapshot_hash="abcd1234abcd1234",
        inferred=inferred,
    )
    return ImpactCandidate(
        item=CourseItem(
            item_id="q1",
            level_id="L1",
            stem="How many days of notice?",
            options=["30", "45"],
            correct_index=0,
        ),
        anchors=[ResolvedAnchor(ref=ref, status=AnchorStatus.RESOLVED, new_span=_span())],
        changes=[change],
        needs_model=preset is None,
        preset_route=preset,
    )


def _adj(
    verdict: Verdict = Verdict.INVALIDATED,
    confidence: float = 0.9,
    *,
    unanimous: bool = True,
    failed: FailureKind | None = None,
) -> AdjudicationResult:
    if failed is not None:
        return AdjudicationResult(
            outcomes=[CallOutcome[Adjudication](ok=False, failure=failed, detail="broke")],
            verdict=None,
            unanimous=False,
        )
    value = Adjudication(
        verdict=verdict, confidence=confidence, rationale="because", evidence=["Notice"]
    )
    return AdjudicationResult(
        outcomes=[CallOutcome[Adjudication](ok=True, value=value, attempts=1)],
        verdict=verdict,
        confidence=confidence,
        unanimous=unanimous,
    )


def _patch(ok: bool = True) -> CallOutcome[Patch]:
    if not ok:
        return CallOutcome[Patch](
            ok=False,
            failure=FailureKind.REJECTED,
            detail="grounded_in is not present in the new source",
            attempts=2,
            raw=['{"stem": "bad"}'],
        )
    value = Patch(
        stem="How many days of notice?",
        options=["45", "30"],
        correct_index=0,
        grounded_in="Notice is 45 days.",
    )
    return CallOutcome[Patch](ok=True, value=value, attempts=1)


# -- one rule at a time ----------------------------------------------------


def test_rule_1_preset_wins_immediately() -> None:
    finding = route_candidate(_candidate(preset=Route.RETIRE_SUGGESTED), None, None, CFG)
    assert finding.route is Route.RETIRE_SUGGESTED


def test_rule_2_missing_adjudication_is_review() -> None:
    finding = route_candidate(_candidate(), None, None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "no adjudication attempted" in finding.reasons[-1]


def test_rule_2_failed_adjudication_names_the_failure() -> None:
    finding = route_candidate(_candidate(), _adj(failed=FailureKind.TIMEOUT), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "TIMEOUT" in finding.reasons[-1]
    assert finding.confidence is None


def test_rule_3_abstention_is_review() -> None:
    finding = route_candidate(_candidate(), _adj(Verdict.UNCERTAIN), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "abstained" in finding.reasons[-1]


def test_rule_4_low_confidence_is_review() -> None:
    finding = route_candidate(_candidate(), _adj(confidence=0.5), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "below the 0.75 floor" in finding.reasons[-1]


def test_rule_5_vote_disagreement_is_review() -> None:
    finding = route_candidate(_candidate(), _adj(unanimous=False), _patch(), Config(votes=3))
    assert finding.route is Route.NEEDS_REVIEW
    assert "votes did not agree" in finding.reasons[-1]


def test_rule_6_inferred_provenance_is_review() -> None:
    finding = route_candidate(_candidate(inferred=True), _adj(), _patch(), CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "provenance was inferred" in finding.reasons[-1]


def test_rule_7_reparenting_is_review() -> None:
    finding = route_candidate(
        _candidate(reparented=True), _adj(Verdict.WEAKENED), _patch(), CFG
    )
    assert finding.route is Route.NEEDS_REVIEW
    assert "moved to a different section" in finding.reasons[-1]


def test_rule_8_unaffected_is_ok() -> None:
    finding = route_candidate(_candidate(), _adj(Verdict.UNAFFECTED), None, CFG)
    assert finding.route is Route.OK
    assert finding.patch is None


def test_rule_9_rejected_patch_is_review_and_keeps_the_attempt() -> None:
    finding = route_candidate(_candidate(), _adj(), _patch(ok=False), CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "no valid fix could be produced" in finding.reasons[-1]
    assert finding.rejected_patch == '{"stem": "bad"}'
    assert finding.patch is None


def test_rule_9_missing_patch_is_review() -> None:
    finding = route_candidate(_candidate(), _adj(), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "no fix attempted" in finding.reasons[-1]


def test_rule_10_validated_patch_is_proposed_not_applied() -> None:
    finding = route_candidate(_candidate(), _adj(), _patch(), CFG)
    assert finding.route is Route.AUTO_FIX_PROPOSED
    assert finding.patch is not None
    assert finding.patch.correct_option == "45"
    assert "passed every validator" in finding.reasons[-1]


# -- precedence ------------------------------------------------------------


def test_preset_beats_everything_below_it() -> None:
    """Rule 1 wins even when a perfectly good patch exists."""
    finding = route_candidate(_candidate(preset=Route.NEEDS_REVIEW), _adj(), _patch(), CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert finding.patch is None


def test_low_confidence_unaffected_is_review_not_ok() -> None:
    """Rule 4 above rule 8: "unaffected, but unsure" is not "fine"."""
    finding = route_candidate(_candidate(), _adj(Verdict.UNAFFECTED, 0.3), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "below the 0.75 floor" in finding.reasons[-1]


def test_inferred_provenance_blocks_ok_as_well_as_auto_fix() -> None:
    """Rule 6 above rule 8: a silent pass built on a guess is still a guess."""
    finding = route_candidate(_candidate(inferred=True), _adj(Verdict.UNAFFECTED), None, CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "provenance was inferred" in finding.reasons[-1]


def test_inferred_provenance_can_never_reach_auto_fix() -> None:
    """The hard rule from the plan, asserted across every broken verdict."""
    for verdict in (Verdict.WEAKENED, Verdict.INVALIDATED):
        finding = route_candidate(_candidate(inferred=True), _adj(verdict, 1.0), _patch(), CFG)
        assert finding.route is not Route.AUTO_FIX_PROPOSED
        assert finding.route is Route.NEEDS_REVIEW


def test_abstention_beats_a_high_confidence_number() -> None:
    finding = route_candidate(_candidate(), _adj(Verdict.UNCERTAIN, 0.99), _patch(), CFG)
    assert finding.route is Route.NEEDS_REVIEW
    assert "abstained" in finding.reasons[-1]


# -- bookkeeping -----------------------------------------------------------


def test_call_counts_cover_adjudication_and_remediation() -> None:
    finding = route_candidate(_candidate(), _adj(), _patch(), CFG)
    assert finding.model_calls == 2


def test_severity_orders_retire_before_review_before_fix_before_ok() -> None:
    order = [Route.RETIRE_SUGGESTED, Route.NEEDS_REVIEW, Route.AUTO_FIX_PROPOSED, Route.OK]
    assert [r.severity for r in order] == sorted(r.severity for r in order)


def test_findings_carry_the_evidence_for_review() -> None:
    finding = route_candidate(_candidate(), _adj(), _patch(), CFG)
    assert finding.rationale == "because"
    assert finding.evidence == ["Notice"]
    assert finding.doc_ids == ["d.md"]
    assert finding.changes
    assert finding.anchors
