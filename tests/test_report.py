"""Phase 7 — report rendering and ordering."""

from __future__ import annotations

from coursedrift.anchor import AnchorStatus, ResolvedAnchor
from coursedrift.config import Config
from coursedrift.diff import ContentOp, PositionOp, SpanChange
from coursedrift.impact import SkipStats
from coursedrift.models import ProvenanceRef, Route, Span
from coursedrift.remediate import Patch
from coursedrift.report import RunReport, render_json, render_markdown, sort_findings
from coursedrift.triage import Finding


def _span(doc: str = "a.md", order: int = 0, section: tuple[str, ...] = ("A",)) -> Span:
    return Span(
        span_id=f"{doc}#p{order}.s0",
        doc_id=doc,
        section_path=section,
        order=order,
        text="Notice is 30 days.",
        norm_text="Notice is 30 days.",
        hash=f"{order:016d}",
    )


def _finding(
    item_id: str,
    route: Route,
    *,
    doc: str = "a.md",
    order: int = 0,
    patch: Patch | None = None,
) -> Finding:
    return Finding(
        item_id=item_id,
        level_id="L1",
        route=route,
        reasons=["because"],
        changes=[
            SpanChange(
                content=ContentOp.MODIFIED,
                position=PositionOp.STABLE,
                old=_span(doc, order),
                new=_span(doc, order),
                inline_diff="Notice is [-30-]{+45+} days.",
                similarity=0.95,
            )
        ],
        patch=patch,
    )


def _report(*findings: Finding) -> RunReport:
    return RunReport(
        course_id="c",
        course_title="Employee Handbook",
        old_root="v1",
        new_root="v2",
        client="fake",
        config=Config(),
        stats=SkipStats(total_items=len(findings), sent_to_model=len(findings)),
        findings=list(findings),
        documents={"modified": ["a.md"], "unchanged": [], "added": [], "deleted": []},
        span_counts={"MODIFIED/STABLE": 1},
        section_counts={"SAME": 1},
    )


# -- ordering --------------------------------------------------------------


def test_sorted_by_severity_then_document_then_position() -> None:
    findings = [
        _finding("q4", Route.OK),
        _finding("q3", Route.AUTO_FIX_PROPOSED),
        _finding("q2", Route.NEEDS_REVIEW, doc="b.md"),
        _finding("q1", Route.NEEDS_REVIEW, doc="a.md", order=5),
        _finding("q0", Route.RETIRE_SUGGESTED),
    ]
    assert [f.item_id for f in sort_findings(findings)] == ["q0", "q1", "q2", "q3", "q4"]


def test_ordering_is_total_so_the_golden_report_is_stable() -> None:
    a = _finding("qa", Route.OK, doc="a.md", order=1)
    b = _finding("qb", Route.OK, doc="a.md", order=1)
    assert [f.item_id for f in sort_findings([b, a])] == ["qa", "qb"]


# -- markdown --------------------------------------------------------------


def test_markdown_leads_with_the_posture() -> None:
    body = render_markdown(_report(_finding("q1", Route.NEEDS_REVIEW)))
    assert "Nothing in this report has been applied" in body


def test_markdown_includes_route_counts_and_level_rollup() -> None:
    body = render_markdown(_report(_finding("q1", Route.NEEDS_REVIEW)))
    assert "| NEEDS_REVIEW | 1 |" in body
    assert "## Levels" in body
    assert "| L1 | 1 | 100% | 1 | 0 | 0 |" in body


def test_markdown_states_why_each_finding_routed() -> None:
    body = render_markdown(_report(_finding("q1", Route.NEEDS_REVIEW)))
    assert "**Why:** because" in body


def test_markdown_marks_the_correct_option_in_a_patch() -> None:
    patch = Patch(
        stem="How many days?",
        options=["30 days", "45 days"],
        correct_index=1,
        grounded_in="Notice is 45 days.",
    )
    body = render_markdown(_report(_finding("q1", Route.AUTO_FIX_PROPOSED, patch=patch)))
    assert "  - 30 days" in body
    assert "  * 45 days" in body
    assert "not applied" in body


def test_markdown_shows_where_a_reparented_span_came_from() -> None:
    finding = Finding(
        item_id="q1",
        level_id="L1",
        route=Route.NEEDS_REVIEW,
        reasons=["moved"],
        changes=[
            SpanChange(
                content=ContentOp.IDENTICAL,
                position=PositionOp.REPARENTED,
                old=_span(section=("Mandatory",)),
                new=_span(section=("Deprecated",)),
            )
        ],
    )
    body = render_markdown(_report(finding))
    assert "was under: Mandatory" in body
    assert "Deprecated" in body


def test_markdown_keeps_a_rejected_patch_as_evidence() -> None:
    finding = _finding("q1", Route.NEEDS_REVIEW)
    finding = finding.model_copy(update={"rejected_patch": '{"stem": "no good"}'})
    body = render_markdown(_report(finding))
    assert "Rejected fix attempt" in body
    assert "no good" in body


def test_markdown_reports_an_ambiguous_anchor_with_its_candidates() -> None:
    ref = ProvenanceRef(
        doc_id="a.md", span_id="a.md#p0.s0", snapshot_text="t", snapshot_hash="h"
    )
    finding = _finding("q1", Route.NEEDS_REVIEW).model_copy(
        update={
            "anchors": [
                ResolvedAnchor(
                    ref=ref,
                    status=AnchorStatus.AMBIGUOUS,
                    candidates=(_span(order=1), _span(order=2)),
                )
            ]
        }
    )
    body = render_markdown(_report(finding))
    assert "AMBIGUOUS, 2 candidates" in body


def test_markdown_says_so_when_there_is_no_drift() -> None:
    body = render_markdown(_report(_finding("q1", Route.OK)))
    assert "No drift found" in body


def test_markdown_carries_no_timestamp() -> None:
    """The golden comparison depends on the body being time-independent."""
    report = _report(_finding("q1", Route.OK))
    report.generated_at = "2026-10-12T12:00:00+00:00"
    assert "2026" not in render_markdown(report)


def test_markdown_ends_with_exactly_one_newline() -> None:
    body = render_markdown(_report(_finding("q1", Route.OK)))
    assert body.endswith("\n")
    assert not body.endswith("\n\n")


# -- json ------------------------------------------------------------------


def test_json_round_trips() -> None:
    report = _report(_finding("q1", Route.NEEDS_REVIEW))
    report.generated_at = "2026-10-12T12:00:00+00:00"
    restored = RunReport.model_validate_json(render_json(report))
    assert restored.findings[0].item_id == "q1"
    assert restored.generated_at == "2026-10-12T12:00:00+00:00"
    assert restored.config == Config()


def test_json_keeps_the_timestamp_the_markdown_omits() -> None:
    report = _report(_finding("q1", Route.OK))
    report.generated_at = "2026-10-12T12:00:00+00:00"
    assert "2026-10-12" in render_json(report)


def test_staleness_is_the_share_of_items_not_plainly_fine() -> None:
    report = _report(
        _finding("q1", Route.OK),
        _finding("q2", Route.OK),
        _finding("q3", Route.NEEDS_REVIEW),
        _finding("q4", Route.RETIRE_SUGGESTED),
    )
    assert report.staleness("L1") == 50
    assert report.staleness("absent") == 0
