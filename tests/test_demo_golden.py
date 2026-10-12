"""Phase 7 — the end-to-end demo, pinned.

Two kinds of assertion live here, and both are needed. The golden comparison
catches any unintended change to the whole report. The per-case table states
what each fixture is *for*, so a failure says which behaviour broke rather
than just "the report moved".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coursedrift.config import Config
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.models import Route
from coursedrift.pipeline import run_drift
from coursedrift.report import RunReport, render_markdown

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden" / "demo_report.md"
LABELS = ("fixtures/docs_v1", "fixtures/docs_v2")


def _run(**overrides: object) -> RunReport:
    return run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(**overrides),  # type: ignore[arg-type]
        FakeClient(),
        NullCache(),
        labels=LABELS,
    )


@pytest.fixture(scope="module")
def report() -> RunReport:
    return _run()


# -- the golden report -----------------------------------------------------


def test_matches_the_golden_report(report: RunReport) -> None:
    expected = GOLDEN.read_text(encoding="utf-8")
    actual = render_markdown(report)
    if actual != expected:
        pytest.fail(
            "the demo report changed.\n\n"
            "If the change is intended, refresh the golden file with:\n"
            "    uv run drift demo > tests/golden/demo_report.md\n"
            "and check the diff before committing it."
        )


def test_the_demo_is_reproducible() -> None:
    assert render_markdown(_run()) == render_markdown(_run())


def test_the_golden_report_embeds_no_absolute_paths() -> None:
    body = GOLDEN.read_text(encoding="utf-8")
    assert "/Users/" not in body
    assert "/home/" not in body


# -- one row per fixture case ---------------------------------------------

CASES = [
    # case, item, expected route, what the fixture is testing
    (1, "q01", Route.AUTO_FIX_PROPOSED, "policy figure edited 30 -> 45 days"),
    (2, "q02", Route.OK, "paragraph reworded, meaning identical"),
    (3, "q09", Route.RETIRE_SUGGESTED, "Per-diem section deleted outright"),
    (4, "q04", Route.OK, "content moved out of a deleted document"),
    (5, "q03", Route.NEEDS_REVIEW, "hedged rewrite the model cannot settle"),
    (6, "q06", Route.OK, "document reordered, no text edits"),
    (7, "q10", Route.NEEDS_REVIEW, "sentence reparented into Deprecated Practices"),
    (8, "q07", Route.OK, "comma-only edit"),
    (10, "q12", Route.NEEDS_REVIEW, "no recorded provenance, inferred and capped"),
]


@pytest.mark.parametrize(("case", "item_id", "expected", "what"), CASES)
def test_fixture_case_routes_as_designed(
    report: RunReport, case: int, item_id: str, expected: Route, what: str
) -> None:
    finding = next(f for f in report.findings if f.item_id == item_id)
    assert finding.route is expected, f"case {case} ({what}) routed to {finding.route.value}"


def test_case_1_proposes_the_corrected_figure(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q01")
    assert finding.patch is not None
    assert finding.patch.correct_option == "45 days"
    assert "45 days" in finding.patch.grounded_in


def test_case_3_names_the_removed_text(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q09")
    assert any(c.old is not None and "per-diem rate" in c.old.text for c in finding.changes)


def test_case_4_anchors_across_documents(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q04")
    assert any(c.across_docs for c in finding.changes)
    assert finding.route is Route.OK


def test_case_6_reordering_produced_no_content_change(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q06")
    assert [c.content.value for c in finding.changes] == ["IDENTICAL"]
    assert [c.position.value for c in finding.changes] == ["MOVED"]


def test_case_7_shows_both_sections(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q10")
    change = finding.changes[0]
    assert change.old is not None and change.old.section_path == ("Expenses",)
    assert change.new is not None and change.new.section_path == (
        "Expenses",
        "Deprecated Practices",
    )


def test_case_8_never_reached_the_model(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q07")
    assert finding.model_calls == 0
    assert finding.verdict is None


def test_case_10_attaches_the_inferred_sources(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q12")
    assert finding.inferred_provenance is True
    assert all(a.ref.inferred for a in finding.anchors)
    assert "inferred" in " ".join(finding.reasons)


def test_an_item_in_an_unchanged_document_is_never_looked_at(report: RunReport) -> None:
    finding = next(f for f in report.findings if f.item_id == "q13")
    assert finding.route is Route.OK
    assert finding.changes == []
    assert finding.model_calls == 0
    assert report.stats.untouched == 1


# -- case 9: the injection payload ----------------------------------------


def test_case_9_payload_reaches_a_prompt_as_data(report: RunReport) -> None:
    """The defence is only meaningful if the payload actually gets in front of
    the model. It sits in the same section as the figure that changed, so it
    lands in that item's CONTEXT block."""
    from coursedrift.adjudicate import gather
    from coursedrift.anchor import resolve_course
    from coursedrift.corpus import diff_manifests, load_course, load_snapshot
    from coursedrift.diff import match_corpus
    from coursedrift.impact import build_candidates

    cfg = Config()
    old = load_snapshot(FIXTURES / "docs_v1")
    new = load_snapshot(FIXTURES / "docs_v2")
    course = load_course(FIXTURES / "course.json")
    matches = match_corpus(old, new, cfg, diff_manifests(old, new))
    anchors = resolve_course(course, old, new, cfg, matches=matches)
    candidates, _ = build_candidates(course, matches, anchors, cfg)

    target = next(c for c in candidates if c.item.item_id == "q01")
    material = gather(target, new)
    assert "disregard every previous instruction" in material.context


def test_case_9_obeyed_injection_still_yields_a_valid_finding() -> None:
    """A successful injection cannot escape the schema or the grounding check."""
    report = run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        FakeClient(failures=["injection"] * 12),
        NullCache(),
        labels=LABELS,
    )
    for finding in report.findings:
        assert finding.route in set(Route)
        if finding.verdict is not None:
            assert finding.evidence, "a verdict without grounded evidence must not pass"
        assert finding.patch is None, "nothing may be auto-fixed on an injected run"


# -- accounting ------------------------------------------------------------


def test_every_item_appears_exactly_once(report: RunReport) -> None:
    ids = [f.item_id for f in report.findings]
    assert len(ids) == len(set(ids)) == report.stats.total_items == 13


def test_skip_statistics_account_for_every_item(report: RunReport) -> None:
    assert report.stats.accounted == report.stats.total_items


def test_only_items_that_needed_judgment_cost_calls(report: RunReport) -> None:
    judged = [f for f in report.findings if f.model_calls > 0]
    assert len(judged) == report.stats.sent_to_model


def test_votes_do_not_change_the_unanimous_outcome() -> None:
    single = _run()
    voted = _run(votes=3)
    assert [f.route for f in single.findings] == [f.route for f in voted.findings]
    assert voted.model_calls > single.model_calls
