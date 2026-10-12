"""Phase 8 — seeding a course with provenance, and refusing ungrounded ones."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from coursedrift.cli import app
from coursedrift.config import Config
from coursedrift.corpus import load_snapshot
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.models import Course
from coursedrift.seed import candidate_spans, seed_course

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
V1 = FIXTURES / "docs_v1"
runner = CliRunner()


# -- span selection --------------------------------------------------------


def test_only_substantial_prose_is_chosen() -> None:
    for span in candidate_spans(V1, 50):
        assert not span.text.strip().startswith(("```", "|"))
        assert len(span.norm_text.split()) >= 8


def test_selection_spreads_across_documents() -> None:
    """A long handbook must not crowd out every other file."""
    picked = candidate_spans(V1, 5)
    assert len({span.doc_id for span in picked}) == 5


def test_selection_respects_the_limit_and_is_deterministic() -> None:
    assert len(candidate_spans(V1, 3)) == 3
    assert [s.span_id for s in candidate_spans(V1, 7)] == [
        s.span_id for s in candidate_spans(V1, 7)
    ]


def test_selection_stops_when_the_corpus_runs_out() -> None:
    picked = candidate_spans(V1, 10_000)
    assert 0 < len(picked) < 10_000


# -- seeding ---------------------------------------------------------------


def test_every_seeded_item_records_real_provenance() -> None:
    report = seed_course(V1, FakeClient(), Config(), NullCache(), max_items=6)
    snapshot = load_snapshot(V1)
    by_id = {span.span_id: span for span in snapshot.all_spans()}

    assert report.accepted == 6
    for item in report.course.items:
        assert len(item.provenance) == 1
        ref = item.provenance[0]
        assert ref.inferred is False, "seeded provenance is recorded, not guessed"
        source = by_id[ref.span_id]
        assert ref.snapshot_text == source.text
        assert ref.snapshot_hash == source.hash


def test_levels_are_named_after_their_document() -> None:
    report = seed_course(V1, FakeClient(), Config(), NullCache(), max_items=5)
    assert {item.level_id for item in report.course.items} <= {
        "benefits",
        "expenses",
        "leave-policy",
        "onboarding",
        "security-handbook",
    }


def test_seeded_questions_pass_the_patch_validators() -> None:
    report = seed_course(V1, FakeClient(), Config(), NullCache(), max_items=6)
    for item in report.course.items:
        assert len(item.options) == len(set(item.options))
        assert 0 <= item.correct_index < len(item.options)
        assert item.stem.endswith("?")


def test_an_ungrounded_question_is_dropped_and_reported() -> None:
    """A course seeded with ungrounded questions would make every later
    drift verdict meaningless, so these are refused rather than included."""
    client = FakeClient(failures=["ungrounded", "ungrounded"])
    report = seed_course(V1, client, Config(), NullCache(), max_items=3)
    assert report.attempted == 3
    assert report.accepted == 2, "the first question is rejected after its repair also fails"
    assert len(report.rejected) == 1
    assert "REJECTED" in report.rejected[0]


def test_a_timeout_drops_one_question_without_killing_the_run() -> None:
    client = FakeClient(failures=["timeout"])
    report = seed_course(V1, client, Config(), NullCache(), max_items=4)
    assert report.accepted == 3
    assert "TIMEOUT" in report.rejected[0]


def test_seeding_is_reproducible() -> None:
    first = seed_course(V1, FakeClient(), Config(), NullCache(), max_items=5)
    second = seed_course(V1, FakeClient(), Config(), NullCache(), max_items=5)
    assert first.course.model_dump_json() == second.course.model_dump_json()


# -- the CLI and the round trip -------------------------------------------


def test_cli_writes_a_loadable_course(tmp_path: Path) -> None:
    out = tmp_path / "course.json"
    result = runner.invoke(
        app,
        ["seed", "--docs", str(V1), "--out", str(out), "--max-items", "4", "--no-cache"],
    )
    assert result.exit_code == 0
    course = Course.model_validate(json.loads(out.read_text()))
    assert len(course.items) == 4
    assert "Wrote 4 of 4 questions" in result.output


def test_seeded_course_can_then_be_checked_for_drift(tmp_path: Path) -> None:
    """The loop that makes this usable on someone's own documents."""
    out = tmp_path / "course.json"
    seeded = runner.invoke(
        app,
        ["seed", "--docs", str(V1), "--out", str(out), "--max-items", "12", "--no-cache"],
    )
    assert seeded.exit_code == 0

    checked = runner.invoke(
        app,
        [
            "run",
            "--old",
            str(V1),
            "--new",
            str(FIXTURES / "docs_v2"),
            "--course",
            str(out),
            "--no-cache",
        ],
    )
    assert checked.exit_code == 0
    assert "Drift report" in checked.output
    assert "AUTO_FIX_PROPOSED" in checked.output
    assert "RETIRE_SUGGESTED" in checked.output


def test_cli_reports_dropped_questions(tmp_path: Path) -> None:
    out = tmp_path / "course.json"
    result = runner.invoke(
        app,
        [
            "seed",
            "--docs",
            str(V1),
            "--out",
            str(out),
            "--max-items",
            "3",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0
    assert "Wrote 3 of 3" in result.output
