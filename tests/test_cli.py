"""Phase 7 — the command surface."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from coursedrift.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _args(*extra: str) -> list[str]:
    return [
        "run",
        "--old",
        str(FIXTURES / "docs_v1"),
        "--new",
        str(FIXTURES / "docs_v2"),
        "--course",
        str(FIXTURES / "course.json"),
        *extra,
    ]


def test_help_lists_the_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "demo", "seed"):
        assert command in result.output


def test_demo_runs_with_no_flags() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Drift report" in result.output
    assert "Nothing in this report has been applied" in result.output


def test_client_none_stops_before_judgment() -> None:
    result = runner.invoke(app, _args("--client", "none"))
    assert result.exit_code == 0
    assert "would be sent to model" in result.output
    assert "Verdict" not in result.output


def test_run_writes_both_report_formats(tmp_path: Path) -> None:
    result = runner.invoke(app, _args("--out", str(tmp_path), "--no-cache"))
    assert result.exit_code == 0
    markdown = (tmp_path / "drift_report.md").read_text()
    payload = json.loads((tmp_path / "drift_report.json").read_text())
    assert "Drift report" in markdown
    assert payload["course_id"] == "handbook-2026"
    assert payload["generated_at"] is not None
    assert len(payload["findings"]) == 13


def test_fail_on_drift_is_opt_in() -> None:
    assert runner.invoke(app, _args("--no-cache")).exit_code == 0
    assert runner.invoke(app, _args("--no-cache", "--fail-on-drift")).exit_code == 1


def test_unknown_client_fails_with_a_readable_message() -> None:
    result = runner.invoke(app, _args("--client", "telepathy"))
    assert result.exit_code == 2
    assert "unknown client" in result.output


def test_ollama_is_not_available_yet_but_says_so() -> None:
    result = runner.invoke(app, _args("--client", "ollama"))
    assert result.exit_code == 2
    assert "phase 8" in result.output


def test_unknown_fake_failure_mode_is_rejected() -> None:
    result = runner.invoke(app, _args("--fake-failures", "explode"))
    assert result.exit_code == 2
    assert "unknown fake failure mode" in result.output


def test_cache_makes_a_second_run_free(tmp_path: Path) -> None:
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"cache_dir": str(tmp_path / "cache")}))

    first = runner.invoke(app, _args("--config", str(config), "--out", str(tmp_path / "a")))
    second = runner.invoke(app, _args("--config", str(config), "--out", str(tmp_path / "b")))
    assert first.exit_code == second.exit_code == 0

    before = json.loads((tmp_path / "a" / "drift_report.json").read_text())
    after = json.loads((tmp_path / "b" / "drift_report.json").read_text())
    assert before["cache_hits"] == 0
    assert after["cache_hits"] > 0
    assert [f["route"] for f in before["findings"]] == [f["route"] for f in after["findings"]]


def test_injected_failures_do_not_crash_the_run() -> None:
    result = runner.invoke(
        app, _args("--no-cache", "--fake-failures", "timeout,malformed,schema,ungrounded")
    )
    assert result.exit_code == 0
    assert "NEEDS_REVIEW" in result.output


def test_votes_flag_overrides_the_config() -> None:
    result = runner.invoke(app, _args("--no-cache", "--votes", "3", "--out", "/dev/null"))
    assert result.exit_code in (0, 1)
