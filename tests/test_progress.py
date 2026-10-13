"""Progress output, and the guarantee that it stays off stdout.

The report goes to stdout and is routinely piped — the golden files are
captured with `drift demo > tests/golden/demo_report.md`. If progress ever
leaked onto stdout those files would be corrupted, so that separation is
asserted here rather than assumed.
"""

from __future__ import annotations

import io
from pathlib import Path

from typer.testing import CliRunner

from coursedrift.cli import app
from coursedrift.config import Config
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.pipeline import run_drift
from coursedrift.progress import NullProgress, TerminalProgress

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
runner = CliRunner()


class Pipe(io.StringIO):
    """A redirected stream: no overwriting, one line per event."""

    def isatty(self) -> bool:
        return False


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _run(progress: object = None) -> object:
    return run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        FakeClient(),
        NullCache(),
        progress=progress,  # type: ignore[arg-type]
    )


# -- the reporter itself ---------------------------------------------------


def test_null_progress_writes_nothing() -> None:
    quiet = NullProgress()
    quiet.stage("a")
    quiet.working("b")
    quiet.done("c")  # must not raise, must not print


def test_pipe_keeps_every_step_on_its_own_line() -> None:
    buf = Pipe()
    reporter = TerminalProgress(buf)
    reporter.stage("Judging")
    reporter.working("[1/2] q01 - asking the model...")
    reporter.done("[1/2] q01: INVALIDATED")
    assert buf.getvalue() == (
        "Judging\n  [1/2] q01 - asking the model...\n  [1/2] q01: INVALIDATED\n"
    )
    assert "\r" not in buf.getvalue()


def test_tty_overwrites_the_in_flight_line() -> None:
    buf = Tty()
    reporter = TerminalProgress(buf)
    reporter.working("asking the model...")
    reporter.done("q01: INVALIDATED")
    written = buf.getvalue()
    assert "\r" in written, "a terminal should replace the in-flight line"
    # What survives after the carriage returns is only the settled line.
    visible = [chunk.split("\r")[-1] for chunk in written.split("\n")]
    assert "  q01: INVALIDATED" in visible


def test_done_without_a_message_just_clears() -> None:
    buf = Tty()
    reporter = TerminalProgress(buf)
    reporter.working("thinking...")
    reporter.done()
    assert buf.getvalue().endswith("\r")


def test_long_lines_are_truncated_so_overwriting_works() -> None:
    buf = Tty()
    TerminalProgress(buf).working("x" * 500)
    assert max(len(part) for part in buf.getvalue().split("\n")) <= 80


# -- the library stays silent ---------------------------------------------


def test_run_drift_prints_nothing_by_default(capsys) -> None:  # type: ignore[no-untyped-def]
    """Library callers and tests must not have to opt out of output."""
    _run()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_progress_reports_each_stage_and_every_model_call() -> None:
    buf = Pipe()
    _run(TerminalProgress(buf))
    output = buf.getvalue()

    for stage in ("Reading documents", "Matching spans", "Anchoring", "Judging", "Done:"):
        assert stage in output

    # One "asking the model" per item that actually reached the model.
    assert output.count("asking the model") == 6
    assert "proposing a fix" in output, "the second call must be visible too"
    assert "[6/6]" in output, "progress must show position within the queue"


def test_each_item_outcome_is_reported() -> None:
    buf = Pipe()
    _run(TerminalProgress(buf))
    assert "q01: INVALIDATED -> AUTO_FIX_PROPOSED" in buf.getvalue()


def test_items_that_never_reach_the_model_are_not_announced() -> None:
    """Only the 6 queued items get per-item lines, not all 14."""
    buf = Pipe()
    _run(TerminalProgress(buf))
    assert buf.getvalue().count(" - asking the model") == 6


# -- the CLI ---------------------------------------------------------------


def test_cli_progress_does_not_contaminate_the_report(tmp_path: Path) -> None:
    """The guarantee the golden files depend on."""
    out = tmp_path / "report"
    result = runner.invoke(
        app,
        [
            "run",
            "--old",
            str(FIXTURES / "docs_v1"),
            "--new",
            str(FIXTURES / "docs_v2"),
            "--course",
            str(FIXTURES / "course.json"),
            "--no-cache",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    written = (out / "drift_report.md").read_text()
    assert written.startswith("# Drift report")
    for noise in ("Reading documents", "asking the model", "Done:"):
        assert noise not in written


def test_quiet_suppresses_progress(tmp_path: Path) -> None:
    args = [
        "run",
        "--old",
        str(FIXTURES / "docs_v1"),
        "--new",
        str(FIXTURES / "docs_v2"),
        "--course",
        str(FIXTURES / "course.json"),
        "--no-cache",
    ]
    loud = runner.invoke(app, args)
    quiet = runner.invoke(app, [*args, "--quiet"])
    assert loud.exit_code == quiet.exit_code == 0
    assert len(quiet.output) < len(loud.output)
    assert "asking the model" not in quiet.output


def test_seed_announces_each_question(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "seed",
            "--docs",
            str(FIXTURES / "docs_v1"),
            "--out",
            str(tmp_path / "course.json"),
            "--max-items",
            "3",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0
    assert "asking the model" in result.output
    assert "[3/3]" in result.output


# -- the point of the whole feature ---------------------------------------


def test_the_call_is_announced_before_it_blocks() -> None:
    """The anti-"looks hung" property, asserted rather than eyeballed.

    A real model call can take tens of seconds. What matters is not that the
    line is printed eventually, but that it is flushed *before* the call
    blocks — otherwise the operator stares at nothing for the whole wait and
    then sees the line appear at the end, which is no help at all.

    This client records what had already been written at the moment it was
    entered, which is exactly what a user would be looking at while waiting.
    """
    import json

    buf = Pipe()
    reply = json.dumps(
        {
            "verdict": "UNAFFECTED",
            "confidence": 0.9,
            "rationale": "nothing changed that matters",
            "evidence": ["Notice must be given within 45 days"],
        }
    )

    class Recording:
        name = "recording"

        def __init__(self) -> None:
            self.visible_at_first_call: str | None = None

        def complete(self, prompt: str, *, timeout: float) -> str:
            if self.visible_at_first_call is None:
                self.visible_at_first_call = buf.getvalue()
            return reply

    client = Recording()
    run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        client,  # type: ignore[arg-type]
        NullCache(),
        progress=TerminalProgress(buf),
    )

    already_shown = client.visible_at_first_call or ""
    assert "Judging" in already_shown
    assert "asking the model" in already_shown, (
        "the operator must see the call announced while it is still in flight"
    )
    assert "[1/6]" in already_shown


def test_vote_count_is_shown_when_self_consistency_is_on() -> None:
    """Three calls per item is worth knowing about while you wait."""
    buf = Pipe()
    run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(votes=3),
        FakeClient(),
        NullCache(),
        progress=TerminalProgress(buf),
    )
    assert "asking the model (3 votes)" in buf.getvalue()
