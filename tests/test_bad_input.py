"""Degenerate and mistaken inputs.

Two of these guard real defects. The rest pin behaviour that happens to be
correct today but is nowhere else asserted, so it could regress silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from coursedrift.cli import app
from coursedrift.config import Config
from coursedrift.corpus import (
    SuspectCorpus,
    UnreadableDocument,
    check_snapshot_pair,
    load_snapshot,
)
from coursedrift.diff import match_corpus
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.models import Route
from coursedrift.pipeline import run_drift

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _dir(tmp_path: Path, name: str, docs: dict[str, str] | None = None) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    for doc_id, body in (docs or {}).items():
        (root / doc_id).write_text(body, encoding="utf-8")
    return root


# -- a document that cannot be decoded ------------------------------------


def test_non_utf8_document_fails_with_the_filename(tmp_path: Path) -> None:
    """Was an uncaught UnicodeDecodeError, so a run died with no report.

    Not handled with errors="replace": a mangled character changes the span
    hash, so the document would report drift no author introduced.
    """
    root = _dir(tmp_path, "v1", {"ok.md": "# A\n\nFine sentence here.\n"})
    (root / "legacy.md").write_bytes(b"# Legacy\n\n\xff\xfe not utf-8\n")

    with pytest.raises(UnreadableDocument) as error:
        load_snapshot(root)
    assert "legacy.md" in str(error.value)
    assert "0xff" in str(error.value)
    assert "Re-save it as UTF-8" in str(error.value)


def test_cli_reports_an_undecodable_document_without_a_traceback(tmp_path: Path) -> None:
    root = _dir(tmp_path, "v1", {"ok.md": "# A\n\nFine.\n"})
    (root / "legacy.md").write_bytes(b"\xff\xfe\n")
    result = runner.invoke(
        app,
        [
            "run",
            "--old",
            str(root),
            "--new",
            str(root),
            "--course",
            str(FIXTURES / "course.json"),
            "--client",
            "none",
        ],
    )
    assert result.exit_code == 2
    assert "legacy.md is not valid UTF-8" in result.output
    assert "Traceback" not in result.output


# -- a snapshot pair that is a mistake ------------------------------------


def test_empty_new_snapshot_is_refused(tmp_path: Path) -> None:
    """The most destructive wrong answer available, so it is refused.

    An empty new snapshot orphans every item and proposes retiring the whole
    course, with no model calls and nothing to suggest anything went wrong.
    """
    old = load_snapshot(_dir(tmp_path, "v1", {"a.md": "# A\n\nContent.\n"}))
    new = load_snapshot(_dir(tmp_path, "v2"))
    with pytest.raises(SuspectCorpus, match="contains no Markdown documents"):
        check_snapshot_pair(old, new, allow_empty_new=False)


def test_the_refusal_names_the_escape_hatch(tmp_path: Path) -> None:
    old = load_snapshot(_dir(tmp_path, "v1", {"a.md": "# A\n\nContent.\n"}))
    new = load_snapshot(_dir(tmp_path, "v2"))
    with pytest.raises(SuspectCorpus, match="--allow-empty-new"):
        check_snapshot_pair(old, new, allow_empty_new=False)


def test_an_emptied_corpus_is_allowed_when_asked_for(tmp_path: Path) -> None:
    old = load_snapshot(_dir(tmp_path, "v1", {"a.md": "# A\n\nContent.\n"}))
    new = load_snapshot(_dir(tmp_path, "v2"))
    check_snapshot_pair(old, new, allow_empty_new=True)  # must not raise


def test_two_empty_snapshots_are_not_suspicious(tmp_path: Path) -> None:
    """Nothing to lose, so nothing to warn about."""
    old = load_snapshot(_dir(tmp_path, "v1"))
    new = load_snapshot(_dir(tmp_path, "v2"))
    check_snapshot_pair(old, new, allow_empty_new=False)


def test_cli_refuses_a_wrong_new_path(tmp_path: Path) -> None:
    empty = _dir(tmp_path, "nothing")
    result = runner.invoke(
        app,
        [
            "run",
            "--old",
            str(FIXTURES / "docs_v1"),
            "--new",
            str(empty),
            "--course",
            str(FIXTURES / "course.json"),
        ],
    )
    assert result.exit_code == 2
    assert "Check the --new path" in result.output
    assert "RETIRE_SUGGESTED" not in result.output


def test_mass_retirement_is_called_out_in_the_report(tmp_path: Path) -> None:
    """The softer case: a path that is real but wrong, taken under protest."""
    empty = _dir(tmp_path, "nothing")
    report = run_drift(
        FIXTURES / "docs_v1",
        empty,
        FIXTURES / "course.json",
        Config(allow_empty_new_snapshot=True),
        FakeClient(),
        NullCache(),
    )
    assert all(f.route is Route.RETIRE_SUGGESTED for f in report.findings)
    assert report.warnings
    assert "wrong --new path" in report.warnings[0]

    from coursedrift.report import render_markdown

    assert "## Check the input" in render_markdown(report)


def test_an_ordinary_run_raises_no_alarm() -> None:
    report = run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        FakeClient(),
        NullCache(),
    )
    assert report.warnings == []


def test_the_alarm_threshold_is_configurable() -> None:
    """One retirement in fourteen is normal; the same run at a 5% floor is not."""
    kwargs = dict(
        old_dir=FIXTURES / "docs_v1",
        new_dir=FIXTURES / "docs_v2",
        course_path=FIXTURES / "course.json",
        client=FakeClient(),
        cache=NullCache(),
    )
    assert run_drift(cfg=Config(), **kwargs).warnings == []  # type: ignore[arg-type]
    assert run_drift(cfg=Config(retire_alarm_fraction=0.05), **kwargs).warnings  # type: ignore[arg-type]


# -- degenerate but valid corpora -----------------------------------------


def test_both_snapshots_empty_produces_no_changes(tmp_path: Path) -> None:
    old = load_snapshot(_dir(tmp_path, "v1"))
    new = load_snapshot(_dir(tmp_path, "v2"))
    assert match_corpus(old, new, Config()).changes == []


def test_a_document_that_is_only_a_heading_has_no_spans(tmp_path: Path) -> None:
    snap = load_snapshot(_dir(tmp_path, "v1", {"h.md": "# Just A Heading\n"}))
    assert list(snap.all_spans()) == []


def test_an_empty_file_has_no_spans(tmp_path: Path) -> None:
    snap = load_snapshot(_dir(tmp_path, "v1", {"e.md": ""}))
    assert list(snap.all_spans()) == []


def test_a_whitespace_only_file_has_no_spans(tmp_path: Path) -> None:
    snap = load_snapshot(_dir(tmp_path, "v1", {"w.md": "   \n\n\t\n"}))
    assert list(snap.all_spans()) == []


def test_a_course_anchored_to_a_nonexistent_document_retires(tmp_path: Path) -> None:
    """Honest: the source cannot be found, so the item cannot be judged."""
    import json

    docs = _dir(tmp_path, "v1", {"real.md": "# Real\n\nActual content here.\n"})
    course = tmp_path / "ghost.json"
    course.write_text(
        json.dumps(
            {
                "course_id": "c",
                "title": "T",
                "items": [
                    {
                        "item_id": "q1",
                        "level_id": "L1",
                        "stem": "Q?",
                        "options": ["a", "b"],
                        "correct_index": 0,
                        "provenance": [
                            {
                                "doc_id": "ghost.md",
                                "span_id": "ghost.md#p0.s0",
                                "snapshot_text": "Text that exists nowhere.",
                                "snapshot_hash": "0" * 16,
                            }
                        ],
                    }
                ],
            }
        )
    )
    report = run_drift(docs, docs, course, Config(), FakeClient(), NullCache())
    assert report.findings[0].route is Route.RETIRE_SUGGESTED
    assert report.model_calls == 0
