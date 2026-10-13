"""Command line entry point."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from coursedrift.config import Config
from coursedrift.corpus import SuspectCorpus, UnreadableDocument
from coursedrift.llm.cache import FileCache, NullCache
from coursedrift.models import Route
from coursedrift.pipeline import Prepared, build_client, prepare, run_drift
from coursedrift.progress import NullProgress, Progress, TerminalProgress
from coursedrift.report import render_json, render_markdown
from coursedrift.seed import seed_course

app = typer.Typer(
    name="drift",
    help="Detect course content drift when source documents change.",
    no_args_is_help=True,
    add_completion=False,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _progress(quiet: bool) -> Progress:
    """Progress goes to stderr, so piping the report to a file stays clean."""
    return NullProgress() if quiet else TerminalProgress()


@contextmanager
def _readable_errors() -> Iterator[None]:
    """Turn bad-input failures into a message and exit 2, not a traceback.

    A corpus that cannot be decoded, or a snapshot pair that looks like a
    wrong path, is the operator's problem to fix. Printing a stack trace
    tells them less than one sentence does.
    """
    try:
        yield
    except (UnreadableDocument, SuspectCorpus, ValidationError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    except (OSError, NotADirectoryError) as error:
        typer.secho(f"Could not read the inputs: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error


@app.command()
def run(
    old: Annotated[Path, typer.Option(help="Directory holding the previous snapshot.")],
    new: Annotated[Path, typer.Option(help="Directory holding the current snapshot.")],
    course: Annotated[Path, typer.Option(help="Course JSON generated from the old snapshot.")],
    client: Annotated[
        str,
        typer.Option(help="Model client: fake, ollama, or none to stop before judgment."),
    ] = "fake",
    model: Annotated[
        str, typer.Option(help="Model name, for the ollama client.")
    ] = "qwen2.5:3b",
    votes: Annotated[int, typer.Option(help="Self-consistency votes per item.")] = 0,
    config: Annotated[
        Path | None, typer.Option(help="Config JSON overriding defaults.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Directory to write the report into.")
    ] = None,
    fake_failures: Annotated[
        str, typer.Option(help="Comma-separated fake failure modes to inject.")
    ] = "",
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the response cache.")
    ] = False,
    allow_empty_new: Annotated[
        bool,
        typer.Option(
            "--allow-empty-new",
            help="Accept a new snapshot with no documents. Retires the whole course.",
        ),
    ] = False,
    fail_on_drift: Annotated[
        bool,
        typer.Option("--fail-on-drift", help="Exit non-zero when drift is found, for CI."),
    ] = False,
    timeout: Annotated[
        float, typer.Option(help="Seconds to allow per model call. 0 keeps the config value.")
    ] = 0.0,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress progress output on stderr.")
    ] = False,
) -> None:
    """Run drift detection over a pair of document snapshots."""
    cfg = Config.load(config)
    if votes:
        cfg = cfg.model_copy(update={"votes": votes})
    if timeout:
        cfg = cfg.model_copy(update={"timeout_s": timeout})
    if allow_empty_new:
        cfg = cfg.model_copy(update={"allow_empty_new_snapshot": True})

    status = _progress(quiet)

    if client == "none":
        with _readable_errors():
            _print_deterministic(prepare(old, new, course, cfg, progress=status))
        return

    failures = [mode.strip() for mode in fake_failures.split(",") if mode.strip()]
    try:
        implementation = build_client(client, model, failures, cfg)
    except (ValueError, RuntimeError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    cache = NullCache() if no_cache else FileCache(cfg.cache_dir)
    with _readable_errors():
        report = run_drift(old, new, course, cfg, implementation, cache, progress=status)
    report.generated_at = datetime.now(UTC).isoformat(timespec="seconds")

    markdown = render_markdown(report)
    typer.echo(markdown, nl=False)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "drift_report.md").write_text(markdown, encoding="utf-8")
        (out / "drift_report.json").write_text(render_json(report), encoding="utf-8")
        typer.secho(f"Wrote {out}/drift_report.md and drift_report.json", fg="green", err=True)

    if fail_on_drift and any(f.route is not Route.OK for f in report.findings):
        raise typer.Exit(code=1)


@app.command()
def demo(
    client: Annotated[str, typer.Option(help="Model client: fake or ollama.")] = "fake",
    model: Annotated[
        str, typer.Option(help="Model name, for the ollama client.")
    ] = "qwen2.5:3b",
    fake_failures: Annotated[
        str, typer.Option(help="Comma-separated fake failure modes to inject.")
    ] = "",
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the machine-readable report instead.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress progress output on stderr.")
    ] = False,
) -> None:
    """Run drift detection over the bundled fixture corpus.

    No flags, no network, no model. The fixture pair exercises every branch
    of the pipeline, so this is the fastest way to see what the system does.
    """
    cfg = Config()
    failures = [mode.strip() for mode in fake_failures.split(",") if mode.strip()]
    implementation = build_client(client, model, failures, cfg)
    report = run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        cfg,
        implementation,
        # Never cached: the demo must show real work every time, and the
        # golden test depends on it being independent of local state.
        NullCache(),
        labels=("fixtures/docs_v1", "fixtures/docs_v2"),
        progress=_progress(quiet),
    )
    typer.echo(render_json(report) if as_json else render_markdown(report), nl=False)


@app.command()
def seed(
    docs: Annotated[Path, typer.Option(help="Directory of Markdown to generate from.")],
    out: Annotated[Path, typer.Option(help="Where to write the course JSON.")],
    client: Annotated[str, typer.Option(help="Model client: fake or ollama.")] = "fake",
    model: Annotated[
        str, typer.Option(help="Model name, for the ollama client.")
    ] = "qwen2.5:3b",
    max_items: Annotated[int, typer.Option(help="How many questions to write.")] = 12,
    title: Annotated[str, typer.Option(help="Course title.")] = "Seeded Course",
    course_id: Annotated[str, typer.Option(help="Course identifier.")] = "seeded-course",
    config: Annotated[
        Path | None, typer.Option(help="Config JSON overriding defaults.")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the response cache.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress progress output on stderr.")
    ] = False,
) -> None:
    """Generate a course with recorded provenance from a document directory.

    Use this to point the drift detector at your own material: seed a course
    from one snapshot, change the documents, then run `drift run` against the
    pair.
    """
    cfg = Config.load(config)
    try:
        implementation = build_client(client, model, [], cfg)
    except (ValueError, RuntimeError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    cache = NullCache() if no_cache else FileCache(cfg.cache_dir)
    report = seed_course(
        docs,
        implementation,
        cfg,
        cache,
        course_id=course_id,
        title=title,
        max_items=max_items,
        progress=_progress(quiet),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.course.model_dump_json(indent=2) + "\n", encoding="utf-8")

    typer.secho(
        f"Wrote {report.accepted} of {report.attempted} questions to {out}",
        fg=typer.colors.GREEN,
        err=True,
    )
    for rejection in report.rejected:
        typer.secho(f"  dropped {rejection}", fg=typer.colors.YELLOW, err=True)
    if not report.accepted:
        raise typer.Exit(code=1)


def _print_deterministic(prepared: Prepared) -> None:
    """Report what the deterministic stages decided, and what they deferred."""
    stats = prepared.stats
    manifest = prepared.manifest

    typer.echo("Documents")
    typer.echo(
        f"  unchanged {len(manifest.unchanged)}  modified {len(manifest.modified)}"
        f"  added {len(manifest.added)}  deleted {len(manifest.deleted)}"
    )

    typer.echo("\nSource changes")
    for label, count in sorted(prepared.matches.counts().items()):
        typer.echo(f"  {label:<24} {count}")

    typer.echo("\nSection changes")
    for label, count in sorted(prepared.matches.sections.counts().items()):
        typer.echo(f"  {label:<24} {count}")

    typer.echo("\nCourse items")
    typer.echo(f"  total                    {stats.total_items}")
    typer.echo(f"  untouched                {stats.untouched}")
    typer.echo(f"  skipped, no impact       {stats.skipped_no_impact}")
    typer.echo(f"  decided without a model  {stats.decided_deterministically}")
    typer.echo(f"  would be sent to model   {stats.sent_to_model}")
    assert stats.accounted == stats.total_items, "skip statistics must account for every item"

    presets: dict[Route, int] = {}
    for candidate in prepared.candidates:
        if candidate.preset_route is not None:
            presets[candidate.preset_route] = presets.get(candidate.preset_route, 0) + 1
    if presets:
        typer.echo("\nDeterministic routes")
        for route in sorted(presets, key=lambda r: r.severity):
            typer.echo(f"  {route.value:<24} {presets[route]}")


if __name__ == "__main__":
    app()
