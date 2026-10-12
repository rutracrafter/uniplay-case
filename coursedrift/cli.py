"""Command line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from coursedrift.config import Config
from coursedrift.models import Route
from coursedrift.pipeline import Prepared, prepare

app = typer.Typer(
    name="drift",
    help="Detect course content drift when source documents change.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    old: Annotated[Path, typer.Option(help="Directory holding the previous snapshot.")],
    new: Annotated[Path, typer.Option(help="Directory holding the current snapshot.")],
    course: Annotated[Path, typer.Option(help="Course JSON generated from the old snapshot.")],
    client: Annotated[
        str, typer.Option(help="Model client: fake, ollama, or none to stop before judgment.")
    ] = "fake",
    config: Annotated[
        Path | None, typer.Option(help="Config JSON overriding defaults.")
    ] = None,
) -> None:
    """Run drift detection over a pair of document snapshots."""
    cfg = Config.load(config)
    prepared = prepare(old, new, course, cfg)

    if client == "none":
        _print_deterministic(prepared)
        return

    typer.echo(f"drift run --client {client}: judgment stages arrive in phase 6", err=True)
    raise typer.Exit(code=1)


@app.command()
def demo() -> None:
    """Run drift detection over the bundled fixture corpus."""
    typer.echo("drift demo: fixtures arrive in phase 7", err=True)
    raise typer.Exit(code=1)


@app.command()
def seed() -> None:
    """Generate a course with recorded provenance from a document directory."""
    typer.echo("drift seed: arrives in phase 8", err=True)
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
