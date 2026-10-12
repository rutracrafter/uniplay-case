"""Command line entry point.

Filled in across phases; phase 0 establishes the command surface so the
package is runnable from the first commit.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="drift",
    help="Detect course content drift when source documents change.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run() -> None:
    """Run drift detection over a pair of document snapshots."""
    raise typer.Exit(code=_unimplemented("run"))


@app.command()
def demo() -> None:
    """Run drift detection over the bundled fixture corpus."""
    raise typer.Exit(code=_unimplemented("demo"))


@app.command()
def seed() -> None:
    """Generate a course with recorded provenance from a document directory."""
    raise typer.Exit(code=_unimplemented("seed"))


def _unimplemented(name: str) -> int:
    typer.echo(f"drift {name}: not implemented yet", err=True)
    return 1


if __name__ == "__main__":
    app()
