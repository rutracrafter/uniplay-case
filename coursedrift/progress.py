"""Telling the operator what is happening, especially during model calls.

Everything here writes to **stderr**, never stdout. The report goes to stdout
and is routinely piped — `drift demo > tests/golden/demo_report.md` has to stay
byte-exact — so progress and output must not share a stream.

The library takes a `Progress` and defaults to one that prints nothing, so
tests and library callers stay silent without having to opt out.
"""

from __future__ import annotations

import sys
from typing import Protocol, TextIO

_MAX_WIDTH = 78


class Progress(Protocol):
    def stage(self, message: str) -> None:
        """Announce a new phase of the run."""

    def working(self, message: str) -> None:
        """Report something in flight. Replaced by the next line on a terminal."""

    def done(self, message: str = "") -> None:
        """Settle whatever was in flight, optionally leaving a line behind."""


class NullProgress:
    """The default. Prints nothing."""

    def stage(self, message: str) -> None:
        return None

    def working(self, message: str) -> None:
        return None

    def done(self, message: str = "") -> None:
        return None


class TerminalProgress:
    """Writes to stderr, overwriting the in-flight line when attached to a tty.

    On a terminal a model call shows as a line that is replaced by its result,
    so the output stays short. Redirected to a file or a CI log there is
    nothing to overwrite, so each step is kept as its own line instead.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._pending = 0

    def _clear(self) -> None:
        if self.tty and self._pending:
            self.stream.write("\r" + " " * self._pending + "\r")
            self._pending = 0

    def stage(self, message: str) -> None:
        self._clear()
        self.stream.write(f"{message}\n")
        self.stream.flush()

    def working(self, message: str) -> None:
        text = f"  {message}"[:_MAX_WIDTH]
        if self.tty:
            self._clear()
            self.stream.write(text)
            self._pending = len(text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def done(self, message: str = "") -> None:
        self._clear()
        if message:
            self.stream.write(f"  {message}\n")
        self.stream.flush()
