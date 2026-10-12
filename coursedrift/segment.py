"""Normalization and segmentation: Markdown in, hashed spans out.

The normalization contract below is load-bearing. Read it before changing
`normalize`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from coursedrift.models import Section, Span

# ---------------------------------------------------------------------------
# Normalization contract
# ---------------------------------------------------------------------------
#
# `normalize` applies exactly three things, in order:
#
#   1. Unicode NFKC
#   2. curly quotes, dashes and exotic spaces folded to ASCII equivalents
#   3. whitespace runs collapsed to one space, ends stripped
#
# It deliberately does NOT lowercase and does NOT strip punctuation. That
# split is what keeps two mechanisms from collapsing into one:
#
#   * whitespace and unicode-variant edits hash IDENTICAL, so Phase A of the
#     matcher absorbs them for free and they never become findings
#   * casing and punctuation edits do NOT hash identical, so they fall through
#     to Phase B and trip the COSMETIC shortcut
#
# Fold punctuation in here and COSMETIC becomes dead code it can never reach;
# leave whitespace out and trivial reflows flood the report. Both halves are
# asserted in tests/test_segment.py.

_FOLD = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",
    "…": "...",
}
_FOLD_TABLE = str.maketrans(_FOLD)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Apply the normalization contract. Idempotent."""
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_FOLD_TABLE)
    return _WS.sub(" ", out).strip()


def span_hash(norm_text: str) -> str:
    """Content identity. Truncated sha256 — collision risk is irrelevant here."""
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_ABBREV = frozenset(
    {
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "no.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "fig.",
        "approx.",
        "cf.",
        "al.",
        "inc.",
        "ltd.",
        "jr.",
        "sr.",
        "st.",
    }
)
_INITIAL = re.compile(r"^[A-Z]\.$")
_BOUNDARY = re.compile(r"([.!?][\"')\]]*)(\s+)(?=[A-Z0-9\"'(\[])")


def _ends_with_abbrev(chunk: str) -> bool:
    tokens = chunk.split()
    if not tokens:
        return False
    last = tokens[-1]
    return last.lower() in _ABBREV or bool(_INITIAL.match(last))


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation, guarding a small abbreviation list.

    Imperfect splitting is acceptable and bounded: a mis-split is stable
    across both snapshots, so it hashes identically on each side and costs
    nothing. This is why no sentence-boundary model is needed.
    """
    out: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start : match.end(1)]
        if _ends_with_abbrev(candidate):
            continue
        stripped = candidate.strip()
        if stripped:
            out.append(stripped)
        start = match.end(2)
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# ---------------------------------------------------------------------------
# Markdown segmentation
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|")


class _Builder:
    """Accumulates spans and sections for one document."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.spans: list[Span] = []
        self.sections: list[Section] = []
        self._order = 0
        self._para = 0
        self._stack: list[tuple[int, str]] = []
        self._section_order = 0
        self._current: dict[str, object] | None = None
        self._open_root()

    # -- sections --------------------------------------------------------

    def _open_root(self) -> None:
        self._current = {
            "path": (),
            "heading": "",
            "level": 0,
            "hashes": set(),
            "order": self._section_order,
        }
        self._section_order += 1

    def open_section(self, level: int, heading: str) -> None:
        self._close_section()
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, heading))
        self._current = {
            "path": tuple(h for _, h in self._stack),
            "heading": heading,
            "level": level,
            "hashes": set(),
            "order": self._section_order,
        }
        self._section_order += 1

    def _close_section(self) -> None:
        cur = self._current
        if cur is None:
            return
        hashes: set[str] = cur["hashes"]  # type: ignore[assignment]
        path: tuple[str, ...] = cur["path"]  # type: ignore[assignment]
        # A root section with no spans never existed as far as the diff cares.
        if not hashes and not path:
            self._current = None
            return
        self.sections.append(
            Section(
                section_id=f"{self.doc_id}#sec{cur['order']}",
                doc_id=self.doc_id,
                path=path,
                heading=str(cur["heading"]),
                level=int(cur["level"]),  # type: ignore[call-overload]
                span_hashes=frozenset(hashes),
                order=int(cur["order"]),  # type: ignore[call-overload]
            )
        )
        self._current = None

    @property
    def section_path(self) -> tuple[str, ...]:
        if self._current is None:
            return ()
        return self._current["path"]  # type: ignore[return-value]

    # -- spans -----------------------------------------------------------

    def add_block(self, text: str, *, split: bool) -> None:
        """Add one block as either sentence spans or a single verbatim span."""
        pieces = split_sentences(text) if split else [text]
        pieces = [p for p in pieces if p.strip()]
        if not pieces:
            return
        for sent_index, piece in enumerate(pieces):
            norm = normalize(piece)
            if not norm:
                continue
            digest = span_hash(norm)
            self.spans.append(
                Span(
                    span_id=f"{self.doc_id}#p{self._para}.s{sent_index}",
                    doc_id=self.doc_id,
                    section_path=self.section_path,
                    para_index=self._para,
                    sent_index=sent_index,
                    order=self._order,
                    text=piece,
                    norm_text=norm,
                    hash=digest,
                )
            )
            self._order += 1
            if self._current is not None:
                hashes: set[str] = self._current["hashes"]  # type: ignore[assignment]
                hashes.add(digest)
        self._para += 1

    def finish(self) -> None:
        self._close_section()


def segment_document(doc_id: str, content: str) -> tuple[list[Span], list[Section]]:
    """Segment Markdown into sentence-level spans grouped under sections.

    Rules, per PLAN.md phase 1:
      * ATX headings open sections and never become spans
      * paragraphs split on blank lines, then into sentences
      * a fenced code block is one span, verbatim, never sentence-split
      * each list item is its own span
      * each table row is its own span
    """
    builder = _Builder(doc_id)
    lines = content.splitlines()
    para: list[str] = []
    fence: list[str] | None = None
    fence_marker = ""

    def flush_para() -> None:
        if para:
            builder.add_block(" ".join(para).strip(), split=True)
            para.clear()

    for line in lines:
        # Inside a fenced block everything is verbatim until the closing fence.
        if fence is not None:
            fence.append(line)
            if _FENCE.match(line) and line.strip().startswith(fence_marker):
                builder.add_block("\n".join(fence), split=False)
                fence = None
            continue

        fence_open = _FENCE.match(line)
        if fence_open:
            flush_para()
            fence_marker = fence_open.group(1)
            fence = [line]
            continue

        heading = _HEADING.match(line)
        if heading:
            flush_para()
            builder.open_section(len(heading.group(1)), heading.group(2).strip())
            continue

        if not line.strip():
            flush_para()
            continue

        if _LIST_ITEM.match(line) or _TABLE_ROW.match(line):
            flush_para()
            builder.add_block(line.strip(), split=False)
            continue

        para.append(line.strip())

    if fence is not None:  # unterminated fence: keep what we have
        builder.add_block("\n".join(fence), split=False)
    flush_para()
    builder.finish()
    return builder.spans, builder.sections
