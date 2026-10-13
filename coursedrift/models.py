"""Domain types shared across the pipeline.

These are the only shapes that cross stage boundaries. Wire types for model
responses live in `adjudicate.py` and `remediate.py`; everything here is
produced and consumed by deterministic code.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Route(StrEnum):
    """Where an item ends up. The only four outcomes the system produces.

    Nothing is ever applied automatically: AUTO_FIX_PROPOSED means a patch
    passed every validator and is ready for a human to approve, not that it
    has been written anywhere.
    """

    RETIRE_SUGGESTED = "RETIRE_SUGGESTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    AUTO_FIX_PROPOSED = "AUTO_FIX_PROPOSED"
    OK = "OK"

    @property
    def severity(self) -> int:
        """Report ordering: lost content first, then work a human must do."""
        return _SEVERITY[self]


# ---------------------------------------------------------------------------
# Source documents
# ---------------------------------------------------------------------------


class Span(BaseModel):
    """One addressable unit of source material: normally a sentence.

    Two fields look like they violate the core invariant and do not, which is
    worth stating since the question comes up:

    `hash` is the identity Phase A matches on, derived from `norm_text` and
    never from position. `span_id` is a *name* for this span within its own
    snapshot, not an identity claim across snapshots — needed because content
    is deliberately not unique (duplicate boilerplate shares a hash), so the
    one-to-one constraint in Phase B, the change table, and deterministic
    sorting all need per-span names. The two places it used to reach across
    the snapshot boundary are gone: it is no longer an anchor tiebreak, and
    `_from_match` verifies the recorded hash before trusting it.

    `order` is the document-wide sentence ordinal, used only for what the
    invariant's second clause permits — classifying movement in Phase C — and
    for deterministic ordering of output.

    Nothing else positional is stored. `para_index` and `sent_index` were
    kept here until they turned out to be written and never read; they were
    decomposed copies of what `span_id` already spells out.
    """

    model_config = ConfigDict(frozen=True)

    span_id: str
    doc_id: str
    section_path: tuple[str, ...] = ()
    order: int
    text: str
    norm_text: str
    hash: str

    def __str__(self) -> str:
        return self.span_id


class Section(BaseModel):
    """A heading and the spans directly beneath it.

    `span_hashes` is the Jaccard input used to match sections across
    snapshots, which is why it is a set of content hashes rather than a list
    of ids: it is order-free by construction.
    """

    model_config = ConfigDict(frozen=True)

    section_id: str
    doc_id: str
    path: tuple[str, ...]
    heading: str
    span_hashes: frozenset[str]
    order: int


class Document(BaseModel):
    """One source file, segmented."""

    doc_id: str
    content: str
    hash: str
    spans: list[Span] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)


class Snapshot(BaseModel):
    """A whole corpus at one point in time."""

    root: Path
    docs: dict[str, Document] = Field(default_factory=dict)

    def all_spans(self) -> Iterator[Span]:
        """Every span in the corpus, in stable document then order sequence."""
        for doc_id in sorted(self.docs):
            yield from self.docs[doc_id].spans

    def all_sections(self) -> Iterator[Section]:
        for doc_id in sorted(self.docs):
            yield from self.docs[doc_id].sections

    def by_hash(self) -> dict[str, list[Span]]:
        """Corpus-wide content index — the Phase A identity map.

        Corpus-wide rather than per-document on purpose: content moved from
        one file into another must match, not read as a delete plus an add.
        """
        index: dict[str, list[Span]] = {}
        for span in self.all_spans():
            index.setdefault(span.hash, []).append(span)
        return index

    def span_by_id(self, span_id: str) -> Span | None:
        doc_id = span_id.split("#", 1)[0]
        doc = self.docs.get(doc_id)
        if doc is None:
            return None
        for span in doc.spans:
            if span.span_id == span_id:
                return span
        return None


class ManifestDiff(BaseModel):
    """Document-level classification between two snapshots."""

    unchanged: set[str] = Field(default_factory=set)
    modified: set[str] = Field(default_factory=set)
    added: set[str] = Field(default_factory=set)
    deleted: set[str] = Field(default_factory=set)

    @property
    def touched(self) -> set[str]:
        """Documents that can contribute changes, so must enter the matcher."""
        return self.modified | self.added | self.deleted


# ---------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------


class ProvenanceRef(BaseModel):
    """Where a course item came from, as recorded when it was generated.

    `snapshot_text` is kept because byte offsets are worthless after an edit:
    relocation works by content, falling back to fuzzy text match. `inferred`
    marks a ref that phase 3 guessed rather than read, and triage treats that
    as a hard cap on how far the item can be auto-approved.
    """

    doc_id: str
    span_id: str
    snapshot_text: str
    snapshot_hash: str
    inferred: bool = False
    infer_score: float | None = None


class CourseItem(BaseModel):
    """One question in a course."""

    item_id: str
    level_id: str
    stem: str
    options: list[str]
    correct_index: int
    provenance: list[ProvenanceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _correct_index_in_range(self) -> CourseItem:
        """Reject a question whose keyed answer does not exist.

        On the model rather than in the loader, so it holds however a
        CourseItem is built — from a file, from `drift seed`, or in a test.

        Without it, `correct_index: 7` against two options validates happily
        and then raises IndexError the first time `correct_answer` is read,
        which is deep inside adjudication or provenance inference. That is an
        uncaught crash mid-run, and the rest of the system is built so that
        failures become routed findings instead. It was also backwards:
        `Patch`, which arrives from a language model, is checked for exactly
        this, while a course file was not checked at all.
        """
        if not self.options:
            raise ValueError(f"{self.item_id}: a question needs at least one option")
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError(
                f"{self.item_id}: correct_index {self.correct_index} is outside "
                f"the {len(self.options)} option(s) supplied"
            )
        return self

    @property
    def correct_answer(self) -> str:
        return self.options[self.correct_index]


class Course(BaseModel):
    """A generated course, as Masterbrain would hand it over."""

    course_id: str
    title: str
    items: list[CourseItem] = Field(default_factory=list)

    def levels(self) -> list[str]:
        seen: list[str] = []
        for item in self.items:
            if item.level_id not in seen:
                seen.append(item.level_id)
        return seen


_SEVERITY: dict[Route, int] = {
    Route.RETIRE_SUGGESTED: 0,
    Route.NEEDS_REVIEW: 1,
    Route.AUTO_FIX_PROPOSED: 2,
    Route.OK: 3,
}
