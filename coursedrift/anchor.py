"""Relocating a course item's provenance in the new corpus.

Byte offsets are worthless after an edit, so relocation is content-first:
exact hash, then fuzzy text, then an honest admission of failure. A silent
wrong anchor is worse than an escalation, so ambiguity is surfaced rather
than guessed away.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from coursedrift.config import Config
from coursedrift.diff import ContentOp, MatchResult, match_key
from coursedrift.models import Course, CourseItem, ProvenanceRef, Snapshot, Span
from coursedrift.segment import normalize, span_hash


class AnchorStatus(StrEnum):
    RESOLVED = "RESOLVED"
    MOVED = "MOVED"
    ORPHANED = "ORPHANED"
    AMBIGUOUS = "AMBIGUOUS"


class ResolvedAnchor(BaseModel):
    """Where a provenance ref landed in the new corpus, and how sure we are."""

    model_config = ConfigDict(frozen=True)

    ref: ProvenanceRef
    status: AnchorStatus
    new_span: Span | None = None
    candidates: tuple[Span, ...] = ()
    similarity: float | None = None

    @property
    def inferred(self) -> bool:
        return self.ref.inferred

    @property
    def located(self) -> bool:
        return self.status in (AnchorStatus.RESOLVED, AnchorStatus.MOVED)


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------


def _narrow(
    candidates: list[Span],
    ref: ProvenanceRef,
    old_span: Span | None,
) -> list[Span]:
    """Apply tiebreaks in order, keeping each one only if it narrows usefully.

    Document, then section path. Each filter is discarded if it would
    eliminate every candidate, so a tiebreak can never turn a findable anchor
    into an orphan.

    Note what is deliberately *not* a tiebreak: the original `span_id`. It
    encodes paragraph position, so preferring the candidate at the same index
    would be using position to establish identity — the one thing the core
    invariant forbids. It also only looks principled: insert a paragraph
    above and the indices shift, making the "match" coincidental. Genuinely
    indistinguishable duplicates escalate as AMBIGUOUS instead.
    """
    tiebreaks: list[Callable[[Span], bool]] = [
        lambda s: s.doc_id == ref.doc_id,
        lambda s: old_span is not None and s.section_path == old_span.section_path,
    ]
    for predicate in tiebreaks:
        if len(candidates) <= 1:
            break
        narrowed = [s for s in candidates if predicate(s)]
        if narrowed:
            candidates = narrowed
    return candidates


def _from_match(
    ref: ProvenanceRef, matches: MatchResult, old_span: Span | None
) -> ResolvedAnchor | None:
    """Translate phase 2's verdict for this span into an anchor, if it has one.

    Returns None — meaning "fall through to the hash lookup" — in two cases.

    The matcher never saw the span, which happens for documents the manifest
    called unchanged, since those are excluded from matching.

    Or the pairing is IDENTICAL, where the matcher's choice is arbitrary and
    must not be treated as knowledge. Inside a duplicate group it zips equal
    hashes in document order because it has to produce *some* change table,
    but "which of these two identical sentences was my source" genuinely has
    no answer, and letting the matcher answer it would silence the ambiguity
    escalation. The hash lookup below sees both candidates and says so.

    What the matcher is trusted for is where it did real work: a similarity
    pairing survived shingle blocking and a one-to-one constraint, and a
    REMOVED verdict means nothing anywhere in the corpus matched.
    """
    change = matches.by_old_span().get(ref.span_id)
    if change is None:
        return None
    if change.content is ContentOp.REMOVED or change.new is None:
        return ResolvedAnchor(ref=ref, status=AnchorStatus.ORPHANED)
    if change.content is ContentOp.IDENTICAL:
        return None
    return ResolvedAnchor(
        ref=ref,
        status=_status_for(change.new, ref, old_span),
        new_span=change.new,
        similarity=change.similarity,
    )


def _status_for(span: Span, ref: ProvenanceRef, old_span: Span | None) -> AnchorStatus:
    if span.doc_id != ref.doc_id:
        return AnchorStatus.MOVED
    if old_span is not None and span.section_path != old_span.section_path:
        return AnchorStatus.MOVED
    return AnchorStatus.RESOLVED


def resolve_anchor(
    ref: ProvenanceRef,
    new: Snapshot,
    cfg: Config,
    old_span: Span | None = None,
    matches: MatchResult | None = None,
) -> ResolvedAnchor:
    """Locate one provenance ref in the new corpus.

    Corpus-wide at every step: content that moved into another document must
    still be found, which is what keeps a reorganization from orphaning a
    course.

    When a `MatchResult` is supplied it is consulted first and its answer is
    final for spans it covers. Phase 2 did the corpus-wide work with shingle
    blocking and a one-to-one constraint, so it knows things a fresh
    similarity lookup cannot reconstruct.

    Letting the two disagree was a real defect, not a tidiness problem.
    Expand a sentence substantially — "Notice must be given within 30 days."
    into "Notice must typically be given within 30 days, though this may
    vary by region." — and the matcher correctly pairs them as MODIFIED while
    an independent fuzz.ratio lookup falls under the 0.85 floor and reports
    ORPHANED. The item would then be routed RETIRE_SUGGESTED: the system
    would propose deleting a question because its source material grew.
    """
    if matches is not None:
        decided = _from_match(ref, matches, old_span)
        if decided is not None:
            return decided

    # 1 — exact content hash.
    by_hash = new.by_hash()
    hits = list(by_hash.get(ref.snapshot_hash, ()))
    if hits:
        narrowed = _narrow(hits, ref, old_span)
        if len(narrowed) == 1:
            return ResolvedAnchor(
                ref=ref,
                status=_status_for(narrowed[0], ref, old_span),
                new_span=narrowed[0],
                similarity=1.0,
            )
        return ResolvedAnchor(
            ref=ref,
            status=AnchorStatus.AMBIGUOUS,
            candidates=tuple(narrowed),
            similarity=1.0,
        )

    # 2 — fuzzy match of the recorded snapshot text.
    #
    # `fuzz.ratio` here, not the `token_set_ratio` Phase B uses for pairing:
    # anchoring asks "is this the same sentence, lightly edited", and set
    # semantics would happily match a different sentence built from the same
    # vocabulary.
    needle = match_key(normalize(ref.snapshot_text))
    scored: list[tuple[float, Span]] = []
    for span in new.all_spans():
        score = fuzz.ratio(needle, match_key(span.norm_text)) / 100.0
        if score >= cfg.anchor_fuzzy:
            scored.append((score, span))

    if not scored:
        return ResolvedAnchor(ref=ref, status=AnchorStatus.ORPHANED)

    best = max(score for score, _ in scored)
    top = [span for score, span in scored if score == best]
    narrowed = _narrow(top, ref, old_span)
    if len(narrowed) == 1:
        return ResolvedAnchor(
            ref=ref,
            status=_status_for(narrowed[0], ref, old_span),
            new_span=narrowed[0],
            similarity=best,
        )
    return ResolvedAnchor(
        ref=ref,
        status=AnchorStatus.AMBIGUOUS,
        candidates=tuple(narrowed),
        similarity=best,
    )


# ---------------------------------------------------------------------------
# Bounded provenance inference
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(match_key(normalize(text)))


class _Bm25:
    """A compact BM25 over span texts.

    Deliberately unambitious — no embeddings, no learned weighting, no
    tuning. Its output is always marked `inferred`, and triage caps how far
    an inferred anchor can take an item.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.lengths = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.lengths) / len(corpus)) if corpus else 0.0
        self.freqs = [Counter(doc) for doc in corpus]
        seen: Counter[str] = Counter()
        for doc in corpus:
            seen.update(set(doc))
        total = len(corpus)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in seen.items()
        }

    def score(self, query: list[str], index: int) -> float:
        if not self.avg_len:
            return 0.0
        freq = self.freqs[index]
        length = self.lengths[index]
        total = 0.0
        for term in query:
            if term not in freq:
                continue
            tf = freq[term]
            denominator = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denominator
        return total


def infer_provenance(item: CourseItem, old: Snapshot, cfg: Config) -> list[ProvenanceRef]:
    """Guess where a question came from, when nothing was recorded.

    Without this the system cannot be retrofitted onto an existing course
    library, which is the common real case. Every ref it returns is marked
    `inferred`, and that mark is a hard cap in triage.
    """
    spans = list(old.all_spans())
    if not spans:
        return []

    index = _Bm25([_tokens(span.norm_text) for span in spans])
    query = _tokens(f"{item.stem} {item.correct_answer}")
    if not query:
        return []

    scored = [(index.score(query, i), spans[i]) for i in range(len(spans))]
    scored = [row for row in scored if row[0] > 0.0]
    if not scored:
        return []

    scored.sort(key=lambda row: (-row[0], row[1].doc_id, row[1].order))
    anchor_section = scored[0][1].section_path
    # Section agreement as the tiebreak: among comparable scores, prefer the
    # spans sitting beside the best hit.
    scored.sort(
        key=lambda row: (
            -row[0],
            0 if row[1].section_path == anchor_section else 1,
            row[1].doc_id,
            row[1].order,
        )
    )

    return [
        ProvenanceRef(
            doc_id=span.doc_id,
            span_id=span.span_id,
            snapshot_text=span.text,
            snapshot_hash=span.hash,
            inferred=True,
            infer_score=round(score, 4),
        )
        for score, span in scored[: cfg.infer_top_k]
    ]


# ---------------------------------------------------------------------------
# Course-level entry point
# ---------------------------------------------------------------------------


class AnchorReport(BaseModel):
    """Resolved anchors for a whole course, plus what had to be inferred."""

    anchors: dict[str, list[ResolvedAnchor]] = Field(default_factory=dict)
    inferred_items: set[str] = Field(default_factory=set)

    def for_item(self, item_id: str) -> list[ResolvedAnchor]:
        return self.anchors.get(item_id, [])


def resolve_course(
    course: Course,
    old: Snapshot,
    new: Snapshot,
    cfg: Config,
    *,
    infer: bool = True,
    matches: MatchResult | None = None,
) -> AnchorReport:
    """Resolve every item's provenance, inferring it where none was recorded.

    Pass `matches` so anchoring and the matcher cannot disagree about what
    happened to a span. See `resolve_anchor` for why that matters.
    """
    report = AnchorReport()
    for item in course.items:
        refs = list(item.provenance)
        if not refs and infer:
            refs = infer_provenance(item, old, cfg)
            if refs:
                report.inferred_items.add(item.item_id)
        report.anchors[item.item_id] = [
            resolve_anchor(ref, new, cfg, old_span=old.span_by_id(ref.span_id), matches=matches)
            for ref in refs
        ]
    return report


def make_ref(span: Span) -> ProvenanceRef:
    """Record provenance for a span as a generator would, at generation time."""
    return ProvenanceRef(
        doc_id=span.doc_id,
        span_id=span.span_id,
        snapshot_text=span.text,
        snapshot_hash=span_hash(span.norm_text),
    )
