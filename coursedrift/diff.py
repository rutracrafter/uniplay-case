"""Span and section matching: the order-independent core of the system.

The invariant this module exists to uphold, from OVERVIEW.md:

    No stage may use positional order to establish identity. Order is used
    only to classify movement and to assemble context.

Phase A matches by content hash, corpus-wide. Phase B matches the residue by
similarity. Only then does Phase C look at position, and only to decide
whether a matched span moved. `tests/test_diff_reorder.py` is the regression
test for the invariant; if it fails, nothing downstream can be trusted.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from coursedrift.config import Config
from coursedrift.lis import lis_indices
from coursedrift.models import ManifestDiff, Section, Snapshot, Span
from coursedrift.segment import normalize

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class ContentOp(StrEnum):
    """What happened to a span's text."""

    IDENTICAL = "IDENTICAL"
    COSMETIC = "COSMETIC"
    MODIFIED = "MODIFIED"
    REMOVED = "REMOVED"
    ADDED = "ADDED"


class PositionOp(StrEnum):
    """What happened to a span's place in the corpus."""

    STABLE = "STABLE"
    MOVED = "MOVED"
    REPARENTED = "REPARENTED"
    NONE = "NONE"


class SectionOp(StrEnum):
    SAME = "SAME"
    RENAMED = "RENAMED"
    SPLIT = "SPLIT"
    MERGED = "MERGED"
    REMOVED = "REMOVED"
    ADDED = "ADDED"


class SpanChange(BaseModel):
    """One span's fate between two snapshots, on both axes."""

    model_config = ConfigDict(frozen=True)

    content: ContentOp
    position: PositionOp
    old: Span | None = None
    new: Span | None = None
    similarity: float | None = None
    """How much of the text actually changed, order included.

    Order-sensitive on purpose. The permissive score below is what decided
    these two spans belong together; this is what a reviewer reads, so it has
    to mean "how different is this text" rather than "do these share a
    vocabulary".
    """

    pair_score: float | None = None
    """The order-insensitive score that paired them. Diagnostic only."""

    reordered: bool = False
    """The same words, rearranged — see `is_reordering`."""

    inline_diff: str | None = None
    across_docs: bool = False

    @property
    def is_change(self) -> bool:
        """Whether this is a change at all, as opposed to stable identity."""
        if self.content is ContentOp.IDENTICAL:
            return self.position is not PositionOp.STABLE
        return True

    def describe(self) -> str:
        where = self.new or self.old
        assert where is not None
        return f"{where.span_id} {self.content.value}/{self.position.value}"


class SectionLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: SectionOp
    old: tuple[Section, ...] = ()
    new: tuple[Section, ...] = ()
    jaccard: float = 0.0


class SectionMap(BaseModel):
    """Section-level restructuring, and the rename resolution it enables."""

    links: list[SectionLink] = Field(default_factory=list)
    resolved: dict[tuple[str, ...], set[tuple[str, ...]]] = Field(default_factory=dict)

    def expected_paths(self, old_path: tuple[str, ...]) -> set[tuple[str, ...]]:
        """Where a span under `old_path` should legitimately end up.

        A span sitting under a *renamed* heading has not been reparented: the
        heading changed around it. A span under a section that *split* may
        legitimately be in any of the resulting sections.
        """
        return self.resolved.get(old_path, {old_path})

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for link in self.links:
            out[link.op.value] += 1
        return dict(out)


class MatchResult(BaseModel):
    """Everything phase 2 produces."""

    changes: list[SpanChange] = Field(default_factory=list)
    sections: SectionMap = Field(default_factory=SectionMap)

    def real_changes(self) -> list[SpanChange]:
        return [c for c in self.changes if c.is_change]

    def by_old_span(self) -> dict[str, SpanChange]:
        return {c.old.span_id: c for c in self.changes if c.old is not None}

    def by_new_span(self) -> dict[str, SpanChange]:
        return {c.new.span_id: c for c in self.changes if c.new is not None}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for change in self.changes:
            out[f"{change.content.value}/{change.position.value}"] += 1
        return dict(out)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]+")


def _skeleton(text: str) -> str:
    """Normalized text with casing and punctuation removed."""
    return _PUNCT.sub("", normalize(text).lower())


def is_cosmetic(old_text: str, new_text: str) -> bool:
    """True when the only differences are casing or punctuation.

    An exact test, not a heuristic: if the two strings agree once casing and
    punctuation are stripped, then casing and punctuation are all that
    changed. Whitespace-only edits never arrive here — they hash identical in
    Phase A — so this is strictly the second half of the normalization
    contract.
    """
    a, b = normalize(old_text), normalize(new_text)
    if a == b:
        return False
    return _skeleton(a) == _skeleton(b)


def is_reordering(old_text: str, new_text: str) -> bool:
    """True when the two texts use exactly the same words in a different order.

    This is a narrow but dangerous class of edit. "Always use a password
    manager, never store passwords in plain text." and "Never use a password
    manager, always store passwords in plain text." contain precisely the
    same words, so every order-insensitive measure calls them identical —
    while they instruct the reader to do opposite things.

    Compared on punctuation-stripped, case-folded tokens as a multiset, so
    neither punctuation nor repeated words can fool it. Strict multiset
    equality keeps the signal crisp and explainable: a reorder that also adds
    or drops a word is an ordinary MODIFIED change and is adjudicated on its
    text like any other.
    """
    old_tokens = shingle_tokens(match_key(normalize(old_text)))
    new_tokens = shingle_tokens(match_key(normalize(new_text)))
    if old_tokens == new_tokens:
        return False
    return sorted(old_tokens) == sorted(new_tokens)


def inline_diff(old_text: str, new_text: str) -> str:
    """Word-level diff rendered as `[-removed-]{+added+}`.

    Presentation only: this feeds the report and the model prompt. Identity
    never depends on it, so using a positional differ here is safe.
    """
    old_words = normalize(old_text).split()
    new_words = normalize(new_text).split()
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        gone = " ".join(old_words[i1:i2])
        came = " ".join(new_words[j1:j2])
        if tag == "equal":
            parts.append(gone)
        elif tag == "delete":
            parts.append(f"[-{gone}-]")
        elif tag == "insert":
            parts.append(f"{{+{came}+}}")
        else:
            parts.append(f"[-{gone}-]{{+{came}+}}")
    return " ".join(p for p in parts if p)


def match_key(text: str) -> str:
    """The string Phase B blocks and scores on.

    Case-folded, because casing is handled on its own axis: a case-only edit
    must stay a near-perfect *match* so the COSMETIC shortcut can claim it.
    Fold case into Phase A's hash instead and casing changes would vanish
    entirely; leave it out of Phase B and they read as delete plus add.
    """
    return text.casefold()


_EDGE_PUNCT = re.compile(r"^\W+|\W+$")


def shingle_tokens(text: str) -> list[str]:
    """Tokens for blocking, with edge punctuation stripped.

    Blocking is a cheap filter, and punctuation must not decide what it
    filters out. Leave it attached and "approve all leave requests." shares
    no trigram with "...all leave requests require...", because one token
    carries a full stop — so a plain reword is never even scored and comes
    out as a delete plus an add.
    """
    out = []
    for word in text.split():
        trimmed = _EDGE_PUNCT.sub("", word)
        if trimmed:
            out.append(trimmed)
    return out


def shingles(text: str, width: int) -> set[tuple[str, ...]]:
    """Word n-grams used to block Phase B candidate pairs."""
    words = shingle_tokens(text)
    if not words:
        return set()
    if len(words) <= width:
        return {tuple(words)}
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def block_width(text: str, max_width: int) -> int:
    """The widest window a single-token edit cannot destroy entirely.

    With `n` words and window `w` there are `n - w + 1` windows, and an edit
    at one position ruins at most `w` of them. Surviving at least one window
    therefore needs `w <= n / 2`.

    This matters more than it looks. At a fixed width of 3, the sentence
    "Notice is 30 days." shares no trigram at all with "Notice is 45 days." —
    the candidate pair is never scored, and an obvious edit comes out as a
    REMOVED plus an ADDED instead of a MODIFIED. Short spans need short
    windows.
    """
    count = len(shingle_tokens(text))
    if count < 2:
        return 1
    return max(1, min(max_width, count // 2))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# ---------------------------------------------------------------------------
# Phase A — global identity match
# ---------------------------------------------------------------------------


class _Pair:
    """A mutable working pair, before it becomes a frozen SpanChange."""

    __slots__ = (
        "across_docs",
        "content",
        "new",
        "old",
        "pair_score",
        "position",
        "reordered",
        "similarity",
    )

    def __init__(
        self,
        old: Span,
        new: Span,
        content: ContentOp,
        similarity: float | None,
        pair_score: float | None = None,
        reordered: bool = False,
    ) -> None:
        self.old = old
        self.new = new
        self.content = content
        self.similarity = similarity
        self.pair_score = pair_score if pair_score is not None else similarity
        self.reordered = reordered
        self.position: PositionOp = PositionOp.STABLE
        self.across_docs = old.doc_id != new.doc_id


def _zip_duplicate_group(
    olds: list[Span], news: list[Span]
) -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    """Pair equal-hash spans, preferring the closest partner available.

    Within a duplicate group every candidate is textually identical, so the
    matcher has to choose — and the choice is arbitrary, which means it must
    not manufacture a signal. Preference runs in three tiers: same document
    and same section, then same document, then anywhere.

    Both of the lower tiers exist to avoid a specific false positive.

    Without the section tier, a sentence appearing once under each of two
    headings pairs across them when those headings are reordered, and
    `_apply_reparenting` reports a span that moved between sections when
    nothing moved at all — each heading still contains its copy. Found by
    `test_properties.py`, which generates exactly this.

    Without the document tier, unrelated boilerplate in another file can
    steal a partner and produce a phantom cross-document move. Whatever is
    left after both tiers pairs across documents, which is a real move.
    """
    pairs: list[tuple[Span, Span]] = []
    pool = list(news)

    def claim(candidates: list[Span], matches: Callable[[Span, Span], bool]) -> list[Span]:
        unclaimed: list[Span] = []
        for old in candidates:
            partner = next((n for n in pool if matches(old, n)), None)
            if partner is None:
                unclaimed.append(old)
                continue
            pool.remove(partner)
            pairs.append((old, partner))
        return unclaimed

    remaining = claim(
        list(olds),
        lambda o, n: o.doc_id == n.doc_id and o.section_path == n.section_path,
    )
    remaining = claim(remaining, lambda o, n: o.doc_id == n.doc_id)

    still_unmatched: list[Span] = []
    for old in remaining:
        if pool:
            pairs.append((old, pool.pop(0)))
        else:
            still_unmatched.append(old)
    return pairs, still_unmatched, pool


def _phase_a_identity(
    old: Snapshot, new: Snapshot
) -> tuple[list[_Pair], list[Span], list[Span]]:
    """Match every span whose content hash appears on both sides.

    Corpus-wide, and position is never consulted. This is what makes a
    reordered document produce matches instead of deletions.
    """
    old_index = old.by_hash()
    new_index = new.by_hash()

    pairs: list[_Pair] = []
    residue_old: list[Span] = []
    residue_new: list[Span] = []

    for digest in sorted(old_index.keys() | new_index.keys()):
        olds = old_index.get(digest, [])
        news = new_index.get(digest, [])
        matched, left_old, left_new = _zip_duplicate_group(olds, news)
        pairs.extend(_Pair(o, n, ContentOp.IDENTICAL, 1.0) for o, n in matched)
        residue_old.extend(left_old)
        residue_new.extend(left_new)

    return pairs, residue_old, residue_new


# ---------------------------------------------------------------------------
# Phase B — similarity alignment of the residue
# ---------------------------------------------------------------------------


def _phase_b_similarity(
    residue_old: Sequence[Span], residue_new: Sequence[Span], cfg: Config
) -> tuple[list[_Pair], list[Span], list[Span]]:
    """Align what Phase A could not, by similarity, one-to-one.

    A shingle inverted index blocks the candidate set so this stays near
    linear instead of quadratic: a pair is only scored if it shares at least
    one word n-gram.
    """
    # New spans are indexed at every width from 1 up, because a query uses
    # the width its own length allows and the two sides may differ in length.
    index: dict[tuple[str, ...], list[Span]] = defaultdict(list)
    for span in residue_new:
        folded = match_key(span.norm_text)
        for width in range(1, cfg.shingle_n + 1):
            for shingle in shingles(folded, width):
                index[shingle].append(span)

    # Two scores, because pairing and reporting want different things.
    #
    # `pair` is order-insensitive, which is what lets a genuine reword find
    # its successor: "Managers must approve all leave requests." scores 0.80
    # against "All leave requests require manager approval." on token sets
    # but only 0.45 on edit distance, so an order-sensitive threshold would
    # split it into a delete plus an add.
    #
    # `text` is order-sensitive, and it is the one that gets reported and
    # that gates COSMETIC. Report the permissive score instead and a
    # meaning-reversing rearrangement shows up as "similarity: 1.00", which
    # tells a reviewer the exact opposite of the truth.
    scored: list[tuple[float, float, Span, Span]] = []
    for old_span in residue_old:
        candidates: dict[str, Span] = {}
        folded_old = match_key(old_span.norm_text)
        for shingle in shingles(folded_old, block_width(folded_old, cfg.shingle_n)):
            for candidate in index.get(shingle, ()):
                candidates[candidate.span_id] = candidate
        for candidate in candidates.values():
            folded_new = match_key(candidate.norm_text)
            pair = fuzz.token_set_ratio(folded_old, folded_new) / 100.0
            if pair < cfg.similarity_match:
                continue
            text = fuzz.ratio(folded_old, folded_new) / 100.0
            scored.append((pair, text, old_span, candidate))

    # Descending pair score, then textual closeness, then span ids: a tie on
    # the permissive score breaks toward the candidate that is actually more
    # similar, and the greedy pass stays deterministic.
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].span_id, row[3].span_id))

    pairs: list[_Pair] = []
    taken_old: set[str] = set()
    taken_new: set[str] = set()
    for pair_score, text_score, old_span, new_span in scored:
        if old_span.span_id in taken_old or new_span.span_id in taken_new:
            continue
        taken_old.add(old_span.span_id)
        taken_new.add(new_span.span_id)
        cosmetic = text_score >= cfg.cosmetic_similarity and is_cosmetic(
            old_span.text, new_span.text
        )
        content = ContentOp.COSMETIC if cosmetic else ContentOp.MODIFIED
        pairs.append(
            _Pair(
                old_span,
                new_span,
                content,
                text_score,
                pair_score=pair_score,
                reordered=is_reordering(old_span.text, new_span.text),
            )
        )

    unmatched_old = [s for s in residue_old if s.span_id not in taken_old]
    unmatched_new = [s for s in residue_new if s.span_id not in taken_new]
    return pairs, unmatched_old, unmatched_new


# ---------------------------------------------------------------------------
# Phase C — move classification
# ---------------------------------------------------------------------------


def _classify_positions(pairs: Iterable[_Pair]) -> None:
    """Assign STABLE or MOVED. Mutates `pairs` in place.

    Cross-document pairs are MOVED unconditionally and never enter the LIS:
    a span that changed file has no position inside the new file to be stable
    within, and including it would corrupt the stable core for its neighbours.
    """
    per_doc: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs:
        if pair.across_docs:
            pair.position = PositionOp.MOVED
        else:
            per_doc[pair.new.doc_id].append(pair)

    for group in per_doc.values():
        group.sort(key=lambda p: p.old.order)
        keep = lis_indices([p.new.order for p in group])
        for offset, pair in enumerate(group):
            pair.position = PositionOp.STABLE if offset in keep else PositionOp.MOVED


def _apply_reparenting(pairs: Iterable[_Pair], sections: SectionMap) -> None:
    """Upgrade STABLE or MOVED to REPARENTED where the section really changed.

    This runs after the section map exists, and that ordering is the point: a
    span under a section that was merely renamed has not been reparented, so
    the rename has to be resolved away before comparing paths.
    """
    for pair in pairs:
        if pair.across_docs:
            # A cross-document move is already the strongest position signal,
            # and heading namespaces in different files are not comparable:
            # "Leave" in leave.md is not a reparenting of "Onboarding" in a
            # document that no longer exists. `across_docs` carries the
            # context change instead, and phase 4 reads it directly.
            continue
        expected = sections.expected_paths(pair.old.section_path)
        if pair.new.section_path not in expected:
            pair.position = PositionOp.REPARENTED


# ---------------------------------------------------------------------------
# Section matching
# ---------------------------------------------------------------------------


def match_sections(old: Snapshot, new: Snapshot, cfg: Config) -> SectionMap:
    """Match sections by heading, then by Jaccard over child span hashes.

    Jaccard over content hashes is order-free and expresses restructuring
    naturally: one old section overlapping two new ones is a split, two old
    overlapping one new is a merge.
    """
    old_secs = list(old.all_sections())
    new_secs = list(new.all_sections())
    links: list[SectionLink] = []
    used_old: set[str] = set()
    used_new: set[str] = set()

    def link(
        op: SectionOp, olds: Sequence[Section], news: Sequence[Section], score: float
    ) -> None:
        links.append(SectionLink(op=op, old=tuple(olds), new=tuple(news), jaccard=score))
        used_old.update(s.section_id for s in olds)
        used_new.update(s.section_id for s in news)

    # Pass 1 — same document, same path.
    new_by_key = {(s.doc_id, s.path): s for s in new_secs}
    for section in old_secs:
        target = new_by_key.get((section.doc_id, section.path))
        if target is not None and target.section_id not in used_new:
            link(
                SectionOp.SAME,
                [section],
                [target],
                jaccard(section.span_hashes, target.span_hashes),
            )

    # Pass 2 — identical heading text, anywhere in the corpus.
    by_heading: dict[str, list[Section]] = defaultdict(list)
    for section in new_secs:
        if section.section_id not in used_new and section.heading:
            by_heading[section.heading].append(section)
    for section in old_secs:
        if section.section_id in used_old or not section.heading:
            continue
        pool = [s for s in by_heading.get(section.heading, ()) if s.section_id not in used_new]
        if pool:
            target = pool[0]
            op = SectionOp.SAME if target.path == section.path else SectionOp.RENAMED
            link(op, [section], [target], jaccard(section.span_hashes, target.span_hashes))

    def remaining_old() -> list[Section]:
        return [s for s in old_secs if s.section_id not in used_old]

    def remaining_new() -> list[Section]:
        return [s for s in new_secs if s.section_id not in used_new]

    # Pass 3 — splits: one old section spread across several new ones.
    for section in remaining_old():
        parts = [
            s
            for s in remaining_new()
            if jaccard(section.span_hashes, s.span_hashes) >= cfg.jaccard_part
        ]
        if len(parts) >= 2:
            best = max(jaccard(section.span_hashes, p.span_hashes) for p in parts)
            link(SectionOp.SPLIT, [section], parts, best)

    # Pass 4 — merges: several old sections folded into one new one.
    for section in remaining_new():
        sources = [
            s
            for s in remaining_old()
            if jaccard(s.span_hashes, section.span_hashes) >= cfg.jaccard_part
        ]
        if len(sources) >= 2:
            best = max(jaccard(s.span_hashes, section.span_hashes) for s in sources)
            link(SectionOp.MERGED, sources, [section], best)

    # Pass 5 — one-to-one by Jaccard, greedy and deterministic.
    scored: list[tuple[float, Section, Section]] = []
    for left in remaining_old():
        for right in remaining_new():
            score = jaccard(left.span_hashes, right.span_hashes)
            if score >= cfg.jaccard_match:
                scored.append((score, left, right))
    scored.sort(key=lambda row: (-row[0], row[1].section_id, row[2].section_id))
    for score, left, right in scored:
        if left.section_id in used_old or right.section_id in used_new:
            continue
        op = SectionOp.SAME if left.path == right.path else SectionOp.RENAMED
        link(op, [left], [right], score)

    # Pass 6 — leftovers.
    for section in remaining_old():
        link(SectionOp.REMOVED, [section], [], 0.0)
    for section in remaining_new():
        link(SectionOp.ADDED, [], [section], 0.0)

    resolved: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for item in links:
        for source in item.old:
            for target in item.new:
                resolved[source.path].add(target.path)

    return SectionMap(links=links, resolved=dict(resolved))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _restrict(snapshot: Snapshot, doc_ids: set[str]) -> Snapshot:
    return Snapshot(
        root=snapshot.root,
        docs={k: v for k, v in snapshot.docs.items() if k in doc_ids},
    )


def match_corpus(
    old: Snapshot,
    new: Snapshot,
    cfg: Config,
    manifest: ManifestDiff | None = None,
) -> MatchResult:
    """Match two snapshots. Identity first, position last.

    Documents the manifest calls unchanged are excluded on both sides: they
    cannot contribute a change, and leaving them in would let their
    boilerplate interfere with duplicate-group pairing elsewhere.
    """
    if manifest is not None:
        keep = manifest.touched
        old = _restrict(old, keep)
        new = _restrict(new, keep)

    pairs, residue_old, residue_new = _phase_a_identity(old, new)
    similar, gone, arrived = _phase_b_similarity(residue_old, residue_new, cfg)
    pairs.extend(similar)

    _classify_positions(pairs)
    sections = match_sections(old, new, cfg)
    _apply_reparenting(pairs, sections)

    changes: list[SpanChange] = []
    for pair in pairs:
        needs_diff = pair.content in (ContentOp.MODIFIED, ContentOp.COSMETIC)
        changes.append(
            SpanChange(
                content=pair.content,
                position=pair.position,
                old=pair.old,
                new=pair.new,
                similarity=pair.similarity,
                pair_score=pair.pair_score,
                reordered=pair.reordered,
                inline_diff=inline_diff(pair.old.text, pair.new.text) if needs_diff else None,
                across_docs=pair.across_docs,
            )
        )
    changes.extend(
        SpanChange(content=ContentOp.REMOVED, position=PositionOp.NONE, old=span)
        for span in gone
    )
    changes.extend(
        SpanChange(content=ContentOp.ADDED, position=PositionOp.NONE, new=span)
        for span in arrived
    )

    changes.sort(key=lambda c: ((c.new or c.old).doc_id, (c.new or c.old).order))  # type: ignore[union-attr]
    return MatchResult(changes=changes, sections=sections)
