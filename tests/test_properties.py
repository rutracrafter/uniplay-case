"""Property-based tests for the matcher.

The core invariant is a *property*: any reordering of a document produces no
content change. `test_diff_reorder.py` asserts it on three hand-picked
arrangements, which is where a bug hides — in the arrangement nobody thought
of. These tests assert the same things over generated documents instead.

The conservation property here is the one the suite was missing entirely: a
Phase A or Phase B bug that silently *dropped* a span would have passed
everything else.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from coursedrift.config import Config
from coursedrift.corpus import load_snapshot
from coursedrift.diff import ContentOp, PositionOp, match_corpus
from coursedrift.lis import lis_indices
from coursedrift.models import Snapshot
from coursedrift.segment import normalize

# A small closed vocabulary keeps generated sentences realistic and makes
# accidental duplicates likely, which is a case worth exercising rather than
# avoiding.
WORDS = st.sampled_from(
    [
        "notice",
        "managers",
        "approve",
        "leave",
        "expenses",
        "receipts",
        "mileage",
        "badges",
        "audit",
        "logs",
        "backups",
        "vendors",
        "training",
        "policy",
        "quarterly",
        "monthly",
        "annual",
        "required",
        "optional",
    ]
)
HEADINGS = st.sampled_from(
    ["Access Control", "Incident Response", "Data Retention", "Leave Policy", "Expenses"]
)

sentences = st.lists(WORDS, min_size=3, max_size=8).map(
    lambda words: " ".join(words).capitalize() + "."
)
sections = st.tuples(HEADINGS, st.lists(sentences, min_size=1, max_size=4))
documents = st.lists(sections, min_size=1, max_size=5)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _render(doc: list[tuple[str, list[str]]]) -> str:
    return "\n\n".join(f"# {heading}\n\n" + " ".join(body) for heading, body in doc) + "\n"


def _snapshot(tmp_path: Path, name: str, body: str) -> Snapshot:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "doc.md").write_text(body, encoding="utf-8")
    return load_snapshot(root)


# -- conservation ----------------------------------------------------------


@SETTINGS
@given(old_doc=documents, new_doc=documents)
def test_every_span_is_accounted_for_exactly_once(
    tmp_path: Path, old_doc: list, new_doc: list
) -> None:
    """The property the suite was missing.

    Every old span must appear exactly once in the change table — as the old
    side of a pair, or as REMOVED. Every new span exactly once — as the new
    side of a pair, or as ADDED. A matcher that quietly dropped a span would
    otherwise pass every other test in the suite.
    """
    old = _snapshot(tmp_path, "v1", _render(old_doc))
    new = _snapshot(tmp_path, "v2", _render(new_doc))
    result = match_corpus(old, new, Config())

    expected_old = sorted(s.span_id for s in old.all_spans())
    expected_new = sorted(s.span_id for s in new.all_spans())
    seen_old = sorted(c.old.span_id for c in result.changes if c.old is not None)
    seen_new = sorted(c.new.span_id for c in result.changes if c.new is not None)

    assert seen_old == expected_old
    assert seen_new == expected_new


@SETTINGS
@given(old_doc=documents, new_doc=documents)
def test_matching_is_one_to_one(tmp_path: Path, old_doc: list, new_doc: list) -> None:
    """No span may be paired twice, in either direction."""
    old = _snapshot(tmp_path, "v1", _render(old_doc))
    new = _snapshot(tmp_path, "v2", _render(new_doc))
    result = match_corpus(old, new, Config())

    paired_old = [c.old.span_id for c in result.changes if c.old and c.new]
    paired_new = [c.new.span_id for c in result.changes if c.old and c.new]
    assert len(paired_old) == len(set(paired_old))
    assert len(paired_new) == len(set(paired_new))


@SETTINGS
@given(doc=documents)
def test_a_snapshot_against_itself_has_no_changes(tmp_path: Path, doc: list) -> None:
    """Reflexivity. Any drift reported here is pure false positive."""
    body = _render(doc)
    result = match_corpus(
        _snapshot(tmp_path, "v1", body), _snapshot(tmp_path, "v2", body), Config()
    )
    assert {c.content for c in result.changes} == {ContentOp.IDENTICAL}
    assert {c.position for c in result.changes} == {PositionOp.STABLE}
    assert result.real_changes() == []


# -- the core invariant, generalised --------------------------------------


@SETTINGS
@given(doc=documents, seed=st.integers(min_value=0, max_value=10_000))
def test_any_reordering_produces_no_content_change(
    tmp_path: Path, doc: list, seed: int
) -> None:
    """The invariant over arbitrary documents and arbitrary permutations.

    `test_diff_reorder.py` covers rotation and reversal by hand. This covers
    the arrangement nobody thought of.
    """
    assume(len(doc) > 1)
    import random

    shuffled = list(doc)
    random.Random(seed).shuffle(shuffled)
    assume(shuffled != doc)

    result = match_corpus(
        _snapshot(tmp_path, "v1", _render(doc)),
        _snapshot(tmp_path, "v2", _render(shuffled)),
        Config(),
    )
    assert {c.content for c in result.changes} == {ContentOp.IDENTICAL}
    assert not [
        c
        for c in result.changes
        if c.content in (ContentOp.MODIFIED, ContentOp.REMOVED, ContentOp.ADDED)
    ]


@SETTINGS
@given(doc=documents, seed=st.integers(min_value=0, max_value=10_000))
def test_reordering_never_reparents(tmp_path: Path, doc: list, seed: int) -> None:
    """Headings travel with their content, so no span changes section."""
    assume(len(doc) > 1)
    import random

    shuffled = list(doc)
    random.Random(seed).shuffle(shuffled)
    result = match_corpus(
        _snapshot(tmp_path, "v1", _render(doc)),
        _snapshot(tmp_path, "v2", _render(shuffled)),
        Config(),
    )
    assert not [c for c in result.changes if c.position is PositionOp.REPARENTED]


# -- normalization ---------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=200))
def test_normalize_is_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once


@settings(max_examples=200, deadline=None)
@given(text=st.text(max_size=200))
def test_normalize_never_introduces_leading_or_trailing_space(text: str) -> None:
    result = normalize(text)
    assert result == result.strip()


# -- LIS against a reference implementation -------------------------------


def _reference_lis_length(seq: list[int]) -> int:
    """Obvious O(n^2) dynamic program, for cross-checking the fast version."""
    if not seq:
        return 0
    best = [1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best)


@settings(max_examples=300, deadline=None)
@given(seq=st.lists(st.integers(min_value=-20, max_value=20), max_size=40))
def test_lis_matches_a_reference_implementation(seq: list[int]) -> None:
    keep = lis_indices(seq)
    assert len(keep) == _reference_lis_length(seq)


@settings(max_examples=300, deadline=None)
@given(seq=st.lists(st.integers(min_value=-20, max_value=20), max_size=40))
def test_lis_result_is_a_strictly_increasing_subsequence(seq: list[int]) -> None:
    picked = [seq[i] for i in sorted(lis_indices(seq))]
    assert all(a < b for a, b in zip(picked, picked[1:], strict=False))
