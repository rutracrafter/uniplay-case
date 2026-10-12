"""Phase 1 — the normalization contract and segmentation rules.

The two halves of the contract are asserted separately and deliberately: a
future tidy-up of `normalize` that folds punctuation would silently make the
COSMETIC shortcut in phase 2 unreachable, and these tests are what catches it.
"""

from __future__ import annotations

from coursedrift.segment import normalize, segment_document, span_hash, split_sentences


def _hash(text: str) -> str:
    return span_hash(normalize(text))


# -- the contract: what MUST fold -------------------------------------------


def test_normalize_is_idempotent() -> None:
    raw = "The  policy’s  notice period — 30 days.\n"
    once = normalize(raw)
    assert normalize(once) == once


def test_whitespace_only_edit_hashes_identical() -> None:
    assert _hash("Notice must be given within 30 days.") == _hash(
        "Notice   must be  given\n  within 30 days."
    )


def test_curly_quote_and_dash_variants_hash_identical() -> None:
    assert _hash("The employee’s notice — 30 days.") == _hash(
        "The employee's notice - 30 days."
    )


def test_nbsp_and_ellipsis_variants_hash_identical() -> None:
    assert _hash("Notice period…") == _hash("Notice period...")


# -- the contract: what MUST NOT fold ---------------------------------------


def test_casing_edit_does_not_hash_identical() -> None:
    """Case changes must reach Phase B so the COSMETIC shortcut can fire."""
    assert _hash("Notice must be given.") != _hash("notice must be given.")


def test_punctuation_edit_does_not_hash_identical() -> None:
    """An added comma must reach Phase B, not vanish into Phase A."""
    assert _hash("Before leaving, submit the form.") != _hash("Before leaving submit the form.")


# -- sentence splitting -----------------------------------------------------


def test_splits_on_terminal_punctuation() -> None:
    assert split_sentences("One thing. Two things! Three?") == [
        "One thing.",
        "Two things!",
        "Three?",
    ]


def test_abbreviations_do_not_split() -> None:
    assert split_sentences("Submit forms, e.g. the T-40. Then wait.") == [
        "Submit forms, e.g. the T-40.",
        "Then wait.",
    ]


def test_initials_do_not_split() -> None:
    assert split_sentences("Contact J. Smith today. He approves.") == [
        "Contact J. Smith today.",
        "He approves.",
    ]


# -- markdown structure ----------------------------------------------------

_DOC = """Intro sentence before any heading.

# Leave Policy

Notice is 30 days. Managers must approve.

## Parental Leave

- Twelve weeks paid
- Extendable by request

```python
rate = 0.5
```

| Band | Days |
| ---- | ---- |
"""


def test_nested_headings_build_section_paths() -> None:
    spans, _ = segment_document("h.md", _DOC)
    paths = {s.text[:20]: s.section_path for s in spans}
    assert paths["Intro sentence befor"] == ()
    assert paths["Notice is 30 days."] == ("Leave Policy",)
    assert paths["- Twelve weeks paid"] == ("Leave Policy", "Parental Leave")


def test_headings_are_not_spans() -> None:
    spans, _ = segment_document("h.md", _DOC)
    assert not any(s.text.startswith("#") for s in spans)


def test_code_fence_is_one_span_with_newlines_intact() -> None:
    spans, _ = segment_document("h.md", _DOC)
    fences = [s for s in spans if s.text.startswith("```")]
    assert len(fences) == 1
    assert "\n" in fences[0].text
    assert "rate = 0.5" in fences[0].text


def test_each_list_item_is_its_own_span() -> None:
    spans, _ = segment_document("h.md", _DOC)
    items = [s for s in spans if s.text.startswith("- ")]
    assert len(items) == 2


def test_each_table_row_is_its_own_span() -> None:
    spans, _ = segment_document("h.md", _DOC)
    rows = [s for s in spans if s.text.startswith("|")]
    assert len(rows) == 2


def test_sections_carry_their_child_hashes() -> None:
    spans, sections = segment_document("h.md", _DOC)
    by_path = {s.path: s for s in sections}
    parental = by_path[("Leave Policy", "Parental Leave")]
    child_hashes = {
        s.hash for s in spans if s.section_path == ("Leave Policy", "Parental Leave")
    }
    assert parental.span_hashes == frozenset(child_hashes)


def test_root_section_only_exists_when_it_has_spans() -> None:
    _, with_intro = segment_document("a.md", _DOC)
    assert any(s.path == () for s in with_intro)
    _, no_intro = segment_document("b.md", "# Only A Heading\n\nBody text.\n")
    assert all(s.path != () for s in no_intro)


# -- stability -------------------------------------------------------------


def test_span_ids_and_order_are_stable_across_loads() -> None:
    first, _ = segment_document("s.md", _DOC)
    second, _ = segment_document("s.md", _DOC)
    assert [s.span_id for s in first] == [s.span_id for s in second]
    assert [s.order for s in first] == [s.order for s in second]
    assert [s.hash for s in first] == [s.hash for s in second]


def test_order_is_document_wide_and_contiguous() -> None:
    spans, _ = segment_document("s.md", _DOC)
    assert [s.order for s in spans] == list(range(len(spans)))
