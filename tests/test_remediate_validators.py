"""Phase 6 — every patch validator rejects its own failure case."""

from __future__ import annotations

from coursedrift.remediate import (
    Patch,
    correct_index_in_range,
    grounded_in_new_document,
    one_correct_option,
    options_distinct,
    options_non_empty,
    patch_validators,
    stem_is_question,
)
from coursedrift.segment import normalize

NEW_TEXT = "Notice must be given within 45 days of the intended start date."
HAYSTACK = normalize(NEW_TEXT).casefold()


def _patch(**overrides: object) -> Patch:
    base: dict[str, object] = {
        "stem": "How many days of notice are required?",
        "options": ["45 days", "30 days", "60 days"],
        "correct_index": 0,
        "grounded_in": NEW_TEXT,
    }
    base.update(overrides)
    return Patch.model_validate(base)


def test_a_clean_patch_passes_every_validator() -> None:
    patch = _patch()
    assert [v(patch) for v in patch_validators(HAYSTACK)] == [None] * 6


# -- one failure per validator --------------------------------------------


def test_correct_index_out_of_range_is_rejected() -> None:
    message = correct_index_in_range(_patch(correct_index=9))
    assert message is not None and "outside the 3 options" in message


def test_empty_option_is_rejected() -> None:
    message = options_non_empty(_patch(options=["45 days", "   "]))
    assert message is not None and "option 1 is empty" in message


def test_duplicate_options_are_rejected() -> None:
    message = options_distinct(_patch(options=["45 days", "45 days", "30 days"]))
    assert message is not None and "not distinct" in message


def test_duplicates_are_caught_despite_casing_and_spacing() -> None:
    assert options_distinct(_patch(options=["45 Days", "45  days"])) is not None


def test_two_correct_answers_are_rejected() -> None:
    """An index-based schema can still describe two correct options."""
    message = one_correct_option(_patch(options=["45 days", "45 days"], correct_index=0))
    assert message is not None and "appears 2 times" in message


def test_statement_stem_is_rejected() -> None:
    message = stem_is_question(_patch(stem="The notice period is forty-five days."))
    assert message is not None and "not phrased as a question" in message


def test_ungrounded_quote_is_rejected() -> None:
    message = grounded_in_new_document(
        _patch(grounded_in="The policy was repealed by the board in 1998."),
        haystack=HAYSTACK,
    )
    assert message is not None and "not present in the new source" in message


# -- the leniency that is deliberate --------------------------------------


def test_the_same_quote_with_different_whitespace_is_accepted() -> None:
    quote = "Notice must be given\n   within 45 days"
    assert grounded_in_new_document(_patch(grounded_in=quote), haystack=HAYSTACK) is None


def test_the_same_quote_with_different_casing_is_accepted() -> None:
    quote = "notice MUST be given within 45 days"
    assert grounded_in_new_document(_patch(grounded_in=quote), haystack=HAYSTACK) is None


def test_a_paraphrase_is_still_rejected() -> None:
    """Leniency covers formatting, not wording."""
    quote = "Notice is required within forty-five days"
    assert grounded_in_new_document(_patch(grounded_in=quote), haystack=HAYSTACK) is not None


def test_interrogative_opening_without_a_question_mark_is_accepted() -> None:
    assert stem_is_question(_patch(stem="Select the correct notice period")) is None
    assert stem_is_question(_patch(stem="Which notice period applies")) is None


# -- schema-level guards ---------------------------------------------------


def test_schema_requires_at_least_two_options() -> None:
    from pydantic import ValidationError

    try:
        _patch(options=["only one"])
    except ValidationError as error:
        assert "options" in str(error)
    else:
        raise AssertionError("a single-option patch must not validate")


def test_schema_forbids_invented_fields() -> None:
    from pydantic import ValidationError

    try:
        Patch.model_validate(
            {
                "stem": "How many?",
                "options": ["a", "b"],
                "correct_index": 0,
                "grounded_in": NEW_TEXT,
                "certainty": "total",
            }
        )
    except ValidationError as error:
        assert "certainty" in str(error)
    else:
        raise AssertionError("an invented field must not validate")


def test_schema_rejects_a_negative_index() -> None:
    from pydantic import ValidationError

    try:
        _patch(correct_index=-1)
    except ValidationError:
        pass
    else:
        raise AssertionError("a negative correct_index must not validate")
