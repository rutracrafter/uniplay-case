"""Phase 5 — tolerant extraction and strict validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from coursedrift.llm.base import (
    FailureKind,
    extract_json,
    parse_and_validate,
    repair_prompt,
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)


# -- extraction ------------------------------------------------------------


def test_plain_object() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_object_inside_a_json_fence() -> None:
    raw = 'Sure thing:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert extract_json(raw) == '{"a": 1}'


def test_object_inside_a_bare_fence() -> None:
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_object_surrounded_by_commentary() -> None:
    assert extract_json('I think {"a": 1} is right.') == '{"a": 1}'


def test_nested_objects_are_kept_whole() -> None:
    raw = '{"a": {"b": {"c": 1}}, "d": 2}'
    assert extract_json(raw) == raw


def test_braces_inside_strings_do_not_confuse_the_scanner() -> None:
    raw = '{"note": "a } brace and a { brace", "ok": true}'
    assert extract_json(raw) == raw


def test_escaped_quote_inside_a_string() -> None:
    raw = '{"note": "she said \\"hi\\" and left"}'
    assert extract_json(raw) == raw


def test_unbalanced_object_is_skipped_for_a_later_one() -> None:
    assert extract_json('{"broken": ') is None


def test_prose_only_is_none() -> None:
    assert extract_json("I would rather explain this in words.") is None


def test_empty_is_none() -> None:
    assert extract_json("") is None


# -- parse and validate ----------------------------------------------------


def test_valid_response_passes() -> None:
    outcome = parse_and_validate('{"verdict": "OK", "confidence": 0.9}', Answer)
    assert outcome.ok
    assert outcome.value is not None and outcome.value.verdict == "OK"
    assert outcome.reason == "ok"


def test_no_json_is_malformed() -> None:
    outcome = parse_and_validate("no json here", Answer)
    assert outcome.failure is FailureKind.MALFORMED
    assert outcome.value is None


def test_wrong_type_is_schema() -> None:
    outcome = parse_and_validate('{"verdict": "OK", "confidence": "high"}', Answer)
    assert outcome.failure is FailureKind.SCHEMA
    assert "confidence" in (outcome.detail or "")


def test_out_of_range_is_schema() -> None:
    outcome = parse_and_validate('{"verdict": "OK", "confidence": 1.4}', Answer)
    assert outcome.failure is FailureKind.SCHEMA


def test_extra_field_is_schema_because_forbid_is_deliberate() -> None:
    """An invented field signals the contract was not followed."""
    outcome = parse_and_validate(
        '{"verdict": "OK", "confidence": 0.9, "certainty": "total"}', Answer
    )
    assert outcome.failure is FailureKind.SCHEMA


def test_validators_run_in_order_and_first_failure_wins() -> None:
    calls: list[str] = []

    def first(value: Answer) -> str | None:
        calls.append("first")
        return None

    def second(value: Answer) -> str | None:
        calls.append("second")
        return "second says no"

    def third(value: Answer) -> str | None:
        calls.append("third")
        return "third says no"

    outcome = parse_and_validate(
        '{"verdict": "OK", "confidence": 0.9}', Answer, [first, second, third]
    )
    assert outcome.failure is FailureKind.REJECTED
    assert outcome.detail == "second says no"
    assert calls == ["first", "second"]


@pytest.mark.parametrize("message", ["", None])
def test_falsy_validator_result_passes(message: str | None) -> None:
    outcome = parse_and_validate(
        '{"verdict": "OK", "confidence": 0.9}', Answer, [lambda _v: message]
    )
    assert outcome.ok


def test_repair_prompt_names_the_error_and_the_bad_output() -> None:
    text = repair_prompt("ORIGINAL", '{"oops": 1}', "confidence is required")
    assert "ORIGINAL" in text
    assert "confidence is required" in text
    assert '{"oops": 1}' in text
    assert "JSON only" in text
