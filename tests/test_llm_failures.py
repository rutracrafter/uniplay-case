"""Phase 5 — every recovery path, exercised rather than asserted.

The point of these tests is that `call_validated` never raises. Each failure
mode produces a FailureKind the caller must route, which is what makes
failing closed to human review a property of the type rather than a habit.
"""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from coursedrift.config import Config
from coursedrift.llm.base import (
    CallBudget,
    FailureKind,
    TransportError,
    call_validated,
)
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.llm.prompts import TASK, TASK_ADJUDICATE, block

GOOD = {"verdict": "UNAFFECTED", "confidence": 0.9, "rationale": "fine", "evidence": ["x"]}


class Verdict(BaseModel):
    """Stands in for phase 6's Adjudication.

    `verdict` is a closed set here, not a bare str: the fake client's
    "schema" injection mode returns PROBABLY_FINE, and a loose str field
    would accept it — which would quietly stop testing anything.
    """

    model_config = ConfigDict(extra="forbid")
    verdict: Literal["UNAFFECTED", "WEAKENED", "INVALIDATED", "UNCERTAIN"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[str]


class Scripted:
    """A client that replays a fixed list of responses, one per call."""

    def __init__(self, *responses: str | BaseException) -> None:
        self.name = "scripted"
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, timeout: float) -> str:
        self.prompts.append(prompt)
        item = self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


def _call(client: object, **kwargs: object) -> object:
    return call_validated(
        client,  # type: ignore[arg-type]
        "PROMPT",
        Verdict,
        cfg=kwargs.get("cfg") or Config(),  # type: ignore[arg-type]
        cache=NullCache(),
        budget=kwargs.get("budget") or CallBudget(10),  # type: ignore[arg-type]
        validators=kwargs.get("validators") or (),  # type: ignore[arg-type]
    )


# -- each failure kind -----------------------------------------------------


def test_timeout_is_returned_not_raised() -> None:
    outcome = _call(Scripted(TimeoutError("took too long")))
    assert outcome.failure is FailureKind.TIMEOUT  # type: ignore[attr-defined]
    assert "took too long" in outcome.reason  # type: ignore[attr-defined]


def test_transport_error_is_returned_not_raised() -> None:
    outcome = _call(Scripted(TransportError("connection refused")))
    assert outcome.failure is FailureKind.TRANSPORT  # type: ignore[attr-defined]


def test_malformed_response() -> None:
    outcome = _call(Scripted("I will explain in prose.", "still prose"))
    assert outcome.failure is FailureKind.MALFORMED  # type: ignore[attr-defined]


def test_schema_violation() -> None:
    bad = json.dumps(
        {"verdict": "UNAFFECTED", "confidence": 3.0, "rationale": "r", "evidence": []}
    )
    outcome = _call(Scripted(bad, bad))
    assert outcome.failure is FailureKind.SCHEMA  # type: ignore[attr-defined]


def test_validator_rejection() -> None:
    outcome = _call(
        Scripted(json.dumps(GOOD), json.dumps(GOOD)),
        validators=[lambda v: "evidence quote is not in the source"],
    )
    assert outcome.failure is FailureKind.REJECTED  # type: ignore[attr-defined]
    assert "not in the source" in (outcome.detail or "")  # type: ignore[attr-defined]


def test_budget_exhaustion_happens_before_any_call() -> None:
    client = Scripted(json.dumps(GOOD))
    outcome = _call(client, budget=CallBudget(0))
    assert outcome.failure is FailureKind.BUDGET  # type: ignore[attr-defined]
    assert outcome.attempts == 0  # type: ignore[attr-defined]
    assert client.prompts == []


# -- tolerance and repair --------------------------------------------------


def test_fenced_json_succeeds_on_the_first_attempt() -> None:
    """Tolerant extraction is asserted, not assumed."""
    client = Scripted(f"Certainly:\n```json\n{json.dumps(GOOD)}\n```")
    outcome = _call(client)
    assert outcome.ok  # type: ignore[attr-defined]
    assert outcome.attempts == 1  # type: ignore[attr-defined]


def test_repair_succeeds_on_the_second_attempt() -> None:
    client = Scripted("prose, sorry", json.dumps(GOOD))
    outcome = _call(client)
    assert outcome.ok  # type: ignore[attr-defined]
    assert outcome.attempts == 2  # type: ignore[attr-defined]
    assert len(outcome.raw) == 2  # type: ignore[attr-defined]
    assert "Error:" in client.prompts[1]


def test_repair_failure_stops_at_max_attempts() -> None:
    client = Scripted("prose", "more prose", json.dumps(GOOD))
    outcome = _call(client)
    assert not outcome.ok  # type: ignore[attr-defined]
    assert outcome.attempts == 2  # type: ignore[attr-defined]
    assert len(client.prompts) == 2, "must not make a third attempt"


def test_transcript_keeps_every_raw_response() -> None:
    client = Scripted("prose", json.dumps(GOOD))
    outcome = _call(client)
    assert outcome.raw == ["prose", json.dumps(GOOD)]  # type: ignore[attr-defined]


def test_timeout_is_not_retried() -> None:
    """Re-asking a model that did not answer in time mostly buys a second wait."""
    client = Scripted(TimeoutError("slow"))
    _call(client)
    assert len(client.prompts) == 1


def test_repair_attempts_consume_budget() -> None:
    budget = CallBudget(1)
    client = Scripted("prose", json.dumps(GOOD))
    outcome = _call(client, budget=budget)
    assert outcome.failure is FailureKind.BUDGET  # type: ignore[attr-defined]
    assert budget.spent == 1


# -- the fake client's injection modes ------------------------------------


def _prompt(task: str = TASK_ADJUDICATE) -> str:
    from coursedrift.llm.prompts import DIFF, SOURCE_NEW, SOURCE_OLD

    return "\n".join(
        [
            block(TASK, task),
            block(SOURCE_OLD, "Notice is 30 days."),
            block(SOURCE_NEW, "Notice is 45 days."),
            block(DIFF, "Notice is [-30-]{+45+} days."),
        ]
    )


def test_fake_rejects_unknown_failure_modes() -> None:
    with pytest.raises(ValueError, match="unknown fake failure mode"):
        FakeClient(failures=["explode"])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("malformed", FailureKind.MALFORMED),
        ("schema", FailureKind.SCHEMA),
        ("timeout", FailureKind.TIMEOUT),
        ("transport", FailureKind.TRANSPORT),
    ],
)
def test_fake_failure_modes_produce_their_failure(mode: str, expected: FailureKind) -> None:
    client = FakeClient(failures=[mode, mode])
    outcome = call_validated(
        client,
        _prompt(),
        Verdict,
        cfg=Config(),
        cache=NullCache(),
        budget=CallBudget(10),
    )
    assert outcome.failure is expected


def test_fake_fenced_mode_still_succeeds() -> None:
    client = FakeClient(failures=["fenced"])
    outcome = call_validated(
        client, _prompt(), Verdict, cfg=Config(), cache=NullCache(), budget=CallBudget(10)
    )
    assert outcome.ok
    assert outcome.attempts == 1


def test_fake_failure_modes_fire_in_listed_order() -> None:
    client = FakeClient(failures=["malformed", "timeout"])
    first = call_validated(
        client,
        _prompt(),
        Verdict,
        cfg=Config(max_attempts=1),
        cache=NullCache(),
        budget=CallBudget(10),
    )
    second = call_validated(
        client,
        _prompt() + " b",
        Verdict,
        cfg=Config(max_attempts=1),
        cache=NullCache(),
        budget=CallBudget(10),
    )
    third = call_validated(
        client,
        _prompt() + " c",
        Verdict,
        cfg=Config(max_attempts=1),
        cache=NullCache(),
        budget=CallBudget(10),
    )
    assert first.failure is FailureKind.MALFORMED
    assert second.failure is FailureKind.TIMEOUT
    assert third.ok


def test_fake_heuristic_flags_a_changed_figure() -> None:
    client = FakeClient()
    outcome = call_validated(
        client, _prompt(), Verdict, cfg=Config(), cache=NullCache(), budget=CallBudget(10)
    )
    assert outcome.ok
    assert outcome.value is not None
    assert outcome.value.verdict == "INVALIDATED"


def test_fake_scripted_response_wins_over_the_heuristic() -> None:
    scripted = [("Notice is 45 days", json.dumps(GOOD))]
    client = FakeClient(scripted=scripted)
    outcome = call_validated(
        client, _prompt(), Verdict, cfg=Config(), cache=NullCache(), budget=CallBudget(10)
    )
    assert outcome.ok
    assert outcome.value is not None and outcome.value.verdict == "UNAFFECTED"


def test_fake_evidence_is_grounded_by_construction() -> None:
    client = FakeClient()
    outcome = call_validated(
        client, _prompt(), Verdict, cfg=Config(), cache=NullCache(), budget=CallBudget(10)
    )
    assert outcome.value is not None
    assert outcome.value.evidence[0] in "Notice is 45 days."


def test_fake_ungrounded_mode_quotes_nothing_real() -> None:
    client = FakeClient(failures=["ungrounded"])
    outcome = call_validated(
        client,
        _prompt(),
        Verdict,
        cfg=Config(max_attempts=1),
        cache=NullCache(),
        budget=CallBudget(10),
    )
    assert outcome.ok, "schema-valid: only a groundedness validator can catch this"
    assert outcome.value is not None
    assert "1998" in outcome.value.evidence[0]
