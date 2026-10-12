"""Phase 5 — memoization that cannot launder an invalid response."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from coursedrift.config import Config
from coursedrift.llm.base import CallBudget, FailureKind, call_validated
from coursedrift.llm.cache import FileCache, NullCache

GOOD = {"verdict": "UNAFFECTED", "confidence": 0.9}


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: str
    confidence: float


class Counting:
    """A client that records how many times it was actually consulted."""

    def __init__(self, response: str) -> None:
        self.name = "counting"
        self.response = response
        self.calls = 0

    def complete(self, prompt: str, *, timeout: float) -> str:
        self.calls += 1
        return self.response


def _call(client: Counting, cache: object, prompt: str = "PROMPT") -> object:
    return call_validated(
        client,
        prompt,
        Answer,
        cfg=Config(),
        cache=cache,  # type: ignore[arg-type]
        budget=CallBudget(10),
    )


def test_key_depends_on_client_and_prompt(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    assert cache.key("a", "p") == cache.key("a", "p")
    assert cache.key("a", "p") != cache.key("b", "p")
    assert cache.key("a", "p") != cache.key("a", "q")


def test_second_call_is_served_from_cache(tmp_path: Path) -> None:
    client = Counting(json.dumps(GOOD))
    cache = FileCache(tmp_path)

    first = _call(client, cache)
    second = _call(client, cache)

    assert client.calls == 1, "the second call must not reach the client"
    assert second.cached is True  # type: ignore[attr-defined]
    assert first.value == second.value  # type: ignore[attr-defined]


def test_cached_values_are_byte_identical(tmp_path: Path) -> None:
    client = Counting(json.dumps(GOOD))
    cache = FileCache(tmp_path)
    first = _call(client, cache)
    second = _call(client, cache)
    assert first.value.model_dump_json() == second.value.model_dump_json()  # type: ignore[union-attr]


def test_different_prompts_do_not_collide(tmp_path: Path) -> None:
    client = Counting(json.dumps(GOOD))
    cache = FileCache(tmp_path)
    _call(client, cache, "one")
    _call(client, cache, "two")
    assert client.calls == 2


def test_failures_are_not_cached(tmp_path: Path) -> None:
    client = Counting("prose only")
    cache = FileCache(tmp_path)
    _call(client, cache)
    before = client.calls
    _call(client, cache)
    assert client.calls > before, "a failed call must be retried, not replayed"


def test_cache_hits_still_pass_through_validation(tmp_path: Path) -> None:
    """A stored response that no longer validates is poison, not a result."""
    cache = FileCache(tmp_path)
    key = cache.key("counting", "PROMPT")
    cache.put(key, json.dumps({"verdict": "UNAFFECTED"}))  # missing confidence

    client = Counting(json.dumps(GOOD))
    outcome = _call(client, cache)

    assert outcome.ok  # type: ignore[attr-defined]
    assert outcome.cached is False  # type: ignore[attr-defined]
    assert client.calls == 1, "the poisoned entry must be dropped and the client called"
    assert cache.get(key) == json.dumps(GOOD)


def test_validator_change_invalidates_a_stored_response(tmp_path: Path) -> None:
    """Tightening a validator must not be defeated by yesterday's cache."""
    client = Counting(json.dumps(GOOD))
    cache = FileCache(tmp_path)
    _call(client, cache)

    outcome = call_validated(
        client,
        "PROMPT",
        Answer,
        cfg=Config(),
        cache=cache,
        budget=CallBudget(10),
        validators=[lambda _v: "now rejected by a stricter rule"],
    )
    assert outcome.failure is FailureKind.REJECTED


def test_corrupt_cache_file_is_treated_as_a_miss(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    key = cache.key("counting", "PROMPT")
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{key}.json").write_text("{not json at all")

    client = Counting(json.dumps(GOOD))
    outcome = _call(client, cache)
    assert outcome.ok  # type: ignore[attr-defined]
    assert client.calls == 1


def test_null_cache_never_serves_anything() -> None:
    client = Counting(json.dumps(GOOD))
    cache = NullCache()
    _call(client, cache)
    _call(client, cache)
    assert client.calls == 2


def test_cache_directory_is_created_on_demand(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper"
    cache = FileCache(target)
    client = Counting(json.dumps(GOOD))
    _call(client, cache)
    assert target.is_dir()
