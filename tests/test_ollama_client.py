"""Phase 8 — the Ollama client, at the transport level. No live daemon."""

from __future__ import annotations

import json

import httpx
import pytest

from coursedrift.config import Config
from coursedrift.llm.base import CallBudget, FailureKind, TransportError, call_validated
from coursedrift.llm.cache import NullCache
from coursedrift.llm.ollama import (
    OllamaClient,
    OllamaModelMissing,
    OllamaUnreachable,
)
from tests.test_llm_failures import Verdict

GOOD = {
    "verdict": "INVALIDATED",
    "confidence": 0.9,
    "rationale": "the figure changed",
    "evidence": ["45 days"],
}


def _client(handler: object) -> OllamaClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OllamaClient(model="qwen2.5:3b", http=httpx.Client(transport=transport))


# -- preflight -------------------------------------------------------------


def test_preflight_passes_when_the_model_is_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})

    _client(handler).preflight()


def test_preflight_matches_an_implicit_latest_tag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b-instruct"}]})

    _client(handler).preflight()


def test_preflight_names_the_pull_command_when_the_model_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    with pytest.raises(OllamaModelMissing) as error:
        _client(handler).preflight()
    assert "ollama pull qwen2.5:3b" in str(error.value)
    assert "llama3:8b" in str(error.value), "should list what is available"


def test_preflight_explains_an_unreachable_daemon() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaUnreachable) as error:
        _client(handler).preflight()
    assert "ollama serve" in str(error.value)
    assert "--client fake" in str(error.value)


def test_preflight_treats_a_server_error_as_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(OllamaUnreachable):
        _client(handler).preflight()


# -- completion ------------------------------------------------------------


def test_request_shape_is_deterministic() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    _client(handler).complete("PROMPT", timeout=5.0)
    assert seen["model"] == "qwen2.5:3b"
    assert seen["stream"] is False
    assert seen["format"] == "json"
    assert seen["options"] == {"temperature": 0, "seed": 0}
    assert seen["prompt"] == "PROMPT"


def test_response_text_is_unwrapped_from_the_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    assert json.loads(_client(handler).complete("p", timeout=5.0))["verdict"] == "INVALIDATED"


def test_timeout_becomes_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(TimeoutError):
        _client(handler).complete("p", timeout=1.0)


def test_connection_failure_becomes_a_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(TransportError):
        _client(handler).complete("p", timeout=1.0)


def test_http_error_status_becomes_a_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with pytest.raises(TransportError):
        _client(handler).complete("p", timeout=1.0)


def test_envelope_without_a_response_field_becomes_a_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    with pytest.raises(TransportError, match="no 'response' string"):
        _client(handler).complete("p", timeout=1.0)


# -- through the guarded call ---------------------------------------------


def test_the_only_new_failure_surface_is_transport() -> None:
    """The point of phase 5: swapping the client changes nothing downstream."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    outcome = call_validated(
        _client(handler),
        "PROMPT",
        Verdict,
        cfg=Config(),
        cache=NullCache(),
        budget=CallBudget(5),
    )
    assert outcome.failure is FailureKind.TRANSPORT
    assert not outcome.ok


def test_a_timeout_through_the_wrapper_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    outcome = call_validated(
        _client(handler),
        "PROMPT",
        Verdict,
        cfg=Config(),
        cache=NullCache(),
        budget=CallBudget(5),
    )
    assert outcome.failure is FailureKind.TIMEOUT
    assert calls["n"] == 1


def test_a_valid_response_flows_through_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": f"```json\n{json.dumps(GOOD)}\n```"})

    outcome = call_validated(
        _client(handler),
        "PROMPT",
        Verdict,
        cfg=Config(),
        cache=NullCache(),
        budget=CallBudget(5),
    )
    assert outcome.ok
    assert outcome.value is not None and outcome.value.verdict == "INVALIDATED"


def test_client_name_carries_the_model_so_the_cache_is_keyed_by_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "{}"})

    assert _client(handler).name == "ollama:qwen2.5:3b"
