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
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        return httpx.Response(200, json={"response": "ok"})

    _client(handler).preflight()
    assert paths == ["/api/tags", "/api/generate"], "preflight checks, then warms"


def test_preflight_can_skip_the_warm_up() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})

    _client(handler).preflight(warm=False)
    assert paths == ["/api/tags"]


def test_a_failed_warm_up_is_not_fatal() -> None:
    """The first real call will report whatever is wrong, properly."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        raise httpx.ConnectError("gone away")

    _client(handler).preflight()  # must not raise


# -- thinking models -------------------------------------------------------
#
# The bug these cover: a thinking model spends almost all of its generation on
# a trace returned in a separate `thinking` field, which this client discards.
# Measured on qwen3.5:9b, one adjudication prompt took 1432 tokens and 36.6s
# against a 30s timeout; with thinking disabled, 78 tokens and 1.7s for the
# same verdict.


def _tags(*models: dict[str, object]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": list(models)})
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    return handler


def test_preflight_records_a_thinking_capability() -> None:
    client = _client(_tags({"name": "qwen3.5:9b", "capabilities": ["completion", "thinking"]}))
    client.model = "qwen3.5:9b"
    client.preflight()
    assert client.thinking_capable is True


def test_preflight_records_the_absence_of_thinking() -> None:
    client = _client(_tags({"name": "qwen2.5:3b", "capabilities": ["completion", "tools"]}))
    client.preflight()
    assert client.thinking_capable is False


def test_thinking_is_disabled_for_a_thinking_model() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3.5:9b", "capabilities": ["thinking"]}]},
            )
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    client = _client(handler)
    client.model = "qwen3.5:9b"
    client.preflight()
    client.complete("PROMPT", timeout=5.0)
    assert bodies[-1]["think"] is False


def test_the_think_field_is_not_sent_to_a_model_without_the_capability() -> None:
    """Gated rather than always-on, so this does not depend on how leniently a
    given Ollama version treats an inapplicable field."""
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    client = _client(handler)
    client.preflight()
    client.complete("PROMPT", timeout=5.0)
    assert "think" not in bodies[-1]


def test_generation_is_capped() -> None:
    """A model that starts repeating itself must not run to the context limit."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(
        model="qwen2.5:3b", http=httpx.Client(transport=transport), max_output_tokens=256
    )
    client.complete("PROMPT", timeout=5.0)
    assert seen["options"] == {"temperature": 0, "seed": 0, "num_predict": 256}


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
    assert seen["options"] == {"temperature": 0, "seed": 0, "num_predict": 1024}
    assert seen["prompt"] == "PROMPT"


def test_response_text_is_unwrapped_from_the_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": json.dumps(GOOD)})

    assert json.loads(_client(handler).complete("p", timeout=5.0))["verdict"] == "INVALIDATED"


def test_timeout_becomes_a_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(TimeoutError, match="--timeout"):
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
