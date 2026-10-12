"""Driving a small local model through Ollama.

Nothing else in the pipeline changes when this replaces the fake client. If
phase 5 did its job, the only new failure surface is TRANSPORT.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from coursedrift.llm.base import TransportError

DEFAULT_HOST = "http://localhost:11434"

WARMUP_TIMEOUT = 180.0
"""Generous: loading a large model from cold is slow, and it happens once."""


class OllamaModelMissing(RuntimeError):
    """The daemon is reachable but the requested model is not pulled."""


class OllamaUnreachable(RuntimeError):
    """No daemon answered at the configured host."""


class OllamaClient:
    """A minimal `/api/generate` client.

    Deterministic by configuration: temperature 0 and a fixed seed, so reruns
    agree with each other and with the response cache.
    """

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        http: httpx.Client | None = None,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"
        self.max_output_tokens = max_output_tokens
        self._http = http or httpx.Client()
        self.thinking_capable: bool | None = None
        """Whether the model emits a hidden reasoning trace. None until preflight."""

    # -- startup checks --------------------------------------------------

    def preflight(self, warm: bool = True) -> None:
        """Fail early, with the command that fixes it.

        A stack trace four stages into a run is a worse way to learn that
        Ollama is not running, so this is called once before any work. It also
        records whether the model thinks, and loads it, for the reasons below.
        """
        try:
            response = self._http.get(f"{self.host}/api/tags", timeout=5.0)
            response.raise_for_status()
            payload: Any = response.json()
        except httpx.HTTPError as error:
            raise OllamaUnreachable(
                f"No Ollama daemon answered at {self.host} ({error}).\n"
                "Start it with `ollama serve`, or pass --client fake to run "
                "without a model."
            ) from error

        entries: list[Any] = [entry for entry in payload.get("models", []) if entry]
        available = {str(entry.get("name", "")) for entry in entries}
        if not self._has_model(available):
            listed = ", ".join(sorted(available)) or "none"
            raise OllamaModelMissing(
                f"Ollama has no model named {self.model!r}.\n"
                f"Pull it with `ollama pull {self.model}`.\n"
                f"Models currently available: {listed}"
            )

        self.thinking_capable = self._advertises_thinking(entries)
        if warm:
            self._warm()

    def _matches(self, name: str) -> bool:
        """Whether an Ollama model name refers to the model we asked for."""
        if not name:
            return False
        if name in (self.model, f"{self.model}:latest"):
            return True
        return name.split(":", 1)[0] == self.model.split(":", 1)[0]

    def _has_model(self, available: set[str]) -> bool:
        """Match with or without an explicit tag, as Ollama's CLI does."""
        return any(self._matches(name) for name in available)

    def _advertises_thinking(self, entries: list[Any]) -> bool:
        """Whether the matched model emits a hidden reasoning trace.

        This matters far more than it sounds. A thinking model spends most of
        its generation on a trace that `/api/generate` returns in a separate
        `thinking` field, which this client discards.

        Measured on qwen3.5:9b with one adjudication prompt: 1432 generated
        tokens, of which roughly 90 were the answer and 5655 characters were
        reasoning we paid for and threw away. 30.0s of generation plus 5.8s of
        model load, against a 30s per-call timeout — so every call failed. The
        same prompt with thinking disabled takes 1.7s and returns the same
        verdict at the same confidence.
        """
        for entry in entries:
            if self._matches(str(entry.get("name", ""))):
                return "thinking" in (entry.get("capabilities") or [])
        return False

    def _warm(self) -> None:
        """Load the model now, so the first real call is not billed for it.

        Ollama reports model load separately from generation, and it can take
        seconds for a large model. Charging it to a per-call timeout means the
        first item of a run fails for a reason that has nothing to do with
        that item.
        """
        body: dict[str, Any] = {"model": self.model, "prompt": "ok", "stream": False}
        if self.thinking_capable:
            body["think"] = False
        try:
            self._http.post(f"{self.host}/api/generate", json=body, timeout=WARMUP_TIMEOUT)
        except httpx.HTTPError:
            # A failed warm-up is not fatal. The first real call will report
            # whatever is wrong, through the normal failure path.
            return

    # -- the client contract ---------------------------------------------

    def _payload(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": 0,
                # A ceiling, not a target. Responses run to about 90 tokens;
                # this stops a model that starts repeating itself from
                # generating until the context window is exhausted, which
                # turns one bad answer into one timeout.
                "num_predict": self.max_output_tokens,
            },
        }
        # Sent only when the model advertises the capability. It is harmless
        # on models that do not, but gating it keeps this from depending on
        # how leniently a given Ollama version treats an inapplicable field.
        if self.thinking_capable:
            body["think"] = False
        return body

    def complete(self, prompt: str, *, timeout: float) -> str:
        try:
            response = self._http.post(
                f"{self.host}/api/generate",
                json=self._payload(prompt),
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise TimeoutError(
                f"Ollama did not respond within {timeout}s. "
                "Raise it with --timeout, or use a smaller model"
            ) from error
        except httpx.HTTPError as error:
            raise TransportError(f"Ollama request failed: {error}") from error

        try:
            payload = response.json()
        except json.JSONDecodeError as error:
            raise TransportError(f"Ollama returned a non-JSON envelope: {error}") from error

        text = payload.get("response")
        if not isinstance(text, str):
            raise TransportError("Ollama response envelope had no 'response' string")
        return text
