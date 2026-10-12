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
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"
        self._http = http or httpx.Client()

    # -- startup checks --------------------------------------------------

    def preflight(self) -> None:
        """Fail early, with the command that fixes it.

        A stack trace four stages into a run is a worse way to learn that
        Ollama is not running, so this is called once before any work.
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

        available = {str(entry.get("name", "")) for entry in payload.get("models", []) if entry}
        if not self._has_model(available):
            listed = ", ".join(sorted(available)) or "none"
            raise OllamaModelMissing(
                f"Ollama has no model named {self.model!r}.\n"
                f"Pull it with `ollama pull {self.model}`.\n"
                f"Models currently available: {listed}"
            )

    def _has_model(self, available: set[str]) -> bool:
        """Match with or without an explicit tag, as Ollama's CLI does."""
        wanted = self.model
        candidates = {wanted, f"{wanted}:latest"}
        if ":" in wanted:
            candidates.add(wanted.split(":", 1)[0])
        return bool(candidates & available) or any(
            name.split(":", 1)[0] == wanted.split(":", 1)[0] for name in available if name
        )

    # -- the client contract ---------------------------------------------

    def complete(self, prompt: str, *, timeout: float) -> str:
        try:
            response = self._http.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "seed": 0},
                },
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise TimeoutError(f"Ollama did not respond within {timeout}s") from error
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
