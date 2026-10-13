"""Tests that talk to a real Ollama daemon. Skipped unless asked for.

Everything else about the Ollama client is tested through
`httpx.MockTransport`, which verifies the request shape and the failure
mapping but cannot catch a model that is simply too slow to answer. That gap
was not theoretical: the thinking-model timeout was found by running the tool
by hand, not by the suite.

Run them with:

    DRIFT_LIVE_OLLAMA=1 uv run pytest -m live

They need `ollama serve` and the model pulled. They are excluded by default so
CI and `make test` stay offline and deterministic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from coursedrift.adjudicate import Adjudication, Verdict
from coursedrift.config import Config
from coursedrift.llm.base import CallBudget, call_validated
from coursedrift.llm.cache import NullCache
from coursedrift.llm.ollama import OllamaClient
from coursedrift.llm.prompts import adjudicate_prompt
from coursedrift.models import Route
from coursedrift.pipeline import run_drift

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("DRIFT_LIVE_OLLAMA") != "1",
        reason="set DRIFT_LIVE_OLLAMA=1 and run `ollama serve` to enable",
    ),
]

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = os.environ.get("DRIFT_LIVE_MODEL", "qwen2.5:3b")

PROMPT = adjudicate_prompt(
    stem="Within how many days must notice of leave be given?",
    options=["30 days", "45 days", "14 days", "90 days"],
    correct="30 days",
    section="leave-policy.md: Leave Policy",
    source_old="Notice must be given within 30 days of the intended start date.",
    source_new="Notice must be given within 45 days of the intended start date.",
    diff="Notice must be given within [-30-]{+45+} days of the intended start date.",
    context="Managers must approve every leave request.",
)


@pytest.fixture(scope="module")
def client() -> OllamaClient:
    live = OllamaClient(model=MODEL)
    try:
        live.preflight()
    except RuntimeError as error:  # unreachable, or model not pulled
        pytest.skip(str(error))
    return live


def test_preflight_reports_the_thinking_capability(client: OllamaClient) -> None:
    """Whatever the answer, it must be decided rather than left unknown."""
    assert client.thinking_capable in (True, False)


def test_a_real_adjudication_validates(client: OllamaClient) -> None:
    outcome = call_validated(
        client,
        PROMPT,
        Adjudication,
        cfg=Config(),
        cache=NullCache(),
        budget=CallBudget(4),
    )
    assert outcome.ok, f"live call failed: {outcome.reason}"
    assert outcome.value is not None
    assert outcome.value.verdict in set(Verdict)
    assert 0.0 <= outcome.value.confidence <= 1.0
    assert outcome.value.evidence


def test_a_real_call_fits_inside_the_default_timeout(client: OllamaClient) -> None:
    """The regression for the thinking-model timeout.

    A thinking model spent 1432 tokens on a hidden trace and took 36.6s
    against the 30s default, so every call in a run failed. This asserts the
    whole call, including any model load, fits the budget the tool ships with.
    """
    cfg = Config()
    started = time.monotonic()
    outcome = call_validated(
        client, PROMPT, Adjudication, cfg=cfg, cache=NullCache(), budget=CallBudget(4)
    )
    elapsed = time.monotonic() - started
    assert outcome.ok, f"live call failed: {outcome.reason}"
    assert elapsed < cfg.timeout_s, (
        f"{MODEL} took {elapsed:.1f}s against a {cfg.timeout_s}s budget. "
        "If it is a thinking model, check that `think: false` is being sent."
    )


def test_the_whole_demo_runs_without_a_single_timeout(client: OllamaClient) -> None:
    """End to end against a real model, which is what the fake cannot prove."""
    report = run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        client,
        NullCache(),
        labels=("fixtures/docs_v1", "fixtures/docs_v2"),
    )
    timed_out = [f.item_id for f in report.findings if "TIMEOUT" in " ".join(f.reasons)]
    assert not timed_out, f"timed out on {timed_out}"
    assert len(report.findings) == 14
    assert {f.route for f in report.findings} <= set(Route)
    assert report.warnings == []


def test_a_real_model_notices_the_changed_figure(client: OllamaClient) -> None:
    """Not a correctness guarantee — a smoke test that the task is understood.

    Deliberately weak: it accepts any verdict other than UNAFFECTED, because
    a small model is allowed to hedge or abstain. Asserting a specific verdict
    would make this flaky for no benefit.
    """
    outcome = call_validated(
        client, PROMPT, Adjudication, cfg=Config(), cache=NullCache(), budget=CallBudget(4)
    )
    assert outcome.ok, f"live call failed: {outcome.reason}"
    assert outcome.value is not None
    assert outcome.value.verdict is not Verdict.UNAFFECTED, (
        "a figure the question is keyed to changed from 30 to 45 days; "
        f"{MODEL} called it UNAFFECTED"
    )
