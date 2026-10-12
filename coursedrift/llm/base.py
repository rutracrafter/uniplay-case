"""The one place where model output becomes trusted data.

Everything here is hostile by default. `call_validated` never raises: every
caller receives a `CallOutcome` it has to inspect, which is what makes "fail
closed to NEEDS_REVIEW" a consequence of the type signature rather than a
discipline someone has to remember.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from coursedrift.config import Config
from coursedrift.llm.cache import Cache

T = TypeVar("T", bound=BaseModel)

Validator = Callable[[T], str | None]
"""A check on a parsed response. Returns an error message, or None to pass."""


class TransportError(RuntimeError):
    """A client could not reach its model. Distinct from a timeout."""


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, *, timeout: float) -> str: ...


class FailureKind(StrEnum):
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    MALFORMED = "MALFORMED"
    SCHEMA = "SCHEMA"
    REJECTED = "REJECTED"
    BUDGET = "BUDGET"


class CallOutcome(BaseModel, Generic[T]):
    """The result of one guarded call. Never an exception."""

    ok: bool
    value: T | None = None
    failure: FailureKind | None = None
    detail: str | None = None
    attempts: int = 0
    raw: list[str] = Field(default_factory=list)
    cached: bool = False

    @property
    def reason(self) -> str:
        """A short, reportable explanation of why this call did not succeed."""
        if self.ok:
            return "ok"
        kind = self.failure.value if self.failure else "UNKNOWN"
        return f"{kind}: {self.detail}" if self.detail else kind


class CallBudget:
    """A whole-run ceiling on model calls, including repair attempts."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self._spent = 0

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self._spent)

    def spend(self) -> bool:
        """Claim one call. False when the budget is exhausted."""
        if self._spent >= self.max_calls:
            return False
        self._spent += 1
        return True


# ---------------------------------------------------------------------------
# Tolerant extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(raw: str) -> str:
    match = _FENCE.search(raw)
    return match.group(1) if match else raw


def _balanced_object(text: str, start: int) -> str | None:
    """Scan one balanced `{...}`, ignoring braces inside string literals."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(raw: str) -> str | None:
    """Pull the first JSON object out of a response.

    Tolerant on purpose: models wrap JSON in fences and bracket it with
    commentary, and rejecting that would mean discarding answers that are
    perfectly good. Tolerance ends at the object boundary — what comes out
    still has to satisfy the schema.
    """
    text = _strip_fences(raw)
    start = text.find("{")
    while start != -1:
        candidate = _balanced_object(text, start)
        if candidate is not None:
            return candidate
        start = text.find("{", start + 1)
    return None


def _short(error: ValidationError, limit: int = 300) -> str:
    parts = []
    for item in error.errors()[:3]:
        location = ".".join(str(piece) for piece in item["loc"]) or "(root)"
        parts.append(f"{location}: {item['msg']}")
    joined = "; ".join(parts)
    return joined[:limit]


def parse_and_validate(
    raw: str, schema: type[T], validators: Sequence[Validator[T]] = ()
) -> CallOutcome[T]:
    """Extract, parse, then run every validator in order."""
    payload = extract_json(raw)
    if payload is None:
        return CallOutcome[T](
            ok=False, failure=FailureKind.MALFORMED, detail="no JSON object in response"
        )
    try:
        value = schema.model_validate_json(payload)
    except ValidationError as error:
        return CallOutcome[T](ok=False, failure=FailureKind.SCHEMA, detail=_short(error))

    for validator in validators:
        message = validator(value)
        if message:
            return CallOutcome[T](ok=False, failure=FailureKind.REJECTED, detail=message)
    return CallOutcome[T](ok=True, value=value)


def repair_prompt(original: str, bad_output: str, error: str) -> str:
    """Re-ask once, naming what was wrong with the previous answer."""
    return (
        f"{original}\n\n"
        "Your previous answer was rejected and must be corrected.\n"
        f"Error: {error}\n"
        f"Previous answer: {bad_output.strip()[:800]}\n\n"
        "Reply with corrected JSON only. No commentary, no code fence."
    )


# ---------------------------------------------------------------------------
# The guarded call
# ---------------------------------------------------------------------------

_RETRYABLE = (FailureKind.MALFORMED, FailureKind.SCHEMA, FailureKind.REJECTED)


def call_validated(
    client: LLMClient,
    prompt: str,
    schema: type[T],
    *,
    cfg: Config,
    cache: Cache,
    budget: CallBudget,
    validators: Sequence[Validator[T]] = (),
) -> CallOutcome[T]:
    """Call a model and return something safe to use, or an honest failure.

    Order of operations:
      1. cache lookup, replayed through the same validation path
      2. budget check
      3. the call, with timeouts and transport errors caught
      4. tolerant extraction, strict parse, then every validator
      5. on a retryable failure, one repair attempt naming the error
      6. only successes are cached

    Timeouts and transport errors are not retried. The plan lists exactly
    three retryable failures, and re-asking a model that did not answer in
    time mostly buys a second timeout.
    """
    key = cache.key(client.name, prompt)
    cached_raw = cache.get(key)
    if cached_raw is not None:
        replay = parse_and_validate(cached_raw, schema, validators)
        if replay.ok:
            return replay.model_copy(update={"cached": True, "raw": [cached_raw]})
        # A cached entry that no longer validates is poison, not a result.
        cache.drop(key)

    attempt_prompt = prompt
    transcript: list[str] = []
    last: CallOutcome[T] | None = None

    for attempt in range(1, cfg.max_attempts + 1):
        if not budget.spend():
            return CallOutcome[T](
                ok=False,
                failure=FailureKind.BUDGET,
                detail=f"run budget of {budget.max_calls} model calls is exhausted",
                attempts=attempt - 1,
                raw=transcript,
            )
        try:
            raw = client.complete(attempt_prompt, timeout=cfg.timeout_s)
        except TimeoutError as error:
            return CallOutcome[T](
                ok=False,
                failure=FailureKind.TIMEOUT,
                detail=str(error) or f"no response within {cfg.timeout_s}s",
                attempts=attempt,
                raw=transcript,
            )
        except TransportError as error:
            return CallOutcome[T](
                ok=False,
                failure=FailureKind.TRANSPORT,
                detail=str(error),
                attempts=attempt,
                raw=transcript,
            )

        transcript.append(raw)
        parsed = parse_and_validate(raw, schema, validators)
        if parsed.ok:
            cache.put(key, raw)
            return parsed.model_copy(update={"attempts": attempt, "raw": transcript})

        last = parsed
        if parsed.failure not in _RETRYABLE:
            break
        attempt_prompt = repair_prompt(prompt, raw, parsed.detail or "invalid output")

    assert last is not None
    return CallOutcome[T](
        ok=False,
        failure=last.failure,
        detail=last.detail,
        attempts=len(transcript),
        raw=transcript,
    )
