"""Asking a model whether a changed span breaks a question.

The response is a closed enum with an explicit abstention, and every quote it
offers as evidence is checked against the text we supplied. That check is the
cheap half of the defence against both hallucination and prompt injection.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from coursedrift.config import Config
from coursedrift.diff import SpanChange
from coursedrift.impact import ImpactCandidate
from coursedrift.llm import prompts
from coursedrift.llm.base import (
    CallBudget,
    CallOutcome,
    FailureKind,
    LLMClient,
    call_validated,
)
from coursedrift.llm.cache import Cache
from coursedrift.models import Snapshot
from coursedrift.segment import normalize


class Verdict(StrEnum):
    UNAFFECTED = "UNAFFECTED"
    WEAKENED = "WEAKENED"
    INVALIDATED = "INVALIDATED"
    UNCERTAIN = "UNCERTAIN"


BROKEN = (Verdict.WEAKENED, Verdict.INVALIDATED)


class Adjudication(BaseModel):
    """The wire schema for one verdict.

    `extra="forbid"` is deliberate: a model that invents a field has not
    followed the contract, and inventing fields correlates with inventing
    content.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, max_length=4)


class AdjudicationResult(BaseModel):
    """One item's adjudication, across however many votes were cast."""

    outcomes: list[CallOutcome[Adjudication]] = Field(default_factory=list)
    verdict: Verdict | None = None
    confidence: float = 0.0
    unanimous: bool = True
    prompt: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None

    @property
    def succeeded(self) -> list[Adjudication]:
        return [o.value for o in self.outcomes if o.ok and o.value is not None]

    @property
    def failure(self) -> FailureKind | None:
        for outcome in self.outcomes:
            if not outcome.ok:
                return outcome.failure
        return None

    @property
    def reason(self) -> str:
        for outcome in self.outcomes:
            if not outcome.ok:
                return outcome.reason
        return "ok"

    @property
    def rationale(self) -> str:
        chosen = self.succeeded
        return chosen[0].rationale if chosen else ""

    @property
    def evidence(self) -> list[str]:
        chosen = self.succeeded
        return list(chosen[0].evidence) if chosen else []

    @property
    def calls(self) -> int:
        return sum(o.attempts for o in self.outcomes)

    @property
    def cache_hits(self) -> int:
        return sum(1 for o in self.outcomes if o.cached)


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


def _haystack(source_old: str, source_new: str, diff: str) -> str:
    return normalize(f"{source_old}\n{source_new}\n{diff}").casefold()


def evidence_is_grounded(value: Adjudication, *, haystack: str) -> str | None:
    """Every evidence quote must appear in the text we supplied.

    Compared after whitespace normalization and case folding: a truthful
    quote should not be rejected for reflowing a line or changing a leading
    capital, but it does have to be the same words in the same order.
    """
    for quote in value.evidence:
        needle = normalize(quote).casefold()
        if not needle:
            return "evidence contains an empty quote"
        if needle not in haystack:
            return f"evidence quote is not present in the source material: {quote[:120]!r}"
    return None


# ---------------------------------------------------------------------------
# Prompt material
# ---------------------------------------------------------------------------


def section_context(new: Snapshot, change: SpanChange, limit: int = 6) -> str:
    """Neighbouring text from the new document, for judgment in context."""
    span = change.new
    if span is None:
        return ""
    document = new.docs.get(span.doc_id)
    if document is None:
        return ""
    siblings = [s for s in document.spans if s.section_path == span.section_path]
    if not siblings:
        return ""
    index = next((i for i, s in enumerate(siblings) if s.span_id == span.span_id), 0)
    start = max(0, index - limit // 2)
    window = siblings[start : start + limit]
    return "\n".join(s.text for s in window)


class PromptMaterial(BaseModel):
    """The text blocks describing what changed under one course item."""

    section: str
    source_old: str
    source_new: str
    diff: str
    context: str

    @property
    def haystack(self) -> str:
        return _haystack(self.source_old, self.source_new, self.diff)


def gather(candidate: ImpactCandidate, new: Snapshot) -> PromptMaterial:
    """Assemble one item's changed spans into prompt blocks.

    One call per item, not one per span: a question whose two sources both
    changed has to be judged on both at once, and splitting it would ask the
    model a question it cannot answer.
    """
    changes = candidate.adjudicable()
    sections: list[str] = []
    olds: list[str] = []
    news: list[str] = []
    diffs: list[str] = []
    contexts: list[str] = []

    for change in changes:
        anchor = change.new or change.old
        if anchor is not None:
            path = " > ".join(anchor.section_path) or "(document root)"
            label = f"{anchor.doc_id}: {path}"
            if label not in sections:
                sections.append(label)
        if change.old is not None:
            olds.append(change.old.text)
        if change.new is not None:
            news.append(change.new.text)
        if change.inline_diff:
            note = ""
            if change.reordered:
                # The model is the only thing that can tell a harmless
                # rearrangement from a reversed instruction, so say plainly
                # that this is what it is looking at.
                note = (
                    "  (NOTE: the new text uses exactly the same words as the old one, "
                    "in a different order. Check whether the meaning has been reversed.)"
                )
            diffs.append(change.inline_diff + note)
        elif change.old is not None and change.new is not None:
            # An identical-but-reparented span has no textual diff; say so
            # rather than leaving the block empty.
            diffs.append(
                f"(text unchanged; moved from "
                f"{' > '.join(change.old.section_path) or 'root'} to "
                f"{' > '.join(change.new.section_path) or 'root'})"
            )
        snippet = section_context(new, change)
        if snippet and snippet not in contexts:
            contexts.append(snippet)

    return PromptMaterial(
        section="\n".join(sections),
        source_old="\n".join(olds),
        source_new="\n".join(news),
        diff="\n".join(diffs),
        context="\n\n".join(contexts),
    )


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def adjudicate(
    candidate: ImpactCandidate,
    new: Snapshot,
    client: LLMClient,
    cfg: Config,
    cache: Cache,
    budget: CallBudget,
) -> AdjudicationResult:
    """Judge one item, casting `cfg.votes` independent votes.

    Any disagreement between votes — including a vote that failed outright —
    leaves `unanimous` False, which triage reads as a review trigger. A
    failed vote is disagreement, not an abstention to be quietly dropped.
    """
    material = gather(candidate, new)
    item = candidate.item
    haystack = material.haystack

    def validator(value: Adjudication) -> str | None:
        return evidence_is_grounded(value, haystack=haystack)

    outcomes: list[CallOutcome[Adjudication]] = []
    first_prompt = ""
    for vote in range(cfg.votes):
        prompt = prompts.adjudicate_prompt(
            stem=item.stem,
            options=item.options,
            correct=item.correct_answer,
            section=material.section,
            source_old=material.source_old,
            source_new=material.source_new,
            diff=material.diff,
            context=material.context,
            vote=vote if cfg.votes > 1 else None,
        )
        if not first_prompt:
            first_prompt = prompt
        outcomes.append(
            call_validated(
                client,
                prompt,
                Adjudication,
                cfg=cfg,
                cache=cache,
                budget=budget,
                validators=[validator],
            )
        )

    values = [o.value for o in outcomes if o.ok and o.value is not None]
    verdicts = [v.verdict for v in values]
    unanimous = bool(values) and len(set(verdicts)) == 1 and len(values) == len(outcomes)

    return AdjudicationResult(
        outcomes=outcomes,
        verdict=verdicts[0] if verdicts else None,
        # The lowest confidence across votes, not the mean: one doubtful vote
        # should pull the item toward review, not be averaged away.
        confidence=min((v.confidence for v in values), default=0.0),
        unanimous=unanimous,
        prompt=first_prompt,
    )
