"""Proposing a fix, and refusing to accept a bad one.

Every validator here is a reason to send an item to a human instead of
shipping a patch. A rejected patch is attached to the finding so the reviewer
can see what was attempted — it is evidence, not a draft to be retried until
it slips through.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from coursedrift.adjudicate import AdjudicationResult, PromptMaterial
from coursedrift.config import Config
from coursedrift.impact import ImpactCandidate
from coursedrift.llm import prompts
from coursedrift.llm.base import (
    CallBudget,
    CallOutcome,
    LLMClient,
    Validator,
    call_validated,
)
from coursedrift.llm.cache import Cache
from coursedrift.models import Snapshot
from coursedrift.segment import normalize

_INTERROGATIVES = (
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "can",
    "should",
    "must",
    "may",
    "will",
    "would",
    "identify",
    "select",
    "choose",
    "name",
)


class Patch(BaseModel):
    """The wire schema for a proposed fix."""

    model_config = ConfigDict(extra="forbid")

    stem: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=6)
    correct_index: int = Field(ge=0)
    grounded_in: str = Field(min_length=1)

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def correct_index_in_range(value: Patch) -> str | None:
    if value.correct_index >= len(value.options):
        return (
            f"correct_index {value.correct_index} is outside the "
            f"{len(value.options)} options supplied"
        )
    return None


def options_non_empty(value: Patch) -> str | None:
    for index, option in enumerate(value.options):
        if not option.strip():
            return f"option {index} is empty"
    return None


def options_distinct(value: Patch) -> str | None:
    seen: set[str] = set()
    for option in value.options:
        key = normalize(option).casefold()
        if key in seen:
            return f"options are not distinct: {option[:80]!r} appears more than once"
        seen.add(key)
    return None


def one_correct_option(value: Patch) -> str | None:
    """The keyed answer must appear exactly once among the options.

    With an index-based schema a second copy of the correct text would make
    two options correct while the index silently points at one of them.
    """
    if value.correct_index >= len(value.options):
        return None  # correct_index_in_range reports this more precisely
    target = normalize(value.correct_option).casefold()
    matches = sum(1 for option in value.options if normalize(option).casefold() == target)
    if matches != 1:
        return f"the correct answer appears {matches} times among the options"
    return None


def stem_is_question(value: Patch) -> str | None:
    stem = value.stem.strip()
    if stem.endswith("?"):
        return None
    first = normalize(stem).casefold().split(" ")[0].strip(",:;")
    if first in _INTERROGATIVES:
        return None
    return "stem is not phrased as a question"


def grounded_in_new_document(value: Patch, *, haystack: str) -> str | None:
    """The supporting quote has to exist in the new source material."""
    needle = normalize(value.grounded_in).casefold()
    if not needle:
        return "grounded_in is empty"
    if needle not in haystack:
        return f"grounded_in is not present in the new source: {value.grounded_in[:120]!r}"
    return None


def patch_validators(haystack: str) -> Sequence[Validator[Patch]]:
    """Every check a patch must survive, in reporting order."""
    return [
        correct_index_in_range,
        options_non_empty,
        options_distinct,
        one_correct_option,
        stem_is_question,
        lambda value: grounded_in_new_document(value, haystack=haystack),
    ]


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def remediate(
    candidate: ImpactCandidate,
    adjudication: AdjudicationResult,
    material: PromptMaterial,
    new: Snapshot,
    client: LLMClient,
    cfg: Config,
    cache: Cache,
    budget: CallBudget,
) -> CallOutcome[Patch]:
    """Ask for a fix for one broken item.

    Grounding is checked against SOURCE_NEW and CONTEXT only. The old text is
    deliberately excluded from the haystack: a patch supported by a quote
    from the *previous* version of the document is exactly the failure this
    stage exists to catch.
    """
    item = candidate.item
    haystack = normalize(f"{material.source_new}\n{material.context}").casefold()
    prompt = prompts.remediate_prompt(
        stem=item.stem,
        options=item.options,
        correct=item.correct_answer,
        section=material.section,
        source_old=material.source_old,
        source_new=material.source_new,
        diff=material.diff,
        context=material.context,
        verdict=adjudication.verdict.value if adjudication.verdict else "BROKEN",
        rationale=adjudication.rationale,
    )
    return call_validated(
        client,
        prompt,
        Patch,
        cfg=cfg,
        cache=cache,
        budget=budget,
        validators=patch_validators(haystack),
    )
