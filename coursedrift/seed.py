"""Generating a course with provenance recorded as it goes.

This closes the loop: it makes the whole system demonstrable on a reader's
own documents rather than only on the bundled fixtures. It is last in the
build order because nothing else depends on it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coursedrift.anchor import make_ref
from coursedrift.config import Config
from coursedrift.corpus import load_snapshot
from coursedrift.llm import prompts
from coursedrift.llm.base import CallBudget, LLMClient, call_validated
from coursedrift.llm.cache import Cache
from coursedrift.models import Course, CourseItem, Span
from coursedrift.progress import NullProgress, Progress
from coursedrift.remediate import Patch, patch_validators
from coursedrift.segment import normalize

MIN_WORDS = 8


class SeedReport(BaseModel):
    """What seeding produced, and what it could not."""

    course: Course
    attempted: int = 0
    rejected: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> int:
        return len(self.course.items)


def candidate_spans(docs: Path, limit: int) -> list[Span]:
    """Pick spans substantial enough to carry a question.

    Round-robin across documents so a long handbook cannot crowd out every
    other file, and skip code fences and table rows: they are kept whole by
    segmentation precisely because they are not prose.
    """
    snapshot = load_snapshot(docs)
    per_doc: dict[str, list[Span]] = {}
    for span in snapshot.all_spans():
        text = span.text.strip()
        if text.startswith(("```", "~~~", "|")):
            continue
        if len(span.norm_text.split()) < MIN_WORDS:
            continue
        per_doc.setdefault(span.doc_id, []).append(span)

    picked: list[Span] = []
    position = 0
    while len(picked) < limit:
        added = False
        for doc_id in sorted(per_doc):
            spans = per_doc[doc_id]
            if position < len(spans):
                picked.append(spans[position])
                added = True
                if len(picked) == limit:
                    return picked
        if not added:
            break
        position += 1
    return picked


def _level_for(span: Span) -> str:
    """One level per document, named after the file."""
    return Path(span.doc_id).stem


def seed_course(
    docs: Path,
    client: LLMClient,
    cfg: Config,
    cache: Cache,
    *,
    course_id: str = "seeded-course",
    title: str = "Seeded Course",
    max_items: int = 12,
    progress: Progress | None = None,
) -> SeedReport:
    """Write one question per selected span, recording where it came from.

    Generated questions go through the same validators a remediation patch
    does, including the groundedness check against the span they were written
    from. A question the model could not ground is dropped and reported, not
    silently included — a course seeded with ungrounded questions would make
    every later drift verdict meaningless.
    """
    budget = CallBudget(cfg.max_calls)
    report = SeedReport(course=Course(course_id=course_id, title=title))
    status = progress or NullProgress()

    spans = candidate_spans(docs, max_items)
    status.stage(f"Writing {len(spans)} question(s) using {client.name}")

    for index, span in enumerate(spans, start=1):
        status.working(f"[{index}/{len(spans)}] {span.span_id} - asking the model...")
        section = " > ".join(span.section_path) or "(document root)"
        prompt = prompts.seed_prompt(source=span.text, section=f"{span.doc_id}: {section}")
        haystack = normalize(span.text).casefold()

        report.attempted += 1
        outcome = call_validated(
            client,
            prompt,
            Patch,
            cfg=cfg,
            cache=cache,
            budget=budget,
            validators=patch_validators(haystack),
        )
        if not outcome.ok or outcome.value is None:
            report.rejected.append(f"{span.span_id}: {outcome.reason}")
            status.done(f"[{index}/{len(spans)}] dropped - {outcome.reason}")
            continue
        status.done(f"[{index}/{len(spans)}] q{index:02d} written")

        written = outcome.value
        report.course.items.append(
            CourseItem(
                item_id=f"q{index:02d}",
                level_id=_level_for(span),
                stem=written.stem,
                options=written.options,
                correct_index=written.correct_index,
                provenance=[make_ref(span)],
            )
        )

    return report
