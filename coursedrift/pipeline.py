"""Wiring the stages together.

Phase 4 contributes the deterministic half: load, match, anchor, narrow. The
model-mediated half is added in phase 6 and the report in phase 7.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coursedrift.adjudicate import BROKEN, adjudicate, gather
from coursedrift.anchor import AnchorReport, resolve_course
from coursedrift.config import Config
from coursedrift.corpus import diff_manifests, load_course, load_snapshot
from coursedrift.diff import MatchResult, match_corpus
from coursedrift.impact import ImpactCandidate, SkipStats, build_candidates
from coursedrift.llm.base import CallBudget, CallOutcome, LLMClient
from coursedrift.llm.cache import Cache
from coursedrift.llm.fake import FakeClient
from coursedrift.models import Course, ManifestDiff, Snapshot
from coursedrift.remediate import Patch, remediate
from coursedrift.report import RunReport, sort_findings
from coursedrift.triage import Finding, route_candidate


class Prepared(BaseModel):
    """Everything the deterministic stages produce, before any model call."""

    old: Snapshot
    new: Snapshot
    course: Course
    manifest: ManifestDiff
    matches: MatchResult
    anchors: AnchorReport
    candidates: list[ImpactCandidate] = Field(default_factory=list)
    stats: SkipStats = Field(default_factory=SkipStats)

    def queued(self) -> list[ImpactCandidate]:
        return [c for c in self.candidates if c.needs_model]


def prepare(old_dir: Path, new_dir: Path, course_path: Path, cfg: Config) -> Prepared:
    """Run stages 1 through 6: no model is involved and none is needed."""
    old = load_snapshot(old_dir)
    new = load_snapshot(new_dir)
    course = load_course(course_path)

    manifest = diff_manifests(old, new)
    matches = match_corpus(old, new, cfg, manifest)
    anchors = resolve_course(course, old, new, cfg, matches=matches)
    candidates, stats = build_candidates(course, matches, anchors, cfg)

    return Prepared(
        old=old,
        new=new,
        course=course,
        manifest=manifest,
        matches=matches,
        anchors=anchors,
        candidates=candidates,
        stats=stats,
    )


def judge_candidate(
    candidate: ImpactCandidate,
    new: Snapshot,
    client: LLMClient,
    cfg: Config,
    cache: Cache,
    budget: CallBudget,
) -> Finding:
    """Stages 7 to 9 for one item: adjudicate, maybe remediate, route.

    Remediation is only attempted for a broken verdict that already clears
    the confidence floor, so no fix is ever paid for on an item that triage
    would send to review regardless.
    """
    if candidate.preset_route is not None:
        return route_candidate(candidate, None, None, cfg)

    adjudication = adjudicate(candidate, new, client, cfg, cache, budget)

    patch_outcome: CallOutcome[Patch] | None = None
    worth_fixing = (
        adjudication.ok
        and adjudication.verdict in BROKEN
        and adjudication.confidence >= cfg.confidence_floor
        and adjudication.unanimous
    )
    if worth_fixing:
        patch_outcome = remediate(
            candidate,
            adjudication,
            gather(candidate, new),
            new,
            client,
            cfg,
            cache,
            budget,
        )

    return route_candidate(candidate, adjudication, patch_outcome, cfg)


def judge_all(
    prepared: Prepared,
    client: LLMClient,
    cfg: Config,
    cache: Cache,
    budget: CallBudget,
) -> list[Finding]:
    """Judge every candidate, in a stable order."""
    return [
        judge_candidate(candidate, prepared.new, client, cfg, cache, budget)
        for candidate in prepared.candidates
    ]


def build_client(name: str, model: str, failures: list[str] | None = None) -> LLMClient:
    """Resolve a client name to an implementation."""
    if name == "fake":
        return FakeClient(failures=failures)
    if name == "ollama":
        from coursedrift.llm.ollama import OllamaClient

        ollama = OllamaClient(model=model)
        ollama.preflight()
        return ollama
    raise ValueError(f"unknown client {name!r}; expected 'fake', 'ollama' or 'none'")


def run_drift(
    old_dir: Path,
    new_dir: Path,
    course_path: Path,
    cfg: Config,
    client: LLMClient,
    cache: Cache,
    *,
    labels: tuple[str, str] | None = None,
) -> RunReport:
    """The whole pipeline, stages 1 through 10.

    `labels` overrides how the two snapshot roots are named in the report.
    The demo passes repo-relative names so its output does not embed an
    absolute path, which would make the golden report machine-specific.
    """
    prepared = prepare(old_dir, new_dir, course_path, cfg)
    budget = CallBudget(cfg.max_calls)
    findings = sort_findings(judge_all(prepared, client, cfg, cache, budget))

    return RunReport(
        course_id=prepared.course.course_id,
        course_title=prepared.course.title,
        old_root=labels[0] if labels else str(old_dir),
        new_root=labels[1] if labels else str(new_dir),
        client=client.name,
        config=cfg,
        stats=prepared.stats,
        findings=findings,
        documents={
            "unchanged": sorted(prepared.manifest.unchanged),
            "modified": sorted(prepared.manifest.modified),
            "added": sorted(prepared.manifest.added),
            "deleted": sorted(prepared.manifest.deleted),
        },
        span_counts=prepared.matches.counts(),
        section_counts=prepared.matches.sections.counts(),
        model_calls=sum(f.model_calls for f in findings),
        cache_hits=sum(f.cache_hits for f in findings),
    )
