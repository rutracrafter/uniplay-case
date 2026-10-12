"""Wiring the stages together.

Phase 4 contributes the deterministic half: load, match, anchor, narrow. The
model-mediated half is added in phase 6 and the report in phase 7.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coursedrift.anchor import AnchorReport, resolve_course
from coursedrift.config import Config
from coursedrift.corpus import diff_manifests, load_course, load_snapshot
from coursedrift.diff import MatchResult, match_corpus
from coursedrift.impact import ImpactCandidate, SkipStats, build_candidates
from coursedrift.models import Course, ManifestDiff, Snapshot


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
    anchors = resolve_course(course, old, new, cfg)
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
