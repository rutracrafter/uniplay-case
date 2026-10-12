"""Rendering a run as JSON for machines and Markdown for the person approving it.

The Markdown report leads with the summary a reviewer needs before any
detail, and every finding states why it routed where it did. A reviewer
should never have to take a route on faith.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from coursedrift.anchor import ResolvedAnchor
from coursedrift.config import Config
from coursedrift.diff import ContentOp, PositionOp, SpanChange
from coursedrift.impact import SkipStats
from coursedrift.models import Route
from coursedrift.triage import Finding

ROUTE_ORDER = [
    Route.RETIRE_SUGGESTED,
    Route.NEEDS_REVIEW,
    Route.AUTO_FIX_PROPOSED,
    Route.OK,
]


class RunReport(BaseModel):
    """One complete run, ready to render."""

    course_id: str
    course_title: str
    old_root: str
    new_root: str
    client: str
    config: Config
    stats: SkipStats
    findings: list[Finding] = Field(default_factory=list)

    documents: dict[str, list[str]] = Field(default_factory=dict)
    span_counts: dict[str, int] = Field(default_factory=dict)
    section_counts: dict[str, int] = Field(default_factory=dict)
    model_calls: int = 0
    cache_hits: int = 0

    # Set after the run. Deliberately absent from the Markdown body so the
    # golden comparison stays meaningful.
    generated_at: str | None = None

    def by_route(self) -> dict[Route, list[Finding]]:
        grouped: dict[Route, list[Finding]] = {route: [] for route in ROUTE_ORDER}
        for finding in self.findings:
            grouped[finding.route].append(finding)
        return grouped

    def route_counts(self) -> dict[Route, int]:
        return {route: len(items) for route, items in self.by_route().items()}

    def levels(self) -> dict[str, dict[Route, int]]:
        out: dict[str, dict[Route, int]] = {}
        for finding in self.findings:
            bucket = out.setdefault(finding.level_id, dict.fromkeys(ROUTE_ORDER, 0))
            bucket[finding.route] += 1
        return out

    def staleness(self, level_id: str) -> int:
        """Share of a level's items that are not plainly fine, as a percentage."""
        counts = self.levels().get(level_id, {})
        total = sum(counts.values())
        if not total:
            return 0
        return round(100 * (total - counts.get(Route.OK, 0)) / total)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Deterministic order: severity, then document, then position, then id.

    Stable ordering is what makes the golden report a real test rather than a
    snapshot of whatever dict order happened to apply.
    """

    def key(finding: Finding) -> tuple[int, str, int, str]:
        spans = [c.new or c.old for c in finding.changes]
        docs = sorted(s.doc_id for s in spans if s is not None)
        orders = sorted(s.order for s in spans if s is not None)
        return (
            finding.route.severity,
            docs[0] if docs else "",
            orders[0] if orders else 0,
            finding.item_id,
        )

    return sorted(findings, key=key)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def render_json(run: RunReport) -> str:
    return run.model_dump_json(indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _options(options: list[str], correct_index: int) -> list[str]:
    return [
        f"  {'*' if index == correct_index else '-'} {option}"
        for index, option in enumerate(options)
    ]


def _describe_change(change: SpanChange) -> list[str]:
    span = change.new or change.old
    assert span is not None
    path = " > ".join(span.section_path) or "(document root)"
    where = f"`{span.doc_id}` &rarr; {path}"
    if change.across_docs and change.old is not None:
        where += f" (moved from `{change.old.doc_id}`)"
    lines = [f"  - {change.content.value} / {change.position.value} in {where}"]
    if change.position is PositionOp.REPARENTED and change.old is not None:
        was = " > ".join(change.old.section_path) or "(document root)"
        lines.append(f"    - was under: {was}")
    if change.inline_diff:
        lines.append(f"    - diff: `{change.inline_diff}`")
    elif change.content is ContentOp.REMOVED and change.old is not None:
        lines.append(f"    - removed text: {change.old.text!r}")
    if change.similarity is not None and change.content is ContentOp.MODIFIED:
        lines.append(f"    - text similarity: {change.similarity:.2f}")
    if change.reordered:
        lines.append(
            "    - **same words, different order** - this can reverse the meaning "
            "while leaving every word in place"
        )
        if change.pair_score is not None:
            lines.append(
                f"    - vocabulary overlap: {change.pair_score:.2f} "
                "(high overlap with lower text similarity is the signature of a rearrangement)"
            )
    return lines


def _describe_anchor(anchor: ResolvedAnchor) -> str:
    extra = ""
    if anchor.ref.inferred:
        score = anchor.ref.infer_score
        extra = f", inferred (BM25 {score:.2f})" if score is not None else ", inferred"
    if anchor.status.value == "AMBIGUOUS":
        extra += f", {len(anchor.candidates)} candidates"
    return f"  - `{anchor.ref.span_id}` &rarr; {anchor.status.value}{extra}"


def _finding_section(finding: Finding) -> list[str]:
    lines = [f"### {finding.item_id} &middot; {finding.route.value}", ""]
    lines.append(f"*Level {finding.level_id}*")
    lines.append("")
    lines.append(f"**Why:** {'; '.join(finding.reasons)}")
    lines.append("")

    if finding.verdict is not None:
        confidence = f"{finding.confidence:.2f}" if finding.confidence is not None else "n/a"
        lines.append(f"**Verdict:** {finding.verdict.value} at confidence {confidence}")
        if not finding.unanimous:
            lines.append("")
            lines.append("**Votes disagreed.**")
        if finding.rationale:
            lines.append("")
            lines.append(f"**Model's reasoning:** {finding.rationale}")
        if finding.evidence:
            lines.append("")
            lines.append("**Evidence quoted from the source:**")
            lines.extend(f"  - {quote!r}" for quote in finding.evidence)
        lines.append("")

    if finding.changes:
        lines.append("**Source changes:**")
        for change in finding.changes:
            lines.extend(_describe_change(change))
        lines.append("")

    if finding.anchors:
        lines.append("**Provenance:**")
        lines.extend(_describe_anchor(anchor) for anchor in finding.anchors)
        lines.append("")

    if finding.patch is not None:
        lines.append("**Proposed fix** (not applied):")
        lines.append("")
        lines.append(f"  stem: {finding.patch.stem}")
        lines.extend(_options(finding.patch.options, finding.patch.correct_index))
        lines.append(f"  grounded in: {finding.patch.grounded_in!r}")
        lines.append("")

    if finding.rejected_patch is not None:
        lines.append("**Rejected fix attempt** (kept as evidence, not a draft):")
        lines.append("")
        lines.append("```")
        lines.append(finding.rejected_patch.strip()[:600])
        lines.append("```")
        lines.append("")

    return lines


def render_markdown(run: RunReport) -> str:
    counts = run.route_counts()
    stats = run.stats
    lines: list[str] = [
        f"# Drift report: {run.course_title}",
        "",
        f"Comparing `{run.old_root}` against `{run.new_root}` using the `{run.client}` client.",
        "",
        "Nothing in this report has been applied. Every proposed fix needs a human.",
        "",
        "## Summary",
        "",
        "| Route | Items |",
        "| --- | --- |",
    ]
    lines.extend(f"| {route.value} | {counts[route]} |" for route in ROUTE_ORDER)
    lines += [
        "",
        "| Course items | Count |",
        "| --- | --- |",
        f"| total | {stats.total_items} |",
        f"| untouched by any change | {stats.untouched} |",
        f"| changed in ways that cannot matter | {stats.skipped_no_impact} |",
        f"| decided without a model | {stats.decided_deterministically} |",
        f"| sent to the model | {stats.sent_to_model} |",
        "",
        f"Model calls: {run.model_calls} (cache hits: {run.cache_hits}). "
        f"Cosmetic spans skipped: {stats.cosmetic_spans}. "
        f"Moved-but-identical spans suppressed: {stats.moved_spans}.",
        "",
        "## Levels",
        "",
        "| Level | Items | Stale | Needs review | Auto-fix | Retire |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for level_id in sorted(run.levels()):
        bucket = run.levels()[level_id]
        total = sum(bucket.values())
        lines.append(
            f"| {level_id} | {total} | {run.staleness(level_id)}% "
            f"| {bucket[Route.NEEDS_REVIEW]} | {bucket[Route.AUTO_FIX_PROPOSED]} "
            f"| {bucket[Route.RETIRE_SUGGESTED]} |"
        )

    lines += ["", "## Structural changes", "", "Documents:", ""]
    for state in ("unchanged", "modified", "added", "deleted"):
        names = run.documents.get(state, [])
        rendered = ", ".join(f"`{name}`" for name in names) if names else "none"
        lines.append(f"  - {state}: {rendered}")

    lines += ["", "Spans:", ""]
    lines.extend(f"  - {label}: {count}" for label, count in sorted(run.span_counts.items()))
    lines += ["", "Sections:", ""]
    lines.extend(f"  - {label}: {count}" for label, count in sorted(run.section_counts.items()))

    lines += ["", "## Findings", ""]
    grouped = run.by_route()
    if not any(grouped[route] for route in ROUTE_ORDER if route is not Route.OK):
        lines.append("No drift found. Every course item still matches its source.")
        lines.append("")

    for route in ROUTE_ORDER:
        items = grouped[route]
        if not items:
            continue
        lines.append(f"## {route.value} ({len(items)})")
        lines.append("")
        for finding in items:
            lines.extend(_finding_section(finding))

    return "\n".join(lines).rstrip() + "\n"
