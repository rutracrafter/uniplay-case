"""The machine-readable report is an interface, so it is pinned.

The markdown report had a golden from the start; the JSON did not. The
consequence showed up in practice: removing three fields from `Span` changed
the JSON output shape while the whole suite stayed green and the markdown
golden stayed byte-identical. Anything consuming `drift_report.json` would
have broken silently.

Two kinds of assertion here on purpose. The key-set checks fail with a
message naming the field that moved, which is what you want for the common
case. The byte-exact golden catches everything else, including value changes
the key sets cannot see.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from coursedrift.config import Config
from coursedrift.llm.cache import NullCache
from coursedrift.llm.fake import FakeClient
from coursedrift.pipeline import run_drift
from coursedrift.report import RunReport, render_json

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden" / "demo_report.json"
LABELS = ("fixtures/docs_v1", "fixtures/docs_v2")

REPORT_KEYS = {
    "cache_hits",
    "client",
    "config",
    "course_id",
    "course_title",
    "documents",
    "findings",
    "generated_at",
    "model_calls",
    "new_root",
    "old_root",
    "section_counts",
    "span_counts",
    "stats",
    "warnings",
}
FINDING_KEYS = {
    "anchors",
    "cache_hits",
    "changes",
    "confidence",
    "evidence",
    "inferred_provenance",
    "item_id",
    "level_id",
    "model_calls",
    "patch",
    "rationale",
    "reasons",
    "rejected_patch",
    "route",
    "unanimous",
    "verdict",
}
SPAN_KEYS = {"doc_id", "hash", "norm_text", "order", "section_path", "span_id", "text"}
CHANGE_KEYS = {
    "across_docs",
    "content",
    "inline_diff",
    "new",
    "old",
    "pair_score",
    "position",
    "reordered",
    "similarity",
}
PATCH_KEYS = {"correct_index", "grounded_in", "options", "stem"}


def _run() -> RunReport:
    return run_drift(
        FIXTURES / "docs_v1",
        FIXTURES / "docs_v2",
        FIXTURES / "course.json",
        Config(),
        FakeClient(),
        NullCache(),
        labels=LABELS,
    )


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(render_json(_run()))


# -- schema ----------------------------------------------------------------


def test_report_keys_are_stable(payload: dict[str, Any]) -> None:
    assert set(payload.keys()) == REPORT_KEYS


def test_finding_keys_are_stable(payload: dict[str, Any]) -> None:
    for finding in payload["findings"]:
        assert set(finding.keys()) == FINDING_KEYS


def test_span_keys_are_stable(payload: dict[str, Any]) -> None:
    """The check that was missing when three Span fields were removed."""
    spans = [
        side
        for finding in payload["findings"]
        for change in finding["changes"]
        for side in (change["old"], change["new"])
        if side is not None
    ]
    assert spans, "the fixture must exercise at least one span"
    for span in spans:
        assert set(span.keys()) == SPAN_KEYS


def test_change_keys_are_stable(payload: dict[str, Any]) -> None:
    changes = [c for f in payload["findings"] for c in f["changes"]]
    assert changes
    for change in changes:
        assert set(change.keys()) == CHANGE_KEYS


def test_patch_keys_are_stable(payload: dict[str, Any]) -> None:
    patches = [f["patch"] for f in payload["findings"] if f["patch"] is not None]
    assert patches, "the fixture must produce at least one proposed fix"
    for patch in patches:
        assert set(patch.keys()) == PATCH_KEYS


def test_every_config_field_is_published(payload: dict[str, Any]) -> None:
    """The report records the thresholds that produced it, in full."""
    assert set(payload["config"].keys()) == set(Config.model_fields)


def test_routes_are_drawn_from_the_closed_set(payload: dict[str, Any]) -> None:
    from coursedrift.models import Route

    assert {f["route"] for f in payload["findings"]} <= {r.value for r in Route}


def test_the_markdown_timestamp_stays_out_but_json_keeps_the_slot(
    payload: dict[str, Any],
) -> None:
    assert "generated_at" in payload
    assert payload["generated_at"] is None, "run_drift must not stamp a time itself"


# -- golden ----------------------------------------------------------------


def test_matches_the_golden_json() -> None:
    expected = GOLDEN.read_text(encoding="utf-8")
    actual = render_json(_run())
    if actual != expected:
        pytest.fail(
            "the machine-readable report changed.\n\n"
            "If the change is intended, refresh the golden file with:\n"
            "    uv run drift demo --json > tests/golden/demo_report.json\n"
            "and check the diff — anything consuming drift_report.json sees it too."
        )


def test_the_golden_json_round_trips() -> None:
    restored = RunReport.model_validate_json(GOLDEN.read_text(encoding="utf-8"))
    assert len(restored.findings) == 14


def test_the_golden_json_embeds_no_absolute_paths() -> None:
    body = GOLDEN.read_text(encoding="utf-8")
    assert "/Users/" not in body
    assert "/home/" not in body
