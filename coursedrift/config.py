"""Every tunable that can affect a verdict lives here.

No magic number outside this module may influence a route. Later phases read
thresholds from a `Config` instance threaded through the pipeline; the CLI can
override any field with `--config path.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    """Thresholds, budgets and limits for one drift run."""

    model_config = ConfigDict(extra="forbid")

    # -- phase 2: matching ------------------------------------------------
    similarity_match: float = Field(default=0.60, ge=0.0, le=1.0)
    """Phase B accepts a candidate pair at or above this similarity."""

    cosmetic_similarity: float = Field(default=0.95, ge=0.0, le=1.0)
    """A pair at or above this similarity may qualify as COSMETIC."""

    shingle_n: int = Field(default=3, ge=1, le=8)
    """Word n-gram width used to block Phase B candidate pairs."""

    jaccard_match: float = Field(default=0.40, ge=0.0, le=1.0)
    """Section identity threshold over child span-hash sets."""

    jaccard_part: float = Field(default=0.25, ge=0.0, le=1.0)
    """Component floor for detecting a SPLIT or MERGE."""

    # -- phase 3: anchoring -----------------------------------------------
    anchor_fuzzy: float = Field(default=0.85, ge=0.0, le=1.0)
    """Relocation of a provenance snapshot by fuzzy text match."""

    infer_top_k: int = Field(default=3, ge=1, le=20)
    """BM25 candidates retained when provenance must be inferred."""

    # -- phase 6: judgment ------------------------------------------------
    confidence_floor: float = Field(default=0.75, ge=0.0, le=1.0)
    """Below this, an adjudication routes to human review."""

    votes: int = Field(default=1, ge=1, le=9)
    """Self-consistency runs per adjudication; any disagreement escalates."""

    # -- phase 5: reliability ---------------------------------------------
    timeout_s: float = Field(default=30.0, gt=0.0)
    """Per-call wall-clock budget handed to the client."""

    max_attempts: int = Field(default=2, ge=1, le=5)
    """Total attempts per call: the first, plus repair attempts."""

    max_calls: int = Field(default=500, ge=0)
    """Whole-run model call budget."""

    max_output_tokens: int = Field(default=1024, ge=16)
    """Ceiling on tokens a single response may generate.

    Our responses run to roughly 90 tokens, so this is slack rather than a
    constraint — its job is to stop a model that starts repeating itself from
    generating until the context window is exhausted, which turns one bad
    answer into one timeout.
    """

    allow_empty_new_snapshot: bool = False
    """Permit a new snapshot with no documents at all.

    Off by default: an empty new snapshot orphans every item and proposes
    retiring the whole course, which is far more often a wrong --new path
    than a real deletion.
    """

    retire_alarm_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    """Share of a course routed to retirement that warrants a warning.

    Above this, the report says plainly that the input looks wrong rather
    than presenting mass retirement as an ordinary result.
    """

    cache_dir: Path = Field(default=Path(".drift_cache"))
    """Where validated responses are memoized, keyed by client and prompt."""

    @classmethod
    def load(cls, path: Path | None) -> Config:
        """Load a config from JSON, or return defaults when `path` is None."""
        if path is None:
            return cls()
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
