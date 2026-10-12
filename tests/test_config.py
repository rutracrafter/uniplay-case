"""Phase 0 — the config object is the single home for tunables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from coursedrift.config import Config


def test_defaults_match_the_plan() -> None:
    cfg = Config()
    assert cfg.similarity_match == 0.60
    assert cfg.cosmetic_similarity == 0.95
    assert cfg.confidence_floor == 0.75
    assert cfg.max_attempts == 2
    assert cfg.votes == 1


def test_cosmetic_floor_is_above_match_floor() -> None:
    """COSMETIC is a subset of matched pairs, so its floor must be higher."""
    cfg = Config()
    assert cfg.cosmetic_similarity > cfg.similarity_match


def test_load_none_returns_defaults() -> None:
    assert Config.load(None) == Config()


def test_load_overrides_from_json(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"votes": 3, "confidence_floor": 0.9}))
    cfg = Config.load(path)
    assert cfg.votes == 3
    assert cfg.confidence_floor == 0.9
    assert cfg.similarity_match == 0.60  # untouched fields keep defaults


def test_unknown_field_is_rejected() -> None:
    """A typo in a config file must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        Config.model_validate({"confidence_flooor": 0.9})


def test_out_of_range_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"confidence_floor": 1.5})
