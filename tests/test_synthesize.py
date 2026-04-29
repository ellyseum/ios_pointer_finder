"""Smoke tests for synthesize.py — runs a tiny synthesis on solid-color
backgrounds, verifies the output schema is right.

Doesn't validate visual quality — that's what the trained model card metrics
are for.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _write_solid_bg(path: Path, color: tuple[int, int, int], w: int = 994, h: int = 2160) -> None:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_synthesize_runs_on_synthetic_bgs(tmp_path: Path, monkeypatch):
    """Generate 20 samples on 4 solid-color backgrounds and verify the
    label schema."""
    bg_dir = tmp_path / "backgrounds_kept"
    bg_dir.mkdir()
    for i, color in enumerate([(20, 20, 20), (80, 80, 80), (160, 160, 160), (240, 240, 240)]):
        _write_solid_bg(bg_dir / f"bg-{i:05d}.png", color)

    out_dir = tmp_path / "dataset"

    # synthesize.py uses module-level paths; monkeypatch them.
    import synthesize

    monkeypatch.setattr(synthesize, "BG_DIR", str(bg_dir))
    monkeypatch.setattr(synthesize, "OUT_DIR", str(out_dir))

    # Drive synthesis through the public main() with arg overrides if available;
    # otherwise call its core compose function directly. The script supports
    # the CLI args we emulate here.
    import sys

    argv = ["synthesize.py", "--out", str(out_dir), "--n", "20"]
    monkeypatch.setattr(sys, "argv", argv)

    if hasattr(synthesize, "main"):
        synthesize.main()
    else:
        pytest.skip("synthesize.py has no main(); skipping smoke test")

    labels_path = out_dir / "labels.jsonl"
    assert labels_path.exists(), "labels.jsonl should be written"

    with open(labels_path) as f:
        labels = [json.loads(line) for line in f if line.strip()]

    assert len(labels) == 20

    # Schema check
    required = {"path", "x", "y", "has_cursor", "sample_type", "bg_id"}
    for ent in labels:
        missing = required - set(ent.keys())
        assert not missing, f"label missing fields {missing}: {ent}"
        assert isinstance(ent["has_cursor"], (int, bool))
        assert ent["sample_type"] in {"normal_pos", "edge_pos", "hard_neg", "plain_neg"}
        # Image file exists
        assert (out_dir / "imgs" / Path(ent["path"]).name).exists() or (
            out_dir / ent["path"]
        ).exists()


def test_synthesize_label_balance(tmp_path: Path, monkeypatch):
    """Across 100 samples we should see at least one positive AND one negative."""
    bg_dir = tmp_path / "backgrounds_kept"
    bg_dir.mkdir()
    for i, color in enumerate([(40, 40, 40), (120, 120, 120), (200, 200, 200)]):
        _write_solid_bg(bg_dir / f"bg-{i:05d}.png", color)

    out_dir = tmp_path / "dataset"

    import synthesize

    monkeypatch.setattr(synthesize, "BG_DIR", str(bg_dir))
    monkeypatch.setattr(synthesize, "OUT_DIR", str(out_dir))

    import sys

    monkeypatch.setattr(sys, "argv", ["synthesize.py", "--out", str(out_dir), "--n", "100"])
    if not hasattr(synthesize, "main"):
        pytest.skip("synthesize.py has no main()")
    synthesize.main()

    with open(out_dir / "labels.jsonl") as f:
        labels = [json.loads(line) for line in f if line.strip()]

    pos = sum(1 for e in labels if e.get("has_cursor"))
    neg = len(labels) - pos
    assert pos > 0, "expected at least one positive sample"
    assert neg > 0, "expected at least one negative sample"
