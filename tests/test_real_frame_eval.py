"""
Real-frame regression gate.

Runs the current model against 8 bundled iPhone-screenshot fixtures and
asserts non-regression. Replaces the dead test_real.py path that pointed
at a non-existent directory and silently exited 0.

Gates:
  1. frame count == 8 (catches the 'no frames loaded' silent failure)
  2. bg-00000 prediction within 50px of hand-annotated GT (hard gate)
  3. all 8 frames produce finite predictions

The other 7 frames have v0.4-bootstrapped GT (~7-25px tolerance);
their distance to bootstrap is logged as a tracking signal, NOT a hard
gate, because v0.7 may legitimately disagree with v0.4 within that
band without representing a regression.

Skipped when no checkpoint is available — this is a regression gate,
not a unit test that must always pass.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURE_DIR = REPO / "tests" / "fixtures" / "real"
GT_PATH = FIXTURE_DIR / "ground_truth.json"

EXPECTED_FRAMES = [
    "bg-00000.png", "bg-00001.png", "bg-00002.png", "bg-00005.png",
    "bg-00006.png", "bg-00007.png", "bg-00008.png", "bg-00009.png",
]

HARD_GT_FRAME = "bg-00000.png"
HARD_GT_TOLERANCE_PX = 50.0


def _find_checkpoint() -> Path | None:
    """Locate a checkpoint to test against. Order: pointer_model.pt
    (rolling current), then any v0.7+ checkpoint, then v0.4 fallback.
    """
    candidates = [
        REPO / "pointer_model.pt",
        REPO / "pointer_model.safetensors",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to any v0.4+ weights present
    for pat in ("pointer_model_v0.7*.pt", "pointer_model_v0.6*.pt",
                "pointer_model_v0.5*.pt", "pointer_model_v0.4*.pt"):
        matches = sorted(REPO.glob(pat))
        if matches:
            return matches[0]
    return None


@pytest.fixture(scope="module")
def gt():
    with open(GT_PATH) as fh:
        data = json.load(fh)
    return data["frames"]


@pytest.fixture(scope="module")
def predictions():
    """Run inference on all 8 fixtures; cache results for the module."""
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint available; real-frame eval needs trained weights")
    from inference import PointerFinder
    finder = PointerFinder(ckpt)
    preds = {}
    for name in EXPECTED_FRAMES:
        fp = FIXTURE_DIR / name
        assert fp.exists(), f"fixture missing: {fp}"
        p = finder.predict(str(fp))
        preds[name] = (float(p.x), float(p.y), float(p.confidence),
                       float(p.heatmap_peak))
    return {"checkpoint": ckpt.name, "preds": preds}


def test_fixture_count_is_eight():
    """Catches the 'no frames loaded' failure mode — root cause of the
    pre-v0.7 silent dead-eval bug.
    """
    found = sorted(p.name for p in FIXTURE_DIR.glob("bg-*.png"))
    assert found == EXPECTED_FRAMES, (
        f"expected exactly 8 fixture frames, got {len(found)}: {found}"
    )


def test_ground_truth_covers_all_fixtures(gt):
    missing = [f for f in EXPECTED_FRAMES if f not in gt]
    assert not missing, f"GT missing for: {missing}"


def test_bg_00000_within_hard_gate(predictions, gt):
    """Hard gate: hand-annotated GT for bg-00000.

    If this fires, the decoder + model combination has drifted more than
    50px from a known-correct cursor location.
    """
    p = predictions["preds"][HARD_GT_FRAME]
    g = gt[HARD_GT_FRAME]
    err = ((p[0] - g["x"]) ** 2 + (p[1] - g["y"]) ** 2) ** 0.5
    assert err <= HARD_GT_TOLERANCE_PX, (
        f"{HARD_GT_FRAME} pred=({p[0]:.1f},{p[1]:.1f}) "
        f"gt=({g['x']},{g['y']}) err={err:.1f}px > {HARD_GT_TOLERANCE_PX}px "
        f"(checkpoint={predictions['checkpoint']})"
    )


def test_all_predictions_finite(predictions):
    import math
    for name, (x, y, conf, peak) in predictions["preds"].items():
        assert all(math.isfinite(v) for v in (x, y, conf, peak)), (
            f"{name}: non-finite prediction {(x, y, conf, peak)}"
        )


def test_log_v04_drift_signal(predictions, gt, capsys):
    """Tracking only — print the per-frame distance to v0.4-bootstrap GT.

    Not a gate: v0.7 may legitimately disagree with v0.4 within ~25px
    on the 7 bootstrapped frames. Useful as a regression signal during
    refactors of the decoder or training loss.
    """
    print(f"\nReal-frame eval ({predictions['checkpoint']}):")
    print(f"  {'frame':<18} {'pred':<24} {'gt':<24} {'err_px':>8} {'src'}")
    for name in EXPECTED_FRAMES:
        x, y, conf, peak = predictions["preds"][name]
        g = gt[name]
        err = ((x - g["x"]) ** 2 + (y - g["y"]) ** 2) ** 0.5
        pred_str = f"({x:.1f},{y:.1f})"
        gt_str = f"({g['x']:.1f},{g['y']:.1f})"
        print(f"  {name:<18} {pred_str:<24} {gt_str:<24} {err:>7.1f}  "
              f"{g['source']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
