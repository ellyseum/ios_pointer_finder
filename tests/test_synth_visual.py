"""
Visual-validation gate for the synth pipeline.

Reproduces the deterministic 16-cell contact sheet committed at
tests/golden/synth_contact_sheet.png and asserts:

  1. Per-cell SSIM >= 0.95 against the golden
     (catches drift in sprite shape, color, or composition logic;
      pixel-histogram alone would not — a UI badge with similar
      luminance can still pass histograms).

  2. Sprite circularity >= 0.85
     (catches "cursor is no longer round" — checkmark-pill or
      square-with-rounded-corners regression. 1.0 = perfect circle.)

If you change the synth pipeline (sprite generation, color picking,
composite math, augmentation) and the SSIM gate fires, REGENERATE the
golden via tools/build_golden.py, visually inspect the new contact
sheet, and commit the new golden in a separate commit before this test
file's expected behavior changes.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import synthesize as S  # noqa: E402

GOLDEN_PATH = REPO / "tests" / "golden" / "synth_contact_sheet.png"
FIXTURE_BG_DIR = REPO / "tests" / "fixtures" / "bg"
FIXTURE_BGS = [
    "bg-00003.png",
    "bg-00018.png",
    "bg-00051.png",
    "bg-00062.png",
]

CELL = 380
COLS = 4
ROWS = 4
PAD = 6


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM (Wang et al. 2004) on grayscale of two images.

    No skimage dependency — a 50-line implementation is enough for the
    coarse 'is the cell visually equivalent' check this gate makes.
    """
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    win = (11, 11)
    mu_a = cv2.GaussianBlur(a, win, 1.5)
    mu_b = cv2.GaussianBlur(b, win, 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, win, 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, win, 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, win, 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float((num / den).mean())


def _render_contact_sheet() -> np.ndarray:
    random.seed(42)
    np.random.seed(42)
    S._REAL_SPRITE = None  # force procedural fallback regardless of env

    canvas = np.full(
        (ROWS * (CELL + PAD) + PAD, COLS * (CELL + PAD) + PAD, 3),
        0,
        dtype=np.uint8,
    )

    for cell_idx in range(16):
        bg = cv2.imread(str(FIXTURE_BG_DIR / FIXTURE_BGS[cell_idx % 4])).copy()
        bh, bw = bg.shape[:2]
        margin = 80
        cx = random.randint(margin, bw - margin)
        cy = random.randint(margin, bh - margin)
        diameter = random.randint(S.CURSOR_PX_MIN, S.CURSOR_PX_MAX)

        sprite_alpha, _ = S.make_pointer_sprite(diameter)
        sh, sw = sprite_alpha.shape
        px0 = max(0, cx - sw // 2)
        py0 = max(0, cy - sh // 2)
        patch = bg[py0 : py0 + sh, px0 : px0 + sw]
        color = S.pick_cursor_color(patch)
        composed, _ = S.composite(bg, sprite_alpha, cx, cy, color)

        cx0 = max(0, cx - CELL // 2)
        cy0 = max(0, cy - CELL // 2)
        cx0 = min(cx0, bw - CELL)
        cy0 = min(cy0, bh - CELL)
        cell = composed[cy0 : cy0 + CELL, cx0 : cx0 + CELL]

        r, c = cell_idx // COLS, cell_idx % COLS
        yy = r * (CELL + PAD) + PAD
        xx = c * (CELL + PAD) + PAD
        canvas[yy : yy + CELL, xx : xx + CELL] = cell
    return canvas


def _split_cells(sheet: np.ndarray) -> list[np.ndarray]:
    cells = []
    for cell_idx in range(16):
        r, c = cell_idx // COLS, cell_idx % COLS
        yy = r * (CELL + PAD) + PAD
        xx = c * (CELL + PAD) + PAD
        cells.append(sheet[yy : yy + CELL, xx : xx + CELL])
    return cells


def test_synth_contact_sheet_matches_golden():
    """Per-cell SSIM >= 0.95 against the committed golden.

    Drift signals: sprite generator changed, pick_cursor_color changed,
    composite math changed, or fixture backgrounds changed.
    """
    assert GOLDEN_PATH.exists(), f"missing golden: {GOLDEN_PATH}"
    golden = cv2.imread(str(GOLDEN_PATH))
    fresh = _render_contact_sheet()
    assert fresh.shape == golden.shape, f"shape drift: fresh={fresh.shape} golden={golden.shape}"

    fresh_cells = _split_cells(fresh)
    golden_cells = _split_cells(golden)
    failures = []
    for i, (f, g) in enumerate(zip(fresh_cells, golden_cells)):
        s = _ssim(f, g)
        if s < 0.95:
            failures.append((i, s))
    assert not failures, (
        "cells below SSIM 0.95: "
        + ", ".join(f"#{i}={s:.3f}" for i, s in failures)
        + ". Synth pipeline changed; regenerate golden if intentional."
    )


def test_procedural_sprite_is_circular():
    """Circularity >= 0.85 on the alpha-thresholded procedural disc.

    A perfect circle is 1.0; the smoothstep disc tests at ~0.93. A
    rounded-rectangle UI badge with a baked-in checkmark falls below
    0.85 — this gate would have caught the v0.5/v0.6 sprite catastrophe.
    """
    S._REAL_SPRITE = None
    diameter = 46
    alpha, _ = S.make_pointer_sprite(diameter)
    # Threshold at 50% of peak alpha to get the sprite footprint
    mask = (alpha > (alpha.max() * 0.5)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert contours, "no contour in procedural sprite alpha"
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    assert perim > 0, "zero-perimeter contour"
    circularity = 4 * np.pi * area / (perim * perim)
    assert circularity >= 0.85, (
        f"procedural sprite not circular enough: {circularity:.3f} < 0.85. "
        f"Sprite generator may be producing a non-disc shape."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
