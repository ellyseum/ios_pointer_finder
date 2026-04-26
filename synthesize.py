"""
synthesize.py — generate labeled training data for the ios-pointer-finder CNN.

Inputs:
  - backgrounds_kept/*.png   real iPhone screen captures, cursor-free
  - at_dot.png               cursor sprite alpha mask (RGB ignored — we
                              synthesize the cursor color per-instance based
                              on local bg luminance, since iOS uses contrast-
                              adaptive cursor tinting)

Outputs:
  - dataset/imgs/NNNNNN.jpg  composited training image
  - dataset/labels.jsonl     one JSON per line: {"path", "x", "y", "has_cursor", "lum_under"}

Each background yields N positive samples (cursor at random position) plus
some negative samples (no cursor — for confidence head training). The cursor
color is chosen per-sample to maximize contrast vs. the bg patch under the
cursor (matches iOS behavior we observed: dark cursor on bright bg, light
cursor on dark bg, white cursor on red, etc.).

Augmentations: brightness ±15%, slight blur, JPEG quality jitter 80-95.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
BG_DIR = os.path.join(ROOT, "backgrounds_kept")
DOT_PATH = os.path.join(ROOT, "at_dot.png")
OUT_DIR = os.path.join(ROOT, "dataset")

# Native iPhone H264 stream resolution (mirror pipeline output)
W, H = 994, 2160
# Empirically measured by counting cursor pixels in real native-res capture:
# the iOS BLE cursor is ~46 px diameter at 994x2160. Sample with small jitter
# (±4 px) so the model is robust to anti-aliasing / encoding artifacts that
# blur the apparent edge slightly.
CURSOR_PX_MIN = 42
CURSOR_PX_MAX = 50
# Empirical match to real cursor: α=0.25 + shift=0.5 + edge_falloff=0.10
# yields c ≈ 205 on dark navy bg, matching observed cursor center brightness
# (~70 BGR) with bg show-through that looks like real iOS rendering.
CURSOR_ALPHA_PEAK = 0.25


def make_pointer_mask(diameter: int = 46, peak_alpha: float = CURSOR_ALPHA_PEAK,
                     edge_falloff: float = 0.10) -> np.ndarray:
    """
    Generate a clean anti-aliased disc mask procedurally — round, no checkmark,
    no square artifacts. Returns (H, W) float32 in [0, 1].

    edge_falloff: fraction of radius over which alpha smoothly drops 1→0 at
    the edge. 0.10 = 10% softness, gives subtle anti-aliased edge without
    making the cursor look fuzzy.
    """
    r = diameter / 2.0
    yy, xx = np.meshgrid(np.arange(diameter), np.arange(diameter), indexing='ij')
    cy = cx = (diameter - 1) / 2.0
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # Smooth edge: alpha = 1 inside (r * (1-falloff)), 0 outside r, smoothstep between
    inner = r * (1.0 - edge_falloff)
    t = np.clip((r - dist) / (r - inner + 1e-6), 0.0, 1.0)
    # smoothstep for nicer falloff
    alpha = t * t * (3.0 - 2.0 * t)
    return (alpha * peak_alpha).astype(np.float32)


def luminance(bgr_patch: np.ndarray) -> float:
    """Mean luminance of a BGR patch in [0, 255]."""
    if bgr_patch.size == 0:
        return 128.0
    b, g, r = bgr_patch[..., 0].mean(), bgr_patch[..., 1].mean(), bgr_patch[..., 2].mean()
    return float(0.299 * r + 0.587 * g + 0.114 * b)


def pick_cursor_color(bg_patch: np.ndarray) -> tuple[int, int, int]:
    """
    iOS contrast-adaptive cursor color.

    Empirically calibrated against real cursor at native res (bg-00000.png):
    real cursor center lum ≈ 67.7 over bg lum ≈ 49.2 → cursor pulls bg toward
    midpoint (128) by ~0.5 of the gap. Solving result = c·α + bg·(1-α) with
    α = CURSOR_ALPHA_PEAK = 0.55 gives the right c per-bg.

    Validated: shift=0.50 produces cursor center brightness within 3 RGB of
    real on tested frames. This is much subtler than the prior "near-white
    on dark / near-black on light" rule, which over-shot brightness.
    """
    SHIFT_FRAC = 0.50
    MIN_CONTRAST = 35  # min RGB delta between cursor center and bg (CRITICAL:
                       # without this, mid-gray bgs make cursor invisible in
                       # the rendered result, killing training-data signal).
    lum = luminance(bg_patch)
    target_result_lum = lum + (128 - lum) * SHIFT_FRAC
    # Enforce min contrast: cursor must be at least MIN_CONTRAST away from bg
    if target_result_lum > lum and target_result_lum - lum < MIN_CONTRAST:
        target_result_lum = lum + MIN_CONTRAST
    elif target_result_lum < lum and lum - target_result_lum < MIN_CONTRAST:
        target_result_lum = lum - MIN_CONTRAST
    elif abs(target_result_lum - lum) < 1:
        # bg exactly at midpoint → arbitrarily pick light cursor
        target_result_lum = lum + MIN_CONTRAST
    # Invert alpha-over: c = (target - bg*(1-α)) / α
    c = (target_result_lum - lum * (1 - CURSOR_ALPHA_PEAK)) / CURSOR_ALPHA_PEAK
    c = max(20, min(235, c))
    j = random.randint(-5, 5)
    v = int(round(c)) + j
    return (v, v, v)


def composite(bg: np.ndarray, sprite_alpha: np.ndarray, cx: int, cy: int,
              cursor_bgr: tuple[int, int, int]) -> np.ndarray:
    """Alpha-over composite of cursor at (cx, cy) onto bg. In-place return."""
    sh, sw = sprite_alpha.shape
    x0 = cx - sw // 2; y0 = cy - sh // 2
    x1 = x0 + sw; y1 = y0 + sh
    # Clip to image bounds (cursor partially off-screen ok)
    bx0 = max(0, x0); by0 = max(0, y0)
    bx1 = min(bg.shape[1], x1); by1 = min(bg.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return bg
    sx0 = bx0 - x0; sy0 = by0 - y0
    sx1 = sx0 + (bx1 - bx0); sy1 = sy0 + (by1 - by0)
    region = bg[by0:by1, bx0:bx1].astype(np.float32)
    a = sprite_alpha[sy0:sy1, sx0:sx1, None]  # (h, w, 1)
    color = np.array(cursor_bgr, dtype=np.float32)[None, None, :]
    out = color * a + region * (1.0 - a)
    bg[by0:by1, bx0:bx1] = np.clip(out, 0, 255).astype(np.uint8)
    return bg


def augment(img: np.ndarray) -> np.ndarray:
    """Brightness ±15%, slight blur, JPEG-recompression noise."""
    img = img.astype(np.float32)
    img *= random.uniform(0.85, 1.15)  # brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # JPEG noise via encode/decode round-trip
    if random.random() < 0.7:
        q = random.randint(78, 95)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-bg", type=int, default=80,
                   help="positive samples per background (cursor present)")
    p.add_argument("--negatives-per-bg", type=int, default=10,
                   help="negative samples per background (no cursor — for confidence head)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=int, default=50,
                   help="keep cursor center >= margin from any edge so disc fully visible")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    bgs = sorted(glob.glob(os.path.join(BG_DIR, "*.png")) +
                 glob.glob(os.path.join(BG_DIR, "*.jpg")))
    if not bgs:
        print(f"FAIL: no backgrounds in {BG_DIR}", file=sys.stderr)
        return 1
    print(f"backgrounds: {len(bgs)}  per_bg={args.per_bg}  negatives_per_bg={args.negatives_per_bg}")
    print(f"expected total: {len(bgs)*(args.per_bg+args.negatives_per_bg)} samples")

    img_dir = os.path.join(args.out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(args.out_dir, "labels.jsonl")

    print(f"cursor sprite (procedural disc): diameter sampled in [{CURSOR_PX_MIN},{CURSOR_PX_MAX}] per instance, peak alpha {CURSOR_ALPHA_PEAK:.2f}")

    idx = 0
    with open(label_path, "w") as labels_f:
        for bg_idx, bg_path in enumerate(bgs):
            bg_orig = cv2.imread(bg_path)
            if bg_orig is None or bg_orig.shape[:2] != (H, W):
                print(f"  skip {bg_path} (bad shape)", file=sys.stderr)
                continue

            # Positive samples — cursor at random position with random size jitter
            for _ in range(args.per_bg):
                cx = random.randint(args.margin, W - args.margin)
                cy = random.randint(args.margin, H - args.margin)
                bg = bg_orig.copy()

                # Per-sample size jitter (real cursor is 46 px ± 4 from anti-alias / encoding)
                diameter = random.randint(CURSOR_PX_MIN, CURSOR_PX_MAX)
                sprite_alpha = make_pointer_mask(diameter=diameter)
                ph = sprite_alpha.shape[0]; pw = sprite_alpha.shape[1]

                py0 = max(0, cy - ph // 2); px0 = max(0, cx - pw // 2)
                py1 = min(H, py0 + ph); px1 = min(W, px0 + pw)
                bg_patch = bg_orig[py0:py1, px0:px1]
                lum_under = luminance(bg_patch)
                cursor_color = pick_cursor_color(bg_patch)

                composite(bg, sprite_alpha, cx, cy, cursor_color)
                bg = augment(bg)
                out_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                cv2.imwrite(out_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                labels_f.write(json.dumps({
                    "path": f"imgs/{idx:06d}.jpg",
                    "x": cx, "y": cy,
                    "has_cursor": 1,
                    "lum_under": round(lum_under, 1),
                    "cursor_v": cursor_color[0],
                }) + "\n")
                idx += 1

            # Negative samples — no cursor (for confidence head training)
            for _ in range(args.negatives_per_bg):
                bg = augment(bg_orig.copy())
                out_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                cv2.imwrite(out_path, bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                labels_f.write(json.dumps({
                    "path": f"imgs/{idx:06d}.jpg",
                    "x": -1, "y": -1,
                    "has_cursor": 0,
                    "lum_under": -1,
                    "cursor_v": -1,
                }) + "\n")
                idx += 1

            if (bg_idx + 1) % 10 == 0:
                print(f"  {bg_idx+1}/{len(bgs)} bgs processed, {idx} total samples")

    print(f"\nwrote {idx} samples → {img_dir}")
    print(f"labels → {label_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
