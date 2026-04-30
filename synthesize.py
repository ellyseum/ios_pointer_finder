"""
synthesize.py — generate labeled training data for the ios-pointer-finder CNN.

v0.3 sample mix:
  - normal_pos  (default 55%): full-cursor positives, random position with margin
  - edge_pos    (default 15%): cursor partially clipped at a screen edge,
                                visible-centroid label, reject if visible
                                fraction < MIN_VISIBLE_FRAC
  - hard_neg    (default 15%): "decoy cursor" composites — wrong-size discs,
                                wrong-alpha discs, hollow rings, ellipses,
                                doubled-dots, I-beam strokes, white wedges.
                                has_cursor=0 — model must learn "cursor-shaped
                                but NOT a real cursor" rejection.
  - plain_neg   (default 15%): unmodified background, has_cursor=0.

Inputs:
  - backgrounds_kept/*.png   real iPhone screen captures, cursor-free

Outputs:
  - dataset/imgs/NNNNNN.jpg
  - dataset/labels.jsonl     {path, x, y, has_cursor, sample_type, bg_id, ...}

bg_id and sample_type fields are read by train.py for bg-level val split and
for sliced metrics (FPR per neg type, pos error per pos type).
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
OUT_DIR = os.path.join(ROOT, "dataset")

# Native iPhone H264 stream resolution
W, H = 994, 2160

# Real cursor is ~46 px diameter — measured empirically from native captures.
# Sample with small jitter for anti-alias / encoding noise robustness.
CURSOR_PX_MIN = 42
CURSOR_PX_MAX = 50
# Real cursor: peak alpha ≈ 0.25 with edge falloff ≈ 0.10 — calibrated against
# observed bg-00000 cursor (center brightness ~70 over bg lum ~49).
CURSOR_ALPHA_PEAK = 0.25
CURSOR_EDGE_FALLOFF = 0.10

# Edge cases: reject if less than this fraction of the cursor's *alpha mass*
# (NOT bounding-box area) survives in-frame. Using box area overstates
# visibility for soft-edged sprites with transparent corners.
MIN_VISIBLE_FRAC = 0.20

# Hard-negative decoy types
HARD_NEG_TYPES = ["wrong_size_disc", "wrong_alpha_disc", "ring",
                  "ellipse", "doubled_dot", "ibeam", "wedge"]


# ============================================================
# Cursor sprite — captured from real iOS screen, with hotspot tracking
# ============================================================

# Path to the captured iOS Pointer-Control sprite (alpha-matted PNG).
# Override with IPF_SPRITE_PATH if shipping a different capture.
SPRITE_PATH = os.environ.get("IPF_SPRITE_PATH",
                             os.path.join(ROOT, "sprites", "at_dot.png"))

# Cached at module load — (alpha_float, hotspot_xy) at the source resolution.
_REAL_SPRITE: tuple[np.ndarray, tuple[float, float]] | None = None


def _load_real_sprite() -> tuple[np.ndarray, tuple[float, float]] | None:
    """Load the captured iOS pointer sprite and compute its alpha-mass hotspot.

    Returns (alpha[H, W] float32 in [0, 1], (hotspot_x, hotspot_y) in source
    pixel coords). The hotspot is the alpha-weighted centroid — i.e. the click
    anchor — which on the real iOS pointer differs from the geometric center
    of the sprite tile by a few pixels. Labeling at the hotspot (not the
    geometric center) eliminates a systematic supervision bias.
    """
    global _REAL_SPRITE
    if _REAL_SPRITE is not None:
        return _REAL_SPRITE
    if not os.path.exists(SPRITE_PATH):
        return None
    img = cv2.imread(SPRITE_PATH, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    h, w = alpha.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    total = float(alpha.sum())
    if total < 1e-6:
        return None
    hx = float((alpha * xx).sum() / total)
    hy = float((alpha * yy).sum() / total)
    _REAL_SPRITE = (alpha, (hx, hy))
    return _REAL_SPRITE


def make_pointer_mask(diameter: int = 46,
                     peak_alpha: float = CURSOR_ALPHA_PEAK,
                     edge_falloff: float = CURSOR_EDGE_FALLOFF) -> np.ndarray:
    """Procedural anti-aliased disc. Used as a fallback when the real sprite
    can't be loaded, and for hard-negative decoy generation."""
    r = diameter / 2.0
    yy, xx = np.meshgrid(np.arange(diameter), np.arange(diameter), indexing='ij')
    cy = cx = (diameter - 1) / 2.0
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    inner = r * (1.0 - edge_falloff)
    t = np.clip((r - dist) / (r - inner + 1e-6), 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
    return (alpha * peak_alpha).astype(np.float32)


def make_pointer_sprite(diameter: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Return (alpha, hotspot_local) for a cursor at the requested diameter.

    Prefers the captured real sprite (resized via INTER_LINEAR); falls back to
    a procedural disc whose hotspot is the geometric center.

    `hotspot_local` is in sprite-local pixel coords: it is the click anchor
    relative to the sprite's top-left, NOT relative to the geometric center.
    Caller computes image-space label as `sprite_top_left + hotspot_local`.
    """
    real = _load_real_sprite()
    if real is None:
        alpha = make_pointer_mask(diameter=diameter)
        c = (diameter - 1) / 2.0
        return alpha, (c, c)
    src_alpha, _ = real
    # Resize to target diameter (square).
    resized = cv2.resize(src_alpha, (diameter, diameter),
                        interpolation=cv2.INTER_LINEAR)
    # Renormalize peak to match CURSOR_ALPHA_PEAK so the contrast math in
    # pick_cursor_color() stays correct against the legacy reference. Scale
    # by the RESIZED peak (not src_peak) — INTER_LINEAR resampling drops the
    # peak slightly, so scaling by source max would leave the resized peak
    # at ~0.227 instead of 0.25.
    resized_peak = float(resized.max())
    if resized_peak > 1e-6:
        resized = resized * (CURSOR_ALPHA_PEAK / resized_peak)
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
    # Hotspot = alpha-mass centroid of the RESIZED sprite (NOT the source-
    # coords scaling). INTER_LINEAR doesn't perfectly preserve centroid
    # under resampling, so use the actual centroid of what we'll composite.
    yy, xx = np.indices(resized.shape, dtype=np.float32)
    total = float(resized.sum())
    if total > 1e-6:
        hx = float((resized * xx).sum() / total)
        hy = float((resized * yy).sum() / total)
    else:
        hx = hy = (diameter - 1) / 2.0
    return resized, (hx, hy)


# ============================================================
# Hard-negative (decoy cursor) sprite generators
# ============================================================

def make_decoy_wrong_size(scale: float | None = None) -> np.ndarray:
    """Disc with wrong diameter — too small (~25 px) or too big (~75 px).
    Same color/alpha logic, just wrong size."""
    if scale is None:
        scale = random.choice([0.5, 0.55, 0.6, 1.5, 1.65, 1.8])
    d = max(8, int(round(46 * scale)))
    return make_pointer_mask(diameter=d, peak_alpha=CURSOR_ALPHA_PEAK)


def make_decoy_wrong_alpha() -> np.ndarray:
    """Disc with right size but very different alpha — too transparent or too opaque."""
    d = random.randint(CURSOR_PX_MIN, CURSOR_PX_MAX)
    a = random.choice([0.06, 0.10, 0.55, 0.70, 0.85])
    return make_pointer_mask(diameter=d, peak_alpha=a)


def make_decoy_ring(diameter: int | None = None) -> np.ndarray:
    """Hollow ring instead of solid disc."""
    if diameter is None:
        diameter = random.randint(CURSOR_PX_MIN, CURSOR_PX_MAX + 8)
    r = diameter / 2.0
    yy, xx = np.meshgrid(np.arange(diameter), np.arange(diameter), indexing='ij')
    cy = cx_ = (diameter - 1) / 2.0
    dist = np.sqrt((yy - cy)**2 + (xx - cx_)**2)
    ring_w = random.randint(4, 8)
    inner_r = max(2, r - ring_w)
    in_outer = (dist <= r).astype(np.float32)
    in_inner = (dist <= inner_r).astype(np.float32)
    ring = in_outer - in_inner
    ring = cv2.GaussianBlur(ring, (3, 3), 0)
    return (ring * CURSOR_ALPHA_PEAK).astype(np.float32)


def make_decoy_ellipse() -> np.ndarray:
    """Ellipse — same general shape but stretched."""
    h = random.randint(CURSOR_PX_MIN - 4, CURSOR_PX_MAX + 4)
    aspect = random.uniform(1.4, 2.2)
    w = int(round(h * aspect))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cy = (h - 1) / 2.0; cx_ = (w - 1) / 2.0
    rx = w / 2.0; ry = h / 2.0
    dist_n = ((yy - cy) / max(1.0, ry)) ** 2 + ((xx - cx_) / max(1.0, rx)) ** 2
    inner = (1.0 - 0.10) ** 2
    t = np.clip((1.0 - dist_n) / (1.0 - inner + 1e-6), 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)
    return (alpha * CURSOR_ALPHA_PEAK).astype(np.float32)


def make_decoy_doubled_dot() -> np.ndarray:
    """Two small dots side-by-side — 'cursor with reflection' look-alike."""
    d_each = random.randint(20, 32)
    gap = random.randint(2, 8)
    canvas_w = d_each * 2 + gap
    canvas_h = d_each
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    dot = make_pointer_mask(diameter=d_each, peak_alpha=CURSOR_ALPHA_PEAK)
    # Dot is square (d_each x d_each); paste at left and right
    canvas[:, :d_each] = dot
    canvas[:, d_each + gap:d_each + gap + d_each] = dot * random.uniform(0.6, 1.0)
    return canvas


def make_decoy_ibeam() -> np.ndarray:
    """Vertical text-cursor I-beam — thin vertical line with optional caps."""
    h = random.randint(36, 56)
    w = random.randint(3, 6)
    canvas = np.zeros((h, w + 6), dtype=np.float32)
    canvas[:, 3:3 + w] = 1.0
    if random.random() < 0.4:
        # add caps
        cap_h = max(1, h // 14)
        canvas[:cap_h, :] = 0
        canvas[:cap_h, 1:-1] = 1.0
        canvas[-cap_h:, :] = 0
        canvas[-cap_h:, 1:-1] = 1.0
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return (canvas * random.uniform(0.20, 0.45)).astype(np.float32)


def make_decoy_wedge() -> np.ndarray:
    """White triangle/wedge — back-button arrow look-alike."""
    h = random.randint(24, 44)
    w = h
    canvas = np.zeros((h, w), dtype=np.float32)
    direction = random.choice(['left', 'right', 'up', 'down'])
    pts = {
        'left':  np.array([[w-1, 0], [w-1, h-1], [0, h//2]], np.int32),
        'right': np.array([[0, 0], [0, h-1], [w-1, h//2]], np.int32),
        'up':    np.array([[0, h-1], [w-1, h-1], [w//2, 0]], np.int32),
        'down':  np.array([[0, 0], [w-1, 0], [w//2, h-1]], np.int32),
    }[direction]
    cv2.fillPoly(canvas, [pts], 1.0)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return (canvas * random.uniform(0.30, 0.55)).astype(np.float32)


def make_decoy(decoy_type: str) -> np.ndarray:
    if decoy_type == "wrong_size_disc":  return make_decoy_wrong_size()
    if decoy_type == "wrong_alpha_disc": return make_decoy_wrong_alpha()
    if decoy_type == "ring":             return make_decoy_ring()
    if decoy_type == "ellipse":          return make_decoy_ellipse()
    if decoy_type == "doubled_dot":      return make_decoy_doubled_dot()
    if decoy_type == "ibeam":            return make_decoy_ibeam()
    if decoy_type == "wedge":            return make_decoy_wedge()
    raise ValueError(f"unknown decoy type: {decoy_type}")


# ============================================================
# Color and compositing
# ============================================================

def luminance(bgr_patch: np.ndarray) -> float:
    if bgr_patch.size == 0:
        return 128.0
    b, g, r = bgr_patch[..., 0].mean(), bgr_patch[..., 1].mean(), bgr_patch[..., 2].mean()
    return float(0.299 * r + 0.587 * g + 0.114 * b)


def pick_cursor_color(bg_patch: np.ndarray,
                      alpha: float = CURSOR_ALPHA_PEAK) -> tuple[int, int, int]:
    """iOS contrast-adaptive cursor color, calibrated to real captures."""
    SHIFT_FRAC = 0.50
    MIN_CONTRAST = 35
    lum = luminance(bg_patch)
    target = lum + (128 - lum) * SHIFT_FRAC
    if target > lum and target - lum < MIN_CONTRAST:
        target = lum + MIN_CONTRAST
    elif target < lum and lum - target < MIN_CONTRAST:
        target = lum - MIN_CONTRAST
    elif abs(target - lum) < 1:
        target = lum + MIN_CONTRAST
    c = (target - lum * (1 - alpha)) / max(1e-6, alpha)
    c = max(20, min(235, c))
    j = random.randint(-5, 5)
    v = int(round(c)) + j
    return (v, v, v)


def composite(bg: np.ndarray, sprite_alpha: np.ndarray, cx: int, cy: int,
              color_bgr: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    """Alpha-over composite at (cx, cy). Returns (bg, visible_pixel_count).
    visible_pixel_count = pixels of sprite that landed inside image bounds."""
    sh, sw = sprite_alpha.shape
    x0 = cx - sw // 2; y0 = cy - sh // 2
    x1 = x0 + sw; y1 = y0 + sh
    bx0 = max(0, x0); by0 = max(0, y0)
    bx1 = min(bg.shape[1], x1); by1 = min(bg.shape[0], y1)
    if bx0 >= bx1 or by0 >= by1:
        return bg, 0
    sx0 = bx0 - x0; sy0 = by0 - y0
    sx1 = sx0 + (bx1 - bx0); sy1 = sy0 + (by1 - by0)
    region = bg[by0:by1, bx0:bx1].astype(np.float32)
    a = sprite_alpha[sy0:sy1, sx0:sx1, None]
    color = np.array(color_bgr, dtype=np.float32)[None, None, :]
    out = color * a + region * (1.0 - a)
    bg[by0:by1, bx0:bx1] = np.clip(out, 0, 255).astype(np.uint8)
    visible_px = (bx1 - bx0) * (by1 - by0)
    return bg, visible_px


def visible_alpha_centroid(sprite_alpha: np.ndarray, cx_geom: int, cy_geom: int
                          ) -> tuple[float, float, float]:
    """Compute the alpha-mass centroid of the *visible* portion of a sprite
    placed at geometric center (cx_geom, cy_geom) in image coords, plus the
    fraction of total alpha mass that survived the in-frame clip.

    Returns (label_x, label_y, visible_frac). Coordinates are floats — the
    centroid is a fractional pixel by construction, and downstream training
    is float-aware (v0.4 fix).
    """
    sh, sw = sprite_alpha.shape
    # Unclipped sprite bbox in image coords (anchored on geometric center).
    x0 = cx_geom - sw // 2
    y0 = cy_geom - sh // 2
    x1 = x0 + sw
    y1 = y0 + sh
    # Clamp to frame.
    px0 = max(0, x0); py0 = max(0, y0)
    px1 = min(W, x1); py1 = min(H, y1)
    if px0 >= px1 or py0 >= py1:
        return float(cx_geom), float(cy_geom), 0.0
    # Sprite-local indices of the visible region.
    sx0 = px0 - x0; sy0 = py0 - y0
    sx1 = sx0 + (px1 - px0); sy1 = sy0 + (py1 - py0)
    visible = sprite_alpha[sy0:sy1, sx0:sx1]
    visible_mass = float(visible.sum())
    total_mass = float(sprite_alpha.sum())
    visible_frac = visible_mass / max(1e-6, total_mass)
    if visible_mass < 1e-6:
        # Degenerate (all-transparent overlap region) — bail out with the
        # geometric-center label so caller can still decide via visible_frac.
        return float(cx_geom), float(cy_geom), visible_frac
    yy = np.arange(sy0, sy1, dtype=np.float32)[:, None]
    xx = np.arange(sx0, sx1, dtype=np.float32)[None, :]
    cy_local = float((visible * yy).sum() / visible_mass)
    cx_local = float((visible * xx).sum() / visible_mass)
    label_x = float(x0) + cx_local
    label_y = float(y0) + cy_local
    return label_x, label_y, visible_frac


def augment(img: np.ndarray) -> np.ndarray:
    """Brightness ±15%, slight blur, JPEG-recompression noise."""
    img = img.astype(np.float32)
    img *= random.uniform(0.85, 1.15)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.7:
        q = random.randint(78, 95)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


# ============================================================
# Sample generators (each returns dict ready for labels.jsonl + the image)
# ============================================================

def gen_normal_pos(bg_orig: np.ndarray, margin: int) -> tuple[np.ndarray, dict]:
    """Full-cursor positive. Label is the alpha-centroid (click anchor),
    NOT the geometric center of the sprite tile — these differ on the real
    iOS pointer by ~1 px in x and ~3 px in y at native resolution."""
    cx_geom = random.randint(margin, W - margin)
    cy_geom = random.randint(margin, H - margin)
    diameter = random.randint(CURSOR_PX_MIN, CURSOR_PX_MAX)
    sprite_alpha, hotspot_local = make_pointer_sprite(diameter)
    ph, pw = sprite_alpha.shape
    label_x = float(cx_geom - pw // 2) + hotspot_local[0]
    label_y = float(cy_geom - ph // 2) + hotspot_local[1]
    py0 = max(0, cy_geom - ph // 2); px0 = max(0, cx_geom - pw // 2)
    py1 = min(H, py0 + ph); px1 = min(W, px0 + pw)
    bg_patch = bg_orig[py0:py1, px0:px1]
    color = pick_cursor_color(bg_patch)
    bg = bg_orig.copy()
    bg, _vis = composite(bg, sprite_alpha, cx_geom, cy_geom, color)
    return augment(bg), {
        "x": label_x, "y": label_y, "has_cursor": 1, "sample_type": "normal_pos",
        "lum_under": round(luminance(bg_patch), 1),
        "cursor_v": color[0], "diameter": diameter,
        # Persist hotspot so train-time crop can use asymmetric sprite-bbox
        # protection (sprite extends [label-hx, label+(d-hx)] × [label-hy,
        # label+(d-hy)], NOT a symmetric radius around the label — the iOS
        # pointer's alpha mass is biased toward the upper-left of its tile).
        "hotspot_x": float(hotspot_local[0]), "hotspot_y": float(hotspot_local[1]),
    }


def gen_edge_pos(bg_orig: np.ndarray, margin: int) -> tuple[np.ndarray, dict] | None:
    """Edge-clipped cursor. Returns None if visible alpha-mass fraction
    (NOT bounding-box fraction) is below MIN_VISIBLE_FRAC. Label is the
    alpha-centroid of the visible portion — the click anchor projected
    onto whatever pixels survived the clip."""
    edge = random.choice(['left', 'right', 'top', 'bottom'])
    diameter = random.randint(CURSOR_PX_MIN, CURSOR_PX_MAX)
    r = diameter // 2
    edge_offset = random.randint(-r - 3, max(0, margin // 2))
    if edge == 'left':
        cx_geom = edge_offset; cy_geom = random.randint(margin, H - margin)
    elif edge == 'right':
        cx_geom = W - 1 - edge_offset; cy_geom = random.randint(margin, H - margin)
    elif edge == 'top':
        cx_geom = random.randint(margin, W - margin); cy_geom = edge_offset
    else:
        cx_geom = random.randint(margin, W - margin); cy_geom = H - 1 - edge_offset

    sprite_alpha, _hotspot = make_pointer_sprite(diameter)
    ph, pw = sprite_alpha.shape
    label_x, label_y, visible_frac = visible_alpha_centroid(
        sprite_alpha, cx_geom, cy_geom)
    if visible_frac < MIN_VISIBLE_FRAC:
        return None
    py0 = max(0, cy_geom - ph // 2); px0 = max(0, cx_geom - pw // 2)
    py1 = min(H, py0 + ph); px1 = min(W, px0 + pw)
    if py0 >= py1 or px0 >= px1:
        return None
    bg_patch = bg_orig[py0:py1, px0:px1]
    color = pick_cursor_color(bg_patch)
    bg = bg_orig.copy()
    bg, _vis = composite(bg, sprite_alpha, cx_geom, cy_geom, color)
    return augment(bg), {
        "x": label_x, "y": label_y, "has_cursor": 1, "sample_type": "edge_pos",
        "lum_under": round(luminance(bg_patch), 1),
        "cursor_v": color[0], "diameter": diameter, "edge": edge,
        "geometric_center": [cx_geom, cy_geom],
        "visible_frac": round(visible_frac, 3),
    }


def gen_hard_neg(bg_orig: np.ndarray, margin: int) -> tuple[np.ndarray, dict]:
    """Composite a 'decoy cursor' (cursor-shaped but wrong) labeled has_cursor=0.
    Forces the model to discriminate real vs. lookalike at the heatmap level."""
    decoy_type = random.choice(HARD_NEG_TYPES)
    sprite = make_decoy(decoy_type)
    ph, pw = sprite.shape
    cx = random.randint(margin, W - margin)
    cy = random.randint(margin, H - margin)
    # Match gen_normal_pos's `py0:py0+ph` patch convention; `cy ± ph // 2`
    # truncates one row/col on odd-height decoys (ibeam, ring), biasing
    # pick_cursor_color by a sub-pixel luminance drift.
    py0 = max(0, cy - ph // 2); px0 = max(0, cx - pw // 2)
    py1 = min(H, py0 + ph); px1 = min(W, px0 + pw)
    bg_patch = bg_orig[py0:py1, px0:px1]
    color = pick_cursor_color(bg_patch)
    bg = bg_orig.copy()
    bg, _vis = composite(bg, sprite, cx, cy, color)
    # Persist decoy size so the train-time crop sampler can protect the
    # decoy footprint. Without this, hard_neg crops can clip large decoys
    # and turn "edge-clipped cursor-like blob = negative" into supervision
    # that directly contradicts edge_pos's "clipped cursor = positive".
    return augment(bg), {
        "x": -1, "y": -1, "has_cursor": 0, "sample_type": "hard_neg",
        "decoy_type": decoy_type, "decoy_pos": [cx, cy],
        "decoy_w": pw, "decoy_h": ph,
    }


def gen_plain_neg(bg_orig: np.ndarray) -> tuple[np.ndarray, dict]:
    """Unmodified background, no cursor, has_cursor=0."""
    return augment(bg_orig.copy()), {
        "x": -1, "y": -1, "has_cursor": 0, "sample_type": "plain_neg",
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-bg", type=int, default=1100,
                   help="total samples per background (split per the fractions below)")
    p.add_argument("--normal-pos-frac", type=float, default=0.55)
    p.add_argument("--edge-pos-frac",   type=float, default=0.15)
    p.add_argument("--hard-neg-frac",   type=float, default=0.15)
    p.add_argument("--plain-neg-frac",  type=float, default=0.15)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=int, default=50,
                   help="min distance from edge for non-edge samples")
    p.add_argument("--limit-bgs", type=int, default=0,
                   help="if >0, only process the first N backgrounds (for quick smoke tests)")
    args = p.parse_args()

    fracs = (args.normal_pos_frac, args.edge_pos_frac,
             args.hard_neg_frac, args.plain_neg_frac)
    if abs(sum(fracs) - 1.0) > 1e-3:
        print(f"FAIL: fractions sum to {sum(fracs):.3f}, must equal 1.0", file=sys.stderr)
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)

    bgs = sorted(glob.glob(os.path.join(BG_DIR, "*.png")) +
                 glob.glob(os.path.join(BG_DIR, "*.jpg")))
    if not bgs:
        print(f"FAIL: no backgrounds in {BG_DIR}", file=sys.stderr)
        return 1
    if args.limit_bgs > 0:
        bgs = bgs[:args.limit_bgs]

    n_normal = int(args.per_bg * args.normal_pos_frac)
    n_edge   = int(args.per_bg * args.edge_pos_frac)
    n_hard   = int(args.per_bg * args.hard_neg_frac)
    n_plain  = args.per_bg - n_normal - n_edge - n_hard  # remainder absorbs rounding

    print(f"backgrounds: {len(bgs)}  per_bg={args.per_bg}")
    print(f"  normal_pos: {n_normal}  edge_pos: {n_edge}  "
          f"hard_neg: {n_hard}  plain_neg: {n_plain}")
    print(f"  expected total ≈ {len(bgs) * args.per_bg} samples (edge rejects "
          f"reduce slightly)")

    img_dir = os.path.join(args.out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(args.out_dir, "labels.jsonl")

    idx = 0
    rejected_edge = 0
    counts = {"normal_pos": 0, "edge_pos": 0, "hard_neg": 0, "plain_neg": 0}
    with open(label_path, "w") as labels_f:
        for bg_idx, bg_path in enumerate(bgs):
            bg_orig = cv2.imread(bg_path)
            if bg_orig is None or bg_orig.shape[:2] != (H, W):
                print(f"  skip {bg_path} (bad shape)", file=sys.stderr)
                continue
            bg_id = os.path.basename(bg_path)

            # Build (gen_fn, count) plan and shuffle so they interleave on disk
            plan: list[tuple[str, callable]] = []
            for _ in range(n_normal):
                plan.append(("normal_pos", lambda b=bg_orig: gen_normal_pos(b, args.margin)))
            for _ in range(n_edge):
                plan.append(("edge_pos", lambda b=bg_orig: gen_edge_pos(b, args.margin)))
            for _ in range(n_hard):
                plan.append(("hard_neg", lambda b=bg_orig: gen_hard_neg(b, args.margin)))
            for _ in range(n_plain):
                plan.append(("plain_neg", lambda b=bg_orig: gen_plain_neg(b)))
            random.shuffle(plan)

            for sample_kind, gen in plan:
                result = gen()
                if result is None:
                    rejected_edge += 1
                    continue
                img, label = result
                out_path = os.path.join(img_dir, f"{idx:06d}.jpg")
                cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                label["path"] = f"imgs/{idx:06d}.jpg"
                label["bg_id"] = bg_id
                labels_f.write(json.dumps(label) + "\n")
                counts[sample_kind] += 1
                idx += 1

            if (bg_idx + 1) % 10 == 0:
                print(f"  {bg_idx + 1}/{len(bgs)} bgs processed, {idx} samples "
                      f"(rejected {rejected_edge} edge-too-small)")

    print(f"\nwrote {idx} samples → {img_dir}")
    print(f"  by type: {counts}")
    print(f"  rejected edge samples (visible<{int(MIN_VISIBLE_FRAC*100)}%): {rejected_edge}")
    print(f"labels → {label_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
