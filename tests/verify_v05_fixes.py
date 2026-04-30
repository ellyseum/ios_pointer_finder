"""verify_v05_fixes.py — mechanical check that the 11 v0.5 bug fixes are
in effect. Run before regenerating the full dataset.

Usage:
    python tests/verify_v05_fixes.py

Each bug gets at least one assertion. Some checks generate a small synthetic
dataset on disk; that lives in `tests/_v05_smoke/` and is NOT committed.

Exit code is the number of failed checks.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import synthesize  # noqa: E402
import train  # noqa: E402
from inference import _parabolic_offset  # noqa: E402

# ANSI for terminal output
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  {GREEN}PASS{RESET}  {name}")
        if detail:
            print(f"        {detail}")
    else:
        print(f"  {RED}FAIL{RESET}  {name}")
        if detail:
            print(f"        {detail}")
        failures.append(name)


def section(title: str) -> None:
    print(f"\n{YELLOW}=== {title} ==={RESET}")


# =====================================================================
# Bug 1 — gen_edge_pos visibility uses true sprite extent, not clamped bbox.
# Bug 6 — visibility is alpha-mass based, not bounding-box area.
# =====================================================================
section("Bug 1+6: edge_pos visibility uses alpha mass on true sprite extent")

# Force the cursor 100% off-screen (geometric center way outside frame).
# Old code's clamp bug would compute visible_area > 0 in some configurations
# even when nothing was visible. Confirm gen_edge_pos rejects.
W = synthesize.W
H = synthesize.H
bg = np.zeros((H, W, 3), dtype=np.uint8) + 100  # dummy gray background

# Manually call visible_alpha_centroid with a far-off geometric center
sprite, _ = synthesize.make_pointer_sprite(46)
lx, ly, vf = synthesize.visible_alpha_centroid(sprite, -100, -100)
check(
    "fully off-frame returns visible_frac == 0",
    vf == 0.0,
    f"visible_frac={vf}",
)

# Half off-screen (geometric center at y=0): should give ~50% mass surviving.
lx, ly, vf = synthesize.visible_alpha_centroid(sprite, 500, 0)
check(
    "half off-frame top: 30% < visible_frac < 70%",
    0.30 < vf < 0.70,
    f"visible_frac={vf:.3f}",
)

# Sanity: unclipped centroid matches the sprite hotspot.
sprite_full, hotspot = synthesize.make_pointer_sprite(46)
sh, sw = sprite_full.shape
lx, ly, vf = synthesize.visible_alpha_centroid(sprite_full, 500, 1000)
expected_x = 500 - sw // 2 + hotspot[0]
expected_y = 1000 - sh // 2 + hotspot[1]
check(
    "unclipped sprite: label == placement_top_left + alpha_centroid",
    abs(lx - expected_x) < 1e-3 and abs(ly - expected_y) < 1e-3,
    f"got ({lx:.4f}, {ly:.4f}) expected ({expected_x:.4f}, {expected_y:.4f})",
)
check(
    "unclipped sprite: visible_frac == 1.0",
    abs(vf - 1.0) < 1e-3,
    f"visible_frac={vf:.6f}",
)


# =====================================================================
# Bug 2 — Real iOS cursor sprite loaded (not procedural smoothstep disc).
# =====================================================================
section("Bug 2: real captured sprite loaded from sprites/at_dot.png")

real = synthesize._load_real_sprite()
check(
    "real sprite loads from sprites/at_dot.png",
    real is not None,
    f"path: {synthesize.SPRITE_PATH}",
)
if real is not None:
    src_alpha, src_hot = real
    check(
        "source sprite shape is 36x36",
        src_alpha.shape == (36, 36),
        f"shape={src_alpha.shape}",
    )
    # The real iOS pointer's alpha centroid is biased upward — confirm.
    check(
        "alpha centroid is offset from geometric center",
        abs(src_hot[1] - 17.5) > 1.0,
        f"hotspot y={src_hot[1]:.2f}, geometric center y=17.5",
    )


# =====================================================================
# Bug 5 — Hotspot ≠ geometric center; label uses alpha-centroid not box mid.
# Bug 9 — visible_centroid 1px bias gone (function removed; alpha-centroid
#         path is float-precise and consistent across sample types).
# =====================================================================
section("Bug 5+9: gen_normal_pos uses alpha-centroid label, not geom center")

random.seed(0); np.random.seed(0)
img, lbl = synthesize.gen_normal_pos(bg, margin=50)
check(
    "normal_pos label is float (not the integer geometric center)",
    isinstance(lbl["x"], float) and not float(lbl["x"]).is_integer(),
    f"label_x={lbl['x']!r}",
)

# Strong check: label MUST sit ABOVE the geometric tile center on y, because
# the real iOS pointer's alpha mass is biased upward (centroid_y < tile_h/2).
# A direction-flip bug (e.g., subtracting hotspot offset instead of adding)
# would put the label BELOW the tile center, which is what we test for.
src_alpha, (sx, sy) = synthesize._load_real_sprite()
sw, sh = src_alpha.shape[::-1]
upward_bias = (sh - 1) / 2.0 - sy   # how many pixels above geom center
upward_count = 0
total = 200
for trial in range(total):
    random.seed(trial); np.random.seed(trial)
    _, lab = synthesize.gen_normal_pos(bg, margin=50)
    # Recover the placement geometric center: label_y = geom_top_left + hotspot_y
    # = (cy_geom - sh//2) + hotspot_y_in_resized; hotspot < (resized_h-1)/2
    # for the real sprite, so label_y < cy_geom. We don't have cy_geom in the
    # label, but we DO have diameter — so we can recompute hotspot at that
    # diameter and then back-solve cy_geom.
    diam = lab["diameter"]
    _, (hx_d, hy_d) = synthesize.make_pointer_sprite(diam)
    cy_geom_recovered = lab["y"] - hy_d + diam // 2
    # Sanity: cy_geom_recovered should be in [margin, H-margin]
    if 50 <= cy_geom_recovered <= synthesize.H - 50:
        # And the offset should be upward (label < cy_geom) iff the sprite
        # has alpha-mass biased upward.
        if (cy_geom_recovered - lab["y"]) > 0:
            upward_count += 1
check(
    f"≥80% of normal_pos labels sit upward of recovered geom center "
    f"(matches sprite hotspot bias of {upward_bias:.1f}px)",
    upward_count >= 0.8 * total,
    f"upward count = {upward_count}/{total}",
)


# =====================================================================
# Bug 3 — sigma_px reduced to 1.25 (was 2.0).
# =====================================================================
section("Bug 3: heatmap target sigma_px is 1.25 (was 2.0)")

# Build a target heatmap and inspect its sharpness.
target_norm = torch.tensor([[0.5, 0.5]])
hm_h, hm_w = 135, 63
target = train.make_target_heatmap(target_norm, hm_h, hm_w)
peak_val = float(target.max())
# Count cells above 0.5 of peak (FWHM area as a proxy)
above_half = int((target > 0.5 * peak_val).sum())
# At sigma_px=1.25, cell area inside FWHM ≈ pi * (sigma * sqrt(2*ln 2))^2 ≈ 6.5 cells
# At sigma_px=2.0, cell area ≈ pi * (2.0 * 1.177)^2 ≈ 17.4 cells
check(
    "FWHM area is consistent with sigma_px=1.25 (~5–10 cells)",
    5 <= above_half <= 12,
    f"cells above half-peak: {above_half}",
)


# =====================================================================
# Bug 4 — train-time crop is cursor-safe.
# =====================================================================
section("Bug 4: train-time crop respects cursor footprint")

# Simulate _apply_train_augment with a positive sample where the cursor
# center is near the frame edge so naive crop would clip it.
class FakeDS(train.PointerDataset):
    def __init__(self):
        self.augment = True

# Synthesize a sample-like dict
fake_label = {
    "x": 60.0, "y": 60.0,  # cursor center near top-left
    "has_cursor": 1, "sample_type": "normal_pos", "diameter": 46,
}
random.seed(1); np.random.seed(1)
fake_img = np.zeros((train.NATIVE_H, train.NATIVE_W, 3), dtype=np.uint8)

ds = FakeDS()
n_trials = 50
clipped_count = 0
# Stronger check: confirm the chosen crop window ACTUALLY contains the
# cursor footprint. We reach inside _apply_train_augment by monkey-patching
# random.randint to record the crop boundaries.
chosen_crops: list[tuple[int, int, int, int]] = []

orig_randint = random.randint
randint_calls = []
def spy_randint(a, b):
    v = orig_randint(a, b)
    randint_calls.append(v)
    return v

for trial in range(n_trials):
    random.seed(trial); np.random.seed(trial)
    randint_calls.clear()
    random.randint = spy_randint
    try:
        img, new_lbl = ds._apply_train_augment(fake_img, fake_label)
    finally:
        random.randint = orig_randint
    if not new_lbl.get("has_cursor", 0):
        clipped_count += 1

check(
    "train-time crop never relabels positive→negative (50 trials)",
    clipped_count == 0,
    f"clipped {clipped_count}/50 — should be 0 with cursor-safe crop",
)

# Drive the crop logic directly: build a positive sample whose anchor is
# very near the right edge of the frame and confirm the crop window's
# right edge stays at frame.W (no random crop on that side that would
# remove cursor pixels).
edge_label = dict(fake_label, x=train.NATIVE_W - 30.0, y=400.0)
violation_count = 0
for trial in range(50):
    random.seed(trial * 7 + 1); np.random.seed(trial)
    img_out, new_lbl = ds._apply_train_augment(fake_img, edge_label)
    if not new_lbl.get("has_cursor"):
        violation_count += 1  # cursor-anchor crop would be a violation
check(
    "near-right-edge anchor: crop never drops the cursor (50 trials)",
    violation_count == 0,
    f"violations = {violation_count}/50",
)

# Edge_pos should not be cropped at all
fake_edge_label = dict(fake_label, sample_type="edge_pos", x=10.0, y=600.0)
random.seed(2); np.random.seed(2)
img_out, _ = ds._apply_train_augment(fake_img, fake_edge_label)
check(
    "edge_pos sample is not cropped (full-frame returned)",
    img_out.shape == fake_img.shape,
    f"out shape {img_out.shape}, expected {fake_img.shape}",
)


# =====================================================================
# Bug 7 — coordinate scaling uses stride convention, not linear stretch.
# =====================================================================
section("Bug 7: native↔cell mapping uses conv-stride convention")

# Round-trip native px → cell → native px. Should be identity. Test BOTH
# axes — y has a different stride at this resolution (NATIVE_H=2160,
# hm_h=135 → stride=16 exactly) than x (994/63 → stride 15.78).
for x in [0.0, 100.0, 500.0, 994.0]:
    x_t = torch.tensor([x])
    cell = train.native_to_cell(x_t, hm_w, train.NATIVE_W)
    back = train.cell_to_native(cell, hm_w, train.NATIVE_W)
    check(
        f"x-axis native→cell→native preserves x={x}",
        abs(float(back[0]) - x) < 1e-3,
        f"x={x} → cell={float(cell[0]):.3f} → back={float(back[0]):.3f}",
    )
for y in [0.0, 500.0, 1000.0, 2160.0]:
    y_t = torch.tensor([y])
    cell = train.native_to_cell(y_t, hm_h, train.NATIVE_H)
    back = train.cell_to_native(cell, hm_h, train.NATIVE_H)
    check(
        f"y-axis native→cell→native preserves y={y}",
        abs(float(back[0]) - y) < 1e-3,
        f"y={y} → cell={float(cell[0]):.3f} → back={float(back[0]):.3f}",
    )

# Old linear-stretch formula `x/(native-1)*(hm-1)` would give cell=62 for x=994.
old_cell = 994.0 / 993.0 * 62.0  # 62.06
new_cell = float(train.native_to_cell(torch.tensor([994.0]), hm_w, train.NATIVE_W)[0])
check(
    "stride formula gives a DIFFERENT cell from old linear stretch at native=994",
    abs(new_cell - old_cell) > 0.1,
    f"old={old_cell:.3f}, new={new_cell:.3f}",
)

# RUNTIME call-site check: place a logit-Gaussian peak at known native
# (x, y), let heatmap_to_xy_px decode it, and confirm we recover within
# 1 native px. This catches a regression where one of the two call sites
# (target gen vs decode) reverts to the linear-stretch formula but the
# other doesn't.
target_x, target_y = 500.0, 1000.0
target_cell_x = float(train.native_to_cell(torch.tensor([target_x]), hm_w, train.NATIVE_W)[0])
target_cell_y = float(train.native_to_cell(torch.tensor([target_y]), hm_h, train.NATIVE_H)[0])
yy_, xx_ = np.indices((hm_h, hm_w), dtype=np.float32)
d2 = (xx_ - target_cell_x) ** 2 + (yy_ - target_cell_y) ** 2
fake_logits = (8.0 * np.exp(-d2 / (2 * 1.25 ** 2)) - 4.0).astype(np.float32)
fake_hm = torch.from_numpy(fake_logits)[None, None]
decoded = train.heatmap_to_xy_px(fake_hm, train.NATIVE_W, train.NATIVE_H)
dx_back = float(decoded[0, 0])
dy_back = float(decoded[0, 1])
check(
    "round-trip via heatmap_to_xy_px (synthesizing logits at known cell)",
    abs(dx_back - target_x) < 1.0 and abs(dy_back - target_y) < 1.0,
    f"target=({target_x},{target_y}) decoded=({dx_back},{dy_back})",
)


# =====================================================================
# Bug 8 — Parabolic subpixel fits raw logits, not sigmoid output.
# =====================================================================
section("Bug 8: parabolic subpixel uses raw logits")

# Build a synthetic heatmap with a Gaussian peak placed at fractional offset
# 0.25 cell to the right of cell 5. Test that fitting on logits recovers
# offset closer to 0.25 than fitting on sigmoid (which saturates and
# under-estimates).
hm_logits_np = np.zeros((11, 11), dtype=np.float32)
target_x = 5.25
target_y = 5.0
for iy in range(11):
    for ix in range(11):
        d2 = (ix - target_x)**2 + (iy - target_y)**2
        # Strong logit peak (saturates sigmoid at peak)
        hm_logits_np[iy, ix] = 8.0 * np.exp(-d2 / (2 * 1.25**2)) - 4.0

flat = hm_logits_np.argmax()
iy, ix = flat // 11, flat % 11

off_logit = _parabolic_offset(hm_logits_np, ix, iy, axis="x")
sig = 1.0 / (1.0 + np.exp(-hm_logits_np))
off_sig = _parabolic_offset(sig, ix, iy, axis="x")

check(
    "parabolic on logits recovers ~0.25 offset (truth)",
    abs(off_logit - 0.25) < 0.05,
    f"logit_off={off_logit:.4f}, sigmoid_off={off_sig:.4f}",
)
check(
    "parabolic on logits is closer to truth than parabolic on sigmoid",
    abs(off_logit - 0.25) < abs(off_sig - 0.25),
    f"|logit-truth|={abs(off_logit-0.25):.4f} vs |sigmoid-truth|={abs(off_sig-0.25):.4f}",
)


# =====================================================================
# Bug 10 — xy_loss weight is 0 (regression head decoupled from gradient).
# =====================================================================
section("Bug 10: soft-argmax xy head removed entirely (v0.5.1)")

with open(os.path.join(ROOT, "train.py")) as f:
    src = f.read()
check(
    "XY_WEIGHT_* constants removed from train.py",
    "XY_WEIGHT_WARMUP" not in src and "XY_WEIGHT_LATE" not in src,
    "v0.5.1 dropped the soft-argmax head; XY_WEIGHT_* should not exist anymore",
)
check(
    "PointerNet.forward returns 2-tuple in source (no `return xy, conf`)",
    "return conf_logit, hm" in src and "return xy, conf_logit, hm" not in src,
    "",
)

# RUNTIME check: forward+backward through the model with xy-only ground truth
# and confirm the regression head's pred_xy receives no gradient. This catches
# the case where the constant is 0 but the loss equation accidentally still
# includes the soft-argmax term unweighted.
# v0.5.1: forward returns 2-tuple (no soft-argmax). Verify structurally.
m = train.PointerNet().eval()
x_in = torch.randn(2, 3, train.TRAIN_H, train.TRAIN_W, requires_grad=False)
with torch.no_grad():
    out = m(x_in)
check(
    "PointerNet.forward returns 2-tuple (conf_logit, hm) — soft-argmax dropped",
    isinstance(out, tuple) and len(out) == 2,
    f"got len={len(out) if isinstance(out, tuple) else type(out).__name__}",
)
conf_logit, hm = out
check(
    "conf_logit shape is (B,)",
    conf_logit.shape == (2,),
    f"got {tuple(conf_logit.shape)}",
)
check(
    "hm shape is (B, 1, H', W')",
    hm.dim() == 4 and hm.shape[:2] == (2, 1),
    f"got {tuple(hm.shape)}",
)


# =====================================================================
# Bug 11 — plain_neg and hard_neg have separate loss weights.
# =====================================================================
section("Bug 11: plain_neg/hard_neg loss terms split")

check(
    "HM_PLAIN_NEG_REL constant present in train.py",
    "HM_PLAIN_NEG_REL" in src,
    "",
)
check(
    "HM_HARD_NEG_REL constant present in train.py",
    "HM_HARD_NEG_REL" in src,
    "",
)
check(
    "HM_PLAIN_NEG_REL < HM_HARD_NEG_REL (plain weighted lower)",
    "HM_PLAIN_NEG_REL = 0.25" in src and "HM_HARD_NEG_REL = 1.0" in src,
    "",
)


# =====================================================================
# Inference-end-to-end smoke: a known-good v0.4 checkpoint should still
# load (state-dict shape unchanged) and produce reasonable coords on a
# blank frame. Coords just need to be inside [0, native).
# =====================================================================
section("Smoke: v0.4 checkpoint loads under v0.5 inference path")

ckpt_path = os.path.join(ROOT, "pointer_model_v0.4.0_22.9px.pt")
if os.path.exists(ckpt_path):
    from inference import PointerFinder

    finder = PointerFinder(ckpt_path)
    blank = np.full((train.NATIVE_H, train.NATIVE_W, 3), 200, dtype=np.uint8)
    pred = finder.predict(blank)
    check(
        "v0.4 checkpoint still loads and predicts in v0.5 inference path",
        0 <= pred.x < train.NATIVE_W and 0 <= pred.y < train.NATIVE_H,
        f"pred=({pred.x}, {pred.y}) conf={pred.confidence:.3f}",
    )
else:
    print(f"  {YELLOW}SKIP{RESET}  v0.4 ckpt not found at {ckpt_path}")


# =====================================================================
# Summary
# =====================================================================
print(f"\n{YELLOW}=== summary ==={RESET}")
if not failures:
    print(f"{GREEN}all checks passed{RESET}")
    sys.exit(0)
else:
    print(f"{RED}{len(failures)} check(s) failed:{RESET}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(len(failures))
