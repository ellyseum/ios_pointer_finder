"""
train.py — train a tiny CNN to find the iPhone Pointer-Control cursor in a
screen capture.

Architecture: small backbone (~338K params) → two heads:
  - heatmap:    1×1 conv at 1/8 of train resolution — primary localization signal
  - confidence: outputs P(cursor present) ∈ [0, 1]

Loss: heatmap BCE-with-logits split into pos / hard_neg / plain_neg terms
(weights 1.0 / 1.0 / 0.25) + BCE on confidence.

Input: native captures are 994 × 2160 (NATIVE_W × NATIVE_H). We downsample 2×
to 497 × 1080 (TRAIN_W × TRAIN_H) — fast inference, still enough detail for the
cursor sprite (~46 native px ≈ 23 px in train space) to form a sharp peak in
the 1/8-stride feature map.

Inference decode: hard argmax + parabolic refinement on raw logits, with a
stride-aware native-pixel mapping. Single canonical implementation lives in
inference.PointerFinder; train.py:heatmap_to_xy_px mirrors it for val.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, "dataset")
WEIGHTS_PATH = os.path.join(ROOT, "pointer_model.pt")
VERSION_PATH = os.path.join(ROOT, "VERSION")


def _stable_bg_hash(bg_id: str) -> int:
    """Deterministic non-cryptographic hash for stable train/val bg-level
    splits across dataset growth. Uses zlib.adler32 (in stdlib, fast,
    deterministic across Python versions and platforms). Whether a given
    `bg_id` lands in val is a per-id property — adding new bgs cannot
    re-shuffle the membership of any existing bg.
    """
    import zlib
    return zlib.adler32(bg_id.encode("utf-8"))


def save_checkpoint(ckpt: dict, path: str) -> None:
    """Suffix-aware checkpoint save. Pickle `.pt` OR safetensors + sidecar.

    For `.safetensors` paths, splits the dict into:
      - tensor weights → safetensors file
      - metadata (epoch, val_pos_err_px, etc.) → `<stem>.config.json` sidecar
        (matches `inference._load_sidecar_config` and
        `scripts/convert_pt_to_safetensors.py`).

    For all other suffixes (default `.pt`), uses `torch.save` (pickle).
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise SystemExit(
                "Saving .safetensors needs `pip install safetensors`."
            ) from e
        save_file(ckpt["model"], path)
        sidecar = os.path.splitext(path)[0] + ".config.json"
        meta = {k: v for k, v in ckpt.items() if k != "model"}
        # JSON can't serialize tuples → list; numpy/torch scalars stay as-is.
        meta = {k: list(v) if isinstance(v, tuple) else v for k, v in meta.items()}
        with open(sidecar, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        torch.save(ckpt, path)


def read_version() -> str:
    """Read semver from the VERSION file at repo root. Used to tag saved
    checkpoints so old versions are never overwritten by future training runs.
    Format: single line like '0.3.1'. Falls back to 'unversioned' if missing."""
    try:
        with open(VERSION_PATH) as f:
            return f.read().strip()
    except OSError:
        return "unversioned"

NATIVE_W, NATIVE_H = 994, 2160
TRAIN_W, TRAIN_H = 497, 1080  # 2x downsample (was 4x — cursor was sub-pixel
                              # at 4x_input × 8x_backbone = 1/32 effective; at
                              # 2x_input × 8x_backbone = 1/16, cursor 46/16 =
                              # ~3 px in feature map, learnable)

# Stride convention. Distinguishes the structural train-resolution stride
# (3 stride-2 conv blocks → 8x downsample, plus 2x train→native upsample =
# 16x effective) from the per-axis native-resolution stride used by the
# decoder (NATIVE / HM_DIM, asymmetric: 994/63=15.778 across X but
# 2160/135=16.0 across Y). The decoder uses the per-axis native stride
# because that's what `cell i → native pixel i*s + (s-1)/2` requires.
STRIDE_TRAIN = 16

# Architecture version pin. Bumped when the model's shape, head count,
# or decoded coordinate convention changes incompatibly. Stored in every
# checkpoint sidecar; loaders assert match before using weights.
#
#   v1 (implicit, pre-v0.7): 2-tuple forward (conf_logit, heatmap),
#       AdaptiveAvgPool conf head, BCE-mean heatmap loss.
#   v2 (v0.7+): same forward signature; added stride/version metadata
#       to the saved sidecar. Future v0.7 commits (#103 BCE-sum, #104
#       conf-head MaxPool) layer on top of this version pin.
ARCHITECTURE_VERSION = 2


# Sample type encoding for metric slicing. Negative type id = unknown/legacy
# (datasets generated before sample_type field was added).
SAMPLE_TYPES = {"normal_pos": 0, "edge_pos": 1, "hard_neg": 2, "plain_neg": 3}
SAMPLE_TYPES_INV = {v: k for k, v in SAMPLE_TYPES.items()}


class PointerDataset(Dataset):
    """
    BG-LEVEL train/val split: a sample-level shuffle leaks backgrounds into
    both train and val and makes val_pos_err artificially low. Instead, hold
    out a fraction of unique bg_ids; all samples from those bgs go to val.
    Falls back to sample-level split with a warning if labels lack bg_id
    (legacy datasets only).
    """
    def __init__(self, dataset_dir: str, train: bool = True, val_frac: float = 0.1,
                 val_bg_ids: set[str] | None = None,
                 augment: bool = False):
        """augment=True turns on train-time random crop + flip + photometric jitter
        in __getitem__. With only ~100 train backgrounds, returning the same baked
        JPEG every epoch lets the network memorize specific bg textures. Online
        augmentation breaks the spatial/photometric memorization shortcut and is
        the highest-leverage anti-overfit change in v0.3.1."""
        with open(os.path.join(dataset_dir, "labels.jsonl")) as f:
            all_labels = [json.loads(line) for line in f if line.strip()]

        has_bg_id = bool(all_labels) and "bg_id" in all_labels[0]
        if has_bg_id:
            if val_bg_ids is None:
                bg_ids = sorted({e["bg_id"] for e in all_labels})
                # Stable hash-based assignment: each bg_id's hash mod 100
                # decides membership against a percentile threshold derived
                # from val_frac. Adding or removing a bg in the source dataset
                # does NOT change which OTHER bgs land in val (membership is
                # a per-id property, not an index-into-list property).
                # `random.Random(42).sample(bg_ids, n)` was index-sensitive
                # and shifted the entire val set on any dataset growth.
                threshold = int(round(val_frac * 100))
                val_bg_ids = {b for b in bg_ids if _stable_bg_hash(b) % 100 < threshold}
                # Keep the legacy "at least one bg in val" guarantee.
                if not val_bg_ids and bg_ids:
                    val_bg_ids = {bg_ids[0]}
            entries = [e for e in all_labels if (e["bg_id"] in val_bg_ids) != train]
            self.val_bg_ids = set(val_bg_ids)
            self.split_kind = "bg-level"
        else:
            print("WARN: dataset missing bg_id field; falling back to sample-level "
                  "split (val_pos_err will be optimistic on this dataset).")
            random.Random(42).shuffle(all_labels)
            n_val = max(1, int(len(all_labels) * val_frac))
            entries = all_labels[n_val:] if train else all_labels[:n_val]
            self.val_bg_ids = set()
            self.split_kind = "sample-level (legacy fallback)"

        self.entries = entries
        self.dataset_dir = dataset_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def _apply_train_augment(self, img_native: np.ndarray, e: dict) -> tuple[np.ndarray, dict]:
        """Train-time augmentation that preserves cursor labels.

        v0.5.1 changes:
        - #12 asymmetric crop: protected region is the actual sprite bbox
          around the hotspot, not a symmetric radius around the label.
          Sprite top-left = label - hotspot; bottom-right = label + (d - hotspot).
          The captured iOS pointer's hotspot is in the upper-left quadrant,
          so the symmetric guard left the sprite bottom-right unprotected.
        - #15 disable H-flip on positives: real captured sprite is left-right
          asymmetric (~31% alpha asymmetry). Flipping shows the model a
          mirrored shape that doesn't exist at inference. Negs still flip.
        - #21 hard_neg crop now respects decoy footprint via persisted
          decoy_w/decoy_h (synthesize.py:gen_hard_neg). Worst-case decoy
          is `make_decoy_ellipse` at 119px wide; MAX_DECOY_EXTENT=120
          provides slack for any per-sample-extent fallback.
        - #28 fallback: when 8 trials exhaust, fall back to no-crop AND
          skip the flip+brightness on positives only (don't compound a
          fallback-cropped state with risky augmentation). Negs continue
          to flip+brightness on the no-crop path.
        """
        h, w = img_native.shape[:2]
        max_crop_x = int(w * 0.07)
        max_crop_y = int(h * 0.07)

        is_pos = bool(e["has_cursor"])
        is_edge_pos = is_pos and e.get("sample_type") == "edge_pos"
        is_hard_neg = (not is_pos) and e.get("sample_type") == "hard_neg"

        # Default crop window: full frame. Track whether we found a real crop.
        crop_x0, crop_y0 = 0, 0
        crop_x1, crop_y1 = w, h
        found_safe_crop = False

        if is_edge_pos:
            # edge_pos cursors sit on the frame boundary; further cropping
            # adds another clip we'd have to re-label. Skip the crop.
            pass
        elif is_pos:
            # Asymmetric protection around hotspot (#12). For a sprite of
            # size d×d placed with top-left at (label - hotspot), the bbox
            # extends [label_x - hx, label_x + (d - hx)] × [label_y - hy, ...].
            # Hotspot defaults to geometric center (d/2) when label dict
            # doesn't carry the actual hotspot — fallback for legacy data.
            d = int(e.get("diameter", 46))
            hx = float(e.get("hotspot_x", d / 2.0))
            hy = float(e.get("hotspot_y", d / 2.0))
            ax = float(e["x"]); ay = float(e["y"])
            margin = 2  # px slack
            req_l = ax - hx - margin; req_r = ax + (d - hx) + margin
            req_t = ay - hy - margin; req_b = ay + (d - hy) + margin
            for _ in range(8):
                c_x0 = random.randint(0, max_crop_x)
                c_y0 = random.randint(0, max_crop_y)
                c_x1 = w - random.randint(0, max_crop_x)
                c_y1 = h - random.randint(0, max_crop_y)
                if (c_x0 <= req_l and req_r <= c_x1
                        and c_y0 <= req_t and req_b <= c_y1):
                    crop_x0, crop_y0, crop_x1, crop_y1 = c_x0, c_y0, c_x1, c_y1
                    found_safe_crop = True
                    break
        elif is_hard_neg:
            # #21: protect the decoy footprint using persisted size, with
            # MAX_DECOY_EXTENT as the worst-case fallback for legacy data.
            MAX_DECOY_EXTENT = 120
            dpos = e.get("decoy_pos", [w // 2, h // 2])
            dx, dy = float(dpos[0]), float(dpos[1])
            dw = int(e.get("decoy_w", MAX_DECOY_EXTENT))
            dh = int(e.get("decoy_h", MAX_DECOY_EXTENT))
            req_l = dx - dw / 2.0 - 2; req_r = dx + dw / 2.0 + 2
            req_t = dy - dh / 2.0 - 2; req_b = dy + dh / 2.0 + 2
            for _ in range(8):
                c_x0 = random.randint(0, max_crop_x)
                c_y0 = random.randint(0, max_crop_y)
                c_x1 = w - random.randint(0, max_crop_x)
                c_y1 = h - random.randint(0, max_crop_y)
                if (c_x0 <= req_l and req_r <= c_x1
                        and c_y0 <= req_t and req_b <= c_y1):
                    crop_x0, crop_y0, crop_x1, crop_y1 = c_x0, c_y0, c_x1, c_y1
                    found_safe_crop = True
                    break
        else:
            # plain_neg — no decoy to preserve, free crop.
            crop_x0 = random.randint(0, max_crop_x)
            crop_y0 = random.randint(0, max_crop_y)
            crop_x1 = w - random.randint(0, max_crop_x)
            crop_y1 = h - random.randint(0, max_crop_y)
            found_safe_crop = True

        img = img_native[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_w = crop_x1 - crop_x0
        crop_h = crop_y1 - crop_y0

        new_label = dict(e)
        if is_pos:
            new_label["x"] = float(e["x"]) - crop_x0
            new_label["y"] = float(e["y"]) - crop_y0

        # #28: skip flip+brightness on positive fallback path. Negs still augment.
        skip_aug_on_fallback = is_pos and not found_safe_crop and not is_edge_pos

        # #15: disable H-flip on positives entirely (real sprite is asymmetric).
        # Negs still flip — bg flip is fine, decoy shapes don't have an
        # orientation to teach.
        if (not skip_aug_on_fallback) and (not is_pos) and random.random() < 0.5:
            img = img[:, ::-1].copy()
            # Negs have no x label to mirror.

        # Photometric: brightness jitter (mild — synth already has ±15% baked).
        # cv2.convertScaleAbs is C-implemented and ~10x faster than the
        # np.float32 round-trip approach we used in the slow first build of
        # v0.3.1 (28 min/epoch). Now ~6 min/epoch.
        # #28: skip on the positive fallback path so we don't compound an
        # already-degraded augmentation choice.
        if not skip_aug_on_fallback:
            b = random.uniform(0.92, 1.08)
            if abs(b - 1.0) > 0.005:
                img = cv2.convertScaleAbs(img, alpha=b)

        # Now resize the crop to TRAIN_W x TRAIN_H. Update label scaling so
        # x/y are still in NATIVE-pixel space (caller divides by NATIVE_W/H).
        # We rescale x_in_crop → x_in_native_equivalent so the existing
        # downstream xn = e["x"] / NATIVE_W normalization still produces the
        # correct in-image position [0, 1] of the cropped frame interpreted
        # as native-resolution.
        # v0.4: preserve floats through the rescale — the previous int(round())
        # truncated subpixel info that the heatmap target Gaussian needs to
        # converge below ~30 px. Downstream `xn = e["x"] / NATIVE_W` works
        # equally well on int or float labels.
        if new_label["has_cursor"]:
            new_label["x"] = new_label["x"] * NATIVE_W / crop_w
            new_label["y"] = new_label["y"] * NATIVE_H / crop_h
        return img, new_label

    def __getitem__(self, i: int):
        e = self.entries[i]
        path = os.path.join(self.dataset_dir, e["path"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f"PointerDataset: failed to read {path} "
                f"(corrupt JPEG or partial regen?)"
            )
        if self.augment:
            img, e = self._apply_train_augment(img, e)
        img = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
        # to torch CHW float32 in [0, 1]
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        # imagenet-ish normalization (loose) — helps a tiny model converge
        x = (x - 0.5) / 0.25
        if e["has_cursor"]:
            xn = e["x"] / NATIVE_W
            yn = e["y"] / NATIVE_H
            target_xy = torch.tensor([xn, yn], dtype=torch.float32)
            target_conf = torch.tensor(1.0)
        else:
            target_xy = torch.tensor([0.0, 0.0], dtype=torch.float32)
            target_conf = torch.tensor(0.0)
        sample_type = torch.tensor(
            SAMPLE_TYPES.get(e.get("sample_type", ""), -1), dtype=torch.long)
        return x, target_xy, target_conf, sample_type


class PointerNet(nn.Module):
    """
    Fully-convolutional cursor finder. Outputs a 1-channel heatmap; argmax +
    parabolic-on-logits + stride-aware decode gives the cursor location at
    sub-cell precision. This preserves spatial information (unlike a
    GAP+regression architecture which collapses 'where' info before the head).

    Backbone: 3 stride-2 convs + 2 stride-1 convs → 1/8 downsample of TRAIN
    resolution (497 × 1080 → 63 × 135 feature map). Supervised with Gaussian-
    target heatmap BCE-with-logits.

    Confidence head: predicts log-probability that any cursor is in frame, from
    a global-pooled sibling branch.

    Forward returns (conf_logit, heatmap_logits) — 2-tuple.
    """
    def __init__(self, dropout_p: float = 0.10):
        super().__init__()
        # 3 stride-2 blocks → 1/8 downsample. With 46-px cursor at 1/4 train
        # resolution = 11.5 px, at 1/8 feature map = ~5-6 pixel peak. Was 4
        # blocks (1/16) which gave only ~3-px peak — too small to learn.
        # v0.3.1: Dropout2d after the last two conv blocks to fight bg-texture
        # memorization at this 338K-param scale.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        # Heatmap head: 1-channel score map at 1/8 train resolution
        # (= 1/16 native resolution after the 2x train→native upsample).
        # Three stride-2 conv blocks above + 2x train→native = 16x effective.
        self.heatmap_head = nn.Conv2d(128, 1, 1)
        # Confidence head: global avg pool + small MLP
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Returns (conf_logit, heatmap_logits). The earlier soft-argmax `xy`
        # output was dropped: it was weighted at 0 in the loss (kept around
        # for back-compat) but cost a softmax over the full heatmap each
        # forward AND used a linear-stretch coordinate convention while the
        # heatmap target used a stride convention — different objectives.
        # Inference path: hard argmax + parabolic refinement on raw logits
        # (`heatmap_to_xy_px`, `inference._parabolic_offset`).
        feat = self.backbone(x)          # (B, 128, H', W')
        hm = self.heatmap_head(feat)     # (B, 1, H', W') — raw logits
        conf_logit = self.conf_head(feat).squeeze(1)
        return conf_logit, hm


def native_to_cell(x_native: torch.Tensor, hm_dim: int, native_dim: int) -> torch.Tensor:
    """Map native-pixel coords to heatmap-cell coords, using the conv-stride
    convention that cell `i` has receptive-field center at native pixel
    `i*stride + (stride-1)/2`, where `stride = native_dim / hm_dim`.

    This replaces the v0.4 linear-stretch formula `x/(native-1) * (hm-1)`,
    which mapped cell 0 → native 0 and cell (hm-1) → native (native-1) and
    forced the model to learn a non-uniform spatial warp to compensate.
    """
    stride = native_dim / hm_dim
    return (x_native - (stride - 1.0) / 2.0) / stride


def cell_to_native(cell: torch.Tensor, hm_dim: int, native_dim: int) -> torch.Tensor:
    """Inverse of native_to_cell. Used to decode heatmap argmax → native px."""
    stride = native_dim / hm_dim
    return cell * stride + (stride - 1.0) / 2.0


def heatmap_to_xy_px(hm_logits: torch.Tensor, native_w: int, native_h: int) -> torch.Tensor:
    """Vectorized argmax + parabolic-subpixel refinement matching deployed inference.

    Given a batch of heatmap logits (B, 1, H, W), returns (B, 2) of (x, y) in
    native pixel coordinates. Mirrors `inference.py:_parabolic_offset` + the
    coordinate mapping in `predict()`.

    v0.5: parabolic fit applied directly to RAW LOGITS, not sigmoid output.
    The training target is a Gaussian `exp(-d²/2σ²)`; logit(target) ≈ -d²/2σ²
    near the peak, which is parabolic in `d`. Fitting a parabola to the
    sigmoid suffers saturation bias near the peak (where sigmoid → 1 and the
    second derivative collapses), suppressing the subpixel offset toward 0.
    """
    hm = hm_logits[:, 0]  # (B, H, W) RAW logits — see v0.5 note above
    B, H, W = hm.shape
    flat = hm.view(B, -1).argmax(dim=1)
    ix = flat % W
    iy = flat // W

    # Parabolic offset along x (zero on horizontal borders or flat fits).
    can_x = (ix > 0) & (ix < W - 1)
    bidx = torch.arange(B, device=hm.device)
    ax = hm[bidx, iy, (ix - 1).clamp(min=0)]
    bx = hm[bidx, iy, ix]
    cx_ = hm[bidx, iy, (ix + 1).clamp(max=W - 1)]
    denom_x = ax - 2 * bx + cx_
    safe_denom_x = torch.where(denom_x.abs() > 1e-9, denom_x, torch.full_like(denom_x, 1e-9))
    off_x = 0.5 * (ax - cx_) / safe_denom_x
    off_x = torch.where(denom_x.abs() > 1e-9, off_x, torch.zeros_like(off_x))
    off_x = off_x.clamp(-0.5, 0.5)
    off_x = torch.where(can_x, off_x, torch.zeros_like(off_x))

    # Parabolic offset along y.
    can_y = (iy > 0) & (iy < H - 1)
    ay = hm[bidx, (iy - 1).clamp(min=0), ix]
    by = hm[bidx, iy, ix]
    cy_ = hm[bidx, (iy + 1).clamp(max=H - 1), ix]
    denom_y = ay - 2 * by + cy_
    safe_denom_y = torch.where(denom_y.abs() > 1e-9, denom_y, torch.full_like(denom_y, 1e-9))
    off_y = 0.5 * (ay - cy_) / safe_denom_y
    off_y = torch.where(denom_y.abs() > 1e-9, off_y, torch.zeros_like(off_y))
    off_y = off_y.clamp(-0.5, 0.5)
    off_y = torch.where(can_y, off_y, torch.zeros_like(off_y))

    rx = ix.float() + off_x
    ry = iy.float() + off_y
    # v0.5: stride-aware cell→native mapping (replaces v0.4 linear stretch).
    cx_px = cell_to_native(rx, W, native_w).round().clamp(0, native_w - 1)
    cy_px = cell_to_native(ry, H, native_h).round().clamp(0, native_h - 1)
    return torch.stack([cx_px, cy_px], dim=1)


def make_target_heatmap(target_xy_norm: torch.Tensor, hm_h: int, hm_w: int,
                        sigma_px: float = 1.25) -> torch.Tensor:
    """
    Build (B, 1, H', W') Gaussian target heatmap centered at each (x_norm, y_norm).
    `target_xy_norm` is (x/NATIVE_W, y/NATIVE_H) — the caller owns native-px
    coords; this function maps native → cell using the v0.5 stride convention.

    sigma_px = stddev in heatmap-cell units. v0.5 default 1.25 (was 2.0):
      - hm cell ≈ 16 native px (stride 16 from native to feature map)
      - sigma_px = 1.25 → 20 native-px stddev → FWHM ≈ 47 native px ≈ cursor
        diameter. Previous 2.0 gave FWHM ~75 px (1.6× cursor) — too diffuse
        to enforce sub-cursor-radius precision.
    """
    B = target_xy_norm.shape[0]
    device = target_xy_norm.device
    # Recover native-px coords, then map to cells via stride convention.
    # We accept the slight redundancy (x_norm * native = native_x; native_x →
    # cell) for clarity over collapsing into a single hidden formula.
    x_native = target_xy_norm[:, 0] * NATIVE_W
    y_native = target_xy_norm[:, 1] * NATIVE_H
    cx = native_to_cell(x_native, hm_w, NATIVE_W)
    cy = native_to_cell(y_native, hm_h, NATIVE_H)
    yy = torch.arange(hm_h, device=device, dtype=torch.float32).view(1, hm_h, 1)
    xx = torch.arange(hm_w, device=device, dtype=torch.float32).view(1, 1, hm_w)
    dy2 = (yy - cy.view(B, 1, 1)) ** 2
    dx2 = (xx - cx.view(B, 1, 1)) ** 2
    target_hm = torch.exp(-(dx2 + dy2) / (2 * sigma_px ** 2))
    return target_hm.unsqueeze(1)  # (B, 1, H, W)


def train_loop(args):
    version = read_version()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ios_pointer_finder v{version}  device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Global RNG seeding. Without this, `torch.initial_seed()` is
    # non-deterministic, which propagates into augmentation, dataloader
    # shuffle order, model init, and dropout. With it, training is
    # bit-exact across runs (as far as floating-point determinism allows;
    # see --strict-determinism for the cudnn knobs).
    #
    # When run inside `train_continuous.sh`'s warm-restart loop, the
    # environment variable `IPF_PASS_ID` is set per pass. We mix it into
    # the seed so each pass sees a different DataLoader shuffle order and
    # augmentation sequence — without this, every warm-restart pass would
    # see identical data ordering, defeating the diversity benefit of
    # warm restarts.
    base_seed = int(args.seed) if args.seed is not None else 42
    pass_id_str = os.environ.get("IPF_PASS_ID", "").strip()
    pass_offset = int(pass_id_str) if pass_id_str.isdigit() else 0
    seed = (base_seed + pass_offset * 1009) % (2 ** 31)  # 1009 prime, decorrelates passes

    # (CUBLAS_WORKSPACE_CONFIG is set in main() before train_loop runs, so
    # cuBLAS sees it at handle init regardless of where torch.cuda gets
    # touched first inside this function.)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(args, "strict_determinism", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # No `warn_only=True`: if a non-deterministic op runs, we want the
        # explicit RuntimeError — that's the signal the run isn't bit-exact.
        torch.use_deterministic_algorithms(True)
    print(f"  base_seed={base_seed}  pass_offset={pass_offset}  effective_seed={seed}  "
          f"strict_determinism={getattr(args, 'strict_determinism', False)}")

    # If resuming, peek at the checkpoint metadata FIRST so we can carry the
    # previous run's val_bg_ids forward — keeps the val_pos_err metric on
    # exactly the same held-out set across resume boundaries.
    persisted_val_bg_ids: set[str] | None = None
    if args.resume and os.path.exists(args.resume):
        suffix = os.path.splitext(args.resume)[1].lower()
        meta = None
        if suffix == ".safetensors":
            sidecar = os.path.splitext(args.resume)[0] + ".config.json"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    meta = json.load(f)
        else:
            # `weights_only=True` is the safe peek path: it loads only
            # tensor data + a small whitelist of plain Python types, so the
            # full pickle (and its arbitrary-code-execution surface) never
            # runs. We just need the metadata fields, not the model state
            # dict — that gets loaded later on `device` via the canonical
            # weights_only=False path.
            try:
                _peek = torch.load(args.resume, map_location="cpu", weights_only=True)
                meta = _peek if isinstance(_peek, dict) else None
                del _peek
            except Exception:
                meta = None
        if isinstance(meta, dict) and meta.get("val_bg_ids"):
            persisted_val_bg_ids = set(meta["val_bg_ids"])
            print(f"  resume: inheriting val_bg_ids from checkpoint ({len(persisted_val_bg_ids)} bgs)")

    train_ds = PointerDataset(args.dataset, train=True, val_frac=args.val_frac,
                              augment=args.augment, val_bg_ids=persisted_val_bg_ids)
    val_ds = PointerDataset(args.dataset, train=False, val_frac=args.val_frac,
                            val_bg_ids=train_ds.val_bg_ids, augment=False)

    # `--limit-val N` for fast regression smoke. Shuffle val entries with a
    # fixed seed BEFORE slicing — `labels.jsonl` is written background-major
    # and PointerDataset preserves order, so a naive `entries[:N]` would land
    # on a single bg and mis-measure generalization.
    if getattr(args, "limit_val", 0) > 0 and args.limit_val < len(val_ds.entries):
        rng = random.Random(424242)
        shuffled = list(val_ds.entries)
        rng.shuffle(shuffled)
        val_ds.entries = shuffled[: args.limit_val]
        print(f"  --limit-val {args.limit_val}: shuffled+sliced val to {len(val_ds.entries)} samples")

    print(f"train: {len(train_ds)}  val: {len(val_ds)}  split: {train_ds.split_kind}  "
          f"train_augment={args.augment}")
    if train_ds.val_bg_ids:
        print(f"  val bgs ({len(train_ds.val_bg_ids)}): {sorted(train_ds.val_bg_ids)[:6]}...")

    # PyTorch DataLoader reseeds *torch* RNGs per worker by default, but NOT
    # Python's `random` or numpy's `np.random`. The augment path uses
    # `random.*`, so without this hook workers fork with identical Python
    # RNG state and produce correlated augmentations across the batch.
    #
    # `torch.initial_seed()` already encodes the worker_id (DataLoader sets
    # `base + worker_id` per worker before our hook runs); using it directly
    # gives each worker a unique, deterministic seed without double-offsetting.
    def _worker_init_fn(worker_id: int) -> None:
        seed = torch.initial_seed() % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=False,
                           worker_init_fn=_worker_init_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True,
                         worker_init_fn=_worker_init_fn)

    model = PointerNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # Resume from existing weights — keeps the model trained across many
    # epoch budgets instead of restarting from scratch every run. Useful for
    # "let it cook overnight" loops where each `train.py` invocation does
    # 30 more epochs on top of whatever was trained before. Optimizer + LR
    # schedule reset on each invocation, which acts like a cosine restart
    # (helps escape local minima).
    resume_best = float("inf")
    if args.resume and os.path.exists(args.resume):
        # Detect `.safetensors` by suffix and load via `safetensors.torch`.
        # Pull metadata (epoch, val_pos_err_px) from a `<stem>.config.json`
        # sidecar if present (matches `inference._load_sidecar_config` and
        # `scripts/convert_pt_to_safetensors.py`).
        suffix = os.path.splitext(args.resume)[1].lower()
        prev_epoch: object = "?"
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise SystemExit(
                    "--resume on .safetensors needs `pip install safetensors`."
                ) from e
            state_dict = load_file(args.resume)
            model.load_state_dict(state_dict)
            sidecar = os.path.splitext(args.resume)[0] + ".config.json"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    meta = json.load(f)
                resume_best = float(meta.get("val_pos_err_px", float("inf")))
                prev_epoch = meta.get("epoch", "?")
        else:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            resume_best = float(ckpt.get("val_pos_err_px", float("inf")))
            prev_epoch = ckpt.get("epoch", "?")
        print(f"resumed from {args.resume} (was epoch {prev_epoch}, "
              f"val_pos_err={resume_best:.1f}px)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss weights (v0.7).
    #
    # v0.7-7b: heatmap BCE switched to .sum(dim=(2,3)) — see hm_loss_per_b
    # below. Pre-v0.7 used .mean(dim=(1,2)) over 8505 cells, which made the
    # effective per-positive-cell heatmap gradient ~1400× weaker than the
    # confidence head's gradient (the model learned 'is something here'
    # fast and 'where exactly' slowly).
    #
    # HM_WEIGHT calibration history:
    #   - 2e-5: chosen by per-head gradient-norm balance at fresh init
    #     (25 steps from random weights). Failed in practice — Cold-Start
    #     #1 P1 stalled at 300 px val_pos_err vs v0.6.2's 36.7 at the same
    #     point. The fresh-init metric biased toward over-suppressing the
    #     heatmap path because BCE-sum gradient is huge at random init
    #     (model uniformly predicts ~0.5 on 8505 cells against a target
    #     that is 0 almost everywhere).
    #   - 2e-3 (current): chosen by total-loss-contribution match. Pre-v0.7
    #     mean-form had hm_loss × HM_WEIGHT ≈ 0.18 early (13% of total);
    #     2e-3 × sum-form hm_loss (~80) = 0.16 early (14% of total). This
    #     puts the optimizer in the same total-objective regime as v0.4
    #     (which hit 22.9 px on procedural disc) while preserving the
    #     v0.7 fix that the per-cell gradient is no longer diluted by the
    #     8505× cell-count denominator.
    HM_WEIGHT = 2e-3
    HM_PLAIN_NEG_REL = 0.25   # trivial backgrounds — small contribution
    HM_HARD_NEG_REL = 1.0     # decoy cursors — full weight, primary neg signal
    HM_NEG_REL_LEGACY = 0.5   # used when sample_type is unknown (legacy datasets)
    CONF_WEIGHT = 2.0         # unchanged from v0.6

    # Carry over the previous run's best so the rolling pointer (--weights-out)
    # only updates when this invocation actually beats the previous global
    # best. The within-pass best (pass_best below) is tracked separately so
    # we always emit a tagged snapshot of THIS pass's best, even if it didn't
    # beat the rolling pointer — surfaces v0.4 SGDR-cycle progress when a hot
    # restart momentarily plateaus above the previous global best.
    best_val_err = resume_best
    pass_best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        train_conf_loss = 0.0
        train_hm_pos_loss = 0.0
        train_hm_plain_neg_loss = 0.0; train_hm_hard_neg_loss = 0.0
        n_train = 0; n_train_pos = 0
        n_train_plain_neg = 0; n_train_hard_neg = 0
        for x, target_xy, target_conf, sample_type in train_dl:
            x = x.to(device, non_blocking=True)
            target_xy = target_xy.to(device, non_blocking=True)
            target_conf = target_conf.to(device, non_blocking=True)
            sample_type = sample_type.to(device, non_blocking=True)
            pred_conf_logit, pred_hm = model(x)
            B, _, H, W = pred_hm.shape

            pos_mask = target_conf
            neg_mask = 1.0 - target_conf
            pos_count = pos_mask.sum().clamp_min(1.0)

            # v0.5: split negatives into plain (trivial bg) vs hard (decoy
            # cursor) so we can weight them differently. Unknown-sample-type
            # entries (legacy datasets) fall back to the v0.4 single-weight
            # path so old datasets still train without changes.
            plain_id = SAMPLE_TYPES["plain_neg"]
            hard_id = SAMPLE_TYPES["hard_neg"]
            plain_mask = (sample_type == plain_id).float() * neg_mask
            hard_mask = (sample_type == hard_id).float() * neg_mask
            unknown_neg_mask = neg_mask * (1.0 - plain_mask) * (1.0 - hard_mask)
            plain_count = plain_mask.sum().clamp_min(1.0)
            hard_count = hard_mask.sum().clamp_min(1.0)
            unknown_neg_count = unknown_neg_mask.sum().clamp_min(1.0)

            # Build target heatmap; ZERO for negatives so no peak is correct.
            # (v0.3 fix — negatives previously had zero loss contribution via
            # the positive-only mask, so the model never learned the "no cursor
            # → flat heatmap" supervision signal.)
            target_hm_pos = make_target_heatmap(target_xy, H, W)
            target_hm = target_hm_pos * pos_mask.view(-1, 1, 1, 1)
            # v0.7-7b: sum over the 8505 cells, not mean. mean diluted the
            # localization gradient by the cell-count denominator; sum keeps
            # per-positive-cell pressure intact, and HM_WEIGHT compensates so
            # heatmap and conf head gradients are comparable at the shared
            # backbone (see HM_WEIGHT comment above).
            hm_loss_per_b = F.binary_cross_entropy_with_logits(
                pred_hm.squeeze(1), target_hm.squeeze(1), reduction='none'
            ).sum(dim=(1, 2))
            hm_pos_loss = (hm_loss_per_b * pos_mask).sum() / pos_count
            hm_plain_neg_loss = (hm_loss_per_b * plain_mask).sum() / plain_count
            hm_hard_neg_loss = (hm_loss_per_b * hard_mask).sum() / hard_count
            hm_unknown_neg_loss = (hm_loss_per_b * unknown_neg_mask).sum() / unknown_neg_count
            hm_loss = (hm_pos_loss
                      + HM_PLAIN_NEG_REL * hm_plain_neg_loss
                      + HM_HARD_NEG_REL * hm_hard_neg_loss
                      + HM_NEG_REL_LEGACY * hm_unknown_neg_loss)

            # v0.5.1: soft-argmax xy_loss removed entirely. PointerNet.forward
            # no longer computes pred_xy.
            conf_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, target_conf)
            loss = hm_loss * HM_WEIGHT + conf_loss * CONF_WEIGHT

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_conf_loss += conf_loss.item() * x.size(0)
            train_hm_pos_loss += hm_pos_loss.item() * pos_count.item()
            train_hm_plain_neg_loss += hm_plain_neg_loss.item() * plain_count.item()
            train_hm_hard_neg_loss += hm_hard_neg_loss.item() * hard_count.item()
            n_train += x.size(0)
            n_train_pos += int(pos_count.item())
            n_train_plain_neg += int(plain_count.item())
            n_train_hard_neg += int(hard_count.item())
        sched.step()

        # ---- Validation: sliced metrics by sample_type ----
        model.eval()
        slice_err_sum: dict[int, float] = {}      # heatmap+parabolic err (deployed inference path)
        slice_err_n: dict[int, int] = {}
        slice_fpr_pred: dict[int, int] = {}   # how many predicted positive (conf > 0.5)
        slice_fpr_n: dict[int, int] = {}
        slice_peak_high: dict[int, int] = {}  # how many had heatmap peak > 0.5
        with torch.no_grad():
            val_conf_correct = 0; n_val = 0
            for x, target_xy, target_conf, sample_type in val_dl:
                x = x.to(device); target_xy = target_xy.to(device)
                target_conf = target_conf.to(device); sample_type = sample_type.to(device)
                pred_conf_logit, pred_hm = model(x)
                pred_conf = torch.sigmoid(pred_conf_logit)
                pred_label = (pred_conf > 0.5).float()
                val_conf_correct += (pred_label == target_conf).sum().item()
                n_val += x.size(0)

                # Per-sample peak prob (heatmap-derived confidence)
                hm_prob = torch.sigmoid(pred_hm).view(x.size(0), -1)
                hm_peak = hm_prob.max(dim=1).values

                # Position error in NATIVE pixels using the SAME path inference uses:
                # heatmap argmax + parabolic subpixel refinement. Soft-argmax
                # (pred_xy) is kept as a secondary signal for backward log
                # comparison only — checkpoint selection uses err_px below.
                hm_pred_xy_px = heatmap_to_xy_px(pred_hm, NATIVE_W, NATIVE_H)
                target_x_px = target_xy[:, 0] * NATIVE_W
                target_y_px = target_xy[:, 1] * NATIVE_H
                e_xn = hm_pred_xy_px[:, 0] - target_x_px
                e_yn = hm_pred_xy_px[:, 1] - target_y_px
                err_px = torch.sqrt(e_xn ** 2 + e_yn ** 2)

                for t_id in sample_type.unique().tolist():
                    sel = (sample_type == t_id)
                    is_pos_type = (t_id == SAMPLE_TYPES["normal_pos"]
                                   or t_id == SAMPLE_TYPES["edge_pos"])
                    if is_pos_type:
                        pos_sel = sel & (target_conf == 1)
                        if pos_sel.any():
                            slice_err_sum[t_id] = slice_err_sum.get(t_id, 0.0) + err_px[pos_sel].sum().item()
                            slice_err_n[t_id] = slice_err_n.get(t_id, 0) + int(pos_sel.sum().item())
                    elif t_id == -1:
                        # Legacy datasets: route by target_conf instead of t_id (#44).
                        pos_sel = sel & (target_conf == 1)
                        neg_sel = sel & (target_conf == 0)
                        if pos_sel.any():
                            slice_err_sum[t_id] = slice_err_sum.get(t_id, 0.0) + err_px[pos_sel].sum().item()
                            slice_err_n[t_id] = slice_err_n.get(t_id, 0) + int(pos_sel.sum().item())
                        if neg_sel.any():
                            slice_fpr_pred[t_id] = slice_fpr_pred.get(t_id, 0) + int(pred_label[neg_sel].sum().item())
                            slice_fpr_n[t_id] = slice_fpr_n.get(t_id, 0) + int(neg_sel.sum().item())
                            slice_peak_high[t_id] = slice_peak_high.get(t_id, 0) + int((hm_peak[neg_sel] > 0.5).sum().item())
                    else:
                        # Negative slice (hard_neg / plain_neg): count false positives
                        slice_fpr_pred[t_id] = slice_fpr_pred.get(t_id, 0) + int(pred_label[sel].sum().item())
                        slice_fpr_n[t_id] = slice_fpr_n.get(t_id, 0) + int(sel.sum().item())
                        slice_peak_high[t_id] = slice_peak_high.get(t_id, 0) + int((hm_peak[sel] > 0.5).sum().item())

        dt = time.monotonic() - t0
        all_err_sum = sum(slice_err_sum.values()); all_err_n = sum(slice_err_n.values())
        # Legacy combined error (normal_pos + edge_pos averaged together).
        # Logged for v0.6.x baseline comparability but no longer drives
        # checkpoint selection — edge_pos labels are noisier (visible-centroid
        # of clipped sprite) and were systematically inflating pass-best.
        legacy_combined_pos_err = all_err_sum / max(1, all_err_n)
        # v0.7 #108: best-on-normal_pos. The metric the rest of the pipeline
        # cares about (real-frame eval, deployment click accuracy) is the
        # full-cursor case; edge_pos is a slice-level diagnostic.
        normal_pos_n = slice_err_n.get(SAMPLE_TYPES["normal_pos"], 0)
        normal_pos_sum = slice_err_sum.get(SAMPLE_TYPES["normal_pos"], 0.0)
        if normal_pos_n > 0:
            mean_pos_err = normal_pos_sum / normal_pos_n
        else:
            # No normal_pos samples in val (e.g., legacy dataset) — fall back
            # to combined so the metric is at least defined.
            mean_pos_err = legacy_combined_pos_err
        conf_acc = val_conf_correct / max(1, n_val)
        slice_lines = []
        for t_id, name in SAMPLE_TYPES_INV.items():
            if t_id in slice_err_n and slice_err_n[t_id] > 0:
                err = slice_err_sum[t_id] / slice_err_n[t_id]
                slice_lines.append(f"{name}_err={err:.1f}px(n={slice_err_n[t_id]})")
            if t_id in slice_fpr_n and slice_fpr_n[t_id] > 0:
                fpr_conf = slice_fpr_pred[t_id] / slice_fpr_n[t_id]
                fpr_peak = slice_peak_high[t_id] / slice_fpr_n[t_id]
                slice_lines.append(f"{name}_fpr_conf={fpr_conf*100:.1f}%_peak={fpr_peak*100:.1f}%(n={slice_fpr_n[t_id]})")
        slice_str = "  ".join(slice_lines) if slice_lines else "(legacy dataset; no slices)"

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_hm_pos={train_hm_pos_loss/max(1,n_train_pos):.4f} "
              f"train_hm_plain={train_hm_plain_neg_loss/max(1,n_train_plain_neg):.4f} "
              f"train_hm_hard={train_hm_hard_neg_loss/max(1,n_train_hard_neg):.4f} "
              f"train_conf={train_conf_loss/n_train:.4f}  "
              f"val_pos_err={mean_pos_err:.1f}px(normal_pos) "
              f"val_combined={legacy_combined_pos_err:.1f}px val_conf_acc={conf_acc*100:.1f}%  "
              f"lr={opt.param_groups[0]['lr']:.5f}  ({dt:.1f}s)")
        if slice_lines:
            print(f"        {slice_str}")

        if mean_pos_err < pass_best:
            pass_best = mean_pos_err
            ckpt = {"model": model.state_dict(),
                    "epoch": epoch,
                    "val_pos_err_px": mean_pos_err,
                    "val_combined_pos_err_px": legacy_combined_pos_err,
                    "val_conf_acc": conf_acc,
                    "native_size": (NATIVE_W, NATIVE_H),
                    "train_size": (TRAIN_W, TRAIN_H),
                    "version": version,
                    "architecture_version": ARCHITECTURE_VERSION,
                    "stride_train": STRIDE_TRAIN,
                    "metric_definition": "normal_pos_only",
                    # Persist val bg membership so resumes measure on the
                    # SAME held-out set, even if the dataset grows. Without
                    # this, even the stable-hash split can drift if the
                    # threshold rounding lands on a boundary case.
                    "val_bg_ids": sorted(train_ds.val_bg_ids)}
            # Per-best snapshot tagged with version AND val_pos_err. Always saved
            # on a within-pass new-best (even if it doesn't beat the rolling
            # global best) — eyeballing the dir shows the descent at a glance:
            # pointer_model_v0.3.1_64.2px.pt → ..._43.0px.pt → ..._30.8px.pt.
            #
            # When run inside the continuous-training loop, IPF_PASS_ID is set
            # to the current pass number — it's injected into the filename so
            # different warm-restart passes that happen to converge to the same
            # rounded val_pos_err don't overwrite each other's weights.
            pass_id = os.environ.get("IPF_PASS_ID", "").strip()
            pass_tag = f"_p{pass_id}" if pass_id else ""
            # Per-best snapshot uses the same suffix as --weights-out so users
            # who pass `--weights-out pointer_model.safetensors` get a
            # consistent set of output files (#51).
            wo_suffix = os.path.splitext(args.weights_out)[1] or ".pt"
            err_path = os.path.join(
                os.path.dirname(args.weights_out),
                f"pointer_model_v{version}{pass_tag}_{mean_pos_err:.1f}px{wo_suffix}")
            save_checkpoint(ckpt, err_path)
            log_extras = []
            # Capture the boolean BEFORE the update so the log tag is honest
            # even on an exact float tie with the previous global best.
            is_global = mean_pos_err < best_val_err
            if is_global:
                best_val_err = mean_pos_err
                save_checkpoint(ckpt, args.weights_out)
                log_extras.append(os.path.basename(args.weights_out))
            log_extras.append(os.path.basename(err_path))
            tag = "✓ saved global-best" if is_global else "↪ saved pass-best"
            print(f"  {tag} (val_pos_err={mean_pos_err:.1f}px) → " + " + ".join(log_extras))

    print(f"\nfinal pass best val_pos_err: {pass_best:.1f}px  "
          f"(rolling global best: {best_val_err:.1f}px)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_DIR)
    p.add_argument("--weights-out", default=WEIGHTS_PATH)
    p.add_argument("--epochs", type=int, default=15,
                   help="v0.3.1 default 15 — overfit window observed past ep 10 on v0.3, "
                        "augmentation should extend that but tighter T_max keeps LR "
                        "from annealing into ultra-low values that lock in memorization")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3,
                   help="v0.3.1 bumped from 1e-4 → 1e-3 to fight bg-texture overfit.")
    p.add_argument("--workers", type=int, default=8,
                   help="DataLoader workers; bumped from 4 to 8 in v0.3.1+ since "
                        "train-time augmentation made data loading more CPU-heavy")
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="fraction of bg_ids to hold out for validation (bg-level split)")
    p.add_argument("--limit-val", type=int, default=0,
                   help="if >0, evaluate on only N (shuffled) val samples per epoch — "
                        "fast regression smoke during dev. Shuffles with a fixed seed "
                        "before slicing so the slice is representative of all val "
                        "backgrounds, not just the first.")
    p.add_argument("--seed", type=int, default=None,
                   help="Global RNG seed for Python random, numpy, and torch "
                        "(default 42 if unset). Combined with --strict-determinism "
                        "for bit-exact repeats across runs.")
    p.add_argument("--strict-determinism", action="store_true",
                   help="Pin cudnn.deterministic=True and torch.use_deterministic_"
                        "algorithms(True). Slower; use for reproducibility audits.")
    p.add_argument("--augment", action="store_true", default=True,
                   help="enable train-time random crop + flip + photometric jitter "
                        "(v0.3.1 default ON — disable with --no-augment for ablation)")
    p.add_argument("--no-augment", action="store_false", dest="augment")
    p.add_argument("--resume", default="",
                   help="load weights from this file before training (continues training "
                        "instead of starting from scratch). Optimizer + LR schedule reset, "
                        "which acts like a cosine restart.")
    args = p.parse_args()
    # Set CUBLAS_WORKSPACE_CONFIG BEFORE any `torch.cuda.*` call (including
    # the device-name print at the top of train_loop, which triggers cuBLAS
    # handle initialization via _lazy_init). cuBLAS reads the env var only
    # at handle creation; setting it later is a no-op.
    if getattr(args, "strict_determinism", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return train_loop(args)


if __name__ == "__main__":
    sys.exit(main())
