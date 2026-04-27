"""
train.py — train a tiny CNN to find the iPhone BLE cursor in a snap.

Architecture: small backbone (~250K params) → two heads:
  - regression: outputs (x_norm, y_norm) ∈ [0, 1]² — cursor center
  - confidence: outputs σ ∈ [0, 1]            — prob cursor present

Loss: MSE on (x, y) for positive samples + BCE on confidence for all.

Input: training images downsampled to TRAIN_W x TRAIN_H. Native captures are
994 x 2160. We downsample 4x to 248 x 540 — fast inference, still enough
detail to localize an 81-px cursor (≈20 px in downsampled space).
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

NATIVE_W, NATIVE_H = 994, 2160
TRAIN_W, TRAIN_H = 497, 1080  # 2x downsample (was 4x — cursor was sub-pixel
                              # at 4x_input × 8x_backbone = 1/32 effective; at
                              # 2x_input × 8x_backbone = 1/16, cursor 46/16 =
                              # ~3 px in feature map, learnable)


# Sample type encoding for metric slicing. Negative type id = unknown/legacy
# (datasets generated before sample_type field was added).
SAMPLE_TYPES = {"normal_pos": 0, "edge_pos": 1, "hard_neg": 2, "plain_neg": 3}
SAMPLE_TYPES_INV = {v: k for k, v in SAMPLE_TYPES.items()}


class PointerDataset(Dataset):
    """
    BG-LEVEL train/val split.
    Hold out a fraction of unique bg_ids; all samples from those bgs go to val.
    Falls back to sample-level split with a warning if labels lack bg_id
    (legacy datasets only).
    """
    def __init__(self, dataset_dir: str, train: bool = True, val_frac: float = 0.1,
                 val_bg_ids: set[str] | None = None):
        with open(os.path.join(dataset_dir, "labels.jsonl")) as f:
            all_labels = [json.loads(line) for line in f if line.strip()]

        has_bg_id = bool(all_labels) and "bg_id" in all_labels[0]
        if has_bg_id:
            if val_bg_ids is None:
                bg_ids = sorted({e["bg_id"] for e in all_labels})
                n_val_bg = max(1, int(round(len(bg_ids) * val_frac)))
                val_bg_ids = set(random.Random(42).sample(bg_ids, n_val_bg))
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

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        img = cv2.imread(os.path.join(self.dataset_dir, e["path"]))
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
    Fully-convolutional cursor finder. Outputs a 1-channel heatmap; argmax gives
    cursor location. This preserves spatial information (unlike a GAP+regression
    architecture which collapses 'where' info before the head).

    Backbone: 4 stride-2 convs → 16x downsample. On 248x540 input the heatmap
    is 16x34. We use soft-argmax (spatial softmax + expected position) so the
    model can be trained with MSE on (x_norm, y_norm) directly — fully
    differentiable, sub-pixel accurate, no need for Gaussian-target heatmap loss.

    Confidence head: predicts log-probability that any cursor is in frame, from
    a global-pooled sibling branch.
    """
    def __init__(self):
        super().__init__()
        # 3 stride-2 blocks → 1/8 downsample. With 46-px cursor at 1/4 train
        # resolution = 11.5 px, at 1/8 feature map = ~5-6 pixel peak. Was 4
        # blocks (1/16) which gave only ~3-px peak — too small to learn.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        # Heatmap head: 1-channel score map at 1/16 resolution
        self.heatmap_head = nn.Conv2d(128, 1, 1)
        # Confidence head: global avg pool + small MLP
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)          # (B, 128, H', W')
        hm = self.heatmap_head(feat)     # (B, 1, H', W') — raw logits
        # Soft-argmax for inference (still differentiable, used as secondary signal)
        B, _, H, W = hm.shape
        flat = hm.view(B, H * W)
        prob = F.softmax(flat, dim=1).view(B, 1, H, W)
        ys = torch.linspace(0, 1, steps=H, device=hm.device).view(1, 1, H, 1)
        xs = torch.linspace(0, 1, steps=W, device=hm.device).view(1, 1, 1, W)
        ex = (prob * xs).sum(dim=(2, 3))
        ey = (prob * ys).sum(dim=(2, 3))
        xy = torch.cat([ex, ey], dim=1)
        conf_logit = self.conf_head(feat).squeeze(1)
        return xy, conf_logit, hm  # hm is raw logits — used directly for heatmap MSE loss


def make_target_heatmap(target_xy_norm: torch.Tensor, hm_h: int, hm_w: int,
                        sigma_px: float = 2.0) -> torch.Tensor:
    """
    Build (B, 1, H', W') Gaussian target heatmap centered at each (x_norm, y_norm).
    sigma_px = stddev in heatmap pixel units. Cursor is ~46 native px = ~3 px
    in feature map at 1/16 resolution, so sigma=2.0 gives a peak slightly
    wider than the cursor itself — enough gradient signal without training
    diffuse predictions.
    """
    B = target_xy_norm.shape[0]
    device = target_xy_norm.device
    cx = target_xy_norm[:, 0] * (hm_w - 1)
    cy = target_xy_norm[:, 1] * (hm_h - 1)
    yy = torch.arange(hm_h, device=device, dtype=torch.float32).view(1, hm_h, 1)
    xx = torch.arange(hm_w, device=device, dtype=torch.float32).view(1, 1, hm_w)
    dy2 = (yy - cy.view(B, 1, 1)) ** 2
    dx2 = (xx - cx.view(B, 1, 1)) ** 2
    target_hm = torch.exp(-(dx2 + dy2) / (2 * sigma_px ** 2))
    return target_hm.unsqueeze(1)  # (B, 1, H, W)


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    train_ds = PointerDataset(args.dataset, train=True, val_frac=args.val_frac)
    val_ds = PointerDataset(args.dataset, train=False, val_frac=args.val_frac,
                            val_bg_ids=train_ds.val_bg_ids)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  split: {train_ds.split_kind}")
    if train_ds.val_bg_ids:
        print(f"  val bgs ({len(train_ds.val_bg_ids)}): {sorted(train_ds.val_bg_ids)[:6]}...")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    model = PointerNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss weights (v0.3 — v0.3 fixes)
    HM_WEIGHT = 10.0          # heatmap BCE total
    HM_NEG_REL = 0.5          # neg term scaled relative to pos (Codex)
    XY_WEIGHT_WARMUP = 5.0    # soft-argmax MSE — full first 5 epochs
    XY_WEIGHT_LATE = 2.5      # demoted after warmup (Codex: hard-argmax inference)
    XY_WARMUP_EPOCHS = 5
    CONF_WEIGHT = 2.0         # bumped from 1.0 (Gemini: conf needs more pressure)

    best_val_err = float("inf")
    for epoch in range(1, args.epochs + 1):
        xy_w = XY_WEIGHT_WARMUP if epoch <= XY_WARMUP_EPOCHS else XY_WEIGHT_LATE
        model.train()
        t0 = time.monotonic()
        train_xy_loss = 0.0; train_conf_loss = 0.0
        train_hm_pos_loss = 0.0; train_hm_neg_loss = 0.0
        n_train = 0; n_train_pos = 0; n_train_neg = 0
        for x, target_xy, target_conf, _stype in train_dl:
            x = x.to(device, non_blocking=True)
            target_xy = target_xy.to(device, non_blocking=True)
            target_conf = target_conf.to(device, non_blocking=True)
            pred_xy, pred_conf_logit, pred_hm = model(x)
            B, _, H, W = pred_hm.shape

            pos_mask = target_conf
            neg_mask = 1.0 - target_conf
            pos_count = pos_mask.sum().clamp_min(1.0)
            neg_count = neg_mask.sum().clamp_min(1.0)

            # Build target heatmap; ZERO for negatives so no peak is correct.
            #
            target_hm_pos = make_target_heatmap(target_xy, H, W, sigma_px=2.0)
            target_hm = target_hm_pos * pos_mask.view(-1, 1, 1, 1)
            hm_loss_per_b = F.binary_cross_entropy_with_logits(
                pred_hm.squeeze(1), target_hm.squeeze(1), reduction='none'
            ).mean(dim=(1, 2))
            # Split pos/neg terms — explicit control as data mix shifts
            hm_pos_loss = (hm_loss_per_b * pos_mask).sum() / pos_count
            hm_neg_loss = (hm_loss_per_b * neg_mask).sum() / neg_count
            hm_loss = hm_pos_loss + HM_NEG_REL * hm_neg_loss

            # Soft-argmax MSE — secondary signal, weight schedule (warmup then demote)
            xy_loss = ((pred_xy - target_xy) ** 2).sum(dim=1) * pos_mask
            xy_loss = xy_loss.sum() / pos_count

            conf_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, target_conf)
            loss = hm_loss * HM_WEIGHT + xy_loss * xy_w + conf_loss * CONF_WEIGHT

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_xy_loss += xy_loss.item() * x.size(0)
            train_conf_loss += conf_loss.item() * x.size(0)
            train_hm_pos_loss += hm_pos_loss.item() * pos_count.item()
            train_hm_neg_loss += hm_neg_loss.item() * neg_count.item()
            n_train += x.size(0)
            n_train_pos += int(pos_count.item())
            n_train_neg += int(neg_count.item())
        sched.step()

        # ---- Validation: sliced metrics by sample_type ----
        model.eval()
        slice_err_sum: dict[int, float] = {}
        slice_err_n: dict[int, int] = {}
        slice_fpr_pred: dict[int, int] = {}   # how many predicted positive (conf > 0.5)
        slice_fpr_n: dict[int, int] = {}
        slice_peak_high: dict[int, int] = {}  # how many had heatmap peak > 0.5
        with torch.no_grad():
            val_conf_correct = 0; n_val = 0
            for x, target_xy, target_conf, sample_type in val_dl:
                x = x.to(device); target_xy = target_xy.to(device)
                target_conf = target_conf.to(device); sample_type = sample_type.to(device)
                pred_xy, pred_conf_logit, pred_hm = model(x)
                pred_conf = torch.sigmoid(pred_conf_logit)
                pred_label = (pred_conf > 0.5).float()
                val_conf_correct += (pred_label == target_conf).sum().item()
                n_val += x.size(0)

                # Per-sample peak prob (heatmap-derived confidence)
                hm_prob = torch.sigmoid(pred_hm).view(x.size(0), -1)
                hm_peak = hm_prob.max(dim=1).values

                # Position error in NATIVE pixels (un-normalize)
                e_xn = (pred_xy[:, 0] - target_xy[:, 0]) * NATIVE_W
                e_yn = (pred_xy[:, 1] - target_xy[:, 1]) * NATIVE_H
                err_px = torch.sqrt(e_xn ** 2 + e_yn ** 2)

                for t_id in sample_type.unique().tolist():
                    sel = (sample_type == t_id)
                    is_pos_type = (t_id == SAMPLE_TYPES["normal_pos"]
                                   or t_id == SAMPLE_TYPES["edge_pos"]
                                   or t_id == -1)  # legacy: assume positive
                    if is_pos_type:
                        pos_sel = sel & (target_conf == 1)
                        if pos_sel.any():
                            slice_err_sum[t_id] = slice_err_sum.get(t_id, 0.0) + err_px[pos_sel].sum().item()
                            slice_err_n[t_id] = slice_err_n.get(t_id, 0) + int(pos_sel.sum().item())
                    else:
                        # Negative slice: count false positives
                        slice_fpr_pred[t_id] = slice_fpr_pred.get(t_id, 0) + int(pred_label[sel].sum().item())
                        slice_fpr_n[t_id] = slice_fpr_n.get(t_id, 0) + int(sel.sum().item())
                        slice_peak_high[t_id] = slice_peak_high.get(t_id, 0) + int((hm_peak[sel] > 0.5).sum().item())

        dt = time.monotonic() - t0
        # Aggregate metrics
        all_err_sum = sum(slice_err_sum.values()); all_err_n = sum(slice_err_n.values())
        mean_pos_err = all_err_sum / max(1, all_err_n)
        conf_acc = val_conf_correct / max(1, n_val)
        # Per-slice formatted lines
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
              f"train_hm_neg={train_hm_neg_loss/max(1,n_train_neg):.4f} "
              f"train_xy={train_xy_loss/n_train:.4f} train_conf={train_conf_loss/n_train:.4f}  "
              f"val_pos_err={mean_pos_err:.1f}px val_conf_acc={conf_acc*100:.1f}%  "
              f"lr={opt.param_groups[0]['lr']:.5f} xy_w={xy_w:.1f}  ({dt:.1f}s)")
        if slice_lines:
            print(f"        {slice_str}")

        if mean_pos_err < best_val_err:
            best_val_err = mean_pos_err
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_pos_err_px": mean_pos_err,
                        "val_conf_acc": conf_acc,
                        "native_size": (NATIVE_W, NATIVE_H),
                        "train_size": (TRAIN_W, TRAIN_H),
                       }, args.weights_out)
            print(f"  ✓ saved best (val_pos_err={mean_pos_err:.1f}px) → {args.weights_out}")

    print(f"\nfinal best val_pos_err: {best_val_err:.1f}px")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_DIR)
    p.add_argument("--weights-out", default=WEIGHTS_PATH)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="fraction of bg_ids to hold out for validation (bg-level split)")
    args = p.parse_args()
    return train_loop(args)


if __name__ == "__main__":
    sys.exit(main())
