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


class PointerDataset(Dataset):
    def __init__(self, dataset_dir: str, train: bool = True, val_frac: float = 0.1):
        with open(os.path.join(dataset_dir, "labels.jsonl")) as f:
            all_labels = [json.loads(line) for line in f if line.strip()]
        random.Random(42).shuffle(all_labels)
        n_val = max(1, int(len(all_labels) * val_frac))
        if train:
            self.entries = all_labels[n_val:]
        else:
            self.entries = all_labels[:n_val]
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
        return x, target_xy, target_conf


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
                        sigma_px: float = 1.5) -> torch.Tensor:
    """
    Build (B, 1, H', W') Gaussian target heatmap centered at each (x_norm, y_norm).
    sigma_px = stddev in heatmap pixel units. With H'=34, W'=16, sigma=1.5
    gives a tight peak (~3-pixel radius) that the model can train against
    directly via MSE — strong gradient signal vs. soft-argmax indirect path.
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

    train_ds = PointerDataset(args.dataset, train=True)
    val_ds = PointerDataset(args.dataset, train=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    model = PointerNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_err = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        train_xy_loss = 0.0; train_conf_loss = 0.0; n_train = 0
        for x, target_xy, target_conf in train_dl:
            x = x.to(device, non_blocking=True)
            target_xy = target_xy.to(device, non_blocking=True)
            target_conf = target_conf.to(device, non_blocking=True)
            pred_xy, pred_conf_logit, pred_hm = model(x)
            mask = target_conf  # 1 if positive, 0 if negative
            # PRIMARY signal: heatmap BCE (better than MSE for sparse Gaussian targets;
            # MSE is dominated by 99% background pixels, gradients toward the peak
            # are too weak. BCE balances pos/neg pixel contributions.)
            B, _, H, W = pred_hm.shape
            target_hm = make_target_heatmap(target_xy, H, W, sigma_px=4.0)
            # BCE expects logits (no sigmoid). pred_hm is logits.
            hm_loss_per_b = F.binary_cross_entropy_with_logits(
                pred_hm.squeeze(1), target_hm.squeeze(1), reduction='none'
            ).mean(dim=(1, 2))
            hm_loss = (hm_loss_per_b * mask).sum() / mask.sum().clamp_min(1.0)
            # SECONDARY: soft-argmax MSE for sub-pixel refinement
            xy_loss = ((pred_xy - target_xy) ** 2).sum(dim=1) * mask
            xy_loss = xy_loss.sum() / mask.sum().clamp_min(1.0)
            conf_loss = F.binary_cross_entropy_with_logits(pred_conf_logit, target_conf)
            loss = hm_loss * 10.0 + xy_loss * 5.0 + conf_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_xy_loss += xy_loss.item() * x.size(0)
            train_conf_loss += conf_loss.item() * x.size(0)
            n_train += x.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pos_err_px = 0.0; n_val_pos = 0
            val_conf_correct = 0; n_val = 0
            for x, target_xy, target_conf in val_dl:
                x = x.to(device); target_xy = target_xy.to(device); target_conf = target_conf.to(device)
                pred_xy, pred_conf_logit, _ = model(x)
                pred_conf = torch.sigmoid(pred_conf_logit)
                # Position error in NATIVE pixels (un-normalize)
                pos_mask = (target_conf == 1)
                if pos_mask.any():
                    e_xn = (pred_xy[:, 0] - target_xy[:, 0]) * NATIVE_W
                    e_yn = (pred_xy[:, 1] - target_xy[:, 1]) * NATIVE_H
                    err_px = torch.sqrt(e_xn ** 2 + e_yn ** 2)
                    val_pos_err_px += err_px[pos_mask].sum().item()
                    n_val_pos += pos_mask.sum().item()
                pred_label = (pred_conf > 0.5).float()
                val_conf_correct += (pred_label == target_conf).sum().item()
                n_val += x.size(0)

        dt = time.monotonic() - t0
        mean_pos_err = val_pos_err_px / max(1, n_val_pos)
        conf_acc = val_conf_correct / max(1, n_val)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_xy={train_xy_loss/n_train:.4f} train_conf={train_conf_loss/n_train:.4f}  "
              f"val_pos_err={mean_pos_err:.1f}px  val_conf_acc={conf_acc*100:.1f}%  "
              f"lr={opt.param_groups[0]['lr']:.5f}  ({dt:.1f}s)")

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
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    return train_loop(args)


if __name__ == "__main__":
    sys.exit(main())
