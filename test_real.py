"""
test_real.py — run trained PointerNet on real_pointer_test/*.png and visualize.

For each real frame:
  - resize 994x2160 -> 497x1080 (training resolution)
  - forward pass → pred_xy (soft-argmax), conf, raw heatmap
  - also compute hard argmax of heatmap (truer "where does the model think
    it sees the cursor"; soft-argmax gets pulled toward image center if
    the heatmap is broad/multi-modal — a textbook synth-to-real failure)

Output to ./eval_out/v02-real-test/:
  - per-frame side-by-side: original w/ PRED markers + heatmap colormap overlay
  - summary printed to stdout: per-frame pred_xy, conf, hm_max, hm_peak_xy
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local import — same dir as train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PointerNet, NATIVE_W, NATIVE_H, TRAIN_W, TRAIN_H

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "real_pointer_test")
OUT_DIR = "./eval_out/v02-real-test"

# Hand-annotated ground truths (only bg-00000 confirmed earlier)
GT = {
    "bg-00000.png": (656, 1424),
}


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """ resize → CHW float32 [0,1] → normalize to (-2, 2) range used in training """
    small = cv2.resize(img_bgr, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = (x - 0.5) / 0.25
    return x.unsqueeze(0)  # add batch dim


def hard_argmax_xy(hm_logits: torch.Tensor) -> tuple[float, float, float]:
    """argmax of raw heatmap logits. Returns (x_norm, y_norm, peak_logit).
    More robust than soft-argmax when the heatmap is broad/multi-modal.
    """
    B, _, H, W = hm_logits.shape
    flat = hm_logits.view(B, H * W)
    peak_logit, idx = flat.max(dim=1)
    iy = (idx // W).float()
    ix = (idx % W).float()
    x_norm = (ix / max(1, W - 1)).item()
    y_norm = (iy / max(1, H - 1)).item()
    return x_norm, y_norm, peak_logit.item()


def heatmap_overlay(img_native: np.ndarray, hm_prob: np.ndarray) -> np.ndarray:
    """Blend a colormap of hm (resized to native) over the image. hm_prob in [0,1]."""
    hm_full = cv2.resize(hm_prob, (img_native.shape[1], img_native.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    hm8 = (np.clip(hm_full, 0, 1) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_native, 0.6, cm, 0.4, 0)


def annotate(img: np.ndarray, pred_soft_xy: tuple[int, int],
             pred_hard_xy: tuple[int, int], gt_xy: tuple[int, int] | None,
             conf: float, peak_logit: float) -> np.ndarray:
    """Draw markers and labels on a copy of img."""
    out = img.copy()
    # Pred (hard argmax) — green circle, prominent
    cv2.circle(out, pred_hard_xy, 30, (0, 255, 0), 3)
    cv2.putText(out, f"argmax", (pred_hard_xy[0] + 35, pred_hard_xy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # Pred (soft-argmax) — yellow circle, secondary
    cv2.circle(out, pred_soft_xy, 22, (0, 220, 220), 2)
    cv2.putText(out, f"soft", (pred_soft_xy[0] + 25, pred_soft_xy[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
    # GT — blue circle if known
    if gt_xy is not None:
        cv2.circle(out, gt_xy, 25, (255, 100, 0), 3)
        cv2.putText(out, f"GT", (gt_xy[0] - 80, gt_xy[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
    # Header bar
    info = f"conf={conf:.3f}  hm_peak_logit={peak_logit:.2f}  hard={pred_hard_xy}  soft={pred_soft_xy}"
    if gt_xy is not None:
        err = np.hypot(pred_hard_xy[0] - gt_xy[0], pred_hard_xy[1] - gt_xy[1])
        info += f"  err={err:.0f}px"
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, info, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(ROOT, "pointer_model.pt"))
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    print(f"checkpoint epoch={ckpt.get('epoch')}  val_pos_err={ckpt.get('val_pos_err_px'):.1f}px  "
          f"train_size={ckpt.get('train_size')}  native_size={ckpt.get('native_size')}")

    model = PointerNet().to(device).eval()
    model.load_state_dict(ckpt["model"])

    frames = sorted(glob.glob(os.path.join(args.test_dir, "*.png")))
    print(f"frames: {len(frames)}\n")

    print(f"{'frame':<18}  {'conf':>6}  {'hm_pk':>6}  {'hard_xy':>14}  {'soft_xy':>14}  {'gt_xy':>14}  {'err':>5}")
    print("-" * 100)

    for fp in frames:
        img = cv2.imread(fp)
        if img is None or img.shape[:2] != (NATIVE_H, NATIVE_W):
            print(f"  skip {fp} (shape {img.shape if img is not None else None})")
            continue
        x = preprocess(img).to(device)
        with torch.no_grad():
            pred_xy, pred_conf_logit, pred_hm = model(x)
        conf = torch.sigmoid(pred_conf_logit).item()
        soft_xn, soft_yn = pred_xy[0].cpu().tolist()
        hard_xn, hard_yn, peak_logit = hard_argmax_xy(pred_hm)

        soft_xy = (int(round(soft_xn * NATIVE_W)), int(round(soft_yn * NATIVE_H)))
        hard_xy = (int(round(hard_xn * NATIVE_W)), int(round(hard_yn * NATIVE_H)))
        name = os.path.basename(fp)
        gt = GT.get(name)
        err_str = "—"
        if gt is not None:
            err = np.hypot(hard_xy[0] - gt[0], hard_xy[1] - gt[1])
            err_str = f"{err:.0f}"
        print(f"{name:<18}  {conf:>6.3f}  {peak_logit:>6.2f}  {str(hard_xy):>14}  {str(soft_xy):>14}  {str(gt):>14}  {err_str:>5}")

        # Visualizations
        hm_prob = torch.sigmoid(pred_hm)[0, 0].cpu().numpy()  # (H', W') in [0,1]
        overlay = heatmap_overlay(img, hm_prob)
        annotated = annotate(img, soft_xy, hard_xy, gt, conf, peak_logit)
        # Side-by-side: annotated original | heatmap overlay
        sxs = np.hstack([annotated, overlay])
        # Half-resolution for easier viewing on Windows
        sxs_half = cv2.resize(sxs, (sxs.shape[1] // 2, sxs.shape[0] // 2),
                              interpolation=cv2.INTER_AREA)
        out_name = name.replace(".png", "_test.jpg")
        cv2.imwrite(os.path.join(args.out_dir, out_name), sxs_half,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"\nwrote visualizations → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
