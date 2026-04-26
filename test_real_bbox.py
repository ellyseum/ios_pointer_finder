"""
test_real_bbox.py — same as test_real.py but draws a data-driven bounding box
from the model's heatmap (not a fixed-size box around argmax).

BBox method:
  1. Sigmoid the raw heatmap logits → prob in [0,1] at 1/8 resolution
  2. Bilinear-upsample to native (994x2160)
  3. Threshold at peak * 0.5 (relative; tolerates frames where peak isn't 1.0)
  4. cv2.boundingRect of the largest connected component
  5. Draw on the image

This shows what the model "sees" as the cursor extent — useful diagnostic
for whether the heatmap is tight (sharp peak, small box) or fuzzy (broad
peak, large box). A real BLE cursor is ~46 px diameter so a clean detection
should give a box ~50-80 px on a side.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PointerNet, NATIVE_W, NATIVE_H, TRAIN_W, TRAIN_H

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT, "real_pointer_test")
OUT_DIR = "/mnt/c/Users/jocel/iphone-snaps/v02-real-bbox"

GT = {"bg-00000.png": (656, 1424)}


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    small = cv2.resize(img_bgr, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = (x - 0.5) / 0.25
    return x.unsqueeze(0)


def heatmap_to_bbox(hm_logits: torch.Tensor, native_w: int, native_h: int,
                    rel_thresh: float = 0.5) -> tuple[tuple[int, int, int, int], tuple[int, int], float]:
    """
    Returns:
      bbox: (x, y, w, h) in native pixels of largest CC above thresh
      center: (cx, cy) argmax of full-res heatmap
      peak: peak prob value [0,1]
    """
    prob = torch.sigmoid(hm_logits)[0, 0].cpu().numpy()  # (H', W')
    prob_full = cv2.resize(prob, (native_w, native_h), interpolation=cv2.INTER_LINEAR)
    peak = float(prob_full.max())
    # argmax in full res
    iy, ix = np.unravel_index(np.argmax(prob_full), prob_full.shape)
    # Threshold relative to peak
    mask = (prob_full >= peak * rel_thresh).astype(np.uint8)
    # Largest connected component containing the peak
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lbl <= 1:
        return (int(ix) - 25, int(iy) - 25, 50, 50), (int(ix), int(iy)), peak
    peak_lbl = lbls[iy, ix]
    if peak_lbl == 0:  # peak fell on background — fallback
        # pick largest non-bg CC
        peak_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    s = stats[peak_lbl]
    x, y, w, h = s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP], s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT]
    return (int(x), int(y), int(w), int(h)), (int(ix), int(iy)), peak


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/tmp/pointer_model_ep28.pt")
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--rel-thresh", type=float, default=0.5,
                   help="bbox = pixels where heatmap >= peak * rel_thresh")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    print(f"checkpoint epoch={ckpt.get('epoch')}  val_pos_err={ckpt.get('val_pos_err_px'):.1f}px")

    model = PointerNet().to(device).eval()
    model.load_state_dict(ckpt["model"])

    frames = sorted(glob.glob(os.path.join(args.test_dir, "*.png")))
    print(f"frames: {len(frames)}\n")
    print(f"{'frame':<18}  {'peak':>5}  {'conf':>5}  {'bbox(x,y,w,h)':>22}  {'center':>14}  {'gt':>14}")
    print("-" * 95)

    for fp in frames:
        img = cv2.imread(fp)
        if img is None or img.shape[:2] != (NATIVE_H, NATIVE_W):
            continue
        x = preprocess(img).to(device)
        with torch.no_grad():
            _, conf_logit, hm = model(x)
        conf = torch.sigmoid(conf_logit).item()
        bbox, center, peak = heatmap_to_bbox(hm, NATIVE_W, NATIVE_H, rel_thresh=args.rel_thresh)
        name = os.path.basename(fp)
        gt = GT.get(name)
        print(f"{name:<18}  {peak:>5.3f}  {conf:>5.3f}  {str(bbox):>22}  {str(center):>14}  {str(gt):>14}")

        # Draw bbox on copy of image
        out = img.copy()
        bx, by, bw, bh = bbox
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
        # crosshair at center (argmax)
        cv2.line(out, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (0, 255, 255), 2)
        cv2.line(out, (center[0], center[1] - 20), (center[0], center[1] + 20), (0, 255, 255), 2)
        if gt is not None:
            cv2.circle(out, gt, 18, (255, 100, 0), 3)
        # Header
        info = f"{name}  conf={conf:.2f}  peak={peak:.2f}  bbox=({bx},{by},{bw}x{bh})"
        cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(out, info, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        # Crop view: 300x300 around center, scaled 2x for clarity
        cx, cy = center
        s = 200
        cx0 = max(0, cx - s); cy0 = max(0, cy - s)
        cx1 = min(NATIVE_W, cx + s); cy1 = min(NATIVE_H, cy + s)
        crop = out[cy0:cy1, cx0:cx1].copy()
        crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

        # Side-by-side: full image (half size) | zoom crop (2x of 400x400 region)
        full_half = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2),
                                interpolation=cv2.INTER_AREA)
        # Pad crop to match height of full_half
        target_h = full_half.shape[0]
        if crop.shape[0] != target_h:
            pad_top = (target_h - crop.shape[0]) // 2 if crop.shape[0] < target_h else 0
            pad_bot = target_h - crop.shape[0] - pad_top if crop.shape[0] < target_h else 0
            if pad_top + pad_bot > 0:
                crop = cv2.copyMakeBorder(crop, pad_top, pad_bot, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(40, 40, 40))
            elif crop.shape[0] > target_h:
                crop = cv2.resize(crop, (int(crop.shape[1] * target_h / crop.shape[0]), target_h),
                                  interpolation=cv2.INTER_AREA)
        sxs = np.hstack([full_half, crop])
        cv2.imwrite(os.path.join(args.out_dir, name.replace(".png", "_bbox.jpg")),
                    sxs, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nwrote → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
