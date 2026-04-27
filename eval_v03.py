"""
eval_v03.py — compare v0.2 vs v0.3 on the failure modes v0.3 review
flagged: localization on real frames (regression check) AND false-positive
behavior on cursor-free frames (the loss-mask fix's primary target).

Usage:
    python eval_v03.py [--v02 path] [--v03 path] [--out dir]

Outputs (per model, side-by-side):
  - real_pointer_test/ frames: pos error, conf, heatmap peak
  - cursor-free real frames (sampled from backgrounds_kept/ that we held
    out of training via bg-level val split — verified by reading the
    val_bg_ids saved in the v0.3 checkpoint metadata if present, otherwise
    falls back to all backgrounds_kept/ frames as cursor-free since they
    were captured cursor-off).

Pass criteria for v0.3 vs v0.2:
  - Real cursor frames: pos err comparable (no regression)
  - Cursor-free frames: conf and peak should DROP markedly
    (previously 1.000/0.9999 false-positive on Settings notification badge,
    Calendar dot, etc.; v0.3 should give low conf and flat heatmap)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from train import PointerNet, NATIVE_W, NATIVE_H, TRAIN_W, TRAIN_H

REAL_DIR = "./real_pointer_test"
BG_POOL = "./backgrounds_kept"


def load_model(path: str, device: torch.device) -> tuple[PointerNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = PointerNet().to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m, ckpt


def preprocess(img: np.ndarray, device: torch.device) -> torch.Tensor:
    small = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = (x - 0.5) / 0.25
    return x.unsqueeze(0).to(device)


def find(model: PointerNet, img: np.ndarray, device: torch.device) -> dict:
    x = preprocess(img, device)
    with torch.no_grad():
        pred_xy, conf_logit, hm = model(x)
    conf = float(torch.sigmoid(conf_logit).item())
    prob = torch.sigmoid(hm)[0, 0].cpu().numpy()  # 1/16 native res
    H_, W_ = prob.shape
    flat = int(prob.argmax())
    iy, ix = flat // W_, flat % W_
    cx = int(round(ix / max(1, W_ - 1) * NATIVE_W))
    cy = int(round(iy / max(1, H_ - 1) * NATIVE_H))
    return {"cx": cx, "cy": cy, "conf": conf,
            "peak": float(prob.max()), "mean_hm": float(prob.mean())}


def evaluate_on_dir(model: PointerNet, paths: list[str], device: torch.device,
                    label: str, gt: dict[str, tuple[int, int]] | None = None) -> dict:
    """Returns dict with per-frame results + summary stats."""
    results = []
    for fp in paths:
        img = cv2.imread(fp)
        if img is None or img.shape[:2] != (NATIVE_H, NATIVE_W):
            continue
        r = find(model, img, device)
        name = os.path.basename(fp)
        gtxy = gt.get(name) if gt else None
        if gtxy is not None:
            err = float(np.hypot(r["cx"] - gtxy[0], r["cy"] - gtxy[1]))
        else:
            err = None
        results.append({**r, "name": name, "gt": gtxy, "err_px": err})
    confs = [r["conf"] for r in results]
    peaks = [r["peak"] for r in results]
    errs = [r["err_px"] for r in results if r["err_px"] is not None]
    summary = {
        "n": len(results),
        "label": label,
        "mean_conf": float(np.mean(confs)) if confs else None,
        "mean_peak": float(np.mean(peaks)) if peaks else None,
        "mean_err_px": float(np.mean(errs)) if errs else None,
        "max_peak": float(np.max(peaks)) if peaks else None,
        "frac_conf_gt05": float(np.mean([c > 0.5 for c in confs])) if confs else None,
        "frac_peak_gt05": float(np.mean([p > 0.5 for p in peaks])) if peaks else None,
    }
    return {"summary": summary, "results": results}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v02", default=os.path.join(ROOT, "pointer_model_v0.2.0.pt"))
    p.add_argument("--v03", default=os.path.join(ROOT, "pointer_model.pt"))
    p.add_argument("--real-dir", default=REAL_DIR)
    p.add_argument("--bg-pool", default=BG_POOL,
                   help="cursor-free real captures (used to test FPR on real distractors)")
    p.add_argument("--n-cursor-free", type=int, default=20,
                   help="how many cursor-free real frames to sample")
    p.add_argument("--out", default="./eval_out/v03-eval")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}\n")

    # Ground truth annotations (only bg-00000 was hand-labeled; rest are
    # tickler-near-stable so the cursor sits within a few px of bg-00000 in all)
    gt = {"bg-00000.png": (656, 1424)}

    real_paths = sorted(glob.glob(os.path.join(args.real_dir, "*.png")))
    bg_pool_paths = sorted(glob.glob(os.path.join(args.bg_pool, "*.png")))[:args.n_cursor_free]

    print(f"real cursor frames:  {len(real_paths)}")
    print(f"cursor-free frames:  {len(bg_pool_paths)}\n")

    rows = []
    for label, weight_path in [("v0.2", args.v02), ("v0.3", args.v03)]:
        if not os.path.exists(weight_path):
            print(f"[skip] {label} not found: {weight_path}")
            continue
        model, ckpt = load_model(weight_path, device)
        print(f"=== {label}: epoch={ckpt.get('epoch')} "
              f"val_pos_err={ckpt.get('val_pos_err_px'):.1f}px ===")
        real_res = evaluate_on_dir(model, real_paths, device, f"{label}_real", gt)
        free_res = evaluate_on_dir(model, bg_pool_paths, device, f"{label}_free")
        for tag, res in [("real_cursor", real_res), ("cursor_free", free_res)]:
            s = res["summary"]
            print(f"  {tag:13s}  n={s['n']:3d}  "
                  f"conf_mean={s['mean_conf']:.3f}  peak_mean={s['mean_peak']:.3f}  "
                  f"%conf>0.5={s['frac_conf_gt05']*100:5.1f}%  "
                  f"%peak>0.5={s['frac_peak_gt05']*100:5.1f}%"
                  + (f"  err_mean={s['mean_err_px']:.1f}px" if s['mean_err_px'] is not None else ""))
        rows.append({"model": label, "real": real_res["summary"], "free": free_res["summary"],
                     "details": {"real": real_res["results"], "free": free_res["results"]}})

    print()
    if len(rows) == 2:
        v02, v03 = rows[0], rows[1]
        print("=== delta (v0.3 vs v0.2) ===")
        print(f"  real_cursor   conf  Δ={v03['real']['mean_conf']-v02['real']['mean_conf']:+.3f}  "
              f"peak Δ={v03['real']['mean_peak']-v02['real']['mean_peak']:+.3f}")
        if v02['real']['mean_err_px'] is not None and v03['real']['mean_err_px'] is not None:
            print(f"  real_cursor   err   Δ={v03['real']['mean_err_px']-v02['real']['mean_err_px']:+.1f}px")
        print(f"  cursor_free   conf  Δ={v03['free']['mean_conf']-v02['free']['mean_conf']:+.3f}  "
              f"peak Δ={v03['free']['mean_peak']-v02['free']['mean_peak']:+.3f}  "
              f"(↓ = improvement; v0.3 should reject distractors)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_v03_results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nfull per-frame results → {args.out}/eval_v03_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
