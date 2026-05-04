#!/usr/bin/env python3
"""
Crop the AssistiveTouch BLE-cursor dot out of a snap, save as RGBA sprite.

OUTPUT MUST BE VISUALLY VERIFIED BEFORE COMMIT TO sprites/.
A previous extraction misidentified a UI badge as the cursor and the
mistake survived five months of training because no one opened the file.
The synthesizer will refuse to load any sprite without an approved
sidecar manifest at <stem>.config.json (see synthesize._load_real_sprite).
The accepted approval flow:
  1. Run this script.
  2. Open the output PNG. Confirm it looks like the iOS Pointer cursor —
     a translucent gray dot with soft falloff, NOT a UI element.
  3. Generate the sidecar with the file's sha256 + your name + ISO date.
  4. Only then commit.

Inputs:
  --snap <path>      JPEG snapshot showing the cursor against ~black background
  --center x,y       JPEG pixel coords where the cursor center is (eyeballed)
  --radius N         crop radius (default 18 → 36x36 sprite)
  --out <path>       output PNG with alpha (default agent/training/at_dot.png)
  --threshold N      brightness 0-255 below which pixel is fully transparent
  --falloff N        soft alpha falloff range above threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snap", required=True)
    p.add_argument("--center", required=True, help='"x,y"')
    p.add_argument("--radius", type=int, default=18)
    p.add_argument("--out", default=str(Path(__file__).parent / "at_dot.png"))
    p.add_argument("--threshold", type=int, default=40)
    p.add_argument("--falloff", type=int, default=40)
    args = p.parse_args()

    cx, cy = map(int, args.center.split(","))
    r = args.radius

    img = Image.open(args.snap).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    crop = arr[y0:y1, x0:x1].copy()

    # Build alpha channel from luminance: pixels below threshold = transparent,
    # above threshold+falloff = opaque, linear ramp between.
    lum = crop.astype(np.float32).mean(axis=2)
    alpha = np.clip((lum - args.threshold) / max(1, args.falloff), 0.0, 1.0)
    alpha = (alpha * 255).astype(np.uint8)

    rgba = np.dstack([crop, alpha])
    out_img = Image.fromarray(rgba, mode="RGBA")
    out_img.save(args.out)
    print(f"saved {args.out}: {out_img.size} (cropped from {args.snap} centered at ({cx},{cy}))")
    print(f"alpha histogram: nonzero={np.count_nonzero(alpha)}/{alpha.size} pixels")


if __name__ == "__main__":
    main()
