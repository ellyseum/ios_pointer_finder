"""
grid_overlay.py — draw a CSS-px grid over a snap so I can pick /tap_css coords
visually instead of eyeballing-then-converting.

CSS is the coordinate space the hands API's /tap_css consumes (440x956).
Native phone-pixel space is 994x2160 (the resolution actually rendered in
the JPEG ring). 1 CSS px ≈ 2.26 native px.

Default behavior:
  python grid_overlay.py            # gridify the newest finalized ring frame
  python grid_overlay.py path.jpg   # gridify a specific file

Output: ./eval_out/grid.jpg (half-res JPEG with grid
overlay drawn in CSS coordinates — major lines + labels every 50 CSS px,
minor gridlines every 25, so any point is within ±5 CSS px by inspection).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

NATIVE_W, NATIVE_H = 994, 2160
CSS_W, CSS_H = 440, 956
JPEG_GLOB = "/tmp/phone-[0-9]*.jpg"
DEFAULT_OUT = "./eval_out/grid.jpg"


def overlay_grid(img_native: np.ndarray,
                 major: int = 50,
                 minor: int = 25,
                 minor_color=(80, 80, 80),
                 major_color=(0, 255, 255),
                 label_color=(255, 255, 255),
                 label_bg=(0, 0, 0),
                 opacity: float = 0.55) -> np.ndarray:
    """Draw a CSS-px grid on a native-resolution image. Returns a copy."""
    h, w = img_native.shape[:2]
    sx = w / CSS_W   # ~2.259
    sy = h / CSS_H   # ~2.259
    overlay = img_native.copy()

    # minor gridlines
    for cx in range(0, CSS_W + 1, minor):
        x = int(round(cx * sx))
        cv2.line(overlay, (x, 0), (x, h - 1), minor_color, 1, cv2.LINE_AA)
    for cy in range(0, CSS_H + 1, minor):
        y = int(round(cy * sy))
        cv2.line(overlay, (0, y), (w - 1, y), minor_color, 1, cv2.LINE_AA)

    # major gridlines
    for cx in range(0, CSS_W + 1, major):
        x = int(round(cx * sx))
        cv2.line(overlay, (x, 0), (x, h - 1), major_color, 2, cv2.LINE_AA)
    for cy in range(0, CSS_H + 1, major):
        y = int(round(cy * sy))
        cv2.line(overlay, (0, y), (w - 1, y), major_color, 2, cv2.LINE_AA)

    # blend grid over image so screen content stays visible
    out = cv2.addWeighted(overlay, opacity, img_native, 1.0 - opacity, 0)

    # labels — solid (no opacity) so they stay readable
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thick = 2
    for cx in range(0, CSS_W + 1, major):
        x = int(round(cx * sx))
        text = str(cx)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        # top edge labels (x-axis)
        cv2.rectangle(out, (x + 2, 8), (x + 8 + tw, 12 + th), label_bg, -1)
        cv2.putText(out, text, (x + 5, 10 + th), font, scale, label_color, thick, cv2.LINE_AA)
    for cy in range(0, CSS_H + 1, major):
        y = int(round(cy * sy))
        text = str(cy)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        # left edge labels (y-axis)
        cv2.rectangle(out, (6, y + 2), (12 + tw, y + 6 + th), label_bg, -1)
        cv2.putText(out, text, (9, y + 4 + th), font, scale, label_color, thick, cv2.LINE_AA)

    # corner badge with screen-space sizes
    badge = f"CSS {CSS_W}x{CSS_H}  native {NATIVE_W}x{NATIVE_H}  major={major}  minor={minor}"
    (tw, th), _ = cv2.getTextSize(badge, font, 0.7, 2)
    cv2.rectangle(out, (0, h - th - 16), (tw + 14, h), label_bg, -1)
    cv2.putText(out, badge, (7, h - 6), font, 0.7, label_color, 2, cv2.LINE_AA)

    return out


def _newest_finalized_ring_frame() -> str | None:
    files = sorted(glob.glob(JPEG_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    return files[1] if len(files) >= 2 else files[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snap", nargs="?", help="path to JPG; default = newest ring frame")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--major", type=int, default=50)
    p.add_argument("--minor", type=int, default=25)
    p.add_argument("--full-res", action="store_true",
                   help="save at native resolution instead of half (default half)")
    args = p.parse_args()

    snap = args.snap or _newest_finalized_ring_frame()
    if snap is None:
        print("FAIL: no snap given and no ring frames in /tmp/phone-*.jpg", file=sys.stderr)
        return 1

    img = cv2.imread(snap)
    if img is None:
        print(f"FAIL: could not read {snap}", file=sys.stderr)
        return 1

    out = overlay_grid(img, major=args.major, minor=args.minor)
    if not args.full_res:
        out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"snap: {snap}")
    print(f"out:  {args.out}  ({out.shape[1]}x{out.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
