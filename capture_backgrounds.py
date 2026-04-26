"""
capture_backgrounds.py — passively saves visually-distinct iPhone screens.

Run this in the foreground while you navigate the iPhone freely (touch, not
mouse — the BLE cursor should be OFF or hidden so backgrounds are cursor-free
training data).

Watches /tmp/phone-NNNNN.jpg ring; saves the 2nd-newest finalized frame
whenever its pixel-diff to the last-kept frame exceeds a threshold. Output
goes to agent/training/backgrounds/bg-NNNNN.jpg.

Stop with Ctrl+C. Aim for 500-2000 backgrounds across diverse apps.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backgrounds")
JPEG_GLOB = "/tmp/phone-[0-9]*.jpg"


def _safe_sorted_jpegs() -> list:
    pairs = []
    for f in glob.glob(JPEG_GLOB):
        try:
            pairs.append((os.path.getmtime(f), f))
        except OSError:
            continue
    pairs.sort(reverse=True)
    return [p[1] for p in pairs]


def avg_hash(img_bgr, size: int = 16) -> np.ndarray:
    """Average-hash perceptual fingerprint. 16x16 = 256 bits, robust to
    small visual differences, sensitive to layout/content."""
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten().astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--lum-delta", type=int, default=25,
                   help="per-pixel luminance delta to count as 'changed'")
    p.add_argument("--diff-threshold", type=float, default=0.04,
                   help="fraction of pixels that must be 'changed' to keep frame")
    p.add_argument("--min-interval-s", type=float, default=1.0,
                   help="minimum seconds between saved frames")
    p.add_argument("--phash-recent", type=int, default=120,
                   help="check perceptual-hash against last N saved frames")
    p.add_argument("--phash-min-bits", type=int, default=32,
                   help="reject if pHash within this many bits of any recent (out of 256)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # resume from existing files
    existing = sorted(glob.glob(os.path.join(args.out_dir, "bg-*.jpg")))
    saved = 0
    if existing:
        try:
            saved = int(os.path.basename(existing[-1]).removeprefix("bg-").removesuffix(".jpg")) + 1
        except Exception:
            saved = len(existing)
    print(f"capture_backgrounds: out={args.out_dir} starting at idx {saved}")
    print(f"diff_threshold={args.diff_threshold:.2f}  lum_delta={args.lum_delta}  min_interval={args.min_interval_s}s")
    print("navigate the iPhone (touch, not BLE mouse). Ctrl+C to stop.\n")

    last_gray = None
    last_save_t = 0.0
    last_seen_path = None
    recent_hashes: list = []
    rejected_phash = 0
    rejected_diff = 0

    # Bootstrap recent_hashes from existing files in out-dir (resume support)
    if existing:
        for f in existing[-args.phash_recent:]:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is not None:
                recent_hashes.append(avg_hash(img))

    try:
        while True:
            files = _safe_sorted_jpegs()
            if len(files) < 2:
                time.sleep(0.05); continue
            snap = files[1]
            if snap == last_seen_path:
                time.sleep(0.05); continue
            last_seen_path = snap

            img = cv2.imread(snap, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            now = time.time()
            changed_frac = 0.0
            if last_gray is not None and gray.shape == last_gray.shape:
                diff = cv2.absdiff(gray, last_gray)
                changed_frac = float((diff > args.lum_delta).sum()) / diff.size
                if changed_frac < args.diff_threshold:
                    rejected_diff += 1
                    time.sleep(0.05)
                    continue
                if (now - last_save_t) < args.min_interval_s:
                    time.sleep(0.05)
                    continue

            # Perceptual-hash dedup against recent saves
            ah = avg_hash(img)
            if recent_hashes:
                min_ham = min(hamming(ah, prev) for prev in recent_hashes)
                if min_ham < args.phash_min_bits:
                    rejected_phash += 1
                    if not args.quiet:
                        print(f"     . dup pHash min_ham={min_ham} (need >= {args.phash_min_bits})")
                    last_gray = gray
                    time.sleep(0.05)
                    continue

            out = os.path.join(args.out_dir, f"bg-{saved:05d}.png")
            cv2.imwrite(out, img)  # PNG = lossless; cursor edges stay crisp for training
            last_gray = gray
            last_save_t = now
            recent_hashes.append(ah)
            if len(recent_hashes) > args.phash_recent:
                recent_hashes.pop(0)
            if not args.quiet:
                print(f"[{saved:5d}] {os.path.basename(snap)} changed={changed_frac:.2%} -> {os.path.basename(out)}")
            saved += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\nstopped. saved {saved} backgrounds (rejected {rejected_diff} below diff, {rejected_phash} as pHash dups)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
