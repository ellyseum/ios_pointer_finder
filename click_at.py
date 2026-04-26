"""
click_at.py — closed-loop "click at (x, y)" using the trained PointerNet.

Given a target pixel (in 994×2160 native-px coords), finds the cursor in the
live JPEG ring with the CNN, computes the displacement vector, sends it to
the BLE-HID hands API, waits for the next frame, repeats. When the cursor
lands within `tolerance` of the target it sends a click.

No probe move, no scale calibration — model gives absolute cursor position
every frame, so we just servo on the pixel error.

CLI:
    python click_at.py X Y [--tolerance 15] [--max-iters 12]

API:
    from click_at import click_at, PointerFinder
    finder = PointerFinder()  # loads weights once, reuse across calls
    result = click_at(656, 1424, finder=finder)

Result: {
    "ok": bool,
    "reason": "converged" | "stale_pipeline" | "max_iters" | "cursor_lost"
              | "hands_dead" | "bad_frame",
    "iters": int,
    "final_xy": (cx, cy) | None,
    "final_err": float | None,
    "history": [{"iter", "cx", "cy", "conf", "peak", "dx", "dy"}, ...]
}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (used implicitly by model fwd)
# `requests` is lazy-imported inside hands API helpers so the inference path
# is usable in envs without it.

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from train import PointerNet  # share architecture definition with training

# ----- constants -----
WEIGHTS_PATH = os.path.join(ROOT, "pointer_model.pt")
JPEG_GLOB = "/tmp/phone-[0-9]*.jpg"
HANDS_URL = "http://127.0.0.1:8765"
NATIVE_W, NATIVE_H = 994, 2160
TRAIN_W, TRAIN_H = 497, 1080

DEFAULT_TOLERANCE = 15        # native px; click fires when |delta| < this
DEFAULT_MAX_ITERS = 12
CONF_THRESHOLD = 0.5          # below this, treat as cursor-lost
PEAK_THRESHOLD = 0.5          # heatmap peak prob; below this, also lost
WAIT_FRAME_TIMEOUT_S = 0.6
SETTLE_BEFORE_CLICK_S = 0.05  # let the last move quiesce before firing
PIPELINE_STALE_S = 3.0        # newest frame older than this → pipeline dead


# ---------- inference ----------

class PointerFinder:
    def __init__(self, weights_path: str = WEIGHTS_PATH, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model = PointerNet().to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.train_size = ckpt.get("train_size", (TRAIN_W, TRAIN_H))
        self.native_size = ckpt.get("native_size", (NATIVE_W, NATIVE_H))

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        small = cv2.resize(img_bgr, self.train_size, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = (x - 0.5) / 0.25
        return x.unsqueeze(0).to(self.device)

    def find(self, img_bgr: np.ndarray) -> tuple[int, int, float, float] | None:
        """
        Returns (cx, cy, conf, peak_prob) in native-px coords.
        cx, cy refined with parabolic sub-pixel interpolation around argmax.
        Returns None if image shape is wrong.
        """
        nh, nw = self.native_size[1], self.native_size[0]
        if img_bgr.shape[:2] != (nh, nw):
            return None
        x = self._preprocess(img_bgr)
        with torch.no_grad():
            _, conf_logit, hm = self.model(x)
        conf = float(torch.sigmoid(conf_logit).item())
        prob = torch.sigmoid(hm)[0, 0].cpu().numpy()  # (H', W')
        H, W = prob.shape
        flat_idx = int(prob.argmax())
        iy, ix = flat_idx // W, flat_idx % W
        sub_x, sub_y = _parabolic_subpixel(prob, ix, iy)
        ix_r = ix + sub_x
        iy_r = iy + sub_y
        x_norm = ix_r / max(1, W - 1)
        y_norm = iy_r / max(1, H - 1)
        cx = int(round(x_norm * nw))
        cy = int(round(y_norm * nh))
        peak = float(prob[iy, ix])
        return cx, cy, conf, peak


def _parabolic_subpixel(hm: np.ndarray, ix: int, iy: int) -> tuple[float, float]:
    """3-tap parabolic fit on the row and column through argmax. Returns
    (Δx, Δy) offsets in [-0.5, 0.5] heatmap-pixel units. Reduces argmax
    quantization error from ±8 native-px to ~±1 native-px."""
    H, W = hm.shape

    def fit_1d(a: float, b: float, c: float) -> float:
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-9:
            return 0.0
        return 0.5 * (a - c) / denom

    sub_x = fit_1d(hm[iy, ix - 1], hm[iy, ix], hm[iy, ix + 1]) if 0 < ix < W - 1 else 0.0
    sub_y = fit_1d(hm[iy - 1, ix], hm[iy, ix], hm[iy + 1, ix]) if 0 < iy < H - 1 else 0.0
    return float(sub_x), float(sub_y)


# ---------- frame source ----------

def _newest_finalized_frame(after_t_wall: float = 0.0,
                             timeout: float = WAIT_FRAME_TIMEOUT_S) -> str | None:
    """Return path of the 2nd-newest JPEG (which is finalized — files[0] may
    be mid-write) whose mtime > after_t_wall. None on timeout. Wall-clock
    comparison since os.path.getmtime is wall-clock; deadline uses monotonic."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = sorted(glob.glob(JPEG_GLOB), key=os.path.getmtime, reverse=True)
        if len(files) >= 2 and os.path.getmtime(files[1]) > after_t_wall:
            return files[1]
        time.sleep(0.02)
    return None


def _pipeline_alive() -> bool:
    files = sorted(glob.glob(JPEG_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return False
    return (time.time() - os.path.getmtime(files[0])) < PIPELINE_STALE_S


# ---------- hands API ----------

def _move_rel(dx: int, dy: int, step: int = 4, delay_ms: int = 3) -> None:
    import requests
    requests.post(
        f"{HANDS_URL}/move_rel",
        json={"dx": int(dx), "dy": int(dy), "step": step, "delay_ms": delay_ms},
        timeout=2.0,
    )


def _click(hold_ms: int = 60) -> None:
    import requests
    requests.post(f"{HANDS_URL}/click", json={"hold_ms": hold_ms}, timeout=2.0)


def _hands_alive() -> bool:
    try:
        import requests
        requests.post(f"{HANDS_URL}/move_rel",
                      json={"dx": 0, "dy": 0, "step": 1, "delay_ms": 1},
                      timeout=1.0)
        return True
    except Exception:
        return False


# ---------- jiggle to wake auto-hidden iOS cursor ----------

def _jiggle_to_wake() -> float:
    """Quick clockwise box ±15 px to wake the auto-hidden cursor. Returns the
    wall-clock timestamp captured BEFORE the first move so the caller can
    use it to gate frame-freshness."""
    t = time.time()
    for dx, dy in [(15, 0), (0, 15), (-15, 0), (0, -15)]:  # box back to start
        _move_rel(dx, dy)
        time.sleep(0.04)
    time.sleep(0.30)  # let iOS render the cursor again + pipeline drain
    return t


# ---------- main loop ----------

def click_at(target_x: int, target_y: int,
             tolerance: int = DEFAULT_TOLERANCE,
             max_iters: int = DEFAULT_MAX_ITERS,
             finder: PointerFinder | None = None,
             allow_jiggle: bool = True,
             verbose: bool = False) -> dict:
    """
    Drive the iPhone cursor to (target_x, target_y) in 994×2160 native-px coords,
    then click. See module docstring for return shape.
    """
    if finder is None:
        finder = PointerFinder()
    if not _hands_alive():
        return {"ok": False, "reason": "hands_dead", "iters": 0,
                "final_xy": None, "final_err": None, "history": []}
    if not _pipeline_alive():
        return {"ok": False, "reason": "pipeline_dead", "iters": 0,
                "final_xy": None, "final_err": None, "history": []}

    history: list[dict] = []
    t_last_move: float = 0.0  # wall-clock of last move command; gates frame freshness
    lost_streak = 0

    for i in range(max_iters):
        snap = _newest_finalized_frame(after_t_wall=t_last_move)
        if snap is None:
            return {"ok": False, "reason": "stale_pipeline", "iters": i,
                    "final_xy": None, "final_err": None, "history": history}
        img = cv2.imread(snap)
        if img is None or img.shape[:2] != (NATIVE_H, NATIVE_W):
            return {"ok": False, "reason": "bad_frame", "iters": i,
                    "final_xy": None, "final_err": None, "history": history}

        result = finder.find(img)
        if result is None:
            return {"ok": False, "reason": "bad_frame", "iters": i,
                    "final_xy": None, "final_err": None, "history": history}
        cx, cy, conf, peak = result

        # cursor-lost path → jiggle + retry
        if conf < CONF_THRESHOLD or peak < PEAK_THRESHOLD:
            lost_streak += 1
            history.append({"iter": i, "cx": cx, "cy": cy, "conf": conf, "peak": peak,
                            "dx": None, "dy": None, "note": "lost"})
            if verbose:
                print(f"  iter {i}: conf={conf:.2f} peak={peak:.2f} → cursor lost"
                      f" (streak {lost_streak})")
            if lost_streak >= 3:
                return {"ok": False, "reason": "cursor_lost", "iters": i + 1,
                        "final_xy": None, "final_err": None, "history": history}
            if allow_jiggle:
                t_last_move = _jiggle_to_wake()
            continue
        lost_streak = 0

        dx = target_x - cx
        dy = target_y - cy
        dist = float(np.hypot(dx, dy))
        history.append({"iter": i, "cx": cx, "cy": cy, "conf": conf, "peak": peak,
                        "dx": dx, "dy": dy, "dist": dist})
        if verbose:
            print(f"  iter {i}: cursor=({cx},{cy}) target=({target_x},{target_y})"
                  f" delta=({dx:+d},{dy:+d}) dist={dist:.1f}px conf={conf:.2f}")

        if dist < tolerance:
            time.sleep(SETTLE_BEFORE_CLICK_S)
            _click()
            return {"ok": True, "reason": "converged", "iters": i + 1,
                    "final_xy": (cx, cy), "final_err": dist, "history": history}

        # send corrective move (absolute pixel vector → BLE HID phone-px)
        t_last_move = time.time()
        _move_rel(dx, dy)

    return {"ok": False, "reason": "max_iters", "iters": max_iters,
            "final_xy": (cx, cy) if 'cx' in dir() else None,
            "final_err": dist if 'dist' in dir() else None,
            "history": history}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("x", type=int, help="target x in native px (0..994)")
    p.add_argument("y", type=int, help="target y in native px (0..2160)")
    p.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    p.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    p.add_argument("--weights", default=WEIGHTS_PATH)
    p.add_argument("--no-jiggle", action="store_true",
                   help="don't auto-wake cursor with a jiggle when lost")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    finder = PointerFinder(args.weights)
    print(f"loaded {os.path.basename(args.weights)} on {finder.device}")
    print(f"target: ({args.x}, {args.y})  tolerance: {args.tolerance}px")

    t0 = time.monotonic()
    result = click_at(args.x, args.y,
                      tolerance=args.tolerance,
                      max_iters=args.max_iters,
                      finder=finder,
                      allow_jiggle=not args.no_jiggle,
                      verbose=args.verbose)
    dt = time.monotonic() - t0
    print(f"\nresult ({dt*1000:.0f} ms total):")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
