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
    "reason": "converged" | "stale_pipeline" | "pipeline_dead" | "max_iters"
              | "cursor_lost" | "static_lock" | "hands_dead" | "bad_frame",
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
PEAK_THRESHOLD = 0.4          # heatmap peak prob; below this, also lost.
                              # v0.7.1: dropped from 0.5 → 0.4 to match v0.7's
                              # calibrated peak distribution. v0.4's mean-BCE
                              # left positive logits unbounded → peaks saturated
                              # at 1.000; v0.7's sum-BCE bounds them, so true-
                              # positive peaks now sit ~0.91 typical and
                              # uncertain frames at ~0.33-0.37. The previous
                              # 0.5 gate false-negatived all uncertain frames;
                              # 0.4 keeps the model's "I'm not confident here"
                              # signal as cursor-lost while passing the
                              # confident-localization band intact.
WAIT_FRAME_TIMEOUT_S = 0.6
SETTLE_BEFORE_CLICK_S = 0.05  # let the last move quiesce before firing
PIPELINE_STALE_S = 3.0        # newest frame older than this → pipeline dead


# ---------- inference ----------
#
# v0.7: PointerFinder is a thin adapter over inference.PointerFinder. The
# previous local class duplicated the decode pipeline (argmax + parabolic +
# stride-aware native mapping); two copies inevitably drift. The adapter
# preserves find()'s tuple return signature so existing callers keep
# working unchanged. Decode logic now lives in decode.argmax_parabolic_native.

from inference import PointerFinder as _InferencePointerFinder


class PointerFinder:
    """Closed-loop deployment adapter over inference.PointerFinder.

    Preserves the (cx, cy, conf, peak) tuple return shape that the servo
    loop and `find()` callers depend on; deprecates the local decoder
    fork that drifted from inference.py during v0.5/v0.6.
    """

    def __init__(self, weights_path: str = WEIGHTS_PATH, device: str | None = None):
        self._finder = _InferencePointerFinder(weights_path, device=device)
        self.native_size = self._finder.native_size
        self.train_size = self._finder.train_size

    def find(self, img_bgr: np.ndarray) -> tuple[int, int, float, float] | None:
        """Returns (cx, cy, conf, peak_prob) in native-px coords, or None
        if the image shape doesn't match the model's expected native size.
        """
        nh, nw = self.native_size[1], self.native_size[0]
        if img_bgr.shape[:2] != (nh, nw):
            return None
        pred = self._finder.predict(img_bgr)
        return pred.x, pred.y, pred.confidence, pred.heatmap_peak


# ---------- frame source ----------

def _safe_sorted_ring() -> list[tuple[float, str]]:
    """List the ring as (mtime, path) tuples, newest first, tolerating
    files that get rotated out between glob and stat (multifilesink with
    max-files=8 deletes oldest as it writes new)."""
    pairs: list[tuple[float, str]] = []
    for f in glob.glob(JPEG_GLOB):
        try:
            pairs.append((os.path.getmtime(f), f))
        except OSError:
            continue
    pairs.sort(reverse=True)
    return pairs


def _newest_finalized_frame(after_t_wall: float = 0.0,
                             timeout: float = WAIT_FRAME_TIMEOUT_S) -> str | None:
    """Return path of the 2nd-newest JPEG (which is finalized — files[0] may
    be mid-write) whose mtime > after_t_wall. None on timeout. Wall-clock
    comparison since os.path.getmtime is wall-clock; deadline uses monotonic."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ring = _safe_sorted_ring()
        if len(ring) >= 2 and ring[1][0] > after_t_wall:
            return ring[1][1]
        time.sleep(0.02)
    return None


def _pipeline_alive() -> bool:
    ring = _safe_sorted_ring()
    if not ring:
        return False
    return (time.time() - ring[0][0]) < PIPELINE_STALE_S


# ---------- hands API ----------

def _move_rel(dx: int, dy: int, step: int = 4, delay_ms: int = 3,
              timeout: float = 20.0) -> None:
    """Send a relative-move command. BLE HID is slow (≈1ms per HID step plus
    BLE write latency) so timeout needs to scale with distance — a 700 px
    diagonal move easily exceeds 2s. 20s gives plenty of headroom."""
    import requests
    requests.post(
        f"{HANDS_URL}/move_rel",
        json={"dx": int(dx), "dy": int(dy), "step": step, "delay_ms": delay_ms},
        timeout=timeout,
    )


def _click(hold_ms: int = 60) -> None:
    import requests
    requests.post(f"{HANDS_URL}/click", json={"hold_ms": hold_ms}, timeout=5.0)


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

    # Per-axis gain — observed_native_px / commanded_phone_px.
    # iOS Tracking Speed makes /move_rel scale unpredictable; we learn it live.
    # Initial guess 0.15 (rough average from earlier observations 0.1-0.5).
    # Updated via probe move below + per-iter EMA refinement.
    gain_x: float = 0.15
    gain_y: float = 0.15
    GAIN_MIN, GAIN_MAX = 0.03, 3.0
    GAIN_EMA_ALPHA = 0.4   # weight for new sample vs prior gain
    PROBE_PHONE_PX = 80    # known probe magnitude
    MIN_OBSERVED_FOR_GAIN = 6  # below this, motion is too small/noisy to trust

    def _update_gain(commanded: int, observed: int, prior: float) -> float:
        """EMA-blend a new gain sample with prior. Reject when sign mismatches
        (means cursor was clamped at edge or hit a snap-target) or motion is
        too small (sub-noise observation)."""
        if abs(commanded) < 10 or abs(observed) < MIN_OBSERVED_FOR_GAIN:
            return prior
        if (commanded > 0) != (observed > 0):
            return prior  # sign mismatch — discard
        sample = observed / commanded
        sample = max(GAIN_MIN, min(GAIN_MAX, sample))
        return (1 - GAIN_EMA_ALPHA) * prior + GAIN_EMA_ALPHA * sample

    # ---- probe phase: send a known move toward target, measure observed delta ----
    snap = _newest_finalized_frame(after_t_wall=0.0)
    if snap is None:
        return {"ok": False, "reason": "stale_pipeline", "iters": 0,
                "final_xy": None, "final_err": None, "history": history}
    img = cv2.imread(snap)
    res0 = finder.find(img) if img is not None else None
    if res0 is not None and (res0[2] >= CONF_THRESHOLD and res0[3] >= PEAK_THRESHOLD):
        sx0, sy0, _, _ = res0
        sign_x = 1 if target_x >= sx0 else -1
        sign_y = 1 if target_y >= sy0 else -1
        probe_dx = sign_x * PROBE_PHONE_PX
        probe_dy = sign_y * PROBE_PHONE_PX
        if verbose:
            print(f"  probe: cursor=({sx0},{sy0}) → /move_rel({probe_dx:+d},{probe_dy:+d})")
        t_last_move = time.time()
        try:
            _move_rel(probe_dx, probe_dy)
        except Exception as e:
            if verbose: print(f"  probe move err: {e}")
        snap1 = _newest_finalized_frame(after_t_wall=t_last_move)
        if snap1 is not None:
            img1 = cv2.imread(snap1)
            res1 = finder.find(img1) if img1 is not None else None
            if res1 is not None:
                sx1, sy1, _, _ = res1
                obs_dx = sx1 - sx0
                obs_dy = sy1 - sy0
                gain_x = _update_gain(probe_dx, obs_dx, gain_x)
                gain_y = _update_gain(probe_dy, obs_dy, gain_y)
                if verbose:
                    print(f"  probe result: observed=({obs_dx:+d},{obs_dy:+d})"
                          f" → gain_x={gain_x:.3f} gain_y={gain_y:.3f}")

    # ---- main loop ----
    prev_cx: int | None = None
    prev_cy: int | None = None
    prev_peak: float | None = None
    prev_cmd_dx: int | None = None
    prev_cmd_dy: int | None = None
    static_lock_streak = 0
    # Init the loop-body locals so the max_iters fallthrough has a
    # well-defined "no detection happened" state without relying on the
    # `'cx' in dir()` introspection idiom.
    cx: int | None = None
    cy: int | None = None
    dist: float | None = None

    # Static-feature lock detection: when the model is locked onto a UI
    # distractor (cursor auto-hidden, etc.), the heatmap peak is identical
    # to many decimal places frame-to-frame because it's a constant scene
    # feature. Real cursor detections vary slightly from compression noise.
    STATIC_LOCK_PEAK_EPS = 1e-5
    STATIC_LOCK_POS_EPS = 1     # native px
    MIN_CMD_FOR_LOCK_CHECK = 50  # only flag lock if we actually commanded a meaningful move

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

        # Static-lock detection: meaningful command issued + position unchanged
        # + peak invariant to high precision = model is hallucinating on a
        # static UI element, not tracking a real cursor.
        is_static_lock = False
        if (prev_cx is not None and prev_peak is not None
                and prev_cmd_dx is not None and prev_cmd_dy is not None):
            obs_dx = cx - prev_cx
            obs_dy = cy - prev_cy
            commanded_meaningful = (abs(prev_cmd_dx) >= MIN_CMD_FOR_LOCK_CHECK
                                    or abs(prev_cmd_dy) >= MIN_CMD_FOR_LOCK_CHECK)
            no_motion = abs(obs_dx) <= STATIC_LOCK_POS_EPS and abs(obs_dy) <= STATIC_LOCK_POS_EPS
            invariant_peak = abs(peak - prev_peak) < STATIC_LOCK_PEAK_EPS
            is_static_lock = commanded_meaningful and no_motion and invariant_peak

            # EMA-refine gain ONLY when observed motion is non-trivial (avoid
            # snap-to-item or edge-clamp from collapsing gain toward zero).
            if not no_motion:
                gain_x = _update_gain(prev_cmd_dx, obs_dx, gain_x)
                gain_y = _update_gain(prev_cmd_dy, obs_dy, gain_y)

        # cursor-lost path → jiggle + retry. Treat static-lock as a lost
        # detection (model is wrong; jiggle will move cursor enough to
        # break out of any UI-element false positive).
        cursor_lost = (conf < CONF_THRESHOLD) or (peak < PEAK_THRESHOLD)
        if cursor_lost or is_static_lock:
            lost_streak += 1
            if is_static_lock:
                static_lock_streak += 1
            note = "static_lock" if is_static_lock else "lost"
            history.append({"iter": i, "cx": cx, "cy": cy, "conf": conf, "peak": peak,
                            "dx": None, "dy": None, "note": note})
            if verbose:
                print(f"  iter {i}: conf={conf:.2f} peak={peak:.4f} → {note}"
                      f" (streak {lost_streak})")
            if lost_streak >= 3:
                reason = "static_lock" if static_lock_streak >= 2 else "cursor_lost"
                return {"ok": False, "reason": reason, "iters": i + 1,
                        "final_xy": None, "final_err": None, "history": history}
            if allow_jiggle:
                t_last_move = _jiggle_to_wake()
            # After jiggle, force the next iteration to NOT think it's still
            # locked — clear the prev tracking so static-lock re-evaluates fresh.
            prev_cx = prev_cy = prev_peak = None
            prev_cmd_dx = prev_cmd_dy = None
            continue
        lost_streak = 0
        static_lock_streak = 0

        dx = target_x - cx
        dy = target_y - cy
        dist = float(np.hypot(dx, dy))
        history.append({"iter": i, "cx": cx, "cy": cy, "conf": conf, "peak": peak,
                        "dx": dx, "dy": dy, "dist": dist,
                        "gain_x": round(gain_x, 3), "gain_y": round(gain_y, 3)})
        if verbose:
            print(f"  iter {i}: cursor=({cx},{cy}) target=({target_x},{target_y})"
                  f" delta=({dx:+d},{dy:+d}) dist={dist:.1f}px"
                  f" gain=({gain_x:.2f},{gain_y:.2f}) conf={conf:.2f}")

        if dist < tolerance:
            time.sleep(SETTLE_BEFORE_CLICK_S)
            _click()
            return {"ok": True, "reason": "converged", "iters": i + 1,
                    "final_xy": (cx, cy), "final_err": dist, "history": history,
                    "gain": (round(gain_x, 3), round(gain_y, 3))}

        # Compute corrective move in phone-px: invert the gain so a delta of
        # N native px gets sent as N/gain phone-px. Cap per-iteration command
        # at ±800 (was 1500) — keeps each BLE move under ~3s so the loop can
        # self-correct quickly instead of one giant move that overshoots.
        cmd_dx = int(round(dx / gain_x))
        cmd_dy = int(round(dy / gain_y))
        cmd_dx = max(-800, min(800, cmd_dx))
        cmd_dy = max(-800, min(800, cmd_dy))
        prev_cx, prev_cy = cx, cy
        prev_peak = peak
        prev_cmd_dx, prev_cmd_dy = cmd_dx, cmd_dy
        t_last_move = time.time()
        try:
            _move_rel(cmd_dx, cmd_dy)
        except Exception as e:
            if verbose: print(f"  move err: {e}")

    return {"ok": False, "reason": "max_iters", "iters": max_iters,
            "final_xy": (cx, cy) if cx is not None else None,
            "final_err": dist if dist is not None else None,
            "gain": (round(gain_x, 3), round(gain_y, 3)),
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
