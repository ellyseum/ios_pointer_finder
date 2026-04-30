"""decode.py — canonical heatmap decoder.

Single source of truth for the argmax + parabolic-subpixel + stride-aware
native-pixel mapping that converts a model heatmap into (x, y) cursor
coordinates. Every consumer (inference.py, click_at.py, test_real.py,
test_real_bbox.py, eval_v03.py, train.py val) must import from here.

Decoder drift between forked copies of this logic was the root cause of
the v0.5/v0.6 path divergence bug — separate clamps on the parabolic
offset, separate stride formulas, separate edge handling. Centralizing
the decode here means a future fix lands once and propagates everywhere.

Parabolic subpixel fits on RAW LOGITS, not the sigmoid heatmap: the
training target exp(-d²/2σ²) implies a parabolic logit profile near the
peak, while the sigmoid saturates at the peak and flattens the second
derivative. Fitting on logits restores the precision the sigmoid would
have lost.

Stride convention: cell ``i`` maps to native pixel ``i*s + (s-1)/2``,
where ``s = native_dim / hm_dim`` per axis. This is the receptive-field
center, not the corner.
"""
from __future__ import annotations

import numpy as np


def parabolic_offset(hm: np.ndarray, ix: int, iy: int, axis: str) -> float:
    """Sub-cell offset of the parabola fit through the argmax cell + 2 neighbors.

    For axis 'x', uses ``hm[iy, ix-1], hm[iy, ix], hm[iy, ix+1]``.
    For axis 'y', uses ``hm[iy-1, ix], hm[iy, ix], hm[iy+1, ix]``.

    Returns 0.0 when the argmax is on the heatmap border (no valid
    neighbor) or when the parabola is degenerate (denominator near zero —
    flat heatmap). Returned offset is clamped to ``[-0.5, 0.5]``: the
    parabola vertex cannot be more than half a cell from the integer
    cell if that cell really is the argmax.
    """
    H, W = hm.shape
    if axis == "x":
        if ix <= 0 or ix >= W - 1:
            return 0.0
        a = float(hm[iy, ix - 1])
        b = float(hm[iy, ix])
        c = float(hm[iy, ix + 1])
    else:
        if iy <= 0 or iy >= H - 1:
            return 0.0
        a = float(hm[iy - 1, ix])
        b = float(hm[iy, ix])
        c = float(hm[iy + 1, ix])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-9:
        return 0.0
    off = 0.5 * (a - c) / denom
    if off > 0.5:
        off = 0.5
    elif off < -0.5:
        off = -0.5
    return off


def argmax_parabolic_native(
    logits: np.ndarray,
    native_w: int,
    native_h: int,
) -> tuple[int, int, float]:
    """Decode a 2-D logit heatmap to native cursor pixel coords.

    Args:
        logits: shape ``(H, W)`` raw logit values (NOT sigmoid output).
        native_w: native screen width in pixels.
        native_h: native screen height in pixels.

    Returns:
        ``(cx, cy, peak_logit)`` where ``cx, cy`` are integer native
        pixel coordinates clamped to ``[0, native_w-1] x [0, native_h-1]``,
        and ``peak_logit`` is the raw logit at the argmax cell (apply
        ``sigmoid`` for a presence-probability score if needed).

    The decode pipeline:
        1. Hard argmax → integer cell ``(ix, iy)``.
        2. Parabolic subpixel offset on raw logits (per-axis).
        3. Stride-aware mapping: cell center ``i*s + (s-1)/2``.
        4. Round + clamp to native bounds.
    """
    H, W = logits.shape
    flat = int(logits.argmax())
    iy, ix = flat // W, flat % W
    rx = float(ix) + parabolic_offset(logits, ix, iy, axis="x")
    ry = float(iy) + parabolic_offset(logits, ix, iy, axis="y")
    stride_x = native_w / W
    stride_y = native_h / H
    rx_native = rx * stride_x + (stride_x - 1.0) / 2.0
    ry_native = ry * stride_y + (stride_y - 1.0) / 2.0
    cx = min(native_w - 1, max(0, int(round(rx_native))))
    cy = min(native_h - 1, max(0, int(round(ry_native))))
    return cx, cy, float(logits[iy, ix])
