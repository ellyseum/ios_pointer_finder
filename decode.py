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

    Standard interior case (axis 'x'): fit through
    ``hm[iy, ix-1], hm[iy, ix], hm[iy, ix+1]``. Parabola vertex offset from
    the integer cell is in ``[-0.5, 0.5]``.

    Border case (v0.7 #14): when the argmax sits at cell 0 or W-1 (no
    valid neighbor on one side), fall back to a one-sided fit through
    the argmax cell + the next two cells inward. Without this, every
    edge_pos sample's error floors at ``(stride-1)/2 ≈ 7-8 px`` because
    the decoder snaps to the cell-center. With it, sub-cell precision
    is recovered up to half a cell beyond the heatmap edge.

    Degenerate parabola (denominator near zero — flat heatmap) returns 0.0.
    """
    H, W = hm.shape
    if axis == "x":
        # Three taps centered on the argmax, or one-sided at the border.
        if ix <= 0:
            if W < 3:
                return 0.0
            a = float(hm[iy, 0])  # argmax (left edge)
            b = float(hm[iy, 1])
            c = float(hm[iy, 2])
            return _parabolic_offset_one_sided(a, b, c, side="left")
        if ix >= W - 1:
            if W < 3:
                return 0.0
            a = float(hm[iy, W - 3])
            b = float(hm[iy, W - 2])
            c = float(hm[iy, W - 1])  # argmax (right edge)
            return _parabolic_offset_one_sided(a, b, c, side="right")
        a = float(hm[iy, ix - 1])
        b = float(hm[iy, ix])
        c = float(hm[iy, ix + 1])
    else:
        if iy <= 0:
            if H < 3:
                return 0.0
            a = float(hm[0, ix])
            b = float(hm[1, ix])
            c = float(hm[2, ix])
            return _parabolic_offset_one_sided(a, b, c, side="left")
        if iy >= H - 1:
            if H < 3:
                return 0.0
            a = float(hm[H - 3, ix])
            b = float(hm[H - 2, ix])
            c = float(hm[H - 1, ix])
            return _parabolic_offset_one_sided(a, b, c, side="right")
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


def _parabolic_offset_one_sided(a: float, b: float, c: float, side: str) -> float:
    """One-sided parabolic fit when the argmax sits at the heatmap border.

    For ``side='left'``: fit y(t) = αt² + βt + γ through points
    ``(0, a), (1, b), (2, c)`` where ``a`` is the argmax. Offset returned
    is the vertex position relative to t=0, clamped to ``[-0.5, 0]``
    (the actual peak cannot lie more than half a cell beyond the heatmap
    edge, and cannot be inside the heatmap or cell 1 would have been the
    argmax).

    For ``side='right'``: fit through ``(0, a), (1, b), (2, c)`` where
    ``c`` is the argmax. Offset returned is relative to t=2, clamped to
    ``[0, 0.5]``.
    """
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-9:
        return 0.0
    if side == "left":
        # Vertex t* = (3a + c - 4b) / (2*(a - 2b + c)); offset from t=0.
        off = (3.0 * a + c - 4.0 * b) / (2.0 * denom)
        if off > 0.0:
            off = 0.0
        elif off < -0.5:
            off = -0.5
        return off
    # side == 'right'; offset from t=2.
    off = (3.0 * c + a - 4.0 * b) / (2.0 * denom)
    if off < 0.0:
        off = 0.0
    elif off > 0.5:
        off = 0.5
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
