"""Unit tests for `inference._parabolic_offset` — vertex offset from a
3-point parabola fit on a heatmap argmax cell + its two neighbors."""

from __future__ import annotations

import numpy as np

from inference import _parabolic_offset


def _hm_with_x_neighbors(a: float, b: float, c: float) -> np.ndarray:
    """Build a 5x5 heatmap with the argmax-row [_, a, b, c, _] at iy=2."""
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[2, 1] = a
    hm[2, 2] = b
    hm[2, 3] = c
    return hm


def test_symmetric_peak_returns_zero():
    """a == c → vertex sits on the integer cell. Offset must be 0."""
    hm = _hm_with_x_neighbors(0.5, 1.0, 0.5)
    assert _parabolic_offset(hm, 2, 2, "x") == 0.0


def test_left_skew_returns_negative():
    """Left neighbor higher than right → vertex is left of the integer cell."""
    hm = _hm_with_x_neighbors(0.8, 1.0, 0.4)
    off = _parabolic_offset(hm, 2, 2, "x")
    assert -0.5 <= off < 0.0


def test_right_skew_returns_positive():
    hm = _hm_with_x_neighbors(0.4, 1.0, 0.8)
    off = _parabolic_offset(hm, 2, 2, "x")
    assert 0.0 < off <= 0.5


def test_classic_quadratic_offset():
    """Closed-form: 0.5*(a-c)/(a-2b+c) = 0.5*(0-1)/(0-4+1) = +1/6.

    Pick a, b, c such that the analytic vertex is exactly on a known fraction,
    then assert the helper returns it to float-precision.
    """
    hm = _hm_with_x_neighbors(0.0, 2.0, 1.0)
    off = _parabolic_offset(hm, 2, 2, "x")
    assert abs(off - (1.0 / 6.0)) < 1e-6


def test_left_border_returns_zero():
    """Argmax on the left column has no a-neighbor — offset must default to 0."""
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[2, 0] = 1.0
    assert _parabolic_offset(hm, 0, 2, "x") == 0.0


def test_right_border_returns_zero():
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[2, 4] = 1.0
    assert _parabolic_offset(hm, 4, 2, "x") == 0.0


def test_top_border_y_axis_returns_zero():
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[0, 2] = 1.0
    assert _parabolic_offset(hm, 2, 0, "y") == 0.0


def test_bottom_border_y_axis_returns_zero():
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[4, 2] = 1.0
    assert _parabolic_offset(hm, 2, 4, "y") == 0.0


def test_flat_heatmap_returns_zero():
    """All neighbors equal → degenerate parabola (denom == 0). Default to 0."""
    hm = np.full((5, 5), 0.7, dtype=np.float32)
    assert _parabolic_offset(hm, 2, 2, "x") == 0.0
    assert _parabolic_offset(hm, 2, 2, "y") == 0.0


def test_offset_is_clamped_to_half_cell():
    """Even pathological neighbors should clamp to [-0.5, 0.5]."""
    # Construct a case where the unclamped formula would exceed 0.5
    # (a much greater than b, b only slightly above c).
    hm = _hm_with_x_neighbors(2.0, 1.0, 0.99)
    off = _parabolic_offset(hm, 2, 2, "x")
    assert -0.5 <= off <= 0.5


def test_y_axis_works_symmetrically_to_x():
    """Vertical equivalent of test_classic_quadratic_offset (vertex at +1/6)."""
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[1, 2] = 0.0
    hm[2, 2] = 2.0
    hm[3, 2] = 1.0
    off = _parabolic_offset(hm, 2, 2, "y")
    assert abs(off - (1.0 / 6.0)) < 1e-6
