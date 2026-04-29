"""Architectural smoke tests — shape, dtype, parameter count."""

from __future__ import annotations

import torch

from train import TRAIN_H, TRAIN_W, PointerNet


def test_model_constructs():
    model = PointerNet()
    assert isinstance(model, torch.nn.Module)


def test_param_count_reasonable():
    """If this fails, somebody changed the architecture without updating the
    model card. 250K-450K is the sane band for this network."""
    model = PointerNet()
    n = sum(p.numel() for p in model.parameters())
    assert 250_000 <= n <= 450_000, f"unexpected param count: {n}"


def test_forward_shape_eval():
    """Forward pass on a unit-batch, native-resized input should return
    (xy [B,2], conf [B,1], heatmap [B,1,h,w])."""
    model = PointerNet().eval()
    x = torch.zeros(1, 3, TRAIN_H, TRAIN_W, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, (tuple, list))
    assert len(out) == 3, f"expected 3 outputs, got {len(out)}"

    xy, conf, hm = out
    assert xy.shape == (1, 2), f"xy shape: {xy.shape}"
    assert conf.shape in {(1, 1), (1,)}, f"conf shape: {conf.shape}"
    assert hm.dim() == 4 and hm.shape[0] == 1 and hm.shape[1] == 1, f"hm shape: {hm.shape}"
    # 1/8 stride of the train resolution (3 stride-2 conv blocks).
    assert hm.shape[2] in {TRAIN_H // 8, TRAIN_H // 8 + 1}, f"hm height: {hm.shape[2]}"
    assert hm.shape[3] in {TRAIN_W // 8, TRAIN_W // 8 + 1}, f"hm width: {hm.shape[3]}"


def test_forward_dtype():
    model = PointerNet().eval()
    x = torch.zeros(1, 3, TRAIN_H, TRAIN_W, dtype=torch.float32)
    with torch.no_grad():
        xy, conf, hm = model(x)
    assert xy.dtype == torch.float32
    assert conf.dtype == torch.float32
    assert hm.dtype == torch.float32


def test_batch_dim_passthrough():
    """A batch of N should produce N outputs, not just 1 — catches squeeze() bugs."""
    model = PointerNet().eval()
    x = torch.zeros(3, 3, TRAIN_H, TRAIN_W, dtype=torch.float32)
    with torch.no_grad():
        xy, conf, hm = model(x)
    assert xy.shape[0] == 3
    assert hm.shape[0] == 3
