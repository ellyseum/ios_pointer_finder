"""CLI surface tests — verifies the `ios-pointer-finder` entry point works."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import cli


def test_version_prints(capsys, repo_root: Path):
    rc = cli.main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out, "version subcommand should print non-empty version"
    # Loosely match a semver-ish string.
    assert out[0].isdigit() or out == "unknown"


def test_predict_uses_local_safetensors(tmp_path: Path, capsys, monkeypatch, native_size):
    """`ios-pointer-finder predict <image> --weights <path>` should run end-to-end
    on a random-init checkpoint and print sensible numbers."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("safetensors not installed")

    # Random-init weights
    from train import PointerNet

    weights = tmp_path / "random.safetensors"
    save_file({k: v.contiguous() for k, v in PointerNet().state_dict().items()}, str(weights))

    # Image: solid grey at native size
    img_path = tmp_path / "frame.jpg"
    img = np.full((native_size[1], native_size[0], 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    rc = cli.main(
        ["predict", str(img_path), "--weights", str(weights), "--device", "cpu", "--json"]
    )
    out = capsys.readouterr().out.strip()
    assert rc == 0

    import json

    data = json.loads(out)
    assert {"x", "y", "confidence", "heatmap_peak", "native_size"} <= set(data.keys())
    assert 0 <= data["x"] <= native_size[0]
    assert 0 <= data["y"] <= native_size[1]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["heatmap_peak"] <= 1.0
