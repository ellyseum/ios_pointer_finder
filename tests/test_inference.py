"""End-to-end inference contract tests.

These tests construct a PointerFinder from a freshly-initialized (random
weights) model so they can run in CI without a checkpoint download. They
verify the API shape, not numerical accuracy. The "is the model still
sharp?" tests live separately in tests/test_inference_weights.py and are
skipped when a real checkpoint isn't present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inference import PointerFinder, PointerPrediction


@pytest.fixture
def random_finder(tmp_path: Path) -> PointerFinder:
    """Build a PointerFinder around a randomly-initialized PointerNet."""
    from train import PointerNet

    weights_path = tmp_path / "random.safetensors"
    try:
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("safetensors not installed in this env")

    model = PointerNet()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(weights_path))
    return PointerFinder(weights_path, device="cpu")


def test_prediction_dataclass_unpacks(random_finder: PointerFinder, native_size):
    """PointerPrediction should be iterable as (x, y, conf, peak)."""
    img = np.zeros((native_size[1], native_size[0], 3), dtype=np.uint8)
    pred = random_finder(img)
    assert isinstance(pred, PointerPrediction)
    x, y, c, p = pred
    assert isinstance(x, int) and isinstance(y, int)
    assert isinstance(c, float) and isinstance(p, float)
    assert 0.0 <= c <= 1.0
    assert 0.0 <= p <= 1.0


def test_prediction_in_native_bounds(random_finder: PointerFinder, native_size):
    img = np.zeros((native_size[1], native_size[0], 3), dtype=np.uint8)
    pred = random_finder(img)
    assert 0 <= pred.x <= native_size[0]
    assert 0 <= pred.y <= native_size[1]
    assert pred.native_size == native_size


def test_predict_accepts_arbitrary_resolution(random_finder: PointerFinder, native_size):
    """Auto-resize should let any HxW image through."""
    weird = np.zeros((1500, 800, 3), dtype=np.uint8)
    pred = random_finder(weird)
    assert pred.native_size == native_size
    assert 0 <= pred.x <= native_size[0]


def test_predict_rejects_wrong_channel_count(random_finder: PointerFinder):
    img = np.zeros((100, 100, 4), dtype=np.uint8)  # alpha channel
    with pytest.raises(ValueError):
        random_finder(img)


def test_predict_from_path_missing(random_finder: PointerFinder, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        random_finder(tmp_path / "does-not-exist.png")


def test_call_is_alias_for_predict(random_finder: PointerFinder, native_size):
    img = np.zeros((native_size[1], native_size[0], 3), dtype=np.uint8)
    a = random_finder(img)
    b = random_finder.predict(img)
    assert a.x == b.x and a.y == b.y


def test_inference_deterministic_on_same_input(random_finder: PointerFinder, native_size):
    img = np.zeros((native_size[1], native_size[0], 3), dtype=np.uint8)
    a = random_finder(img)
    b = random_finder(img)
    assert (a.x, a.y) == (b.x, b.y)
    assert a.confidence == pytest.approx(b.confidence, abs=1e-5)


@pytest.mark.weights
def test_published_checkpoint_loads_if_present(repo_root: Path):
    """If the canonical .safetensors is in the repo (local dev), it should load.

    Skipped on CI / fresh clones where weights aren't downloaded yet.
    """
    candidates = list(repo_root.glob("pointer_model_v*.safetensors"))
    if not candidates:
        pytest.skip("no .safetensors checkpoint present (run convert_pt_to_safetensors.py)")
    finder = PointerFinder(candidates[0], device="cpu")
    assert finder.model is not None
