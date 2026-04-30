"""Export a trained ios_pointer_finder checkpoint to Apple CoreML (.mlpackage).

Optional macOS-only target. Requires `coremltools`:
    pip install coremltools

Usage:
    python scripts/export_coreml.py pointer_model_v0.3.4.safetensors
    python scripts/export_coreml.py pointer_model_v0.3.4.pt --out ios_pointer_finder.mlpackage

CoreML compute units default to ALL (Neural Engine + GPU + CPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add repo root so we can import inference + train.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import PointerFinder
from train import TRAIN_H, TRAIN_W


def export(weights: Path, out: Path) -> Path:
    try:
        import coremltools as ct
    except ImportError as e:
        raise SystemExit("Install coremltools first (Mac-only): pip install coremltools") from e

    finder = PointerFinder(weights, device="cpu")
    model = finder.model.eval()

    dummy = torch.zeros(1, 3, TRAIN_H, TRAIN_W, dtype=torch.float32)
    traced = torch.jit.trace(model, dummy)

    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape, dtype=ct.TensorType.float32)],
        # v0.5.1: dropped "xy" output — see CHANGELOG. PointerNet.forward now
        # returns (conf_logit, heatmap_logits); decode argmax+parabolic on the
        # heatmap externally (or via inference.PointerFinder).
        outputs=[
            ct.TensorType(name="conf"),
            ct.TensorType(name="heatmap"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = f"ios_pointer_finder ({weights.name})"
    mlmodel.save(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("weights", help=".safetensors or .pt checkpoint to export.")
    p.add_argument("--out", default=None, help="Output .mlpackage path.")
    args = p.parse_args(argv)

    weights = Path(args.weights)
    out = Path(args.out) if args.out else weights.with_suffix(".mlpackage")
    result = export(weights, out)
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
