"""Export a trained ios_pointer_finder checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py pointer_model_v0.3.4_30.5px.safetensors
    python scripts/export_onnx.py pointer_model_v0.3.4_30.5px.pt --out pointer_model_v0.3.4.onnx --opset 17
    python scripts/export_onnx.py pointer_model.pt --check  # numerical parity vs PyTorch

The exported ONNX has:
  - input  "image":   float32 [1, 3, TRAIN_H, TRAIN_W] in [-2, 2] normalized space
  - output "xy":      float32 [1, 2]   — regression head (kept for back-compat)
  - output "conf":    float32 [1]      — confidence logit
  - output "heatmap": float32 [1, 1, H/16, W/16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add repo root to path so we can import inference / train.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import PointerFinder
from train import TRAIN_H, TRAIN_W


def export(weights: Path, out: Path, opset: int = 17, check: bool = False) -> Path:
    finder = PointerFinder(weights, device="cpu")
    model = finder.model

    dummy = torch.zeros(1, 3, TRAIN_H, TRAIN_W, dtype=torch.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["xy", "conf", "heatmap"],
        dynamic_axes={
            "image": {0: "batch"},
            "xy": {0: "batch"},
            "conf": {0: "batch"},
            "heatmap": {0: "batch"},
        },
        opset_version=opset,
    )

    if check:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit("Install onnxruntime first: pip install 'ios-pointer-finder[onnx]'") from e
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, TRAIN_H, TRAIN_W).astype(np.float32)
        ort_outs = sess.run(None, {"image": x})

        with torch.no_grad():
            torch_outs = model(torch.from_numpy(x))
        torch_outs = [t.cpu().numpy() for t in torch_outs]

        for name, t, o in zip(["xy", "conf", "heatmap"], torch_outs, ort_outs, strict=False):
            diff = float(np.max(np.abs(t - o)))
            print(f"  parity check {name:>7}: max |Δ| = {diff:.2e}")
            assert diff < 1e-4, f"ONNX export drifted on {name}: {diff}"

    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("weights", help=".safetensors or .pt checkpoint to export.")
    p.add_argument("--out", default=None, help="Output .onnx path.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument(
        "--check",
        action="store_true",
        help="Run a parity check against the PyTorch model (requires onnxruntime).",
    )
    args = p.parse_args(argv)

    weights = Path(args.weights)
    out = Path(args.out) if args.out else weights.with_suffix(".onnx")
    result = export(weights, out, opset=args.opset, check=args.check)
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
