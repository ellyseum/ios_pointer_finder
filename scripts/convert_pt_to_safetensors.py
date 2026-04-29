"""Convert legacy .pt checkpoints to .safetensors + sidecar config.json.

The .pt format we used during training packs:
  - "model": OrderedDict[str, Tensor]  — the weights
  - "epoch": int
  - "val_pos_err_px": float
  - "version": str               — semver from VERSION
  - "native_size": (W, H)        — native iPhone resolution
  - "train_size": (W, H)         — input resolution to the network
  - "val_bg_ids": list[str]      — held-out background filenames

safetensors only stores tensors, so we extract the model state_dict into the
.safetensors file and emit the metadata into a sibling .config.json file.

Usage:
    python scripts/convert_pt_to_safetensors.py path/to/pointer_model.pt
    python scripts/convert_pt_to_safetensors.py path/to/pointer_model.pt --out-dir dist/

Or batch-convert everything:
    python scripts/convert_pt_to_safetensors.py *.pt --out-dir dist/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def convert_one(pt_path: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or pt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_out = out_dir / pt_path.with_suffix(".safetensors").name
    config_out = out_dir / pt_path.with_suffix(".config.json").name

    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise SystemExit(
            "Install safetensors first: pip install 'ios-pointer-finder[safetensors]'"
        ) from e

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit(
            f"{pt_path} is not a recognized ios_pointer_finder checkpoint "
            f"(missing 'model' key). Got top-level keys: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}"
        )

    state = {k: v.contiguous() for k, v in ckpt["model"].items()}
    save_file(state, str(weights_out))

    metadata = {k: _jsonable(v) for k, v in ckpt.items() if k != "model"}
    metadata["source_pt"] = pt_path.name
    metadata["param_count"] = sum(t.numel() for t in state.values())
    config_out.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return weights_out, config_out


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return repr(value)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", help="One or more .pt checkpoint paths.")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write outputs. Defaults to alongside each input.",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else None
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"  skip (not found): {raw}", file=sys.stderr)
            continue
        if path.suffix.lower() != ".pt":
            print(f"  skip (not .pt):  {raw}", file=sys.stderr)
            continue
        weights, config = convert_one(path, out_dir)
        print(f"{path.name}  →  {weights.name} + {config.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
