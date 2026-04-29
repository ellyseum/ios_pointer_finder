"""Command-line entry point for ios-pointer-finder.

Installed as the `ipf` console script via pyproject.toml.

    ipf predict <image> [--weights PATH] [--device DEV] [--json]
    ipf version

For training / synthesis / eval, run the underlying scripts directly:
    python train.py ...
    python synthesize.py ...
    python eval_v03.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION_PATH = Path(__file__).parent / "VERSION"


def _read_version() -> str:
    try:
        return VERSION_PATH.read_text().strip()
    except OSError:
        return "unknown"


def _cmd_predict(args: argparse.Namespace) -> int:
    from inference import PointerFinder

    finder = PointerFinder.from_pretrained(args.weights, device=args.device)
    result = finder(args.image)

    if args.json:
        print(
            json.dumps(
                {
                    "x": result.x,
                    "y": result.y,
                    "confidence": result.confidence,
                    "heatmap_peak": result.heatmap_peak,
                    "native_size": list(result.native_size),
                }
            )
        )
    else:
        print(
            f"({result.x}, {result.y})  "
            f"conf={result.confidence:.3f}  "
            f"peak={result.heatmap_peak:.3f}  "
            f"native={result.native_size[0]}x{result.native_size[1]}"
        )
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_read_version())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ios-pointer-finder",
        description="Find the iPhone Pointer-Control cursor in an image.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Run inference on an image.")
    p_pred.add_argument("image", help="Path to a BGR image readable by OpenCV.")
    p_pred.add_argument(
        "--weights",
        default="ellyseum/ios_pointer_finder",
        help="HF repo id or path to .safetensors / .pt (default: ellyseum/ios_pointer_finder).",
    )
    p_pred.add_argument(
        "--device",
        default=None,
        help="torch device (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    p_pred.add_argument("--json", action="store_true", help="Output JSON.")
    p_pred.set_defaults(func=_cmd_predict)

    p_ver = sub.add_parser("version", help="Print model version.")
    p_ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
