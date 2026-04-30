"""Public inference API for ios_pointer_finder.

Example:
    >>> from inference import PointerFinder
    >>> finder = PointerFinder.from_pretrained("pointer_model_v0.3.4_30.5px.safetensors")
    >>> result = finder("snap.jpg")
    >>> print(result.x, result.y, result.confidence)

PointerFinder loads a checkpoint (.safetensors preferred, .pt accepted), runs the
model on a BGR uint8 image (any resolution — auto-resized to native), and returns
a PointerPrediction dataclass.

The on-disk checkpoint contract:
- .safetensors file with the model state_dict (no optimizer state).
- Optional sidecar `<name>.config.json` carrying training metadata
  (version, val_pos_err_px, train_size, native_size, epoch).
- For legacy .pt files, the full pickle is loaded and the "model" key is used.

This file is the single import point library users should rely on. Internal
training code (train.py, synthesize.py) is not stable API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# PointerNet definition lives in train.py for now — re-export here as part of
# the public surface so users don't need to import from train internals.
from train import NATIVE_H, NATIVE_W, TRAIN_H, TRAIN_W, PointerNet

PathLike = str | os.PathLike | Path


def _parabolic_offset(hm: np.ndarray, ix: int, iy: int, axis: str) -> float:
    """Sub-cell offset of the parabola fit through the argmax cell + 2 neighbors.

    For axis 'x', uses hm[iy, ix-1], hm[iy, ix], hm[iy, ix+1].
    For axis 'y', uses hm[iy-1, ix], hm[iy, ix], hm[iy+1, ix].

    Returns 0.0 when the argmax is on the heatmap border (no valid neighbor)
    or when the parabola is degenerate (denominator near zero — flat heatmap).
    Returns a clamped offset in [-0.5, 0.5] (the parabola vertex can't be
    further from the integer cell than half a cell width if the cell really
    is the argmax).
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


@dataclass
class PointerPrediction:
    """Result of one ios_pointer_finder inference call.

    Attributes:
        x: predicted cursor x in native screen pixels (994 px wide by default).
        y: predicted cursor y in native screen pixels (2160 px tall by default).
        confidence: P(cursor present) in [0, 1]. Set a threshold (e.g. 0.5) to
            decide whether to trust (x, y).
        heatmap_peak: max of the sigmoid heatmap in [0, 1]. Independent signal
            from confidence — both should agree for a real detection.
        native_size: (W, H) the prediction is expressed in.
    """

    x: int
    y: int
    confidence: float
    heatmap_peak: float
    native_size: tuple[int, int]

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.confidence
        yield self.heatmap_peak


class PointerFinder:
    """Stateful cursor detector. Construct once, call many times."""

    def __init__(
        self,
        weights: PathLike,
        device: str | torch.device | None = None,
        config: dict | None = None,
    ):
        """Load weights from a .safetensors or .pt file.

        Args:
            weights: path to checkpoint. Format detected by suffix.
            device: torch device. Defaults to CUDA if available, else CPU.
            config: optional metadata dict. If provided, overrides the sidecar
                config.json. Used by from_pretrained().
        """
        self.weights_path = Path(weights)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = PointerNet().to(self.device).eval()
        state_dict = self._load_state_dict(self.weights_path)
        self.model.load_state_dict(state_dict)

        self.config = config or self._load_sidecar_config(self.weights_path)
        self.native_size = tuple(self.config.get("native_size", (NATIVE_W, NATIVE_H)))
        self.train_size = tuple(self.config.get("train_size", (TRAIN_W, TRAIN_H)))

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        filename: str = "pointer_model.safetensors",
        config_filename: str = "config.json",
        revision: str | None = None,
        **kwargs,
    ) -> PointerFinder:
        """Load by HF repo id, local path, or bare checkpoint name.

        Resolution order:
            1. If `name_or_path` exists as a local file → load directly.
            2. If it looks like an HF repo id (contains "/") → download via
               huggingface_hub (requires `pip install ios-pointer-finder[hub]`).
            3. Otherwise → treat as a relative path under cwd.

        Args:
            name_or_path: HF repo id ("ellyseum/ios_pointer_finder"), local
                checkpoint path, or bare filename relative to cwd.
            filename: weight file inside the HF repo. Default `pointer_model.safetensors`
                points at the canonical "current" weights. Override to pin a
                specific historical version (e.g. `pointer_model_v0.3.4_30.5px.safetensors`).
            config_filename: sidecar metadata file inside the HF repo.
                Set to None to skip config download.
            revision: HF repo branch, tag, or commit SHA. Default uses the
                main branch's HEAD. Pin a tag (e.g. `v0.3.4`) for reproducibility.
            **kwargs: forwarded to PointerFinder.__init__ (device, etc.).
        """
        candidate = Path(name_or_path)
        if candidate.is_file():
            return cls(candidate, **kwargs)

        if "/" in name_or_path and not candidate.exists():
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "Loading from Hugging Face Hub requires `pip install ios-pointer-finder[hub]`."
                ) from e
            weights = hf_hub_download(
                repo_id=name_or_path,
                filename=filename,
                revision=revision,
            )
            config = None
            if config_filename:
                try:
                    config_path = hf_hub_download(
                        repo_id=name_or_path,
                        filename=config_filename,
                        revision=revision,
                    )
                    with open(config_path) as f:
                        config = json.load(f)
                except Exception:
                    config = None
            return cls(weights, config=config, **kwargs)

        return cls(candidate, **kwargs)

    def predict(self, image: PathLike | np.ndarray) -> PointerPrediction:
        """Run inference on a single image.

        Args:
            image: file path or a BGR uint8 numpy array. Any resolution —
                auto-resized to the model's native size.

        Returns:
            PointerPrediction with (x, y) in native pixel coordinates.
        """
        if isinstance(image, (str, os.PathLike, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        else:
            img = np.asarray(image)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape {img.shape}")

        nw, nh = self.native_size
        if img.shape[:2] != (nh, nw):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        tw, th = self.train_size
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - 0.5) / 0.25).unsqueeze(0).to(self.device)

        with torch.no_grad():
            conf_logit, hm = self.model(x)  # v0.5.1: forward returns 2-tuple

        conf = float(torch.sigmoid(conf_logit).item())
        # v0.5: parabolic subpixel fit is applied to RAW LOGITS, not sigmoid
        # output. The training target is exp(-d²/2σ²); logit(target) is
        # parabolic in `d` near the peak, while the sigmoid saturates and
        # collapses the second derivative. Fitting on logits restores subpixel
        # precision lost to saturation bias.
        logits = hm[0, 0].cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid for heatmap_peak
        H, W = logits.shape
        flat = int(logits.argmax())
        iy, ix = flat // W, flat % W
        # Parabolic subpixel refinement on logits.
        rx = float(ix) + _parabolic_offset(logits, ix, iy, axis="x")
        ry = float(iy) + _parabolic_offset(logits, ix, iy, axis="y")
        # v0.5: stride-aware cell→native mapping (replaces v0.4 linear stretch
        # `rx/(W-1)*nw` which forced the model to learn a non-uniform spatial
        # warp). Cell `i` has receptive-field center at native pixel
        # `i*stride + (stride-1)/2` where stride = native_dim / hm_dim.
        stride_x = nw / W
        stride_y = nh / H
        rx_native = rx * stride_x + (stride_x - 1.0) / 2.0
        ry_native = ry * stride_y + (stride_y - 1.0) / 2.0
        cx = min(nw - 1, max(0, int(round(rx_native))))
        cy = min(nh - 1, max(0, int(round(ry_native))))
        return PointerPrediction(
            x=cx,
            y=cy,
            confidence=conf,
            heatmap_peak=float(prob.max()),
            native_size=(nw, nh),
        )

    def __call__(self, image: PathLike | np.ndarray) -> PointerPrediction:
        return self.predict(image)

    @staticmethod
    def _load_state_dict(path: Path) -> dict:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "Reading .safetensors checkpoints requires "
                    "`pip install ios-pointer-finder[safetensors]`."
                ) from e
            return load_file(str(path))
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            return ckpt["model"]
        return ckpt

    @staticmethod
    def _load_sidecar_config(weights_path: Path) -> dict:
        config_path = weights_path.with_suffix(".config.json")
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        # Try the legacy .pt embedded metadata.
        if weights_path.suffix.lower() == ".pt":
            try:
                ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    return {
                        k: v
                        for k, v in ckpt.items()
                        if k in {"version", "val_pos_err_px", "epoch", "native_size", "train_size"}
                    }
            except Exception:
                pass
        return {}


__all__ = ["PointerFinder", "PointerPrediction"]
