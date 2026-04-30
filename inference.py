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
from train import (
    ARCHITECTURE_VERSION,
    NATIVE_H,
    NATIVE_W,
    TRAIN_H,
    TRAIN_W,
    PointerNet,
)
from decode import argmax_parabolic_native, parabolic_offset

PathLike = str | os.PathLike | Path

# Backwards-compatible alias for callers that imported the private name.
_parabolic_offset = parabolic_offset


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
        self._check_architecture_version(self.config)

    @staticmethod
    def _check_architecture_version(config: dict) -> None:
        """Assert the checkpoint's architecture version matches this code.

        v0.7+ checkpoints stamp ``architecture_version`` in their sidecar.
        Older checkpoints predate the field; load them with a warning so
        legacy weights still work but the user knows they're outside the
        version contract.
        """
        ckpt_v = config.get("architecture_version")
        if ckpt_v is None:
            import warnings
            warnings.warn(
                "Checkpoint has no architecture_version field — likely a "
                "pre-v0.7 weight file. Loading anyway, but predictions may "
                "differ if the checkpoint was trained against a different "
                "model shape than the current PointerNet.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if ckpt_v != ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"Checkpoint architecture_version={ckpt_v} but this code "
                f"expects {ARCHITECTURE_VERSION}. The model shape, head "
                f"count, or coordinate convention has changed incompatibly. "
                f"Either re-train, or check out the matching code revision."
            )

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
        logits = hm[0, 0].cpu().numpy()
        cx, cy, _peak_logit = argmax_parabolic_native(logits, nw, nh)
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid for heatmap_peak
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
