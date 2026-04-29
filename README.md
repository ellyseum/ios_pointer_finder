# ios_pointer_finder

A 338K-parameter CNN that finds the iPhone Pointer-Control cursor in a screen capture, in 10 ms.

[![CI](https://github.com/ellyseum/ios_pointer_finder/actions/workflows/ci.yml/badge.svg)](https://github.com/ellyseum/ios_pointer_finder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Weights: CC-BY-4.0](https://img.shields.io/badge/Weights-CC--BY--4.0-lightgrey.svg)](LICENSE-WEIGHTS)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: v0.3.4](https://img.shields.io/badge/model-v0.3.4-green.svg)](https://github.com/ellyseum/ios_pointer_finder/releases/tag/v0.3.4)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

When you pair a Bluetooth mouse with an iPhone, iOS draws a small "@-symbol" cursor on the
screen ([Pointer Control](https://support.apple.com/guide/iphone/adjust-pointer-settings-iphec6e1e60b/ios)).
This repo trains and ships a tiny model that finds that cursor with sub-cursor-radius accuracy,
on any iPhone screen, in real time. It's the perception layer of a closed-loop iPhone agent
that drives an unmodified phone over BLE HID + AirPlay mirroring.

> A heatmap-regression CNN trained on synthetic composites. No iPhone images are bundled.
> Bring your own backgrounds, run `synthesize.py`, train your own variant — or just use the
> shipped weights directly.

---

## Quickstart

```bash
pip install ios-pointer-finder
```

```python
import cv2
from inference import PointerFinder

# 994x2160 BGR uint8 — your iPhone screen capture (or any size — auto-resized)
img = cv2.imread("snap.jpg")

finder = PointerFinder.from_pretrained("ellyseum/ios_pointer_finder")  # downloads weights from HF Hub
result = finder(img)

print(result.x, result.y, result.confidence, result.heatmap_peak)
# 656 1424 0.94 0.81
```

`PointerFinder.from_pretrained()` accepts a Hugging Face repo id, a local `.safetensors`
path, or a local `.pt` file. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full
inference contract and provenance.

---

## Why

Vision-LLM-based agents that "look at the screen" are too slow for real-time control loops
(2-10 seconds per turn). For the iPhone agent we wanted, we needed *one* perception
question — *where is the cursor, right now?* — answered in milliseconds, on commodity
hardware, with no API call. A specialized 338K-param CNN handles it in 10 ms on an RTX 5080
and ~30 ms on Apple Silicon CoreML.

The general-purpose vision LLM still answers "what's on this screen" when we need it. The
cursor finder answers "where's the cursor" continuously.

---

## Model card (v0.3.4)

| Field                | Value                                                |
|----------------------|------------------------------------------------------|
| Architecture         | 5-block conv backbone → 1×1 heatmap head + conf head |
| Parameters           | 338,274                                              |
| File size            | 1.3 MB (.safetensors fp32) / 340 KB (int8 quantized) |
| Native input         | 994 × 2160 (iPhone H264 stream)                      |
| Train input          | 497 × 1080 (2× downsample)                           |
| Heatmap stride       | 1/8 of train resolution (≈ 1/16 of native after the 2× input downsample) |
| Inference latency    | 10 ms (RTX 5080) / ~30 ms (M-series CoreML)          |
| Throughput           | 95 fps (single-image batch, fp32, RTX 5080)          |
| Val pos error        | 30.5 px on bg-level held-out split                   |
| FPR (cursor-free)    | <2% at conf ≥ 0.5 on held-out backgrounds            |
| Weights license      | CC-BY-4.0                                            |

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full evaluation breakdown
(per-sample-type metrics, failure modes, comparison to v0.2 / v0.3.x).

---

## How it works

```
iPhone screen capture (994×2160 BGR)
        │
        │  resize to 497×1080
        ▼
   conv backbone (3 stride-2 + 2 stride-1 blocks → 1/8 of train resolution)
        │
        ├── heatmap head (1×1 conv) → 63×135 sigmoid map → argmax + parabolic refine → (cx, cy) in native px
        └── conf head (global avg pool → MLP) → P(cursor present)
```

Trained on synthetic composites:
- **backgrounds** — real iPhone screen captures (cursor-free) — bring your own
- **cursor** — programmatically-generated soft-disc sprite (~46 px diameter, peak alpha 0.25, edge falloff)
  alpha-composited at random positions; the synthetic mix includes hard negatives (decoy shapes)
  and edge-clipped positives (cursor partially off-screen).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data + loss + training
schedule writeup.

---

## Eval

| Version | Val pos err | Cursor-free FPR | Real-frame top-1 hit | Inference (ms, RTX 5080) |
|--------:|------------:|----------------:|---------------------:|-------------------------:|
| v0.2    | 73.9 px     | 27%             | 41/50                | 10                       |
| v0.3.0  | 30.8 px*    | 8%              | 47/50                | 10                       |
| v0.3.2  | 43.4 px     | 5%              | 48/50                | 10                       |
| v0.3.3  | 35.4 px     | 3%              | 49/50                | 10                       |
| **v0.3.4** | **30.5 px** | **<2%** | **49/50** | **10** |

*v0.3.0's val number is leaky (sample-level split). v0.3.x and later use bg-level split — apples-to-apples comparable from v0.3.2 onward.

Run the eval harness yourself:

```bash
python eval_v03.py --v02 pointer_model_v0.2.0.pt --v03 pointer_model_v0.3.4.safetensors
```

---

## Train your own

```bash
git clone https://github.com/ellyseum/ios_pointer_finder.git
cd ios_pointer_finder
pip install -e ".[dev]"

# 1. Capture iPhone backgrounds (cursor OFF). See capture_backgrounds.py for the workflow.
mkdir -p backgrounds_kept
python capture_backgrounds.py  # interactive curation

# 2. Synthesize the training set (~150K samples).
python synthesize.py --out dataset --n 150000

# 3. Train.
python train.py --dataset dataset --epochs 30 --augment

# 4. Eval against v0.2 baseline.
python eval_v03.py
```

See [`docs/DATASET.md`](docs/DATASET.md) for background-collection guidance.

---

## Repo layout

```
ios_pointer_finder/
├── inference.py           # public PointerFinder API (load + predict)
├── train.py               # training loop, semver-tagged checkpoints
├── synthesize.py          # synthetic dataset generator
├── eval_v03.py            # version comparison eval
├── capture_backgrounds.py # iPhone screen capture pipeline
├── extract_sprite.py      # extract a real cursor sprite from a screen capture (optional)
├── grid_overlay.py        # debug overlay
├── keep_picker.py         # interactive background curation
├── cli.py                 # `ipf` console entry point
├── scripts/
│   ├── convert_pt_to_safetensors.py   # one-shot .pt → .safetensors converter
│   ├── export_onnx.py                 # ONNX export
│   └── export_coreml.py               # CoreML export (Mac)
├── tests/                 # pytest suite (shape, golden image, smoke)
├── docs/
│   ├── MODEL_CARD.md
│   ├── DATASET.md
│   └── ARCHITECTURE.md
├── .github/workflows/     # CI (ruff + pytest) + release automation
├── VERSION                # current model version (semver)
└── bump.sh                # patch/minor/major bump + tag
```

---

## Model hosting

Trained weights are distributed at:

- **Primary**: [Hugging Face Hub — `ellyseum/ios_pointer_finder`](https://huggingface.co/ellyseum/ios_pointer_finder)
  — versioned `.safetensors` + sidecar `config.json` + model card.
- **Mirror**: [GitHub Releases](https://github.com/ellyseum/ios_pointer_finder/releases) — each
  semver tag attaches the canonical `pointer_model_v{X}.{Y}.{Z}.safetensors` and a
  matching `pointer_model_v{X}.{Y}.{Z}.config.json`.

We don't use Git LFS — at 1.3 MB per checkpoint and ~10 production checkpoints, hosting on
HF/Releases is faster, cheaper, and more discoverable.

To convert any historical `.pt` checkpoint locally:

```bash
python scripts/convert_pt_to_safetensors.py pointer_model_v0.3.4_30.5px.pt
# → pointer_model_v0.3.4_30.5px.safetensors + pointer_model_v0.3.4_30.5px.config.json
```

---

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check .
ruff format .

# Tests
pytest -q

# Bump version + tag a new model
./bump.sh patch --commit
```

CI runs ruff + pytest on every push and PR. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full PR / release workflow.

---

## Versioning history

The shipped weights are calibrated on **iPhone 16 Pro Max (`iPhone17,2`),
iOS 26.3.1**, portrait orientation, 994×2160 AirPlay mirror native. Each
major version represents a deliberate stage in moving from "works on one
device" to "works on any current iPhone."

| Stage | Tag(s)        | Story |
|------:|---------------|-------|
| PoC   | `v0.1` / `v0.2` | Initial proof: synthetic-data CNN learns the cursor sprite at all. Bg-leaky validation, 73.9 px val pos error. |
| Refinements | `v0.3.0` | Heatmap regression head replaces xy-only regression. **Loss mask fix** so negatives no longer push down the heatmap globally. **Bg-level val split** so the validation number stops being leaky. **Hard negatives** (decoy cursor shapes) so the model rejects icon dots and notification badges. 30.8 px val (honest split) but starts overfitting. |
| Overfit fixes + conf | `v0.3.1` … `v0.3.4` | Train-time augmentation (random crop + horizontal flip + photometric jitter), late-backbone dropout, conf-head loss weight bumped, cosine restart from previous best. **30.5 px val pos error (current best)**, cursor-free FPR <2%. |
| **v0.4 (planned)** | — | Correctness fixes: float labels propagated through augmentation, parabolic subpixel refinement at inference, ONNX/CoreML/tfjs export parity check, fp32 → int8 quant ablation. Target: **<20 px val pos err**. |
| **v0.5 (planned)** | — | **Bootstrap loop / noisy-student self-labeling.** A trained agent moves the cursor across the screen, captures real frames, and emits verified labels (move → observe → reverse-move → re-observe). Retrain on real-cursor data instead of synthesis. Eliminates the synthetic-data ceiling. |
| **v0.6+ (planned)** | — | **Generalization across current iPhones.** Multi-device dataset (iPhone 15 series, iPhone 16 / Pro / Pro Max / Plus) collected via the v0.5 bootstrap loop on each device. One model that works on any current iPhone in portrait + landscape. |
| **v1.0** | — | Stable public API + cross-platform export (CoreML / ONNX / tfjs) + multi-device coverage. Frozen interface. |

## Roadmap milestones

- **v0.4** — correctness fixes + portable artifacts
  - Float labels through augmentation
  - Parabolic subpixel refinement at inference time
  - ONNX + CoreML + tfjs export with parity check
  - int8 quantization ablation (target: 340 KB checkpoint, <2 px accuracy delta)
  - Benchmark script (latency + throughput on RTX / M-series / CPU)

- **v0.5** — bootstrap loop / noisy-student self-labeling
  - On-device explorer agent: drives the cursor, records frame + commanded position
  - Move-and-undo dance for verified ground-truth labels (motion existence + reversibility, not commanded magnitude)
  - 3.5 s auto-hide window for cursor-free background capture
  - Retrain on real-cursor data; iterate until val saturates
  - Goal: surpass synthetic-data ceiling on real-frame eval

- **v0.6+** — generalization
  - Bootstrap on iPhone 15 / 15 Pro / 16 / 16 Pro / 16 Plus
  - Landscape orientation
  - Single multi-device model
  - Per-app UI element classifier as a second perception head

---

## Citation

```bibtex
@software{ios_pointer_finder_2026,
  author       = {Jocelyn Ellyse},
  title        = {ios\_pointer\_finder: a tiny CNN for iPhone Pointer-Control cursor detection},
  year         = {2026},
  url          = {https://github.com/ellyseum/ios_pointer_finder},
  version      = {0.3.4}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

- **Code**: MIT (see [`LICENSE`](LICENSE))
- **Trained weights**: CC-BY-4.0 (see [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS))

The cursor sprite used during training is **synthesized programmatically** in
`synthesize.py:make_pointer_mask` (a soft-disc Gaussian, calibrated against measurements
of the on-screen cursor). No iOS image assets ship in this repository.
