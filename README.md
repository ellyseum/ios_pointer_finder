# ios_pointer_finder

A 338K-parameter CNN that finds the iPhone Pointer-Control cursor in a screen capture, in 10 ms.

[![CI](https://github.com/ellyseum/ios_pointer_finder/actions/workflows/ci.yml/badge.svg)](https://github.com/ellyseum/ios_pointer_finder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Weights: CC-BY-4.0](https://img.shields.io/badge/Weights-CC--BY--4.0-lightgrey.svg)](LICENSE-WEIGHTS)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: v0.7.0](https://img.shields.io/badge/model-v0.7.0-green.svg)](https://github.com/ellyseum/ios_pointer_finder/releases/tag/v0.7.0)
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

> **Status:** PyPI publish + Hugging Face Hub model repo are not live yet —
> both are gated on the v0.7 retrain finishing (see [Roadmap](#roadmap-milestones)).
> Today's install path is from a clone:

```bash
git clone https://github.com/ellyseum/ios_pointer_finder.git
cd ios_pointer_finder
pip install -e ".[hub,safetensors]"
```

Then either:

**A) Train your own weights** (see [Train your own](#train-your-own) below) — produces
`pointer_model_v{X}.{Y}.{Z}_{err}px.{pt|safetensors}`. Convert legacy `.pt` to
`.safetensors`:

```bash
python scripts/convert_pt_to_safetensors.py pointer_model_v0.7.0.pt
# → pointer_model_v0.7.0.safetensors  +  pointer_model_v0.7.0.config.json
```

`train.py` can also write `.safetensors` directly via `--weights-out *.safetensors`.

**B) Once the HF repo is up** (gated on the v0.7 retrain), use the one-liner the
package is designed for:

```python
import cv2
from inference import PointerFinder

img = cv2.imread("snap.jpg")  # any resolution — auto-resized to native (994×2160)
finder = PointerFinder.from_pretrained("ellyseum/ios_pointer_finder")
result = finder(img)
print(result.x, result.y, result.confidence, result.heatmap_peak)
# 656 1424 0.94 0.81
```

For now (path A), call `PointerFinder` with a local `.safetensors` or `.pt` path:

```python
from inference import PointerFinder
finder = PointerFinder.from_pretrained("./pointer_model_v0.7.0.safetensors")
```

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full inference contract and provenance.

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

## Model card (v0.7.0)

| Field                | Value                                                |
|----------------------|------------------------------------------------------|
| Architecture         | 5-block conv backbone → 1×1 heatmap head + conf head |
| Parameters           | 338,274                                              |
| File size            | 1.3 MB (.safetensors fp32)                           |
| Native input         | 994 × 2160 (iPhone H264 stream)                      |
| Train input          | 497 × 1080 (2× downsample)                           |
| Heatmap stride       | 1/8 of train resolution (≈ 1/16 of native after the 2× input downsample) |
| Inference latency    | 10 ms (RTX 5080) / ~30 ms (M-series CoreML)          |
| Throughput           | 95 fps (single-image batch, fp32, RTX 5080)          |
| Val pos error        | TBD — v0.7.0 cold-start retrain pending. v0.5 reached 18.9 px on bg-level honest split before the v0.6 fixes landed. |
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
        ├── heatmap head (1×1 conv) → 63×135 sigmoid map → argmax → (cx, cy) in native px
        └── conf head (global avg pool → MLP) → P(cursor present)
```

Trained on synthetic composites:
- **backgrounds** — real iPhone screen captures (cursor-free) — bring your own
- **cursor** — procedural smoothstep disc (peak alpha ≈ 0.25, luminance-matched
  to the local background), resized to ~46 native px and composited at random
  positions. v0.7 ships on this canonical synth target. A captured sprite at
  `sprites/at_dot.png` may be substituted, but only when accompanied by an
  approved sidecar manifest (sha256 + approver) — the loader fails hard
  otherwise. The synthetic mix includes hard negatives (decoy shapes designed
  to look cursor-like at a distance) and edge-clipped positives (cursor
  partially off-screen).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data + loss + training
schedule writeup.

---

## Eval

| Version | Val pos err | Cursor-free FPR | Real-frame top-1 hit | Inference (ms, RTX 5080) |
|--------:|------------:|----------------:|---------------------:|-------------------------:|
| v0.2    | 73.9 px     | 27%             | 41/50                | 10                       |
| v0.3.4  | 30.5 px     | 1.7%            | 49/50                | 10                       |
| v0.4.0  | 22.9 px     | <2%             | —                    | 10                       |
| v0.5.0  | 18.9 px     | <2%             | —                    | 10                       |
| **v0.7.0** | **TBD** | **TBD**         | **—**                | **10**                   |

All versions train on synthetic data (real backgrounds + composited cursor
sprite) with a bg-level honest val split — apples-to-apples comparable from
v0.3.x onward. v0.4.0 added correctness fixes (float labels through
augmentation, parabolic subpixel refinement at inference). v0.5.0/v0.6.x
shipped on a sprite asset that turned out to be a UI badge, not the cursor —
v0.7 reverts to the v0.4 procedural smoothstep disc as the canonical synth
target, adds a visual-validation gate that catches the failure pattern, and
re-enables the previously-silent real-frame regression eval. v0.7 also
unifies the decoder across all aux scripts (single canonical `decode.py`),
switches the heatmap BCE reduction from mean to sum with a calibrated
HM_WEIGHT (the prior mean form diluted the localization gradient ~1400×
relative to the confidence head), and swaps confidence-head pooling from
average to max (the avg-pool washed out the cursor signal at the head).
v0.7.0 number lands here once retrain completes.

Run the eval harness yourself:

```bash
python eval_v03.py --v02 pointer_model_v0.2.0.pt --v07 pointer_model_v0.7.0.safetensors
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
| Heatmap head + honest val | `v0.3.0` … `v0.3.4` | Heatmap regression replaces xy-only regression. **Loss mask fix** so negatives no longer push down the heatmap globally. **Bg-level val split** so validation stops leaking. **Hard negatives** (decoy cursor shapes) so the model rejects icon dots and badges. **Train-time augmentation** + cosine restart for the overfit-fix wave. v0.3.4: 30.5 px val, FPR <2%. |
| Correctness wave I | `v0.4.0` | Float labels through augmentation, parabolic subpixel refinement at inference, mask-aware heatmap eval. 22.9 px val. |
| Correctness wave II | `v0.5.0` | Real captured iOS pointer sprite (replaces procedural smoothstep disc — still synthetic compositing), alpha-mass-centroid labels (replaces geometric-tile-center labels), stride-aware coord mapping (replaces linear stretch), parabolic on raw logits (replaces sigmoid-domain), tighter Gaussian σ, plain/hard neg loss split. 18.9 px val. |
| **v0.6.0** (current) | `v0.6.0` | Forward signature simplified — drops the unused soft-argmax head — so `PointerNet.forward(x)` returns `(conf_logit, heatmap)`. Asymmetric cursor-safe crop matching the real-sprite hotspot. H-flip disabled on positives (real sprite is left-right asymmetric). Hard-negative crop guard. Single canonical decoder reused by all aux scripts and exporters. `.safetensors` round-trip with `<stem>.config.json` sidecar. **Breaking change** for the exported `.onnx` / `.mlpackage` schema. |
| **Bootstrap loop (planned)** | — | Use a trained-enough agent to drive the cursor and emit verified real labels (move → observe → reverse-move → re-observe). Retrain on synthesis + verified-real. Eliminates the synthetic-data ceiling without manual labels. |
| **Generalization (planned)** | — | Multi-device dataset (iPhone 15 series, 16 / Pro / Pro Max / Plus) collected via the bootstrap loop on each device. One model that works on any current iPhone in portrait + landscape. Per-app UI element classifier as a second perception head. |
| **v1.0** | — | Stable public API + cross-platform export (CoreML / ONNX) + multi-device coverage. Frozen interface. |

## Roadmap milestones

- **Bootstrap loop / noisy-student self-labeling** (next)
  - On-device explorer agent: drives the cursor, records frame + commanded position
  - Move-and-undo dance for verified ground-truth labels (motion existence + reversibility, NOT commanded magnitude — iOS Tracking Speed is per-device-tunable)
  - 3.5 s auto-hide window for cursor-free background capture
  - Retrain on synthesis + verified-real mix; iterate until val saturates
  - Goal: surpass the synthetic-only ceiling on real-frame eval

- **Generalization across iPhones**
  - Bootstrap on iPhone 15 / 15 Pro / 16 / 16 Pro / 16 Plus / 16 Pro Max
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
  version      = {0.6.0}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

- **Code**: MIT (see [`LICENSE`](LICENSE))
- **Trained weights**: CC-BY-4.0 (see [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS))

The cursor sprite used during training (`sprites/at_dot.png`, 36×36 BGRA) was
captured from a single high-resolution screenshot of the iOS Pointer-Control
cursor and alpha-matted by hand. It ships in this repository as a small
training artifact; iOS asset reproduction at this scale and form is fair-use
research/utility, not a redistribution of any Apple image set. Hard-negative
decoy sprites are still procedurally generated in `synthesize.py`.
