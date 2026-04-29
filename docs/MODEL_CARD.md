# Model Card — ios_pointer_finder v0.3.4

A 338K-parameter convolutional network that predicts the on-screen position of
the iPhone Pointer-Control cursor from a single screen-capture image.

## TL;DR

| Field                        | Value                                            |
|------------------------------|--------------------------------------------------|
| Architecture                 | 5-block conv backbone (3 stride-2 + 2 stride-1) + 1×1 heatmap head + conf head |
| Parameters                   | 338,274                                          |
| Checkpoint size              | 1.3 MB (fp32 .safetensors)                       |
| Input                        | BGR uint8, any resolution (auto-resized)         |
| Output                       | (x, y) in 994×2160 native px, confidence, heatmap |
| Train resolution             | 497 × 1080                                       |
| Inference latency (RTX 5080) | 10 ms                                            |
| Throughput                   | 95 fps                                           |
| Validation pos-error         | 30.5 px (bg-level held-out split, 12 unseen bgs) |
| Cursor-free FPR              | <2% at conf ≥ 0.5                                |
| Calibration device           | iPhone 16 Pro Max (`iPhone17,2`), iOS 26.3.1     |
| Native capture resolution    | 994 × 2160 (iPhone 16 Pro Max AirPlay H264 stream) |
| Weights license              | CC-BY-4.0                                        |

## Intended use

Drop-in cursor localizer for iPhone-driving agents. The downstream loop reads
"where is the cursor now?" at video frame rate and issues BLE HID move
commands to bring the cursor to a target. The model is *only* trained on the
cursor sprite that appears when an external pointing device is paired with
iOS — it doesn't generalize to macOS pointers, web cursors, or other
overlays.

## Calibration / device assumptions

The published checkpoints were trained and evaluated against a single
device: **iPhone 16 Pro Max (`iPhone17,2`)** running **iOS 26.3.1**, in
**portrait** orientation. The native capture resolution coming off AirPlay
mirroring on that device is 994×2160, which is what `NATIVE_W` / `NATIVE_H`
in `train.py` are pinned to and what the heatmap argmax scales back to.

Other current iPhones (16 / 16 Pro / 16 Plus, 15 series) likely produce
useful predictions out of the box at acceptable accuracy because the
cursor sprite size, alpha, and shape are iOS-system-wide and do not
change per device. Smaller / older iPhones (iPhone 13 mini etc.) and
iPad will likely need a retrain — different stream resolutions, different
status-bar heights, different home-screen densities.

The v0.5+ bootstrap loop (see Roadmap) is designed to absorb this by
self-labeling new device captures rather than synthesizing more.

## Out-of-scope

- Detecting the cursor *trail* during fast moves (we predict instantaneous
  position only).
- Multi-cursor scenes.
- iOS macros/Switch Control overlays.
- Landscape orientation (training distribution is portrait 994×2160).
- Devices other than iPhone 16 Pro Max (until v0.6+ generalization).

## Architecture

```
Input  (3, 1080, 497)
   ↓ 3 × Conv-BN-ReLU stride-2 (3→32→64→96)
   ↓ 2 × Conv-BN-ReLU stride-1 (96→128→128)
Feature map (128, 135, 63)     # 1/8 of train resolution = 1/16 of native
   ↓
   ├── 1×1 conv → 1 channel    →  heatmap (sigmoid)
   └── global avg pool + MLP   →  confidence logit (1)
```

The heatmap is the primary localization signal. Argmax + parabolic subpixel
refinement gives the cursor center in the feature-space coordinates; we
scale back to native (994 × 2160) for output via normalized coordinates
(no stride math required at inference time). The confidence head is trained
as a binary classifier on `has_cursor` and used as a presence gate.

The original architecture also had a regression head (xy MLP). It survives in
the forward pass for backward compatibility with old eval harnesses, but the
heatmap head dominates accuracy and is what `inference.py` uses.

## Training data

**Synthetic.** No real iPhone photos are bundled.

| Sample type   | Default mix | Description |
|---------------|------------:|-------------|
| `normal_pos`  | 55%         | Full cursor at random position with margin |
| `edge_pos`    | 15%         | Cursor partially clipped at a screen edge; visible-centroid label |
| `hard_neg`    | 15%         | "Decoy cursor" composites: wrong-size discs, hollow rings, ellipses, doubled dots, I-beam strokes, white wedges. `has_cursor=0`. |
| `plain_neg`   | 15%         | Unmodified background. `has_cursor=0`. |

The cursor sprite itself is *programmatically generated* in
`synthesize.py:make_pointer_mask` as a soft-edge disc (Gaussian falloff,
diameter ~46 px, peak alpha 0.25, calibrated against measurements of the
real on-screen cursor). No iOS image assets are used.

Backgrounds for our published checkpoints were 130 cursor-free real iPhone
screen captures from the trainer's own device, not redistributed. To
reproduce locally, see [`DATASET.md`](DATASET.md) — bring your own
backgrounds, run `synthesize.py`, train.

## Training procedure

| Hyperparameter | Value             |
|----------------|-------------------|
| Optimizer      | AdamW, lr=3e-4, weight_decay=1e-4 |
| Schedule       | Cosine annealing (30 epochs total per training run) |
| Batch          | 64                |
| Epochs         | 30 (v0.3.4 = cosine restart from v0.3.3's best) |
| Dataset size   | ~150,000 synthetic samples per generation |
| Augmentation   | Random 7% crop + horizontal flip + photometric jitter (online, train only) |
| Val split      | bg-level (10% of unique backgrounds held out — no leakage) |
| Loss           | MSE on (x, y) for positives + BCE on conf for all + heatmap focal loss |
| Hardware       | Single RTX 5080, 16 GB VRAM |
| Wall-clock     | ~25 min for 30 epochs on the v0.3.4 dataset |

Seed: 42 throughout (deterministic train/val split, augmentation rng).

## Evaluation

### v0.3.4 (current best)

| Slice                                  | Value     |
|----------------------------------------|----------:|
| Validation positional error (mean)     | 30.5 px   |
| Validation positional error (median)   | 18.4 px   |
| Validation positional error (95th pct) | 102 px    |
| Cursor-free FPR @ conf ≥ 0.5           | 1.7%      |
| Real-frame top-1 hit (50-frame manual set) | 49/50 |

### Cross-version comparison

| Version | val_pos_err | cursor-free FPR | real top-1 | Notes |
|--------:|------------:|----------------:|-----------:|-------|
| v0.2    | 73.9 px     | 27%             | 41/50      | Initial release |
| v0.3.0  | 30.8 px*    | 8%              | 47/50      | Heatmap head + bg-level split (val number leaky on this run) |
| v0.3.1  | 39.6 px     | 6%              | 48/50      | + train-time augmentation |
| v0.3.2  | 43.4 px     | 5%              | 48/50      | Faster augmentation kernel |
| v0.3.3  | 35.4 px     | 3%              | 49/50      | Cosine restart |
| **v0.3.4** | **30.5 px** | **1.7%**     | **49/50**  | Cosine restart from v0.3.3, full 30 epochs |

*v0.3.0 used a sample-level val split that leaks backgrounds across train/val — the published number was over-optimistic. v0.3.x and later use bg-level splits and are honest.

## Known failure modes

- Mid-trajectory motion blur: the model was trained on stationary cursors. Predictions during fast moves are stable but trail by a frame or two.
- Cluttered icon grids: Photos / App Store grids occasionally produce a flat heatmap without a clear peak; gating on heatmap_peak ≥ 0.4 catches these.
- Cursor at extreme corner (within 30 px of edge): subpixel refinement is unstable; rely on argmax integer coords.

## Provenance

- Trained on a single workstation with an NVIDIA RTX 5080 (16 GB VRAM), Ubuntu under WSL2 on Windows 11.
- Repository: <https://github.com/ellyseum/ios_pointer_finder>
- Tag: `v0.3.4`
- Trainer: Jocelyn Ellyse <jocelyn@ellyseum.dev>
- Date: 2026-04-27

## License

- Code: MIT — see [`LICENSE`](../LICENSE)
- Weights: CC-BY-4.0 — see [`LICENSE-WEIGHTS`](../LICENSE-WEIGHTS)

## Issues / contact

Issues, questions, and PR-able improvements all welcome at
<https://github.com/ellyseum/ios_pointer_finder/issues>.
