# Model Card — ios_pointer_finder v0.7.0

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
| Validation pos-error         | TBD (cold-start retrain pending — see Roadmap). v0.5 reached 18.9 px on the same dataset before the v0.6.0 fixes landed. |
| Cursor-free FPR              | <2% at conf ≥ 0.5 (v0.5 measurement; re-measured after v0.6.0 retrain) |
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

The v0.7+ bootstrap loop (see Roadmap) is designed to absorb this by
self-labeling new device captures rather than synthesizing more.

## Out-of-scope

- Detecting the cursor *trail* during fast moves (we predict instantaneous
  position only).
- Multi-cursor scenes.
- iOS macros/Switch Control overlays.
- Landscape orientation (training distribution is portrait 994×2160).
- Devices other than iPhone 16 Pro Max (until v0.7+ generalization).

## Architecture

```
Input  (3, 1080, 497)
   ↓ 3 × Conv-BN-ReLU stride-2 (3→32→64→96)
   ↓ 2 × Conv-BN-ReLU stride-1 (96→128→128)
Feature map (128, 135, 63)     # 1/8 of train resolution = 1/16 of native
   ↓
   ├── 1×1 conv → 1 channel    →  heatmap logits
   └── global avg pool + MLP   →  confidence logit (1)
```

The heatmap is the primary localization signal. Inference applies argmax +
parabolic-subpixel refinement on **raw logits** (not sigmoid output — the
sigmoid saturates near the peak and collapses the second derivative the
parabola fit relies on), then maps the cell index to native pixels via the
**stride convention** (cell `i` has receptive-field center at native pixel
`i*stride + (stride-1)/2`). This replaces the v0.4 linear-stretch mapping
that quietly forced the model to learn a non-uniform spatial warp.

The confidence head is a separate global-average-pool branch trained as a
binary classifier on `has_cursor`.

**v0.6.0 forward signature** (BREAKING from v0.5): `PointerNet.forward(x)`
returns `(conf_logit, heatmap_logits)`. The earlier soft-argmax `xy` head
was removed — it had been weighted at 0 since v0.5 and used a different
coordinate convention than the deployed inference path. Decoders should
import `inference.PointerFinder` (or use argmax + parabolic on the raw
heatmap logits) to recover `(x, y)`.

## Training data

**Synthetic.** No real iPhone photos are bundled.

| Sample type   | Default mix | Description |
|---------------|------------:|-------------|
| `normal_pos`  | 55%         | Full cursor at random position with margin |
| `edge_pos`    | 15%         | Cursor partially clipped at a screen edge; alpha-mass-centroid label |
| `hard_neg`    | 15%         | "Decoy cursor" composites: wrong-size discs, hollow rings, ellipses, doubled dots, I-beam strokes, white wedges. `has_cursor=0`. |
| `plain_neg`   | 15%         | Unmodified background. `has_cursor=0`. |

The cursor sprite is a **procedural smoothstep disc** (peak alpha 0.25,
luminance-matched to the local background patch via
`synthesize.pick_cursor_color`). v0.7 ships on this canonical synth
target — the same target used in v0.3 and v0.4. A captured iOS pointer
sprite may be substituted via `sprites/at_dot.png`, but only when paired
with an approved sidecar manifest (`<stem>.config.json` with sha256,
approved_by, approved_at); the loader fails hard otherwise. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md#asset-integrity-gate) for the
review flow.

Backgrounds for our published checkpoints were ~120 cursor-free real
iPhone screen captures from the trainer's own device, not redistributed.
To reproduce locally, see [`DATASET.md`](DATASET.md) — bring your own
backgrounds, run `synthesize.py`, train.

## Training procedure

| Hyperparameter | Value             |
|----------------|-------------------|
| Optimizer      | AdamW, lr=1e-3, weight_decay=1e-3 |
| Schedule       | Cosine annealing per pass; warm restarts via `train_continuous.sh` (SGDR-style) |
| Batch          | 64                |
| Epochs/pass    | 15 (configurable; SGDR T_max = 15 by default) |
| Auto-stop      | 3 consecutive stale passes (no global-best improvement) |
| Dataset size   | ~125,000 synthetic samples per generation |
| Augmentation   | Cursor-safe random crop (asymmetric protection around hotspot) + photometric jitter (online, train only). H-flip enabled on negatives only — the real sprite is left-right asymmetric. |
| Val split      | bg-level (10% of unique backgrounds held out — no leakage) |
| Loss           | BCE on heatmap logits (split into pos / hard_neg / plain_neg with weights 1.0 / 1.0 / 0.25) + BCE on confidence |
| Heatmap target | Gaussian, σ=1.25 cells (FWHM ≈ cursor diameter at native resolution) |
| Hardware       | Single RTX 5080, 16 GB VRAM |
| Wall-clock     | ~7 min/epoch, ~100 min per 15-epoch pass |

Seed: 42 throughout (deterministic train/val split + per-worker RNG seeding
via `worker_init_fn` so DataLoader workers don't share `random` state).

## Evaluation

### v0.6.0 — pending

The first v0.6.0 cold-start training run is queued. The expected baseline is
the v0.5 best (18.9 px) plus whatever lift comes from the v0.6.0 fixes
(canonical decoder unification across `inference.py`/`click_at.py`/eval
scripts; H-flip disabled on positives; asymmetric crop respecting the real
sprite hotspot; hard-negative crop guard; coordinate-system + parabolic-
domain corrections). Updated numbers will land here when the run completes.

### Cross-version comparison (synthetic val)

| Version | val_pos_err | cursor-free FPR | real top-1 | Notes |
|--------:|------------:|----------------:|-----------:|-------|
| v0.2    | 73.9 px     | 27%             | 41/50      | Initial release. Sample-level val split (leaky). Procedural sprite. |
| v0.3.4  | 30.5 px     | 1.7%            | 49/50      | Heatmap head + bg-level honest split + hard negatives. Procedural sprite, linear-stretch coords, soft-argmax MSE. |
| v0.4.0  | 22.9 px     | <2%             | —          | Correctness wave: float labels through augmentation, parabolic subpixel, mask-aware heatmap eval. Still procedural sprite. |
| v0.5.0  | 18.9 px     | <2%             | —          | Real captured sprite, alpha-centroid labels, stride coord convention, parabolic on logits, sigma 1.25, plain/hard neg loss split. |
| **v0.6.0** | **TBD**  | **TBD**         | **TBD**    | v0.5 plus: forward signature simplified (conf, heatmap); H-flip off for positives; asymmetric crop; hard-neg footprint protection; canonical decoder used by all aux scripts. |

The v0.4 → v0.5 jump (22.9 → 18.9) came from a coherent set of label-correctness
and decoder-domain fixes: real captured sprite + alpha-centroid labels (replacing
a procedural disc + geometric-center labels), stride-aware coordinate mapping
(replacing a linear stretch that forced a non-uniform spatial warp), parabolic
subpixel refinement on raw logits (replacing sigmoid-domain refinement that
saturated near peaks), and tighter Gaussian-target sigma. v0.5 → v0.6 followed
up with an architectural simplification (dropping the unused soft-argmax head)
and more augmentation correctness work (asymmetric cursor-safe crop matching
the real-sprite hotspot, H-flip disabled on positives since the real sprite is
left-right asymmetric, hard-negative crop guard, decoder-path consolidation).
Empirical v0.6 numbers will land here once the cold-start retrain completes.

## Known failure modes

- Mid-trajectory motion blur: the model was trained on stationary cursors. Predictions during fast moves are stable but trail by a frame or two.
- Cluttered icon grids: Photos / App Store grids occasionally produce a flat heatmap without a clear peak; gating on `heatmap_peak ≥ 0.4` catches these.
- Cursor at extreme corner: the stride-aware coord mapping covers the central ~native−7 px of each axis. Cursors in the outer ~7-px border can't be predicted exactly there. v0.7+ will add asymmetric border-parabolic refinement.

## Provenance

- Trained on a single workstation with an NVIDIA RTX 5080 (16 GB VRAM), Ubuntu under WSL2 on Windows 11.
- Repository: <https://github.com/ellyseum/ios_pointer_finder>
- Tag: `v0.6.0`
- Trainer: Jocelyn Ellyse <jocelyn@ellyseum.dev>
- Date: 2026-04-30

## License

- Code: MIT — see [`LICENSE`](../LICENSE)
- Weights: CC-BY-4.0 — see [`LICENSE-WEIGHTS`](../LICENSE-WEIGHTS)

## Issues / contact

Issues, questions, and PR-able improvements all welcome at
<https://github.com/ellyseum/ios_pointer_finder/issues>.
