# Architecture

The full design rationale for the ios_pointer_finder model and training pipeline.

## Constraints we designed against

1. **Real-time inference.** A vision-LLM call costs 2-10 seconds per turn. A
   useful agent loop wants <50 ms per cursor query.
2. **Locked-down deployment target.** The downstream consumer is an iPhone
   automation agent that may run on a Mac (CoreML, NPU/GPU/CPU as available),
   a Linux box (CUDA), or a Windows tray app (CPU at minimum). The model has
   to be portable and small.
3. **Synthetic-only training data.** We can't realistically collect 100K
   real "cursor at known (x, y)" labels. Synthesis on real backgrounds gives
   us infinite labeled samples at the cost of a sim-to-real gap to manage.
4. **Imbalanced negatives.** The cursor is auto-hidden after 3 seconds idle.
   At runtime, most frames have no cursor at all. False positives on
   cursor-free frames break the closed-loop control logic.

## Architectural choices

### Heatmap regression, not bounding box

A 46-px cursor in a 994×2160 image is small and circular. A standard
detection head (anchors, NMS, classification + box regression) is overkill —
all of its capacity goes into rejecting non-cursor regions, but our
"non-cursor" is "everything else on the screen" (which we already screen out
with the confidence head).

Heatmap regression — a single channel that's hot where the cursor center is —
collapses the localization problem to argmax + subpixel refinement. Smaller
head, fewer hyperparameters, more accurate at sub-cursor-pixel resolution.

We output the heatmap at 1/8 of the train resolution (which is itself 2×
downsampled from native, so 1/16 native overall). The 46-px native cursor
becomes ~6 px in train space and ~3 px at the heatmap stride — large enough
for a peak to be sharp, small enough that the model learns localization
rather than a global "cursor exists" classifier.

### Confidence head independent from heatmap

The confidence head is a global average pool + 2-layer MLP that emits one
scalar (sigmoid). Trained as binary "is there a cursor visible?" against the
synthetic mix's `has_cursor` label.

It's deliberately not derived from `heatmap.max()`, because the heatmap can
be locally hot on a cursor-shaped UI element (a notification dot, a Settings
badge) when no real cursor is present. The conf head, looking at the whole
image globally, learns to discount these. In practice the two signals are
complementary; the closed-loop gate is `conf ≥ 0.5 AND heatmap_peak ≥ 0.5`.

### Tiny backbone

5 conv blocks (3 stride-2 + 2 stride-1), Conv-BN-ReLU, channel widths
[32, 64, 96, 128, 128]. No attention, no skip connections, no fancy pooling.
Total downsample is 1/8 of the train input (= 1/16 of native after the 2×
input resize). The receptive field at the final layer comfortably covers a
46-px cursor with surrounding context.

| Block | Stride | In ch | Out ch | Spatial output (1080×497 input) |
|------:|-------:|------:|-------:|---------------------------------|
| 1     | 2      | 3     | 32     | 540×249                         |
| 2     | 2      | 32    | 64     | 270×125                         |
| 3     | 2      | 64    | 96     | 135×63                          |
| 4     | 1      | 96    | 128    | 135×63                          |
| 5     | 1      | 128   | 128    | 135×63                          |

Heatmap head: 1×1 conv → 1 channel (raw logits, no sigmoid in `forward`).
Confidence head: AdaptiveAvgPool2d(1) → Linear(128, 32) → ReLU → Linear(32, 1).

`PointerNet.forward(x)` returns `(conf_logit, heatmap_logits)`.

## Loss

```
L_total = HM_WEIGHT · L_heatmap  +  CONF_WEIGHT · L_conf
```

- `L_heatmap`: per-cell BCE-with-logits between predicted heatmap and a
  Gaussian target centered on the cursor (σ = 1.25 cells ≈ FWHM matches
  cursor diameter at native resolution). Split into three weighted terms:
  - **positives**: weight 1.0
  - **hard negatives** (decoy cursor shapes): weight 1.0
  - **plain negatives** (untouched background): weight 0.25

  Plain-negs are trivial supervision (model learns to emit a flat heatmap
  on any unfamiliar bg within a few epochs); hard_neg supervision is what
  forces the model to discriminate cursor-from-distractor and gets the
  full weight. Without the split, ~half the negative gradient goes to
  samples the model already gets right.

- `L_conf`: BCE-with-logits on the confidence head against `has_cursor`.

The mask isolates negative supervision so the model isn't pushed to flatten
the heatmap globally (the v0.3 fix that flipped accuracy). Negatives still
contribute to the heatmap loss, but with target=0 — which is the correct
"no cursor anywhere" signal — and at the appropriate weight per neg type.

## Train-time augmentation

Each `__getitem__` (when `augment=True`):

1. **Cursor-safe random crop.** Up to 7% of each axis. The protected region
   is the asymmetric sprite footprint around the click anchor: the iOS
   pointer's alpha mass is biased toward the upper-left of its tile, so the
   guard uses `[label_x - hx, label_x + (d - hx)] × [label_y - hy, label_y + (d - hy)]`
   not a symmetric radius. Hard-negative samples carry the persisted decoy
   bbox; their crop respects the decoy footprint too. If 8 random trials
   fail to find a valid crop window, the sample skips augmentation
   (full-frame, no flip / no brightness jitter).
2. **Horizontal flip 50% — negatives only.** The real captured cursor sprite
   is left-right asymmetric (~31% alpha asymmetry), so flipping a positive
   shows the model a mirrored shape that doesn't exist at inference.
   Backgrounds (negs) flip freely.
3. **Photometric jitter.** Brightness ± 8% (synth pre-bakes ±15%, so this
   is a small additional shift, not the dominant variance source).

Per-worker seeding via `worker_init_fn` ensures DataLoader workers don't
share Python `random` / numpy RNG state — without this, all workers fork
with identical state and produce correlated augmentations across the batch.

## Inference

```python
img = cv2.imread("snap.jpg")                  # BGR uint8, native size
small = cv2.resize(img, (TRAIN_W, TRAIN_H), AREA)
x = ((small / 255.0) - 0.5) / 0.25            # standardize to ~[-2, 2]
conf_logit, heatmap = model(x)                # 2-tuple, raw logits
conf = sigmoid(conf_logit)                    # presence gate
logits = heatmap[0, 0]                        # H×W
flat = argmax(logits)
iy, ix = unravel_index(flat)
# Parabolic subpixel refinement on RAW LOGITS (sigmoid saturates near peak
# and collapses the second derivative the parabola fit needs).
sub_x = parabolic_offset(logits, ix, iy, axis="x")
sub_y = parabolic_offset(logits, ix, iy, axis="y")
rx, ry = ix + sub_x, iy + sub_y
# Stride-aware decode: cell i has receptive-field center at native pixel
# i*stride + (stride-1)/2, where stride = native_dim / hm_dim.
stride_x = NATIVE_W / W
stride_y = NATIVE_H / H
cx = round(rx * stride_x + (stride_x - 1.0) / 2.0)
cy = round(ry * stride_y + (stride_y - 1.0) / 2.0)
```

The single canonical decoder lives in `inference.PointerFinder.predict()`.
Auxiliary scripts (`click_at.py`, `eval_v03.py`, `test_real.py`) all import
from there to avoid drift.

The resize from native (994×2160) to train (497×1080) is a 2× downsample
chosen so the cursor stays large enough at the final feature-map stride to
form a clear peak. Earlier 4× experiments lost the cursor below the
sub-pixel boundary.

## Versioning

Every retrain bumps `VERSION`. Checkpoint files are saved locally as
`pointer_model_v{X}.{Y}.{Z}_{val_pos_err}px.{pt|safetensors}` so old runs
are never overwritten. Each commit message tags the version. Each git tag
attaches the canonical checkpoint to a GitHub Release.

For `.safetensors` checkpoints, metadata (epoch, val_pos_err_px,
native/train sizes, version) lives in a sibling `<stem>.config.json`
sidecar — the same convention `inference.PointerFinder` and
`scripts/convert_pt_to_safetensors.py` use.

## Road from here

The shipped checkpoints are calibrated on a single device — iPhone 16 Pro
Max, iOS 26.3.1, portrait. The model evolves from there in stages:

**Bootstrap loop (planned).** Eliminate the synthetic-data ceiling by
self-labeling. A trained-enough agent drives the cursor across the screen,
captures real frames, and emits verified labels via a move-and-undo dance
(motion existence + reversibility supplies ground truth — *not* the
commanded BLE-HID magnitude, which is iOS-tracking-speed-dependent).
Retrain on real-cursor data alongside synthesis. Iterate.

**Confidence head architecture (planned).** Global-average pooling over
the cursor's ~5×5 feature-map signal in an 8505-cell map dilutes the
discriminative signal into the background prior. Replace with max-pooling
or derive confidence from `heatmap_peak` directly.

**Generalization across iPhones (planned).** Run the bootstrap loop on
each currently-shipping iPhone (15 / 15 Pro / 16 / 16 Pro / 16 Plus /
16 Pro Max). Combine the collected datasets, train one model that handles
any current iPhone in portrait and landscape. Add a second perception
head for per-app UI elements (buttons, text fields, dialogs) so the agent
layer can address elements by name instead of pixel coordinates.
