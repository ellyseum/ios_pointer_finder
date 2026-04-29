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

| Block | Stride | In ch | Out ch | Spatial output (497×1080 input) |
|------:|-------:|------:|-------:|---------------------------------|
| 1     | 2      | 3     | 32     | 540×248                         |
| 2     | 2      | 32    | 64     | 270×124                         |
| 3     | 2      | 64    | 96     | 135×62                          |
| 4     | 1      | 96    | 128    | 135×62                          |
| 5     | 1      | 128   | 128    | 135×62                          |

Heatmap head: 1×1 conv → 1 channel.
Confidence head: AdaptiveAvgPool2d(1) → Linear(128, 32) → ReLU → Linear(32, 1).

## Loss

```
L_total = L_heatmap + λ_xy · L_xy + λ_conf · L_conf
```

- `L_heatmap`: focal-modulated MSE between predicted heatmap and a Gaussian
  centered on the cursor (σ matches cursor radius). Computed only on
  positives. Mask zeros out heatmap loss for negatives so the model isn't
  punished for hot spots when no cursor exists.
- `L_xy`: MSE on (x_norm, y_norm) ∈ [0, 1]² for positive samples only. Kept
  for back-compat with v0.2 inference paths; deprecated in favor of heatmap
  argmax.
- `L_conf`: BCE on the global confidence head against `has_cursor`.

The mask is the v0.3 fix that flipped accuracy: in v0.2 the heatmap was
regressed against zeros for negatives, which trained the model to push
down activations everywhere, including in regions that legitimately have
cursor-like features. With the mask, the negative supervision is limited
to the conf head and the model is free to express "looks cursor-shaped"
in the heatmap as long as conf says no.

## Train-time augmentation

Each `__getitem__` (when augment=True):

1. Random crop ~7% of each axis before resize. Forces spatial invariance —
   the model can't memorize "exact pixel pattern at exact location".
2. Random horizontal flip (cursor sprite is rotationally symmetric; just mirror x).
3. Photometric jitter: brightness ± 0.1, optional small Gaussian noise.

This was the v0.3.1 anti-overfit fix. With only ~100 train backgrounds,
returning the same baked synthesis every epoch let the network memorize
JPEG noise + bg textures as cursor cues. Online augmentation broke that
shortcut: v0.3.0 → v0.3.1 went from clearly-overfitting (train loss near
zero, val plateau at 8 px) to a useful generalizer (val 30 px, real top-1
hit 47/50).

## Inference

```python
img = cv2.imread("snap.jpg")                    # BGR uint8, native size
small = cv2.resize(img, (TRAIN_W, TRAIN_H), AREA)
x = ((small / 255.0) - 0.5) / 0.25              # standardize to ~[-2, 2]
xy_unused, conf_logit, heatmap = model(x)
conf = sigmoid(conf_logit)
prob = sigmoid(heatmap)[0, 0]                    # 135×63 in feature space
iy, ix = unravel_index(argmax(prob))
cx_native = ix / (W - 1) * NATIVE_W              # W = heatmap width
cy_native = iy / (H - 1) * NATIVE_H              # H = heatmap height
```

The resize from native (994×2160) to train (497×1080) is a 2× downsample
chosen so the cursor stays large enough at the final feature-map stride to
form a clear peak. Earlier 4× experiments lost the cursor below the
sub-pixel boundary.

## Versioning

Every retrain bumps `VERSION`. We commit checkpoint files locally as
`pointer_model_v{X}.{Y}.{Z}_{val_pos_err}px.pt` so old runs are never
overwritten. Each commit message tags the version. Each git tag (`v0.3.4`)
attaches the canonical checkpoint to a GitHub Release.

## Road from here

- **v0.4** — float-label propagation through augmentation; parabolic subpixel
  refinement at inference time; honest comparison vs the published v0.3.4.
  Target: <20 px val pos err.
- **v0.5** — bootstrap loop: a trained agent runs the cursor across the
  screen and records "cursor was here at frame N" as verified labels. This
  lifts the synthetic-data ceiling because we get supervision from real
  iPhone behavior rather than synthesis assumptions.
- **v0.6+** — multi-cursor support, landscape, additional perception heads
  (UI element classifier).
