# Changelog

All notable changes to ios_pointer_finder are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per `bump.sh` policy (see `CONTRIBUTING.md`).

## [0.7.1] - Unreleased

### Changed
- **HM_WEIGHT 2e-3 → 8e-3** (calibrated by 3-point log sweep against the same fixed seed: 5e-4 / 2e-3 / 8e-3). Result: 184.7 / 36.1 / 20.8 px best P1 val_pos_err respectively. 8e-3 also lifts the heatmap peak distribution out of v0.7.0's marginal-confidence band — the two real-frame fixtures that previously failed the deployment gate at peak < 0.4 (bg-00007: peak 0.371 → 0.664; bg-00009: peak 0.337 → 0.671) now pass. bg-00009's prediction error also halved (27 px → 14 px) without any architectural change.
- **PEAK_THRESHOLD 0.5 → 0.4** in `click_at.py`. v0.7's BCE-sum loss bounds positive logits via real gradient pressure from the 8500+ background cells; true-positive peaks now sit ~0.91-0.96 typical (vs v0.4's saturated 1.000), and uncertain frames at ~0.34-0.37. The 0.5 gate was calibrated for the saturated regime and false-negatived all uncertain frames; 0.4 keeps the model's "I'm not confident here" signal as cursor-lost while passing the confident-localization band intact. Note: with HM_WEIGHT=8e-3 the peak distribution shifts higher again, so bg-00007 / bg-00009 now pass at 0.664 / 0.671 — the threshold drop is defensive headroom, not the load-bearing fix.

### Added
- **Per-epoch backbone-gradient diagnostic** (`train.py: _measure_backbone_grad_ratio`). Logs ‖∂L_h/∂backbone‖ for each head h after every epoch, plus the ratio. Reusable anchor for any future loss-reduction A/B; would have caught v0.7's HM_WEIGHT=2e-5 calibration miss in minutes instead of via a wasted 8 h cold-start. Diagnostic is non-invasive — sets BatchNorm modules to eval mode for the off-path forward so running stats don't drift.
- **bg-00009 hand-annotated GT** at `tests/fixtures/real/ground_truth.json` (was previously v0.4-bootstrapped). v0.4 was correct on this frame; v0.7.0 errs by 27 px with peak 0.337, v0.7.1 (8e-3) errs by 14 px with peak 0.671.
- **Soft-gate on bg-00009** (`tests/test_real_frame_eval.py: SOFT_GT_GATES`) at 60 px tolerance. Catches further regressions on this hand-annotated frame without forcing every future model to immediately solve every existing failure mode.

### Fixed
- `_measure_backbone_grad_ratio` no longer mutates BatchNorm running stats. Initial implementation called `model.train()` before the diagnostic forward, drifting eval statistics on every epoch from off-path data.
- `eval_v03.py` and `train.py:slice_peak_high` previously hardcoded `peak > 0.5` for false-positive accounting — now updated to track the deployment threshold (0.4) so val-time slice metrics match what would actually leak through `click_at.PEAK_THRESHOLD`.
- `docs/ARCHITECTURE.md` closed-loop gate description updated to match the new threshold (was `peak ≥ 0.5`).
- `tests/test_parabolic.py` border tests reframed to assert the v0.7-5 one-sided parabolic behavior (offset in `[-0.5, 0]` at left/top edges, `[0, 0.5]` at right/bottom) instead of the pre-v0.7 default-zero.
- `tests/test_inference.py` PointerPrediction-unpacks test updated for v0.7-10's 5-field `__iter__` (was 4-field).

## [0.7.0] - Unreleased

### Fixed (data integrity — see "What broke" below)

- **Synthesis pipeline targeted the wrong asset.** `sprites/at_dot.png` shipped under the description "captured iOS Pointer-Control sprite" but was actually a rounded-rectangle iOS UI badge with a baked-in checkmark. Five months of training (v0.5 → v0.6.2) descended cleanly on the wrong target — validation curves looked healthy because the supervision signal was *consistent*, just incorrect. Discovered when a contact sheet of the synth output was rendered for the first time. Removed the asset; `synthesize.py` now requires any sprite at `SPRITE_PATH` to ship with an approved sidecar manifest at `<stem>.config.json` (sha256, approved_by, approved_at) — absence of the sidecar fails hard. The procedural smoothstep disc is the canonical synth target; a captured sprite remains an option but is gated on visual review (see `CONTRIBUTING.md`).

- **Real-frame regression eval was a silent no-op.** `test_real.py` referenced a `real_pointer_test/` directory that did not exist in the repo, exiting 0 with zero frames loaded. Real-frame quality has been unmeasured for an unknown number of releases — only synthetic validation was actually being checked. Replaced by `tests/test_real_frame_eval.py`: bundles eight iPhone-screenshot fixtures into `tests/fixtures/real/` with a hand-annotated and bootstrapped `ground_truth.json`, asserts frame count == 8, and gates on bg-00000 prediction error ≤ 50 px. The "frame count == 8" assertion is specifically there to catch the original failure mode.

- **Decoder duplication regressed across deployment paths.** `click_at.py` carried its own `PointerFinder` class and `_parabolic_subpixel` function that diverged from `inference.PointerFinder` — different border behavior, different clamp, different stride math, slowly drifting from each other. Extracted the canonical decode path into `decode.py`: `argmax_parabolic_native(logits, native_w, native_h) -> (x, y, peak_logit)`. Every consumer now imports from there: `inference.py`, `click_at.py`, `test_real.py`, `test_real_bbox.py`, `eval_v03.py`. A pre-flight regression fixture pins the decoder output on three test frames so future refactors cannot silently change deployment behavior.

### Changed

- **Heatmap BCE reduction switched from mean to sum.** `.mean(dim=(2,3))` over 8505 cells made the per-positive-cell localization gradient ~1400× weaker than the confidence-head gradient at the shared backbone. The model learned "is something here" fast and "where exactly" slowly, exactly tracking the v0.5/v0.6 confidence-accuracy plateau near the dataset prior. New form: `.sum(dim=(2,3))` with `HM_WEIGHT=2e-3`. The first calibration attempt (`HM_WEIGHT=2e-5`) was chosen by per-head gradient-norm balance at random init; that heuristic failed in practice (P1 stalled at ~300 px val_pos_err vs v0.6.2's 36.7 px on the same architecture) because BCE-sum gradients at random init are dominated by the 8500+ near-zero-target cells, biasing the calibration toward an over-small weight that under-trains the heatmap path in steady state. The shipped value (`2e-3`) was chosen to match the same total-objective contribution as the pre-v0.7 mean-form regime that v0.4 trained under, while keeping the per-cell gradient no longer diluted by the 8505× cell-count denominator (which is the actual fix). Naive `HM_WEIGHT = 10 / 8505 ≈ 1.2e-3` is algebraically identity-equivalent to the old code; `2e-3` is meaningfully above that floor. A proper sweep is queued for v0.7.1.

- **Confidence-head plateau broken (still on AdaptiveAvgPool).** The historical ~68% val_conf_acc plateau (near the dataset positive prior across every prior version) lifted to 73-76% in v0.7 with the loss reshape alone, before any pooling change. The planned `AdaptiveAvgPool → AdaptiveMaxPool` swap (#104) is deferred to v0.7.1 since the plateau already broke; its independent contribution will be measured as a clean A/B with the queued HM_WEIGHT sweep.

- **Outer ~7-px border parabolic refinement.** When the heatmap argmax landed at cell 0 or W-1, the previous decoder returned offset 0 (no neighbor on one side), flooring `edge_pos` slice error at `(stride-1)/2 ≈ 7-8 px`. New one-sided parabolic fit recovers sub-cell precision up to half a cell beyond the heatmap edge.

- **Validation metric: `normal_pos_err` only for checkpoint selection.** Previously combined `normal_pos` and `edge_pos` errors. `edge_pos` ground truth is the visible-centroid of clipped sprites — noisier label, and the slice that doesn't drive deployment click accuracy. Both are still logged each epoch (`val_pos_err=...(normal_pos)` + `val_combined=...`) so the v0.6.x baseline can still be compared.

- **Stride convention: `STRIDE_TRAIN=16` (structural) + per-axis native stride (derived).** Three stride-2 conv blocks × 2x train→native give 16x effective downsample at train resolution; native stride is `NATIVE_DIM / HM_DIM` per axis (asymmetric: 994/63=15.778 across X, 2160/135=16.0 across Y). Code now distinguishes the two so future fixes target the right one.

### Added

- **Architecture-version pin in checkpoint sidecars.** `architecture_version=2` field in `<stem>.config.json`. Loaders assert match; pre-v0.7 checkpoints (no field) load with a `RuntimeWarning`, future incompatible-shape changes fail loudly with a "re-train or check out matching code revision" message.

- **Visual-validation gate (`tests/test_synth_visual.py`).** Renders a deterministic 16-cell contact sheet from `synthesize.py` and asserts per-cell SSIM ≥ 0.95 against `tests/golden/synth_contact_sheet.png` plus circularity ≥ 0.85 on the procedural sprite footprint. The circularity check is what would have caught the v0.5/v0.6 sprite mistake — a checkmark-pill footprint scores ~0.78, the smoothstep disc scores ~0.91. Future synth-pipeline changes that legitimately move the golden require regenerating it via the helper script and visually approving the new contact sheet in a separate commit.

- **Decoder baseline snapshot (`tests/baselines/v07_decoder_baseline.json`).** Records v0.4.0 22.9px and v0.6.2 (broken-sprite) predictions on the eight bundled real fixtures through the v0.7 canonical decoder. Reference for future regression checks: any decoder change that moves v0.4's bg-00000 result more than 5 px from the recorded 7.3 px requires written justification.

- `_run_v07.sh` — committed cold-start training recipe. Previously a working-tree-only scratch script that would have been lost on `rm -rf /tmp` or a reboot.

### What broke and how it was caught

Recording this for future contributors: between v0.5 and v0.6.2, every release shipped on a synth target that was not the iOS Pointer-Control cursor. The asset shipped with a name and code comments declaring it the cursor; ten rounds of code review missed it because reviews looked at code, not pixels; loss curves descended cleanly the whole way because BCE doesn't care whether the target *is* the right thing, only whether the model converges to *some* fixed thing. The mistake surfaced when a contact sheet of the synth output was rendered for human-readable explanation purposes — and a glance at the alpha-thresholded sprite immediately showed it was a rounded rectangle with a green checkmark, not a translucent disc. The v0.7 `test_synth_visual.py` gate, the `tests/fixtures/real/` regression suite, and the `synthesize.py` sidecar requirement are the structural fixes meant to keep this class of mistake from happening twice.

## [0.6.2] - Unreleased

### Fixed
- Warm-restart diversity restored. v0.6.1's global RNG seeding made every warm-restart pass reuse seed=42, producing the same DataLoader shuffle order and augmentation sequence pass-after-pass — defeating the diversity benefit of SGDR. `train.py` now mixes `IPF_PASS_ID` (set per pass by `train_continuous.sh`) into the effective seed via `(base_seed + pass_offset * 1009) % 2^31` so each pass sees a unique data ordering while remaining bit-exact reproducible.
- `--strict-determinism` now actually deterministic on CUDA. `CUBLAS_WORKSPACE_CONFIG` was being set AFTER `torch.cuda.manual_seed_all()`, but cuBLAS reads that env var only at handle creation — so the env-var was a no-op. Moved the env-var set before any `torch.cuda.*` call. Also dropped `warn_only=True` from `torch.use_deterministic_algorithms` so non-deterministic ops raise rather than silently warn.
- Hard-negative crop guard correctly protects asymmetric decoys. v0.6.1 changed `decoy_pos` from canvas-center to the alpha-mass centroid; the train-time guard's symmetric `± decoy_w/2` math then under-protected decoys whose mass is offset within the canvas (`doubled_dot`'s right dot scaled 0.6-1.0 → centroid LEFT of canvas; directional `wedge`). Reverted `decoy_pos` to canvas-center.
- `train.py` resume metadata peek now uses `weights_only=True`. Avoids re-executing the pickle (and its arbitrary-code surface) twice for the val_bg_ids inheritance path.
- `test_real.py` and `test_real_bbox.py` now accept `.safetensors` checkpoints (were `.pt`-only). Same suffix-detect pattern as v0.6.1's `click_at.py` / `eval_v03.py` fixes.

## [0.6.1] - Unreleased

### Added
- `--seed` and `--strict-determinism` flags on `train.py`. Global RNG seeding now happens at startup (Python `random`, numpy, torch CPU + CUDA). With `--strict-determinism`, also pins `cudnn.deterministic=True` and `torch.use_deterministic_algorithms(True)` for bit-exact repeats.
- `val_bg_ids` is persisted in the checkpoint (and config-json sidecar). On resume, the val membership is inherited from the checkpoint so val_pos_err stays measured against the same held-out bgs across resume boundaries.
- `safetensors` resume in `click_at.py` (was `.pt`-only).

### Changed
- bg-level train/val split now uses a stable hash-based assignment (`zlib.adler32(bg_id) % 100 < val_frac*100`). Adding or removing a background to the dataset no longer reshuffles which OTHER backgrounds land in val. Replaces the index-sensitive `random.Random(42).sample`.
- `_worker_init_fn` no longer double-offsets by `worker_id` — `torch.initial_seed()` already encodes it via DataLoader's per-worker reseed.
- `gen_edge_pos` now rejects samples that landed fully on-frame instead of tagging them `edge_pos`. Cleaner slice metric, and removes a quiet "tagged edge but actually unclipped" path that lost crop augmentation in `train.py`.
- `gen_hard_neg` margin now scales with the largest decoy dimension (e.g., wide ellipses up to 119 px wide). Previously, default `margin=50` allowed wide decoys to compose-clip at the frame edge and contradict the `edge_pos` slice's "clipped cursor = positive" supervision.
- `gen_hard_neg` `decoy_pos` is now the alpha-mass centroid of the decoy, not the canvas center. Fixes mis-protection of asymmetric decoys (`doubled_dot` especially — canvas center was the gap between two dots).
- `test_real_bbox.py:heatmap_to_bbox` reports the canonical decoder's center (argmax + parabolic on logits + stride-aware) instead of the upsampled-grid argmax. Bbox CC analysis unchanged.
- Updated module / class docstrings to reflect the v0.6 forward signature (2-tuple) and removal of the soft-argmax regression head.

### Fixed
- `eval_v03.py` crashed on `.safetensors` checkpoints with no sidecar metadata (`f"{None:.1f}"` raised `TypeError`). Now prints `n/a` for missing fields.
- ONNX export docstring now correctly states the heatmap stride (1/8 of train, 1/16 of native) — was incorrectly listed as 1/16 of train.
- `tests/test_model.py:test_forward_shape_eval` tightens the conf shape assertion to `(B,)` only — the legacy `(B, 1)` tolerance accepted a shape the current forward doesn't produce.

## [0.5.1] - Unreleased

### Changed (BREAKING)
- `PointerNet.forward` now returns `(conf_logit, heatmap)` instead of `(xy, conf_logit, heatmap)`. The soft-argmax `xy` head was unused (XY_WEIGHT=0 since v0.5) and used a different coordinate convention than the deployed inference path. Decoders should use `inference.PointerFinder.predict()` (or argmax + parabolic on raw logits) to recover `(x, y)`.
- Exported `.onnx` / `.mlpackage` artifacts no longer carry an `xy` output. Downstream consumers binding by name must update.

### Added
- Stride-aware coordinate mapping in target generation, training validation, and inference (replaces linear-stretch).
- Parabolic subpixel refinement on raw heatmap logits (not sigmoid output).
- Real iOS pointer sprite (`sprites/at_dot.png`) with alpha-centroid labeling; replaces procedural smoothstep disc.
- Asymmetric crop protection in train-time augmentation (true sprite bbox around hotspot).
- Cursor-safe crop also protects hard-negative decoy footprints via persisted `decoy_w`/`decoy_h`.
- `.safetensors` resume + save with `<stem>.config.json` sidecar metadata.
- `--limit-val N` flag for fast regression smoke (shuffle-then-slice).
- DataLoader `worker_init_fn` to break Python `random` / numpy RNG state correlation across workers.
- `set -o pipefail` in `train_continuous.sh` so `train.py` crashes don't get swallowed by `tee`.

### Fixed
- `gen_edge_pos` visible_frac is alpha-mass-based, not bounding-box area.
- `gen_hard_neg` color patch off-by-one on odd-height decoys.
- `capture_backgrounds.py` resume glob (was `bg-*.jpg`, now `bg-*.png` matching write path).
- Legacy negs (`sample_type` missing) now route correctly to neg-FPR in val metrics.
- H-flip disabled for positives (real sprite is left-right asymmetric).

### Removed
- `val_soft_err_px` from training logs and saved checkpoints.
- `XY_WEIGHT_*` constants and the soft-argmax head from `PointerNet.forward`.
- Soft-argmax marker in `test_real.py` visualizations.

## [Unreleased]

### Added
- `inference.py` — public `PointerFinder` class for clean library usage.
- `cli.py` — `ipf` console entry point (`predict`, `version`).
- `scripts/convert_pt_to_safetensors.py` — one-shot .pt → .safetensors + sidecar config.json conversion.
- `scripts/export_onnx.py` — ONNX export with optional parity check.
- `scripts/export_coreml.py` — Apple CoreML export.
- `LICENSE` (MIT) for the code.
- `LICENSE-WEIGHTS` (CC-BY-4.0) for the trained weights.
- `README.md` with badges, model card snippet, quickstart, and eval table.
- `docs/MODEL_CARD.md`, `docs/DATASET.md`, `docs/ARCHITECTURE.md`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.
- `tests/` — pytest suite with model-shape, inference, and synthesize smoke tests.
- `.github/workflows/ci.yml` — ruff + pytest on push and PR.
- `.github/workflows/release.yml` — auto-upload .safetensors + config.json on tag push.
- `pyproject.toml` — installable package, ruff/pytest config.

## [0.3.4] - 2026-04-27

### Added
- v0.3.4 checkpoint (full 30-epoch cosine restart from v0.3.3).

### Changed
- Best validation positional error: 30.5 px on bg-level honest split (down from v0.3.3's 35.4 px).

## [0.3.3] - 2026-04-27

### Added
- v0.3.3 cosine restart from v0.3.2's 43.4 px best (15 more epochs).

### Changed
- Validation positional error: 35.4 px.

## [0.3.2] - 2026-04-27

### Added
- v0.3.2 fast-augmentation training run.

### Changed
- Replaced slow noise + alpha kernels with `cv2.convertScaleAbs` for ~3× faster augmentation.
- Validation positional error: 43.4 px.

## [0.3.1] - 2026-04-27

### Added
- Train-time augmentation: random 7% crop + horizontal flip + photometric jitter, applied online in `__getitem__`.
- Anti-overfitting fixes: deeper data shuffle, warmup schedule, dropout after the last two conv blocks.

## [0.3.0] - 2026-04-27

### Added
- Heatmap regression head (replaces the v0.2 xy regression as primary localization signal).
- Loss-mask fix — heatmap loss now masked to positives only.
- Bg-level train/val split (no leakage of backgrounds across splits).
- `sigma_px` parameter for heatmap target Gaussian.
- Hard-negatives in synthesis (decoy cursor shapes — wrong-size discs, rings, ellipses, etc.).
- Edge-positive samples (cursor partially clipped at screen edges, visible-centroid label).
- `bg_id` and `sample_type` fields in `dataset/labels.jsonl` for sliced metrics.
- `eval_v03.py` — comparison harness for model versions on real frames + cursor-free frames.

## [0.2.0] - 2026-04-26

### Added
- Initial public release: training pipeline, synthesis pipeline, basic PointerNet architecture.
- `train.py`, `synthesize.py`, `click_at.py` (closed-loop controller).
- `bump.sh` and `VERSION` for semver discipline.
