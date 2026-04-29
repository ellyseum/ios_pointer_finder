# Changelog

All notable changes to ios_pointer_finder are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning per `bump.sh` policy (see `CONTRIBUTING.md`).

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
