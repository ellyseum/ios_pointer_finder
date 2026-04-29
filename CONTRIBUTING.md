# Contributing to ios_pointer_finder

Thanks for considering a contribution. The goal of this repo is a *small,
sharp, reproducible* cursor detector — every PR should leave the model
either more accurate, more portable, or more obviously honest.

## Setup

```bash
git clone https://github.com/ellyseum/ios_pointer_finder.git
cd ios_pointer_finder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Sanity check
ruff check .
pytest -q
```

## What we welcome

- **Bug fixes** — anything where current behavior contradicts the README or model card.
- **Cross-platform support** — CoreML/ONNX/TF.js exports that produce identical predictions to the PyTorch reference.
- **New eval slices** — failure-mode analyses that surface real-world weaknesses (cluttered backgrounds, dark mode, edge clipping, etc.).
- **Synthetic data improvements** — better hard negatives, more realistic alpha falloff, edge-case scenarios.
- **Performance** — tighter inference paths (quantization, pruning) that don't sacrifice accuracy.
- **Documentation** — anything that helps a stranger get from `git clone` to "I trained my own variant" in under an hour.

## What we'd push back on

- New architectures that materially grow the parameter count without a clear accuracy or robustness win. The point of this model is small and fast.
- Vendored binaries / large data files. Weights belong on Hugging Face Hub or GitHub Releases, not git.
- Anti-patterns: `print()` debug spam in checked-in code, unused imports, `from x import *`.
- iOS image assets (Apple's UI sprites, screenshots from iOS PR materials) — keep the repo Apple-IP-clean.

## PR workflow

1. Fork + branch. Branch names: `fix/...`, `feat/...`, `docs/...`, `model/v0.X.Y`.
2. Run `ruff check .` and `ruff format .` before pushing.
3. Run `pytest -q` and add tests for new code paths. The bar is "shape + golden image + smoke" — see `tests/` for examples.
4. Update `CHANGELOG.md` under `## [Unreleased]`.
5. Open a PR against `main`. Describe what changed and why; link any related issues.
6. CI runs ruff + pytest on every push. Green CI is required before merge.

## Releasing a new model version

1. Train and validate. The checkpoint should land at `pointer_model_v{X}.{Y}.{Z}_{val_err}px.pt`.
2. `python scripts/convert_pt_to_safetensors.py pointer_model_v{X}.{Y}.{Z}_*.pt` to emit the public .safetensors + sidecar config.
3. Update `docs/MODEL_CARD.md` with the new metrics row.
4. Update `README.md` model card snippet and badges.
5. Bump `VERSION` via `./bump.sh patch --commit` (or `minor`/`major` as appropriate).
6. `git tag v{X}.{Y}.{Z}` and `git push --tags`. The release workflow attaches the .safetensors + config.json to the GitHub Release.
7. Push the same .safetensors to the Hugging Face Hub model repo (`huggingface-cli upload ellyseum/ios_pointer_finder ...`).

## Versioning policy

We follow a slightly tighter semver:

- **patch** — retrain with same/similar code, hyperparameter tweak, identical interface
- **minor** — substantive algorithm or data change (new loss term, new augmentation, new training stage)
- **major** — breaking interface change (e.g. coord-space change, new input format)

`bump.sh` enforces this — see `bump.sh` for the full policy.

## Code style

- Python 3.10+. Type hints on public functions.
- Imports: stdlib, third-party, local — separated, alphabetized within groups.
- Docstrings on public functions and classes; one-liners for private helpers if non-obvious.
- No `print()` in library code; use the `logging` module if you need diagnostic output.
- Tests use pytest. Mark slow tests `@pytest.mark.slow`.
- Commits: imperative mood ("Add foo", not "Added foo"). One concept per commit.

## Reporting issues

Use the GitHub Issues bug-report template. The most useful issues include:

- A concrete failure case (image + expected vs actual prediction).
- The model version (`ios-pointer-finder version`) and runtime (`python -V`, `pip show torch`).
- Whether you can reproduce on the published checkpoint or only your own trained one.

## Code of Conduct

By participating you agree to abide by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
