# Dataset Guide

`ios_pointer_finder` trains on **synthetic data**: real iPhone backgrounds + a
programmatically-generated cursor sprite alpha-composited at random
positions. No real cursor photos are used; the published checkpoints'
backgrounds were the trainer's own iPhone screens and are not redistributed.

This guide tells you how to collect your own backgrounds and synthesize a
training set locally.

## What you need

- An unmodified iPhone running iOS 13+ (Pointer Control supported). The
  published checkpoints were calibrated on **iPhone 16 Pro Max
  (`iPhone17,2`), iOS 26.3.1**; if you train on a different device the
  values of `NATIVE_W` / `NATIVE_H` in `train.py` may need to change to
  match your device's mirror resolution.
- A way to capture screen frames from the iPhone (any pixel-faithful path
  works — AirPlay receiver, USB tethered capture, screen recording, etc.).
- 100+ unique screen captures with the cursor **OFF** (Bluetooth pointer
  not paired, or "Hide Cursor" enabled).
- ~3 GB free disk for a ~150K-sample synthetic dataset.

## Why cursor-OFF backgrounds?

We composite a synthetic cursor into the background ourselves. If the
background already contains a cursor, the model learns to localize the
real one and to ignore the synthesized one — exactly backwards. Always
capture with the cursor invisible.

In iOS: Settings → Accessibility → Pointer Control → toggle the AssistiveTouch
pointer off, *or* simply unpair the Bluetooth mouse before capturing.

## Variety matters more than volume

A diverse 100-bg set beats a homogenous 1000-bg set. Aim for:

- All native iOS Settings panels (Wi-Fi, Bluetooth, Battery, ...)
- Home screen with a few app layouts (folders open and closed)
- Lock screen, notifications, Control Center
- A handful of common third-party apps (Photos, Messages, Notes, App Store, Maps)
- Browser pages (Safari with various sites)
- Dark mode and light mode coverage
- Low-light / night-shift coverage if your downstream agent runs in those conditions

The model doesn't see a cursor sprite per background until synthesis time, so
you don't need to position cursors thoughtfully — you just need diverse
*pixels under the cursor*.

## Background format

```
backgrounds_kept/
├── bg-00000.png    # 994 × 2160 BGR (or whatever your iPhone native is)
├── bg-00001.png
├── ...
```

PNG (lossless) is preferred over JPEG to avoid baking compression artifacts
into ground truth. 994 × 2160 is the iPhone H264 stream resolution; if you
capture at a different resolution, update `NATIVE_W` / `NATIVE_H` in
`train.py` and `synthesize.py` to match.

A dedicated capture script lives at `capture_backgrounds.py` for the
specific WSL+AirPlay+UxPlay capture pipeline this project was originally driven from. For other
capture paths, just produce equivalently-sized PNGs in `backgrounds_kept/`.

## Synthesis

Once `backgrounds_kept/` is populated:

```bash
python synthesize.py --out dataset --n 150000
```

This writes:

```
dataset/
├── imgs/000000.jpg
├── imgs/000001.jpg
├── ...
└── labels.jsonl       # one JSON per line: {path, x, y, has_cursor, sample_type, bg_id}
```

By default the mix is:

| Type        | Default | Purpose |
|-------------|--------:|---------|
| `normal_pos`| 55%     | Cursor visible at random position |
| `edge_pos`  | 15%     | Cursor partially clipped at a screen edge |
| `hard_neg`  | 15%     | Decoy shape (wrong-size disc, ring, ellipse, etc.) — `has_cursor=0` |
| `plain_neg` | 15%     | Untouched background — `has_cursor=0` |

Override the mix via CLI (`python synthesize.py --help`).

## What the trainer holds out

`train.py` does a **bg-level** train/val split: 10% of unique `bg_id`s
(determined from the source PNG filename) are held out entirely. Every sample
generated from those backgrounds goes to validation. This avoids the leakage
you'd get with a sample-level shuffle — synthesis variants of the same
background ending up in both splits and inflating the val score.

The held-out `bg_id`s are saved into the checkpoint metadata so re-evals are
reproducible.

## Provided demo backgrounds

The repo ships *no* real iPhone screen captures. If you want a tiny set to
verify the pipeline runs end-to-end without your own device, public iOS
screenshots from Apple's PR kit work well — see
<https://www.apple.com/newsroom/> or use any 994×2160 image you have rights
to redistribute. Drop a few into `backgrounds_kept/` and run synthesis with
`--n 1000` for a quick sanity loop.

## Reproducing the published checkpoints

You cannot fully reproduce the v0.3.x checkpoints' validation numbers
without the trainer's private background set, because the bg-level split
holds out *specific* backgrounds. You can:

- Train an architecturally-identical model on your own backgrounds and
  compare relative metrics across versions.
- Compare against the shipped weights on a shared real-frame test set.

## Privacy

Background captures are screenshots of the device they were taken on. They
contain whatever was on the screen — messages, photos, browsing history,
account names. Treat your `backgrounds_kept/` directory as private; do not
publish it. The repo's `.gitignore` already excludes images.
