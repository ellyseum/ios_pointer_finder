#!/bin/bash
# v0.7 cold-start training pipeline.
#
# Synth regen is NOT run here — the procedural-disc dataset/ generated for
# v0.7 is reused (no synthesize.py change in v0.7 affects synth output).
# This script only drives the cold-start: 30 epochs/pass, SGDR warm-restart
# loop, auto-stop after 3 stale passes.
#
# Cold-Start #1 measures the v0.7-7b BCE-sum + HM_WEIGHT=2e-5 loss change
# in isolation. After it lands, conf-head MaxPool (#104) triggers
# Cold-Start #2 from the same dataset.
set -e
set -o pipefail
cd "$(dirname "$0")"

PY=/home/jocel/miniforge3/envs/cursor-ml/bin/python3
LOG_DIR=/tmp/v07-train
mkdir -p "$LOG_DIR"

# Cold-start: ensure pointer_model.pt is absent so train.py inits fresh.
# (Previously seeding from v0.4 weights would warm-start the new loss,
# which entangles "did the loss reshape help?" with "did it help RELATIVE
# to v0.4's already-optimized weights for the OLD loss?")
if [ -f pointer_model.pt ]; then
    BACKUP_NAME="pointer_model_pre_v07_cold_$(date +%Y%m%d_%H%M%S).pt"
    mv pointer_model.pt "$BACKUP_NAME"
    [ -f pointer_model.config.json ] && mv pointer_model.config.json "${BACKUP_NAME%.pt}.config.json"
    echo "[$(date -Iseconds)] cold-start: backed up rolling pointer to $BACKUP_NAME"
fi

echo "[$(date -Iseconds)] === v0.7 cold-start training starting ==="
SAMPLES=$(ls dataset/imgs/ 2>/dev/null | wc -l)
echo "[$(date -Iseconds)] dataset: $SAMPLES samples"
echo "[$(date -Iseconds)] training: 30 ep/pass, batch 64, 23 workers, auto-stop=3"

EPOCHS_PER_PASS=30 \
BATCH_SIZE=64 \
WORKERS=23 \
PASSES_NO_IMPROVE_STOP=3 \
LOG_PREFIX="$LOG_DIR/train" \
PYTHON="$PY" \
./train_continuous.sh

echo "[$(date -Iseconds)] === v0.7 cold-start DONE ==="
