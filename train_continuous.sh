#!/bin/bash
# train_continuous.sh — train the model in short passes with warm-restart LR
# until something kills this script (e.g. user runs `pkill -f train_continuous`
# or the underlying train.py crashes).
#
# Each pass:
#   train.py --resume pointer_model.pt --epochs $EPOCHS_PER_PASS
# train.py re-instantiates AdamW + CosineAnnealingLR on every invocation, so
# each pass = a fresh LR cosine from the configured peak down — Loshchilov-style
# warm restart (SGDR). The model weights carry forward via --resume; the
# scheduler does NOT carry forward, which is what we want.
#
# Usage:
#   ./train_continuous.sh [WAIT_FOR_PID]
#
# Env overrides (defaults shown):
#   EPOCHS_PER_PASS=15
#   BATCH_SIZE=32
#   WORKERS=6
#   LOG_PREFIX=/tmp/ipf-train
#
# Logs: ${LOG_PREFIX}-pass-N.log per pass.

set -e
cd "$(dirname "$0")"

EPOCHS_PER_PASS="${EPOCHS_PER_PASS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-6}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/ipf-train}"

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
    echo "[continuous] waiting for PID $WAIT_PID to finish before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[continuous] PID $WAIT_PID exited. starting continuous loop."
fi

PASS=1
while true; do
    LOG="${LOG_PREFIX}-pass-${PASS}.log"
    echo "[continuous] === pass ${PASS} (${EPOCHS_PER_PASS} epochs) starting at $(date -Iseconds) ==="
    echo "[continuous] log → $LOG"
    "${PYTHON:-python3}" train.py \
        --resume pointer_model.pt \
        --epochs "$EPOCHS_PER_PASS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        2>&1 | tee "$LOG"
    echo "[continuous] === pass ${PASS} finished at $(date -Iseconds) ==="
    PASS=$((PASS + 1))
    sleep 5
done
