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
# Stop after this many consecutive passes that fail to improve the rolling
# pointer's mtime by even a second (= no new global-best saved). 0 disables.
PASSES_NO_IMPROVE_STOP="${PASSES_NO_IMPROVE_STOP:-5}"
ROLLING_POINTER="${ROLLING_POINTER:-pointer_model.pt}"

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
    echo "[continuous] waiting for PID $WAIT_PID to finish before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[continuous] PID $WAIT_PID exited. starting continuous loop."
fi

PASS=1
STALL=0
while true; do
    LOG="${LOG_PREFIX}-pass-${PASS}.log"
    # Snapshot the rolling-pointer mtime before the pass; the pass updated
    # the rolling pointer iff its mtime moves forward.
    MTIME_BEFORE=$(stat -c %Y "$ROLLING_POINTER" 2>/dev/null || echo 0)

    echo "[continuous] === pass ${PASS} (${EPOCHS_PER_PASS} epochs) starting at $(date -Iseconds) ==="
    echo "[continuous] log → $LOG"
    # Export pass id so train.py can inject it into per-best filenames; without
    # it, two passes that converge to the same rounded val_pos_err would
    # collide on disk and overwrite each other's weights.
    IPF_PASS_ID="$PASS" \
    "${PYTHON:-python3}" train.py \
        --resume "$ROLLING_POINTER" \
        --epochs "$EPOCHS_PER_PASS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        2>&1 | tee "$LOG"
    echo "[continuous] === pass ${PASS} finished at $(date -Iseconds) ==="

    MTIME_AFTER=$(stat -c %Y "$ROLLING_POINTER" 2>/dev/null || echo 0)
    if [ "$MTIME_AFTER" -gt "$MTIME_BEFORE" ]; then
        STALL=0
        echo "[continuous] pass ${PASS} produced a new global-best (rolling pointer updated)"
    else
        STALL=$((STALL + 1))
        echo "[continuous] pass ${PASS} did not improve global best (stall ${STALL}/${PASSES_NO_IMPROVE_STOP})"
        if [ "$PASSES_NO_IMPROVE_STOP" -gt 0 ] && [ "$STALL" -ge "$PASSES_NO_IMPROVE_STOP" ]; then
            echo "[continuous] reached PASSES_NO_IMPROVE_STOP=${PASSES_NO_IMPROVE_STOP}; stopping."
            exit 0
        fi
    fi

    PASS=$((PASS + 1))
    sleep 5
done
