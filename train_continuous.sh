#!/bin/bash
# train_continuous.sh — keep training the v0.3 model in 30-epoch passes
# until something kills this script (e.g. user runs `pkill -f train_continuous`
# or the underlying train.py crashes).
#
# Usage:
#   ./train_continuous.sh [WAIT_FOR_PID]
# If WAIT_FOR_PID is given, wait for that PID to exit before starting.
# Each pass: train.py --resume pointer_model.pt --epochs 30
# That keeps the model warm and acts like cosine LR restarts between passes.
# Logs go to /tmp/v03-train.log (current pass) and /tmp/v03-train-pass-N.log

set -e
cd "$(dirname "$0")"

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
    echo "[continuous] waiting for PID $WAIT_PID to finish before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "[continuous] PID $WAIT_PID exited. starting continuous loop."
fi

PASS=1
while true; do
    LOG="/tmp/v03-train-pass-${PASS}.log"
    echo "[continuous] === pass ${PASS} starting at $(date -Iseconds) ==="
    echo "[continuous] log → $LOG"
    "${PYTHON:-python3}" train.py \
        --resume pointer_model.pt \
        --epochs 30 \
        --batch-size 32 \
        --workers 6 \
        2>&1 | tee "$LOG"
    echo "[continuous] === pass ${PASS} finished at $(date -Iseconds) ==="
    PASS=$((PASS + 1))
    sleep 5
done
