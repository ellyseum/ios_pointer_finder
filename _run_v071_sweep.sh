#!/bin/bash
# v0.7.1 HM_WEIGHT sweep — 3 cold-start P1s, sequential (one GPU).
#
# For each candidate HM_WEIGHT, train a fresh model for 30 epochs and
# save the per-epoch metrics + final checkpoint. Compare descent shapes
# + best val_pos_err + bg-00009 peak across the three runs.
set -e
set -o pipefail
cd "$(dirname "$0")"

PY=/home/jocel/miniforge3/envs/cursor-ml/bin/python3
LOG_DIR=/tmp/v071-sweep
mkdir -p "$LOG_DIR"

WEIGHTS=(5e-4 2e-3 8e-3)

echo "[$(date -Iseconds)] === v0.7.1 HM_WEIGHT sweep starting ==="
SAMPLES=$(ls dataset/imgs/ 2>/dev/null | wc -l)
echo "[$(date -Iseconds)] dataset: $SAMPLES samples; sweep: ${WEIGHTS[*]}"

for W in "${WEIGHTS[@]}"; do
    TAG=$(echo "$W" | tr -d 'e+-' | tr 'e' 'e')
    LOG="$LOG_DIR/sweep-$W.log"
    CKPT_OUT="pointer_model_v0.7.1_sweep_${W}.pt"

    # Cold-start: ensure no rolling pointer is loaded
    if [ -f pointer_model.pt ]; then
        mv pointer_model.pt "pointer_model_pre_sweep_${W}_$(date +%s).pt"
        [ -f pointer_model.config.json ] && mv pointer_model.config.json "pointer_model_pre_sweep_${W}_$(date +%s).config.json"
    fi

    echo ""
    echo "[$(date -Iseconds)] === HM_WEIGHT=$W starting (cold-start P1 / 30 epochs) ==="
    echo "[$(date -Iseconds)] log → $LOG"

    IPF_HM_WEIGHT="$W" \
    IPF_PASS_ID="sweep-$W" \
        "$PY" train.py \
        --resume pointer_model.pt \
        --epochs 30 \
        --batch-size 64 \
        --workers 23 \
        --weights-out "$CKPT_OUT" \
        2>&1 | tee "$LOG"

    echo "[$(date -Iseconds)] === HM_WEIGHT=$W finished — ckpt: $CKPT_OUT ==="
done

echo ""
echo "[$(date -Iseconds)] === v0.7.1 sweep DONE ==="
echo ""
echo "best per HM_WEIGHT:"
for W in "${WEIGHTS[@]}"; do
    BEST=$(grep "saved global-best" "$LOG_DIR/sweep-$W.log" | tail -1 | grep -oP 'val_pos_err=\K[\d.]+')
    echo "  HM_WEIGHT=$W → best val_pos_err=${BEST:-N/A}px"
done
