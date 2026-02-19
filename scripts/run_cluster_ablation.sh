#!/bin/bash
# Cluster-swap ablation: train each variant with v2 settings, then fast-gate test.
# Runs sequentially (GPU shared).

MODEL=/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16
TRAIN_DATA=data/training/answer_masked_train.jsonl
EVAL_DATA=data/training/answer_masked_val.jsonl
BASE_OUTPUT=/Volumes/CodeCypher/experiments/ablation-cluster

for CLUSTER in crt algebra tricky hs; do
    echo ""
    echo "============================================================"
    echo "CLUSTER: ${CLUSTER}"
    echo "============================================================"

    RETENTION=data/training/ablation/retention_swap_${CLUSTER}.jsonl
    OUTPUT=${BASE_OUTPUT}/${CLUSTER}

    mkdir -p "${OUTPUT}"

    echo "Training with ${CLUSTER} cluster swap..."
    poetry run mc train run \
        --model "${MODEL}" \
        -d "${TRAIN_DATA}" \
        --eval-data "${EVAL_DATA}" \
        --answer-mask \
        --retention-data "${RETENTION}" \
        --retention-fraction 0.2 \
        --max-epochs 7 \
        --budget-cap 0.775 \
        --output "${OUTPUT}" \
        2>&1 | tail -5

    echo ""
    echo "Fast attractor gate for ${CLUSTER}..."
    poetry run python scripts/fast_attractor_gate.py \
        --model "${MODEL}" \
        --adapter "${OUTPUT}" \
        2>&1

    GATE_EXIT=$?
    if [ ${GATE_EXIT} -eq 0 ]; then
        echo ">>> ${CLUSTER}: PASS"
    else
        echo ">>> ${CLUSTER}: FAIL (attractor degeneration)"
    fi

    echo ""
done

echo "============================================================"
echo "ABLATION COMPLETE"
echo "============================================================"
