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
    # Experimental flags use service API directly (not exposed in CLI).
    poetry run python -c "
from modelcypher.cli.composition import get_dataset_training_service
svc = get_dataset_training_service()
svc.train_from_dataset(
    model_path='${MODEL}',
    dataset_path='${TRAIN_DATA}',
    eval_dataset_path='${EVAL_DATA}',
    output_path='${OUTPUT}',
    answer_mask=True,
    retention_dataset_path='${RETENTION}',
    retention_fraction=None,
    max_epochs=7,
    budget_cap=0.775,
)
" 2>&1 | tail -5

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
