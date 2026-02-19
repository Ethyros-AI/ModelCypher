#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Build cluster-swap ablation variants of retention_replay.jsonl.
#
# Control: lines 1-200 (the original v2 retention that works).
# Each variant REPLACES equal-count samples from the end of the control
# with one new cluster, keeping total at 200.
#
# Clusters (lines 201-235):
#   CRT:     lines 201-210 (10 samples)
#   Algebra: lines 211-220 (10 samples)
#   Tricky:  lines 221-230 (10 samples)
#   HS:      lines 231-235 (5 samples)

import json
import random
from pathlib import Path

RETENTION_FILE = Path("data/training/retention_replay.jsonl")
OUTPUT_DIR = Path("data/training/ablation")

CLUSTERS = {
    "crt":     (200, 210),  # lines 201-210 (0-indexed: 200-209)
    "algebra": (210, 220),  # lines 211-220
    "tricky":  (220, 230),  # lines 221-230
    "hs":      (230, 235),  # lines 231-235
}


def main():
    # Load all 235 samples
    with open(RETENTION_FILE) as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    assert len(all_samples) == 235, f"Expected 235, got {len(all_samples)}"

    control = all_samples[:200]  # v2 control set
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write control file
    control_path = OUTPUT_DIR / "retention_control.jsonl"
    with open(control_path, "w") as f:
        for s in control:
            f.write(json.dumps(s) + "\n")
    print(f"Control: {control_path} ({len(control)} samples)")

    # For each cluster, replace last N samples of control with the cluster
    for cluster_name, (start, end) in CLUSTERS.items():
        cluster_samples = all_samples[start:end]
        n_cluster = len(cluster_samples)

        # Replace last n_cluster samples from control
        variant = control[:200 - n_cluster] + cluster_samples
        assert len(variant) == 200, f"{cluster_name}: expected 200, got {len(variant)}"

        # Shuffle for training (deterministic seed per cluster)
        random.seed(42 + hash(cluster_name))
        random.shuffle(variant)

        variant_path = OUTPUT_DIR / f"retention_swap_{cluster_name}.jsonl"
        with open(variant_path, "w") as f:
            for s in variant:
                f.write(json.dumps(s) + "\n")
        print(f"{cluster_name}: {variant_path} ({n_cluster} swapped, {200 - n_cluster} retained)")


if __name__ == "__main__":
    main()
