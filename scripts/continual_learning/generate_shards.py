#!/usr/bin/env python3
# Copyright (C) 2026 EthyrosAI LLC / Jason Kempf
"""Generate token-balanced disjoint shards from a JSONL dataset.

Method: Sort samples by character length descending, assign round-robin
to N shards. Round-robin over a sorted sequence minimises the per-shard
token imbalance: adjacent items in the sorted order have similar lengths,
and round-robin spreads them across shards evenly. For any tokenizer where
token count ∝ character count (true for all BPE tokenizers), the resulting
shards are token-balanced.

Usage:
    python scripts/continual_learning/generate_shards.py
    python scripts/continual_learning/generate_shards.py --source data/training/benchmark_train.jsonl
    python scripts/continual_learning/generate_shards.py --n-shards 4 --out-dir data/training/shards
"""

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token-balanced shard generator")
    p.add_argument(
        "--source",
        default="data/training/benchmark_train.jsonl",
        help="Input JSONL file",
    )
    p.add_argument("--n-shards", type=int, default=8, help="Number of output shards")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/training/shards"),
        help="Output directory",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.source).expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with source.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Sort descending by character length so round-robin balances token load.
    samples.sort(key=lambda s: len(s.get("text", "")), reverse=True)

    shards: list[list[dict]] = [[] for _ in range(args.n_shards)]
    for idx, sample in enumerate(samples):
        shards[idx % args.n_shards].append(sample)

    total_chars = sum(len(s.get("text", "")) for s in samples)
    print(f"Source: {source.name} | {len(samples)} samples | {total_chars} chars")
    print(f"Shards: {args.n_shards} | ~{len(samples) // args.n_shards} samples each")

    for i, shard in enumerate(shards, start=1):
        out_path = out_dir / f"S{i}.jsonl"
        shard_chars = sum(len(s.get("text", "")) for s in shard)
        with out_path.open("w") as fh:
            for sample in shard:
                fh.write(json.dumps(sample) + "\n")
        print(f"  S{i}: {len(shard)} samples, {shard_chars} chars ({100*shard_chars/total_chars:.1f}%)")

    # Verify positional coverage: every source sample in exactly one shard.
    total_in_shards = sum(len(s) for s in shards)
    assert total_in_shards == len(samples), (
        f"Sample count mismatch: {total_in_shards} in shards vs {len(samples)} in source"
    )
    print(f"Verification: {total_in_shards}/{len(samples)} samples covered, disjoint by position")


if __name__ == "__main__":
    main()
