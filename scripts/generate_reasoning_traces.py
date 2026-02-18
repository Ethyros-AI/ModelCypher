#!/usr/bin/env python3
"""Generate explicit reasoning trace training data for modus tollens.

The 350M model can do MP (forward: see A, conclude B) but not MT (backward:
see ¬B, conclude ¬A). Analysis shows the model's attention collapses on MT —
it has no representation of backward reasoning.

This script generates training data with EXPLICIT reasoning chains that spell
out the logical operation. The traces are programmatically derived (provably
correct), not model-generated.

Key insight: The model needs to learn to GENERATE the backward reasoning
trajectory, not just the final answer. CE loss on "Not A" teaches the answer
format. CE loss on the full trace teaches the reasoning path.

Usage:
    python scripts/generate_reasoning_traces.py --output data/training/
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Import premise pairs from novel_problems.py
sys.path.insert(0, str(Path(__file__).parent))
from novel_problems import PREMISE_PAIRS


# ---------------------------------------------------------------------------
# Reasoning trace templates
# ---------------------------------------------------------------------------

# MT templates: each produces a different phrasing of the same logical operation
MT_TRACE_TEMPLATES = [
    # Template 1: Formal rule naming
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is not the case that {B}.\n"
        "What can we conclude?\n"
        "The premise states: if {A}, then {B}. We observe that {B} is not the case. "
        "By modus tollens, when the consequent is false, the antecedent must also be false. "
        "Therefore, it is not the case that {A}."
    ),
    # Template 2: Contrapositive explanation
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is not the case that {B}.\n"
        "What can we conclude?\n"
        "We know that {A} implies {B}. The contrapositive is equally valid: "
        "if not {B}, then not {A}. Since {B} is not the case, "
        "it follows that it is not the case that {A}."
    ),
    # Template 3: Step-by-step with rule
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is not the case that {B}.\n"
        "What can we conclude?\n"
        "Premise: if {A} then {B}. "
        "Observation: not {B}. "
        "If the consequent ({B}) is denied, the antecedent ({A}) must be denied. "
        "Conclusion: it is not the case that {A}."
    ),
    # Template 4: Direct and concise
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is not the case that {B}.\n"
        "What can we conclude?\n"
        "Since {B} is not the case, and {A} would require {B}, "
        "it is not the case that {A}."
    ),
]

# MP templates: reinforce forward reasoning with explicit traces
MP_TRACE_TEMPLATES = [
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is the case that {A}.\n"
        "What can we conclude?\n"
        "The premise states: if {A}, then {B}. We observe that {A} is the case. "
        "By modus ponens, when the antecedent is true, the consequent must be true. "
        "Therefore, {B}."
    ),
    (
        "Apply logical reasoning:\n"
        "If {A}, then {B}. It is the case that {A}.\n"
        "What can we conclude?\n"
        "We know that {A} implies {B}. Since {A} is the case, "
        "it follows that {B}."
    ),
]

# Answer-start markers (what comes after the question)
MT_ANSWER_TEMPLATES = [
    "The premise states: if {A}, then {B}.",
    "We know that {A} implies {B}.",
    "Premise: if {A} then {B}.",
    "Since {B} is not the case,",
]

MP_ANSWER_TEMPLATES = [
    "The premise states: if {A}, then {B}.",
    "We know that {A} implies {B}.",
]


def capitalize_first(s: str) -> str:
    """Capitalize the first letter of a string."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def generate_mt_traces(premise_pairs: list, seed: int = 42) -> list[dict]:
    """Generate MT reasoning traces for all premise pairs."""
    rng = random.Random(seed)
    samples = []

    for domain, A, B in premise_pairs:
        # Use each template
        for t_idx, template in enumerate(MT_TRACE_TEMPLATES):
            text = template.format(A=A, B=B)
            answer_start = MT_ANSWER_TEMPLATES[t_idx].format(A=A, B=B)

            samples.append({
                "text": text,
                "answer_start": answer_start,
                "logic_id": "modus_tollens",
                "template_id": f"mt_trace_{domain}_{t_idx}",
                "pair_type": "mt_distilled",
            })

    rng.shuffle(samples)
    return samples


def generate_mp_traces(premise_pairs: list, seed: int = 42) -> list[dict]:
    """Generate MP reasoning traces for all premise pairs."""
    rng = random.Random(seed)
    samples = []

    for domain, A, B in premise_pairs:
        A_cap = capitalize_first(A)
        for t_idx, template in enumerate(MP_TRACE_TEMPLATES):
            text = template.format(A=A, B=B, A_cap=A_cap)
            answer_start = MP_ANSWER_TEMPLATES[t_idx].format(A=A, B=B)

            samples.append({
                "text": text,
                "answer_start": answer_start,
                "logic_id": "modus_ponens",
                "template_id": f"mp_trace_{domain}_{t_idx}",
                "pair_type": "mp_distilled",
            })

    rng.shuffle(samples)
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate reasoning trace training data")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of premise pairs held out for validation")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Split premise pairs into train/val by DOMAIN to ensure no leakage
    domains = sorted({p[0] for p in PREMISE_PAIRS})
    rng.shuffle(domains)
    n_val_domains = max(1, int(len(domains) * args.val_fraction))
    val_domains = set(domains[:n_val_domains])
    train_domains = set(domains[n_val_domains:])

    train_pairs = [p for p in PREMISE_PAIRS if p[0] in train_domains]
    val_pairs = [p for p in PREMISE_PAIRS if p[0] in val_domains]

    print(f"Domains: {len(domains)} total, {len(train_domains)} train, {len(val_domains)} val")
    print(f"Train domains: {sorted(train_domains)}")
    print(f"Val domains: {sorted(val_domains)}")
    print(f"Premise pairs: {len(train_pairs)} train, {len(val_pairs)} val")

    # Generate traces
    mt_train = generate_mt_traces(train_pairs, seed=args.seed)
    mp_train = generate_mp_traces(train_pairs, seed=args.seed + 1)
    mt_val = generate_mt_traces(val_pairs, seed=args.seed + 2)
    mp_val = generate_mp_traces(val_pairs, seed=args.seed + 3)

    # Combine MP and MT for training (model needs both directions)
    train_samples = mt_train + mp_train
    val_samples = mt_val + mp_val
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Count by type
    mt_train_count = sum(1 for s in train_samples if s["logic_id"] == "modus_tollens")
    mp_train_count = sum(1 for s in train_samples if s["logic_id"] == "modus_ponens")
    mt_val_count = sum(1 for s in val_samples if s["logic_id"] == "modus_tollens")
    mp_val_count = sum(1 for s in val_samples if s["logic_id"] == "modus_ponens")

    print(f"\nTraining samples: {len(train_samples)} ({mt_train_count} MT, {mp_train_count} MP)")
    print(f"Validation samples: {len(val_samples)} ({mt_val_count} MT, {mp_val_count} MP)")

    # Write files
    train_path = output_dir / "reasoning_traces_train.jsonl"
    val_path = output_dir / "reasoning_traces_val.jsonl"

    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nWritten:")
    print(f"  {train_path} ({len(train_samples)} samples)")
    print(f"  {val_path} ({len(val_samples)} samples)")

    # Print a few examples
    print(f"\n{'='*60}")
    print("SAMPLE MT TRACE:")
    print(f"{'='*60}")
    print(mt_train[0]["text"])
    print(f"\n{'='*60}")
    print("SAMPLE MP TRACE:")
    print(f"{'='*60}")
    print(mp_train[0]["text"])


if __name__ == "__main__":
    main()
