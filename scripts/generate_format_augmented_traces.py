#!/usr/bin/env python3
"""Generate format-augmented reasoning trace training data.

The pure CE adapter (v1) learned MT successfully but only fires from a narrow
prompt format matching the training template. This generator produces DIVERSE
prompt formats to teach format invariance.

Format dimensions varied:
1. Header: "Apply logical reasoning:", "Consider the following:", none
2. Negation: "It is not the case that B", "B is not true", "Not B", natural neg
3. Question: "What can we conclude?", "What follows?", "Therefore, what?"
4. Layout: newline before question, or inline
5. Premise structure: conditional ("If A then B"), universal ("All X have Y",
   "Every X who A also B"), bidirectional ("A implies B")

Each premise pair × logic direction generates ~8-12 samples across format
combinations + ~4 universal quantifier variants. Reasoning traces vary.

Usage:
    python scripts/generate_format_augmented_traces.py --output data/training/
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from novel_problems import PREMISE_PAIRS


# ---------------------------------------------------------------------------
# Format components (vary independently)
# ---------------------------------------------------------------------------

HEADERS = [
    "Apply logical reasoning:\n",
    "Consider the following:\n",
    "Logical reasoning:\n",
    "",  # no header
]

# Negation format for observation: {B} is the variable
# Returns (observation_text, normalized_B_neg) — the normalized form is used in traces
def make_negations(B: str) -> list[str]:
    """Generate diverse negation phrasings of B."""
    return [
        f"It is not the case that {B}.",
        f"{B[0].upper() + B[1:]} is not the case.",
        f"It is not true that {B}.",
        f"It is false that {B}.",
    ]


# For MP: affirmation formats
def make_affirmations(A: str) -> list[str]:
    """Generate diverse affirmation phrasings of A."""
    A_cap = A[0].upper() + A[1:]
    return [
        f"It is the case that {A}.",
        f"{A_cap}.",
        f"We observe that {A}.",
    ]


QUESTIONS = [
    "What can we conclude?",
    "What follows?",
    "What can be deduced?",
]

LAYOUTS = [
    "newline",   # observation\nquestion
    "inline",    # observation question  (same line)
]


# ---------------------------------------------------------------------------
# Reasoning trace templates (the ANSWERS — format-invariant)
# ---------------------------------------------------------------------------

MT_TRACES = [
    # Formal rule naming
    (
        "The premise states: if {A}, then {B}. We observe that {B} is not the case. "
        "By modus tollens, when the consequent is false, the antecedent must also be false. "
        "Therefore, it is not the case that {A}."
    ),
    # Contrapositive explanation
    (
        "We know that {A} implies {B}. The contrapositive is equally valid: "
        "if not {B}, then not {A}. Since {B} is not the case, "
        "it follows that it is not the case that {A}."
    ),
    # Step-by-step
    (
        "Premise: if {A} then {B}. "
        "Observation: not {B}. "
        "If the consequent ({B}) is denied, the antecedent ({A}) must be denied. "
        "Conclusion: it is not the case that {A}."
    ),
    # Concise
    (
        "Since {B} is not the case, and {A} would require {B}, "
        "it is not the case that {A}."
    ),
]

MP_TRACES = [
    (
        "The premise states: if {A}, then {B}. We observe that {A} is the case. "
        "By modus ponens, when the antecedent is true, the consequent must be true. "
        "Therefore, {B}."
    ),
    (
        "We know that {A} implies {B}. Since {A} is the case, "
        "it follows that {B}."
    ),
]

# ---------------------------------------------------------------------------
# Premise structure templates (vary the logical framing of "If A then B")
# ---------------------------------------------------------------------------

# Conditional: "If A, then B" (base format)
# Universal: "All/Every X that A also B" or "Anything that A also B"
# These require A/B to be noun-compatible — use a subset of premise pairs

PREMISE_STRUCTURES = [
    # Standard conditional
    ("conditional", "If {A}, then {B}."),
    # Universal quantifier (active)
    ("universal_all", "All cases where {A} result in {B}."),
    # Universal (every)
    ("universal_every", "Whenever {A}, {B}."),
]


def generate_samples(premise_pairs: list, seed: int = 42) -> list[dict]:
    """Generate format-augmented training samples."""
    rng = random.Random(seed)
    samples = []

    for domain, A, B in premise_pairs:
        negations = make_negations(B)
        affirmations = make_affirmations(A)

        # MT: each premise pair gets diverse format combinations
        for neg_text in negations:
            for header in HEADERS:
                # Pick a question, layout, and premise structure randomly
                question = rng.choice(QUESTIONS)
                layout = rng.choice(LAYOUTS)
                _, premise_fmt = rng.choice(PREMISE_STRUCTURES)
                premise = premise_fmt.format(A=A, B=B)

                if layout == "newline":
                    prompt = f"{header}{premise} {neg_text}\n{question}\n"
                else:
                    prompt = f"{header}{premise} {neg_text} {question}\n"

                # Pick a trace template
                trace = rng.choice(MT_TRACES).format(A=A, B=B)
                text = prompt + trace

                samples.append({
                    "text": text,
                    "logic_id": "modus_tollens",
                    "domain": domain,
                })

        # MP: fewer combos (forward reasoning already works)
        for aff_text in affirmations[:2]:  # only 2 affirmation formats
            header = rng.choice(HEADERS)
            question = rng.choice(QUESTIONS)
            layout = rng.choice(LAYOUTS)
            _, premise_fmt = rng.choice(PREMISE_STRUCTURES)
            premise = premise_fmt.format(A=A, B=B)

            if layout == "newline":
                prompt = f"{header}{premise} {aff_text}\n{question}\n"
            else:
                prompt = f"{header}{premise} {aff_text} {question}\n"

            trace = rng.choice(MP_TRACES).format(A=A, B=B)
            text = prompt + trace

            samples.append({
                "text": text,
                "logic_id": "modus_ponens",
                "domain": domain,
            })

    rng.shuffle(samples)
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate format-augmented reasoning traces")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Split by domain
    domains = sorted({p[0] for p in PREMISE_PAIRS})
    rng.shuffle(domains)
    n_val = max(1, int(len(domains) * args.val_fraction))
    val_domains = set(domains[:n_val])
    train_domains = set(domains[n_val:])

    train_pairs = [p for p in PREMISE_PAIRS if p[0] in train_domains]
    val_pairs = [p for p in PREMISE_PAIRS if p[0] in val_domains]

    print(f"Domains: {len(domains)} total, {len(train_domains)} train, {len(val_domains)} val")
    print(f"Val domains: {sorted(val_domains)}")
    print(f"Premise pairs: {len(train_pairs)} train, {len(val_pairs)} val")

    train_samples = generate_samples(train_pairs, seed=args.seed)
    val_samples = generate_samples(val_pairs, seed=args.seed + 100)

    # Count
    mt_train = sum(1 for s in train_samples if s["logic_id"] == "modus_tollens")
    mp_train = sum(1 for s in train_samples if s["logic_id"] == "modus_ponens")
    mt_val = sum(1 for s in val_samples if s["logic_id"] == "modus_tollens")
    mp_val = sum(1 for s in val_samples if s["logic_id"] == "modus_ponens")

    print(f"\nTraining: {len(train_samples)} ({mt_train} MT, {mp_train} MP)")
    print(f"Validation: {len(val_samples)} ({mt_val} MT, {mp_val} MP)")

    # Write text-only (pure CE format — no structured fields)
    train_path = output_dir / "format_augmented_train.jsonl"
    val_path = output_dir / "format_augmented_val.jsonl"

    for path, data in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps({"text": s["text"]}) + "\n")

    print(f"\nWritten:")
    print(f"  {train_path} ({len(train_samples)} samples)")
    print(f"  {val_path} ({len(val_samples)} samples)")

    # Show format diversity
    print(f"\n{'='*60}")
    print("FORMAT DIVERSITY EXAMPLES (first 6 MT):")
    print(f"{'='*60}")
    mt_examples = [s for s in train_samples if s["logic_id"] == "modus_tollens"][:6]
    for i, s in enumerate(mt_examples):
        print(f"\n--- Example {i+1} ({s['domain']}) ---")
        # Show just the prompt part (before the trace)
        lines = s["text"].split("\n")
        for line in lines[:4]:
            print(f"  {line[:120]}")
        if len(lines) > 4:
            print(f"  ...")


if __name__ == "__main__":
    main()
