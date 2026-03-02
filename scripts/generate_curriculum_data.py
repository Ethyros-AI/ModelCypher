#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate isolated training data for curriculum skill nodes.

Generates the five JSONL files referenced by skill_dag.py that had no
training data:

  - modus_tollens_train.jsonl          (logic: MT)
  - disj_syllogism_train.jsonl         (logic: DS)
  - universal_instantiation_train.jsonl (logic: UI)
  - arithmetic_add_train.jsonl         (math: A + B)
  - arithmetic_div_train.jsonl         (math: A / B)

Logic files use the same domain-split train/val pattern and
{"text", "answer_start", "logic_id"} format as generate_reasoning_traces.py.

Arithmetic files use the integer answer_start offset format from
retention_replay.jsonl.

Usage:
    poetry run python scripts/generate_curriculum_data.py
    poetry run python scripts/generate_curriculum_data.py --output data/training --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_reasoning_traces import MT_ANSWER_TEMPLATES, MT_TRACE_TEMPLATES  # noqa: E402
from novel_problems import PREMISE_PAIRS  # noqa: E402


def generate_mt(pairs: list, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B in pairs:
        for t_idx, template in enumerate(MT_TRACE_TEMPLATES):
            text = template.format(A=A, B=B)
            answer_start = MT_ANSWER_TEMPLATES[t_idx].format(A=A, B=B)
            samples.append({
                "text": text,
                "answer_start": answer_start,
                "logic_id": "modus_tollens",
                "template_id": f"mt_{domain}_{t_idx}",
            })
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Disjunctive Syllogism — new DS trace templates parallel to MT
# ---------------------------------------------------------------------------

DS_TRACE_TEMPLATES = [
    # Template 1: Formal rule naming
    (
        "Apply logical reasoning:\n"
        "Either {A} or {B}. It is not the case that {A}.\n"
        "What can we conclude?\n"
        "The premises state: either {A} or {B}. We observe that {A} is not the case. "
        "By disjunctive syllogism, when one disjunct is false the other must be true. "
        "Therefore, {B}."
    ),
    # Template 2: Elimination explanation
    (
        "Apply logical reasoning:\n"
        "Either {A} or {B}. It is not the case that {A}.\n"
        "What can we conclude?\n"
        "We have a disjunction: {A} or {B}. Eliminating the false disjunct — "
        "{A} is not the case — we are left with the only remaining option. "
        "Therefore, {B}."
    ),
    # Template 3: Step-by-step with rule
    (
        "Apply logical reasoning:\n"
        "Either {A} or {B}. It is not the case that {A}.\n"
        "What can we conclude?\n"
        "Disjunction: {A} or {B}. "
        "Negation: not {A}. "
        "The false disjunct ({A}) is eliminated, leaving the other disjunct. "
        "Conclusion: {B}."
    ),
    # Template 4: Direct and concise
    (
        "Apply logical reasoning:\n"
        "Either {A} or {B}. It is not the case that {A}.\n"
        "What can we conclude?\n"
        "Since {A} is not the case, and exactly one of the two must hold, {B}."
    ),
]

DS_ANSWER_TEMPLATES = [
    "The premises state: either {A} or {B}.",
    "We have a disjunction: {A} or {B}.",
    "Disjunction: {A} or {B}.",
    "Since {A} is not the case,",
]


def generate_ds(pairs: list, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B in pairs:
        for t_idx, template in enumerate(DS_TRACE_TEMPLATES):
            text = template.format(A=A, B=B)
            answer_start = DS_ANSWER_TEMPLATES[t_idx].format(A=A, B=B)
            samples.append({
                "text": text,
                "answer_start": answer_start,
                "logic_id": "disjunctive_syllogism",
                "template_id": f"ds_{domain}_{t_idx}",
            })
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Universal Instantiation — self-contained premise data + trace templates
# ---------------------------------------------------------------------------
#
# Each entry: (domain, universal_fact, specific_fact, conclusion)
# Prompt pattern: "{U}. {S}.\nWhat can we conclude?"
# Answer: "{C}."
#
# UI doesn't map to PREMISE_PAIRS (A→B conditionals), so we provide
# domain-specific tuples here.

UI_PAIRS: list[tuple[str, str, str, str]] = [
    # biology
    ("biology", "All mammals are warm-blooded", "Dolphins are mammals", "Dolphins are warm-blooded"),
    ("biology", "All vertebrates have a spinal cord", "Trout are vertebrates", "Trout have a spinal cord"),
    ("biology", "All flowering plants produce seeds", "Roses are flowering plants", "Roses produce seeds"),
    ("biology", "All reptiles are cold-blooded", "Lizards are reptiles", "Lizards are cold-blooded"),
    # astronomy
    ("astronomy", "All stars undergo nuclear fusion", "The Sun is a star", "The Sun undergoes nuclear fusion"),
    ("astronomy", "All planets orbit a host star", "Jupiter is a planet", "Jupiter orbits a host star"),
    ("astronomy", "All moons are natural satellites", "Europa is a moon", "Europa is a natural satellite"),
    ("astronomy", "All red giants have exhausted their core hydrogen", "Betelgeuse is a red giant",
     "Betelgeuse has exhausted its core hydrogen"),
    # chemistry
    ("chemistry", "All acids have a pH below 7", "Hydrochloric acid is an acid", "Hydrochloric acid has a pH below 7"),
    ("chemistry", "All noble gases are chemically inert", "Argon is a noble gas", "Argon is chemically inert"),
    ("chemistry", "All isotopes of an element share the same atomic number",
     "Carbon-14 is an isotope of carbon", "Carbon-14 shares the atomic number of carbon"),
    # physics
    ("physics", "All objects with mass exert gravitational attraction",
     "The Moon has mass", "The Moon exerts gravitational attraction"),
    ("physics", "All conductors allow electric current to flow",
     "Copper is a conductor", "Copper allows electric current to flow"),
    ("physics", "All moving objects possess kinetic energy",
     "A thrown ball is a moving object", "A thrown ball possesses kinetic energy"),
    # geography
    ("geography", "All peninsulas are surrounded by water on three sides",
     "The Iberian Peninsula is a peninsula", "The Iberian Peninsula is surrounded by water on three sides"),
    ("geography", "All archipelagos consist of a group of islands",
     "The Philippines is an archipelago", "The Philippines consists of a group of islands"),
    ("geography", "All deserts receive less than 250 mm of rainfall per year",
     "The Atacama is a desert", "The Atacama receives less than 250 mm of rainfall per year"),
    # cooking
    ("cooking", "All leavening agents cause dough to rise",
     "Baking powder is a leavening agent", "Baking powder causes dough to rise"),
    ("cooking", "All emulsifiers help bind oil and water",
     "Lecithin is an emulsifier", "Lecithin helps bind oil and water"),
    # music
    ("music", "All octaves span twelve semitones",
     "The interval from C to C is an octave", "The interval from C to C spans twelve semitones"),
    ("music", "All percussion instruments produce sound by being struck",
     "A snare drum is a percussion instrument", "A snare drum produces sound by being struck"),
]

UI_TRACE_TEMPLATES = [
    # Template 1: Formal rule naming
    "{U}. {S}.\nWhat can we conclude?\n"
    "By universal instantiation: the universal premise states {U}. "
    "We have a particular instance: {S}. "
    "Applying the universal to this instance: {C}.",
    # Template 2: Direct inference
    "{U}. {S}.\nWhat follows?\n"
    "The universal rule applies to every member of the category. "
    "Since {S}, and the universal tells us {U}, it follows that {C}.",
    # Template 3: Step-by-step
    "{U}. {S}.\nWhat must be true?\n"
    "Universal premise: {U}. "
    "Particular: {S}. "
    "The particular instance falls under the universal scope. "
    "Conclusion: {C}.",
    # Template 4: Direct and concise
    "{U}. {S}.\nWhat can we conclude?\n"
    "{C}.",
]

UI_ANSWER_TEMPLATES = [
    "By universal instantiation:",
    "The universal rule applies to every member of the category.",
    "Universal premise:",
    "",  # Template 4 has no prefix — answer_start IS the conclusion
]


def generate_ui(pairs: list, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, U, S, C in pairs:
        for t_idx, template in enumerate(UI_TRACE_TEMPLATES):
            text = template.format(U=U, S=S, C=C)
            ans_prefix = UI_ANSWER_TEMPLATES[t_idx]
            # For template 4, answer_start is the conclusion itself
            answer_start = ans_prefix if ans_prefix else C
            samples.append({
                "text": text,
                "answer_start": answer_start,
                "logic_id": "universal_instantiation",
                "template_id": f"ui_{domain}_{t_idx}",
            })
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Arithmetic Add — A + B = C, integer range
# ---------------------------------------------------------------------------

def generate_add(seed: int) -> list[dict]:
    """Generate integer addition examples.

    DATA DESIGN CHOICE (not an algorithmic threshold):
    Range A, B ∈ [1, 25] defines the scope of addition facts taught —
    single-digit through two-digit addends, sums up to 50. This is a
    curriculum coverage decision about which facts are "in scope", not
    a convergence parameter. Example count is the exhaustive consequence
    of this range: 325 unique unordered pairs × 2 phrasings (commutativity)
    minus 25 diagonal pairs with a=b = 625 total.

    Two phrasings per unique pair to surface commutativity.
    """
    rng = random.Random(seed)
    samples = []
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for a in range(1, 26):
        for b in range(1, 26):
            pair = (min(a, b), max(a, b))
            if pair not in seen:
                seen.add(pair)
                candidates.append((a, b))

    rng.shuffle(candidates)

    for a, b in candidates:
        c = a + b
        text1 = f"What is {a} + {b}?\n{c}"
        samples.append({
            "text": text1,
            "answer_start": len(f"What is {a} + {b}?\n"),
            "logic_id": "arithmetic_add",
        })
        if a != b:
            text2 = f"What is {b} + {a}?\n{c}"
            samples.append({
                "text": text2,
                "answer_start": len(f"What is {b} + {a}?\n"),
                "logic_id": "arithmetic_add",
            })

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Arithmetic Divide — (A * B) / A = B, exact integer division
# ---------------------------------------------------------------------------

def generate_div(seed: int) -> list[dict]:
    """Generate integer division examples with exact results.

    DATA DESIGN CHOICE (not an algorithmic threshold):
    Divisor [2, 15] × quotient [2, 12] defines the scope of division facts
    taught — the standard multiplication table range, giving 14×11=154
    unique (divisor, quotient) pairs. All results are exact integers by
    construction (dividend = divisor × quotient). Example count is the
    exhaustive consequence of this scope: 154 pairs × 2 phrasings = 308.

    All pairs guarantee exact integer results (no remainders).
    Two phrasings per pair for format variation.
    """
    rng = random.Random(seed)
    samples = []
    for divisor in range(2, 16):
        for quotient in range(2, 13):
            dividend = divisor * quotient
            text1 = f"What is {dividend} / {divisor}?\n{quotient}"
            samples.append({
                "text": text1,
                "answer_start": len(f"What is {dividend} / {divisor}?\n"),
                "logic_id": "arithmetic_divide",
            })
            text2 = f"{dividend} divided by {divisor} is?\n{quotient}"
            samples.append({
                "text": text2,
                "answer_start": len(f"{dividend} divided by {divisor} is?\n"),
                "logic_id": "arithmetic_divide",
            })

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _domain_split(
    pairs: list,
    val_fraction: float,
    seed: int,
) -> tuple[list, list]:
    """Split premise pairs by domain to prevent train/val leakage."""
    rng = random.Random(seed)
    domains = sorted({p[0] for p in pairs})
    rng.shuffle(domains)
    n_val = max(1, int(len(domains) * val_fraction))
    val_domains = set(domains[:n_val])
    train_pairs = [p for p in pairs if p[0] not in val_domains]
    val_pairs = [p for p in pairs if p[0] in val_domains]
    return train_pairs, val_pairs


def _write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Wrote {len(samples):4d} examples → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curriculum training data")
    parser.add_argument("--output", default="data/training", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    # val_fraction=0.15: DATA DESIGN CHOICE — matches generate_reasoning_traces.py
    # (established precedent across this codebase). _domain_split operates on domains,
    # not examples: PREMISE_PAIRS has 12 domains → n_val_domains = max(1, int(12×0.15))
    # = 1 held-out domain → 3–6 premise pairs × 4 templates = 12–24 val examples per
    # logic type (seed-dependent). UI_PAIRS has 7 domains → 1 val domain → 12–16
    # examples. This produces fewer examples than the ≥20 Clopper-Pearson floor for
    # auto-regime CI; these val files are for curriculum monitoring, not regime gating.
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    out = Path(args.output)
    seed = args.seed
    vf = args.val_fraction

    # ── Logic files: domain-split from PREMISE_PAIRS ──────────────────────
    train_pairs, val_pairs = _domain_split(PREMISE_PAIRS, vf, seed)
    print(f"PREMISE_PAIRS: {len(PREMISE_PAIRS)} total, "
          f"{len(train_pairs)} train, {len(val_pairs)} val")

    print("\n[modus_tollens]")
    _write_jsonl(out / "modus_tollens_train.jsonl", generate_mt(train_pairs, seed))
    _write_jsonl(out / "modus_tollens_val.jsonl",   generate_mt(val_pairs,   seed + 1))

    print("\n[disjunctive_syllogism]")
    _write_jsonl(out / "disj_syllogism_train.jsonl", generate_ds(train_pairs, seed + 2))
    _write_jsonl(out / "disj_syllogism_val.jsonl",   generate_ds(val_pairs,   seed + 3))

    print("\n[universal_instantiation]")
    ui_train, ui_val = _domain_split(UI_PAIRS, vf, seed + 4)
    print(f"  UI_PAIRS: {len(UI_PAIRS)} total, {len(ui_train)} train, {len(ui_val)} val")
    _write_jsonl(out / "universal_instantiation_train.jsonl", generate_ui(ui_train, seed + 5))
    _write_jsonl(out / "universal_instantiation_val.jsonl",   generate_ui(ui_val,   seed + 6))

    # ── Arithmetic files: generated integers ──────────────────────────────
    print("\n[arithmetic_add]")
    _write_jsonl(out / "arithmetic_add_train.jsonl", generate_add(seed + 7))

    print("\n[arithmetic_divide]")
    _write_jsonl(out / "arithmetic_div_train.jsonl", generate_div(seed + 8))

    print("\nDone.")


if __name__ == "__main__":
    main()
