#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generate held-out eval JSONL files for all curriculum logic and arithmetic nodes.

Creates data/eval/ and writes 100 samples per node. These files are consumed by
evaluate_skill_mastery() (adapters/curriculum_eval_adapter.py).

Format: {"text": "PROMPT\nAnswer: EXPECTED"} — the "Answer:" split is used by
evaluate_skill_mastery to separate prompt from expected answer, so no answer_start
field is needed.

Logic eval strategy:
  - Uses same PREMISE_PAIRS as training but a different random seed (99) for ordering.
  - Uses SHORT answer-only templates (not full reasoning traces) to test rule application
    without testing trace-format memorization.
  - For nodes without PREMISE_PAIRS support (HS, rule_recognition, concise_reasoning,
    chain_reasoning): uses purpose-built eval pairs defined below.

Arithmetic eval strategy:
  - Generates from wider integer ranges than training (adds coverage, not repetition).
  - Exact integer arithmetic — no oracle required.

Usage:
    poetry run python scripts/generate_curriculum_evals.py
    poetry run python scripts/generate_curriculum_evals.py --output data/eval --seed 99
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from novel_problems import PREMISE_PAIRS  # noqa: E402

EVAL_SEED = 99  # distinct from training seed (42)
N_SAMPLES = 100  # per node; well above the 50-sample CI floor from skill_dag.md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap(s: str) -> str:
    """Capitalize first letter of a string."""
    return s[0].upper() + s[1:] if s else s


def _write_jsonl(path: Path, samples: list[dict], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  [{label}] {len(samples):4d} samples → {path}")


# ---------------------------------------------------------------------------
# Modus Ponens: (A→B, A) ⊢ B
# ---------------------------------------------------------------------------
# Short eval templates: give premises + affirmed antecedent, expect consequent.
# Training data used full reasoning traces; eval tests the rule without trace format.

MP_PROMPT_TEMPLATES = [
    "If {A}, then {B}. {capA}.\nWhat can we conclude?\nAnswer: {capB}.",
    "Premise: if {A}, then {B}. Observation: {A}.\nWhat follows?\nAnswer: {capB}.",
    "Given: if {A}, then {B}. Also: {A}.\nConclusion?\nAnswer: {capB}.",
    "{capA}. If {A}, then {B}.\nWhat must be true?\nAnswer: {capB}.",
]


def generate_mp_eval(pairs: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B in pairs:
        for tmpl in MP_PROMPT_TEMPLATES:
            text = tmpl.format(A=A, B=B, capA=_cap(A), capB=_cap(B))
            samples.append({"text": text, "logic_id": "modus_ponens"})
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Modus Tollens: (A→B, ¬B) ⊢ ¬A
# ---------------------------------------------------------------------------
# Training used trace templates; eval uses direct short-form templates.

MT_PROMPT_TEMPLATES = [
    # All four templates use the same canonical answer: "It is not the case that {A}."
    # This ensures substring matching works regardless of how the model rephrases conclusions.
    "If {A}, then {B}. It is not the case that {B}.\nWhat can we conclude?\nAnswer: It is not the case that {A}.",
    "If {A}, then {B}. {capB} is not the case.\nWhat follows?\nAnswer: It is not the case that {A}.",
    "Premise: if {A} then {B}. Observation: not {B}.\nConclusion?\nAnswer: It is not the case that {A}.",
    "{capA} implies {B}. {B} does not hold.\nWhat must be true?\nAnswer: It is not the case that {A}.",
]


def generate_mt_eval(pairs: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B in pairs:
        for tmpl in MT_PROMPT_TEMPLATES:
            text = tmpl.format(A=A, B=B, capA=_cap(A), capB=_cap(B))
            samples.append({"text": text, "logic_id": "modus_tollens"})
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Disjunctive Syllogism: (A∨B, ¬A) ⊢ B
# ---------------------------------------------------------------------------

DS_PROMPT_TEMPLATES = [
    "Either {A} or {B}. It is not the case that {A}.\nWhat can we conclude?\nAnswer: {capB}.",
    "Either {A} or {B}. {capA} is not the case.\nWhat follows?\nAnswer: {capB}.",
    "Disjunction: {A} or {B}. {capA} is ruled out.\nConclusion?\nAnswer: {capB}.",
    "We know: {A} or {B}. Not {A}.\nWhat must be the case?\nAnswer: {capB}.",
]


def generate_ds_eval(pairs: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B in pairs:
        for tmpl in DS_PROMPT_TEMPLATES:
            text = tmpl.format(A=A, B=B, capA=_cap(A), capB=_cap(B))
            samples.append({"text": text, "logic_id": "disjunctive_syllogism"})
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Hypothetical Syllogism: (A→B, B→C) ⊢ A→C
# ---------------------------------------------------------------------------
# Each tuple: (domain, A, B, C) — "If A then B; if B then C; therefore if A then C."

HS_TRIPLES: list[tuple[str, str, str, str]] = [
    # science chains
    ("physics", "a spring is compressed", "it stores potential energy", "it can do work when released"),
    ("physics", "a wire carries current", "a magnetic field surrounds it", "a compass needle nearby deflects"),
    ("physics", "temperature rises", "particle kinetic energy increases", "collision frequency increases"),
    ("chemistry", "a catalyst lowers activation energy", "more molecules can react", "reaction rate increases"),
    ("chemistry", "a solution becomes acidic", "pH decreases below 7", "litmus paper turns red"),
    ("chemistry", "an element gains electrons", "it becomes negatively charged", "it is called an anion"),
    ("biology", "a cell receives a growth signal", "it enters the cell cycle", "DNA replication begins"),
    ("biology", "a predator enters a habitat", "prey population decreases", "vegetation recovers"),
    ("biology", "a plant absorbs sunlight", "photosynthesis occurs", "glucose is produced"),
    # engineering/technology chains
    ("engineering", "a software update is deployed", "existing bugs are patched", "security vulnerabilities decrease"),
    ("engineering", "voltage across a resistor doubles", "current through it doubles", "power dissipated quadruples"),
    ("engineering", "a gear ratio increases", "output speed decreases", "output torque increases"),
    # economics/social chains
    ("economics", "a central bank cuts interest rates", "borrowing becomes cheaper", "consumer spending increases"),
    ("economics", "oil prices rise", "transport costs increase", "prices of goods rise"),
    ("economics", "productivity increases", "production costs fall", "prices can be reduced"),
    # ecology chains
    ("ecology", "rainfall decreases", "soil moisture drops", "plant growth slows"),
    ("ecology", "temperature rises in a lake", "oxygen solubility decreases", "fish populations decline"),
    ("ecology", "a wildfire clears vegetation", "sunlight reaches the soil", "pioneer species establish"),
    # medicine chains
    ("medicine", "blood glucose rises sharply", "insulin is secreted", "glucose is taken up by cells"),
    ("medicine", "a virus enters a cell", "viral replication occurs", "new virions are released"),
    ("medicine", "cortisol levels stay elevated", "immune function is suppressed", "infection risk increases"),
    # astronomy chains
    ("astronomy", "a star's core runs out of hydrogen", "fusion shifts to the shell", "the outer layers expand"),
    ("astronomy", "a massive star explodes as a supernova", "heavy elements are ejected", "interstellar gas is enriched"),
    ("astronomy", "two galaxies approach each other", "gravitational tides intensify", "star formation bursts occur"),
    # law/logic chains
    ("law", "a contract is signed under duress", "it lacks free consent", "it may be voided by a court"),
    ("law", "evidence is improperly obtained", "it violates due process", "a judge may exclude it"),
    ("law", "a patent expires", "the invention enters the public domain", "anyone can manufacture it"),
    # environment chains
    ("environment", "carbon dioxide concentrations rise", "more heat is trapped in the atmosphere", "average temperatures increase"),
    ("environment", "glaciers melt", "freshwater runoff increases", "ocean salinity decreases near coasts"),
]

HS_PROMPT_TEMPLATES = [
    "If {A}, then {B}. If {B}, then {C}.\nWhat can we conclude about the relationship between {A} and {C}?\nAnswer: If {A}, then {C}.",
    "Premise 1: if {A}, then {B}. Premise 2: if {B}, then {C}.\nWhat follows?\nAnswer: If {A}, then {C}.",
    "Given: {A} implies {B}, and {B} implies {C}.\nWhat is the combined conclusion?\nAnswer: {capA} implies {C}.",
    "If {A} then {B}. If {B} then {C}. {capA} is the case.\nWhat follows?\nAnswer: {capC}.",
]


def generate_hs_eval(triples: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, A, B, C in triples:
        for tmpl in HS_PROMPT_TEMPLATES:
            text = tmpl.format(A=A, B=B, C=C, capA=_cap(A), capB=_cap(B), capC=_cap(C))
            samples.append({"text": text, "logic_id": "hypothetical_syllogism"})
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Universal Instantiation: (∀x P(x), a ∈ domain) ⊢ P(a)
# ---------------------------------------------------------------------------
# Reuse UI_PAIRS from generate_curriculum_data.py but with different seed and templates.

sys.path.insert(0, str(Path(__file__).parent))
from generate_curriculum_data import UI_PAIRS  # noqa: E402

UI_EVAL_TEMPLATES = [
    "{U}. {S}.\nWhat can we conclude?\nAnswer: {C}.",
    "Universal rule: {U}. Specific case: {S}.\nConclusion?\nAnswer: {C}.",
    "{U}. Given that {S}, what follows?\nAnswer: {C}.",
    "Apply universal instantiation: {U}. {S}.\nWhat must be true?\nAnswer: {C}.",
]


def generate_ui_eval(pairs: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for domain, U, S, C in pairs:
        for tmpl in UI_EVAL_TEMPLATES:
            text = tmpl.format(U=U, S=S, C=C)
            samples.append({"text": text, "logic_id": "universal_instantiation"})
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Rule Recognition: given (premises, conclusion), identify the inference rule.
# ---------------------------------------------------------------------------

def generate_rule_recognition_eval(pairs: list, seed: int, n: int) -> list[dict]:
    """Generate problems asking which inference rule was applied.

    Three rule types, roughly equal: MP (affirming antecedent → consequent),
    MT (denying consequent → deny antecedent), DS (disjunction minus one → other).
    Expected answers are the rule names.
    """
    rng = random.Random(seed)
    samples = []

    mp_tmpl = (
        "Argument: If {A}, then {B}. {capA}. Therefore, {B}.\n"
        "Which inference rule was applied?\nAnswer: modus ponens"
    )
    mt_tmpl = (
        "Argument: If {A}, then {B}. It is not the case that {B}. Therefore, it is not the case that {A}.\n"
        "Which inference rule was applied?\nAnswer: modus tollens"
    )
    ds_tmpl = (
        "Argument: Either {A} or {B}. It is not the case that {A}. Therefore, {B}.\n"
        "Which inference rule was applied?\nAnswer: disjunctive syllogism"
    )

    for domain, A, B in pairs:
        samples.append({"text": mp_tmpl.format(A=A, B=B, capA=_cap(A)), "logic_id": "rule_recognition"})
        samples.append({"text": mt_tmpl.format(A=A, B=B, capA=_cap(A)), "logic_id": "rule_recognition"})
        samples.append({"text": ds_tmpl.format(A=A, B=B, capA=_cap(A)), "logic_id": "rule_recognition"})

    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Concise Reasoning: apply inference rules with minimal scaffolding.
# ---------------------------------------------------------------------------
# Tests same skills as rule_recognition but in production form:
# short prompts, expect answer-only (no trace).

def generate_concise_eval(pairs: list, seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []

    mp_t = "If {A}, then {B}. {capA}.\nAnswer: {capB}."
    mt_t = "If {A}, then {B}. Not {B}.\nAnswer: It is not the case that {A}."
    ds_t = "{capA} or {B}. Not {A}.\nAnswer: {capB}."

    for domain, A, B in pairs:
        samples.append({"text": mp_t.format(A=A, B=B, capA=_cap(A), capB=_cap(B)), "logic_id": "concise_reasoning"})
        samples.append({"text": mt_t.format(A=A, B=B, capA=_cap(A)), "logic_id": "concise_reasoning"})
        samples.append({"text": ds_t.format(A=A, B=B, capA=_cap(A), capB=_cap(B)), "logic_id": "concise_reasoning"})

    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Chain Reasoning: multi-step deductions combining ≥3 rules.
# ---------------------------------------------------------------------------
# Each eval problem chains ≥3 inferences. Two main patterns:
#   1. HS + MP: A→B, B→C, A ⊢ C  (chain application)
#   2. HS + MT: A→B, B→C, ¬C ⊢ ¬A  (chain contrapositive)
# Uses HS_TRIPLES defined above.

CHAIN_TEMPLATES = [
    # chain application (HS then MP)
    "If {A}, then {B}. If {B}, then {C}. {capA}.\nWhat can we conclude?\nAnswer: {capC}.",
    # chain contrapositive (HS then MT)
    "If {A}, then {B}. If {B}, then {C}. It is not the case that {C}.\nWhat can we conclude?\nAnswer: It is not the case that {A}.",
    # three-step with DS start
    "Either {A} or {B}. It is not the case that {A}. If {B}, then {C}.\nWhat can we conclude?\nAnswer: {capC}.",
    # mixed: DS + MT
    "Either {A} or {B}. It is not the case that {A}. If {B}, then {C}. It is not the case that {C}.\n"
    "What can we conclude?\nAnswer: It is not the case that {B}.",
]


def generate_chain_eval(triples: list, pairs: list, seed: int, n: int) -> list[dict]:
    """Generate multi-step chaining eval problems.

    Uses HS_TRIPLES for templates 1 and 2; uses PREMISE_PAIRS for templates 3 and 4
    (pairing a DS problem with a follow-on conditional).
    """
    rng = random.Random(seed)
    samples = []

    # Templates 1 and 2: chain via HS_TRIPLES
    for domain, A, B, C in triples:
        t1 = CHAIN_TEMPLATES[0].format(A=A, B=B, C=C, capA=_cap(A), capC=_cap(C))
        t2 = CHAIN_TEMPLATES[1].format(A=A, B=B, C=C, capA=_cap(A))
        samples.append({"text": t1, "logic_id": "chain_reasoning"})
        samples.append({"text": t2, "logic_id": "chain_reasoning"})

    # Templates 3 and 4: DS start, then chained conditional
    # Pair consecutive pairs from same domain as the "follow-on" B→C
    pairs_by_domain: dict[str, list] = {}
    for domain, A, B in pairs:
        pairs_by_domain.setdefault(domain, []).append((A, B))

    for domain, domain_pairs in pairs_by_domain.items():
        if len(domain_pairs) >= 2:
            for i in range(len(domain_pairs) - 1):
                A, _B = domain_pairs[i]
                B_alt, C = domain_pairs[i + 1]
                t3 = CHAIN_TEMPLATES[2].format(A=A, B=B_alt, C=C, capA=_cap(A), capC=_cap(C))
                t4 = CHAIN_TEMPLATES[3].format(A=A, B=B_alt, C=C, capA=_cap(A))
                samples.append({"text": t3, "logic_id": "chain_reasoning"})
                samples.append({"text": t4, "logic_id": "chain_reasoning"})

    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Arithmetic: addition, multiplication, division
# ---------------------------------------------------------------------------

def generate_add_eval(seed: int, n: int) -> list[dict]:
    """Integer addition eval. Wider range (1-99) than training (1-25)."""
    rng = random.Random(seed)
    samples = []
    seen: set[tuple[int, int]] = set()
    candidates = []
    for a in range(1, 100):
        for b in range(1, 100):
            pair = (min(a, b), max(a, b))
            if pair not in seen:
                seen.add(pair)
                candidates.append((a, b))
    rng.shuffle(candidates)
    for a, b in candidates[:n * 2]:
        c = a + b
        samples.append({"text": f"What is {a} + {b}?\nAnswer: {c}", "logic_id": "arithmetic_add"})
    rng.shuffle(samples)
    return samples[:n]


def generate_mul_eval(seed: int, n: int) -> list[dict]:
    """Integer multiplication eval (1-12 × 1-12 = multiplication table)."""
    rng = random.Random(seed)
    samples = []
    for a in range(1, 13):
        for b in range(1, 13):
            c = a * b
            samples.append({"text": f"What is {a} × {b}?\nAnswer: {c}", "logic_id": "arithmetic_multiply"})
    rng.shuffle(samples)
    return samples[:n]


def generate_div_eval(seed: int, n: int) -> list[dict]:
    """Integer division eval (exact division only)."""
    rng = random.Random(seed)
    samples = []
    for divisor in range(2, 20):
        for quotient in range(2, 15):
            dividend = divisor * quotient
            samples.append({
                "text": f"What is {dividend} ÷ {divisor}?\nAnswer: {quotient}",
                "logic_id": "arithmetic_divide",
            })
    rng.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curriculum eval data")
    parser.add_argument("--output", default="data/eval", help="Output directory (default: data/eval)")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="Random seed (default: 99)")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help="Samples per node (default: 100)")
    args = parser.parse_args()

    out = Path(args.output)
    seed = args.seed
    n = args.n

    print(f"Generating {n} eval samples per node → {out}/")
    print(f"Using PREMISE_PAIRS: {len(PREMISE_PAIRS)} pairs, seed={seed}\n")

    # ── Logic nodes ───────────────────────────────────────────────────────
    _write_jsonl(out / "modus_ponens_eval.jsonl",
                 generate_mp_eval(PREMISE_PAIRS, seed, n), "modus_ponens")

    _write_jsonl(out / "modus_tollens_eval.jsonl",
                 generate_mt_eval(PREMISE_PAIRS, seed + 1, n), "modus_tollens")

    _write_jsonl(out / "disjunctive_syllogism_eval.jsonl",
                 generate_ds_eval(PREMISE_PAIRS, seed + 2, n), "disjunctive_syllogism")

    _write_jsonl(out / "hypothetical_syllogism_eval.jsonl",
                 generate_hs_eval(HS_TRIPLES, seed + 3, n), "hypothetical_syllogism")

    _write_jsonl(out / "universal_instantiation_eval.jsonl",
                 generate_ui_eval(UI_PAIRS, seed + 4, n), "universal_instantiation")

    _write_jsonl(out / "rule_recognition_eval.jsonl",
                 generate_rule_recognition_eval(PREMISE_PAIRS, seed + 5, n), "rule_recognition")

    _write_jsonl(out / "concise_reasoning_eval.jsonl",
                 generate_concise_eval(PREMISE_PAIRS, seed + 6, n), "concise_reasoning")

    _write_jsonl(out / "chain_reasoning_eval.jsonl",
                 generate_chain_eval(HS_TRIPLES, PREMISE_PAIRS, seed + 7, n), "chain_reasoning")

    # ── Math nodes ────────────────────────────────────────────────────────
    _write_jsonl(out / "arithmetic_add_eval.jsonl",
                 generate_add_eval(seed + 8, n), "arithmetic_add")

    _write_jsonl(out / "arithmetic_multiply_eval.jsonl",
                 generate_mul_eval(seed + 9, n), "arithmetic_multiply")

    _write_jsonl(out / "arithmetic_divide_eval.jsonl",
                 generate_div_eval(seed + 10, n), "arithmetic_divide")

    print(f"\nDone. {11} eval files written.")
    print("Next: install datasets package and run profile_gsm8k_difficulty.py")
    print("  for word_problem_1step and word_problem_multi eval files.")


if __name__ == "__main__":
    main()
