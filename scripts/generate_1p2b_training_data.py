#!/usr/bin/env python3
"""Generate training data for LFM2-1.2B geometric optimization.

Combines existing data with newly generated reasoning traces to create
a comprehensive ~3000 sample training set.

Existing data sources:
- format_augmented_train.jsonl (990) — modus ponens/tollens, diverse formats
- expansion_train.jsonl (2000) — mathematical reasoning with verification
- ce_reasoning_traces_train.jsonl (330) — CE reasoning traces

New data to generate:
- Hypothetical syllogism (if A->B and B->C, then A->C)
- Disjunctive syllogism (A or B, not A, therefore B)
- Universal/existential quantifier reasoning
- More format templates per logic type
- Chain reasoning (multi-step)

Output: data/training/1p2b_reasoning_foundation_{train,val}.jsonl

Usage:
    poetry run python scripts/generate_1p2b_training_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path("data/training")


# ── Template Library ──

HYPOTHETICAL_SYLLOGISM_TEMPLATES = [
    {
        "premise1": "If {A}, then {B}.",
        "premise2": "If {B}, then {C}.",
        "observation": "{A_true}.",
        "conclusion": "{C_true}.",
        "trace": (
            "We have two rules:\n"
            "Rule 1: {A} implies {B}.\n"
            "Rule 2: {B} implies {C}.\n"
            "Observation: {A_true}.\n"
            "By Rule 1, since {A_true}, it follows that {B_true}.\n"
            "By Rule 2, since {B_true}, it follows that {C_true}.\n"
            "Conclusion: {C_true}."
        ),
    },
    {
        "premise1": "Whenever {A}, {B}.",
        "premise2": "Whenever {B}, {C}.",
        "observation": "It is the case that {A_lower}.",
        "conclusion": "Therefore, {C_lower}.",
        "trace": (
            "From the first rule, {A_lower} implies {B_lower}.\n"
            "From the second rule, {B_lower} implies {C_lower}.\n"
            "Since {A_lower} is the case, {B_lower} follows.\n"
            "Since {B_lower} is the case, {C_lower} follows.\n"
            "Conclusion: {C_lower}."
        ),
    },
]

DISJUNCTIVE_SYLLOGISM_TEMPLATES = [
    {
        "disjunction": "Either {A} or {B}.",
        "negation": "It is not the case that {A_lower}.",
        "conclusion": "Therefore, {B_lower}.",
        "trace": (
            "We know: {A} or {B} (at least one must hold).\n"
            "We observe: it is not the case that {A_lower}.\n"
            "Since one must hold and {A_lower} does not, {B_lower} must be the case.\n"
            "Conclusion: {B_lower}."
        ),
    },
    {
        "disjunction": "{A} or {B}. One of these must be true.",
        "negation": "{A} is false.",
        "conclusion": "What follows?\n{B}.",
        "trace": (
            "Given the disjunction: {A} or {B}.\n"
            "Given the negation: {A} is false.\n"
            "By disjunctive syllogism, if one disjunct is false, the other must be true.\n"
            "Conclusion: {B}."
        ),
    },
]

UNIVERSAL_QUANTIFIER_TEMPLATES = [
    {
        "universal": "Every {category} has property {P}.",
        "instance": "{X} is a {category_lower}.",
        "conclusion": "Therefore, {X} has property {P_lower}.",
        "trace": (
            "Universal rule: all members of {category_lower} have {P_lower}.\n"
            "{X} is a member of {category_lower}.\n"
            "By universal instantiation, {X} has {P_lower}.\n"
            "Conclusion: {X} has property {P_lower}."
        ),
    },
]

# ── Content Library ──

CHAIN_SCENARIOS = [
    {
        "A": "it rains", "A_true": "It rains", "A_lower": "it rains",
        "B": "the streets get wet", "B_true": "The streets get wet", "B_lower": "the streets get wet",
        "C": "cars slow down", "C_true": "Cars slow down", "C_lower": "cars slow down",
    },
    {
        "A": "the temperature drops below freezing", "A_true": "The temperature drops below freezing", "A_lower": "the temperature drops below freezing",
        "B": "water turns to ice", "B_true": "Water turns to ice", "B_lower": "water turns to ice",
        "C": "the pipes may burst", "C_true": "The pipes may burst", "C_lower": "the pipes may burst",
    },
    {
        "A": "a student studies consistently", "A_true": "A student studies consistently", "A_lower": "a student studies consistently",
        "B": "they understand the material", "B_true": "They understand the material", "B_lower": "they understand the material",
        "C": "they perform well on the exam", "C_true": "They perform well on the exam", "C_lower": "they perform well on the exam",
    },
    {
        "A": "the soil is fertile", "A_true": "The soil is fertile", "A_lower": "the soil is fertile",
        "B": "crops grow well", "B_true": "Crops grow well", "B_lower": "crops grow well",
        "C": "the harvest is abundant", "C_true": "The harvest is abundant", "C_lower": "the harvest is abundant",
    },
    {
        "A": "there is a power outage", "A_true": "There is a power outage", "A_lower": "there is a power outage",
        "B": "the servers go offline", "B_true": "The servers go offline", "B_lower": "the servers go offline",
        "C": "the website becomes inaccessible", "C_true": "The website becomes inaccessible", "C_lower": "the website becomes inaccessible",
    },
    {
        "A": "exercise is performed regularly", "A_true": "Exercise is performed regularly", "A_lower": "exercise is performed regularly",
        "B": "cardiovascular health improves", "B_true": "Cardiovascular health improves", "B_lower": "cardiovascular health improves",
        "C": "the risk of heart disease decreases", "C_true": "The risk of heart disease decreases", "C_lower": "the risk of heart disease decreases",
    },
    {
        "A": "the code is well-tested", "A_true": "The code is well-tested", "A_lower": "the code is well-tested",
        "B": "bugs are caught early", "B_true": "Bugs are caught early", "B_lower": "bugs are caught early",
        "C": "the software is reliable", "C_true": "The software is reliable", "C_lower": "the software is reliable",
    },
    {
        "A": "deforestation continues", "A_true": "Deforestation continues", "A_lower": "deforestation continues",
        "B": "habitats are destroyed", "B_true": "Habitats are destroyed", "B_lower": "habitats are destroyed",
        "C": "species go extinct", "C_true": "Species go extinct", "C_lower": "species go extinct",
    },
]

DISJUNCTIVE_SCENARIOS = [
    {"A": "The package was delivered", "A_lower": "the package was delivered",
     "B": "The package was returned to sender", "B_lower": "the package was returned to sender"},
    {"A": "The alarm was triggered by an intruder", "A_lower": "the alarm was triggered by an intruder",
     "B": "The alarm malfunctioned", "B_lower": "the alarm malfunctioned"},
    {"A": "The flight was canceled due to weather", "A_lower": "the flight was canceled due to weather",
     "B": "The flight was canceled due to a mechanical issue", "B_lower": "the flight was canceled due to a mechanical issue"},
    {"A": "The candidate has a PhD", "A_lower": "the candidate has a PhD",
     "B": "The candidate has 10 years of experience", "B_lower": "the candidate has 10 years of experience"},
    {"A": "The error is in the frontend code", "A_lower": "the error is in the frontend code",
     "B": "The error is in the backend code", "B_lower": "the error is in the backend code"},
    {"A": "The signal is analog", "A_lower": "the signal is analog",
     "B": "The signal is digital", "B_lower": "the signal is digital"},
]

UNIVERSAL_SCENARIOS = [
    {"category": "Mammal", "category_lower": "mammal", "P": "warm blood", "P_lower": "warm blood", "X": "A dolphin"},
    {"category": "Planet in our solar system", "category_lower": "planet in our solar system", "P": "orbiting the Sun", "P_lower": "orbiting the Sun", "X": "Mars"},
    {"category": "Even number", "category_lower": "even number", "P": "divisibility by 2", "P_lower": "divisibility by 2", "X": "The number 14"},
    {"category": "Rectangle", "category_lower": "rectangle", "P": "four right angles", "P_lower": "four right angles", "X": "A square"},
    {"category": "Citizen over 18", "category_lower": "citizen over 18", "P": "the right to vote", "P_lower": "the right to vote", "X": "A 25-year-old citizen"},
    {"category": "Conductor", "category_lower": "conductor", "P": "allowing electric current to flow", "P_lower": "allowing electric current to flow", "X": "Copper wire"},
]

QUESTION_HEADERS = [
    "Question: ", "Consider the following:\n", "", "Problem:\n",
    "Analyze this:\n", "Given:\n", "Premise:\n", "Reasoning task:\n",
]

CONCLUSION_HEADERS = [
    "What can we conclude?", "What follows?", "What is the conclusion?",
    "Determine the result.", "What must be true?", "What can be deduced?",
]


def generate_hypothetical_syllogism() -> list[dict]:
    examples = []
    for scenario in CHAIN_SCENARIOS:
        for template in HYPOTHETICAL_SYLLOGISM_TEMPLATES:
            for qh in random.sample(QUESTION_HEADERS, min(3, len(QUESTION_HEADERS))):
                for ch in random.sample(CONCLUSION_HEADERS, min(2, len(CONCLUSION_HEADERS))):
                    text = (
                        f"{qh}"
                        f"{template['premise1'].format(**scenario)} "
                        f"{template['premise2'].format(**scenario)} "
                        f"{template['observation'].format(**scenario)}\n"
                        f"{ch}\n"
                        f"{template['trace'].format(**scenario)}"
                    )
                    examples.append({"text": text})
    return examples


def generate_disjunctive_syllogism() -> list[dict]:
    examples = []
    for scenario in DISJUNCTIVE_SCENARIOS:
        for template in DISJUNCTIVE_SYLLOGISM_TEMPLATES:
            for qh in random.sample(QUESTION_HEADERS, min(4, len(QUESTION_HEADERS))):
                text = (
                    f"{qh}"
                    f"{template['disjunction'].format(**scenario)} "
                    f"{template['negation'].format(**scenario)}\n"
                    f"{template['conclusion'].format(**scenario)}\n"
                    f"{template['trace'].format(**scenario)}"
                )
                examples.append({"text": text})
    return examples


def generate_universal_quantifier() -> list[dict]:
    examples = []
    for scenario in UNIVERSAL_SCENARIOS:
        for template in UNIVERSAL_QUANTIFIER_TEMPLATES:
            for qh in random.sample(QUESTION_HEADERS, min(4, len(QUESTION_HEADERS))):
                text = (
                    f"{qh}"
                    f"{template['universal'].format(**scenario)} "
                    f"{template['instance'].format(**scenario)}\n"
                    f"{template['conclusion'].format(**scenario)}\n"
                    f"{template['trace'].format(**scenario)}"
                )
                examples.append({"text": text})
    return examples


def load_existing(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Generating 1.2B training data...")

    # Load existing data
    format_augmented = load_existing("format_augmented_train.jsonl")
    expansion = load_existing("expansion_train.jsonl")
    ce_traces = load_existing("ce_reasoning_traces_train.jsonl")

    print(f"  Existing format_augmented: {len(format_augmented)}")
    print(f"  Existing expansion (math): {len(expansion)}")
    print(f"  Existing CE traces: {len(ce_traces)}")

    # Generate new reasoning data
    hyp_syl = generate_hypothetical_syllogism()
    disj_syl = generate_disjunctive_syllogism()
    univ_quant = generate_universal_quantifier()

    print(f"  New hypothetical syllogism: {len(hyp_syl)}")
    print(f"  New disjunctive syllogism: {len(disj_syl)}")
    print(f"  New universal quantifier: {len(univ_quant)}")

    # Combine all reasoning (format_augmented + new logic types)
    all_reasoning = format_augmented + hyp_syl + disj_syl + univ_quant

    # Combine with math (sample to balance)
    # Target: ~2000 reasoning + ~1000 math = ~3000 total
    math_sample = random.sample(expansion, min(1000, len(expansion)))

    all_data = all_reasoning + math_sample + ce_traces
    random.shuffle(all_data)

    print(f"  Total combined: {len(all_data)}")

    # 80/20 split
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Write output
    train_path = DATA_DIR / "1p2b_reasoning_foundation_train.jsonl"
    val_path = DATA_DIR / "1p2b_reasoning_foundation_val.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print("\nOutput:")
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Val:   {val_path} ({len(val_data)} samples)")


if __name__ == "__main__":
    main()
