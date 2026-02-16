#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Generate paired reasoning data for constrained geometric training.

Each logical form gets multiple surface templates. This creates:
- Invariance pairs: same logic, different template
- Counterfactual pairs: same template, different logic

The constraint optimizer uses these pairs to force the model to encode
relational structure rather than surface form.

Usage:
    python scripts/generate_paired_data.py \
        --output data/training/paired_reasoning_train.jsonl \
        --val-output data/training/paired_reasoning_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PairedSample:
    text: str
    answer_start: str
    logic_id: str
    template_id: str
    pair_type: str  # anchor | invariance | counterfactual


# =============================================================================
# Template definitions
# =============================================================================

# Each logic form is a function that takes a template dict and returns
# (full_text, answer_text). Templates provide variable names and contexts.


def _modus_ponens(t: dict) -> tuple[str, str]:
    """If P then Q. P is true. Therefore Q."""
    premise = f"If {t['P']}, then {t['Q']}. {t['P_true']}."
    answer = f"Therefore, {t['Q_conclusion']}."
    return f"{premise}\n{answer}", answer


def _modus_tollens(t: dict) -> tuple[str, str]:
    """If P then Q. Not Q. Therefore not P."""
    premise = f"If {t['P']}, then {t['Q']}. {t['not_Q']}."
    answer = f"Therefore, {t['not_P']}."
    return f"{premise}\n{answer}", answer


def _disjunctive_syllogism(t: dict) -> tuple[str, str]:
    """P or Q. Not P. Therefore Q."""
    premise = f"Either {t['P']} or {t['Q']}. {t['not_P']}."
    answer = f"Therefore, {t['Q_conclusion']}."
    return f"{premise}\n{answer}", answer


def _hypothetical_syllogism(t: dict) -> tuple[str, str]:
    """If P then Q. If Q then R. Therefore if P then R."""
    premise = f"If {t['P']}, then {t['Q']}. If {t['Q']}, then {t['R']}."
    answer = f"Therefore, if {t['P']}, then {t['R']}."
    return f"{premise}\n{answer}", answer


def _chain_contrapositive(t: dict) -> tuple[str, str]:
    """If A then B, if B then C. Not C. Therefore not A and not B."""
    premise = (
        f"If {t['A']}, then {t['B']}. "
        f"If {t['B']}, then {t['C']}. "
        f"{t['not_C']}."
    )
    answer = f"Therefore, {t['not_A']} and {t['not_B']}."
    return f"{premise}\n{answer}", answer


def _conjunction_elimination(t: dict) -> tuple[str, str]:
    """P and Q. Therefore P. Therefore Q."""
    premise = f"{t['P']} and {t['Q']}."
    answer = f"We can conclude {t['P_alone']}. We can also conclude {t['Q_alone']}."
    return f"{premise}\n{answer}", answer


def _affirming_consequent_fallacy(t: dict) -> tuple[str, str]:
    """If P then Q. Q is true. CANNOT conclude P (fallacy)."""
    premise = f"If {t['P']}, then {t['Q']}. {t['Q_true']}."
    answer = f"We cannot conclude that {t['P_conclusion']}. This is the fallacy of affirming the consequent."
    return f"{premise}\n{answer}", answer


def _denying_antecedent_fallacy(t: dict) -> tuple[str, str]:
    """If P then Q. Not P. CANNOT conclude not Q (fallacy)."""
    premise = f"If {t['P']}, then {t['Q']}. {t['not_P']}."
    answer = f"We cannot conclude that {t['not_Q']}. This is the fallacy of denying the antecedent."
    return f"{premise}\n{answer}", answer


def _biconditional(t: dict) -> tuple[str, str]:
    """P if and only if Q. P is true. Therefore Q."""
    premise = f"{t['P']} if and only if {t['Q']}. {t['P_true']}."
    answer = f"Therefore, {t['Q_conclusion']}."
    return f"{premise}\n{answer}", answer


def _disjunction_intro(t: dict) -> tuple[str, str]:
    """P is true. Therefore P or Q."""
    premise = f"{t['P_true']}."
    answer = f"Therefore, {t['P']} or {t['Q']}."
    return f"{premise}\n{answer}", answer


# =============================================================================
# Templates per logic form
# =============================================================================

MODUS_PONENS_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "P_true": "A is true", "Q_conclusion": "B is true",
    },
    {
        "id": "rain",
        "P": "it is raining", "Q": "the ground is wet",
        "P_true": "It is raining", "Q_conclusion": "the ground is wet",
    },
    {
        "id": "study",
        "P": "you study hard", "Q": "you will pass the exam",
        "P_true": "You studied hard", "Q_conclusion": "you will pass the exam",
    },
    {
        "id": "fire",
        "P": "there is fire", "Q": "there is smoke",
        "P_true": "There is fire", "Q_conclusion": "there is smoke",
    },
    {
        "id": "citizen",
        "P": "someone is born in France", "Q": "they are a French citizen",
        "P_true": "Marie was born in France",
        "Q_conclusion": "Marie is a French citizen",
    },
]

MODUS_TOLLENS_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "not_Q": "B is not true", "not_P": "A is not true",
    },
    {
        "id": "rain",
        "P": "it is raining", "Q": "the ground is wet",
        "not_Q": "The ground is not wet", "not_P": "it is not raining",
    },
    {
        "id": "study",
        "P": "you study hard", "Q": "you will pass the exam",
        "not_Q": "You did not pass the exam", "not_P": "you did not study hard",
    },
    {
        "id": "battery",
        "P": "the battery is charged", "Q": "the device turns on",
        "not_Q": "The device does not turn on",
        "not_P": "the battery is not charged",
    },
    {
        "id": "mammal",
        "P": "an animal is a mammal", "Q": "it has a backbone",
        "not_Q": "The animal does not have a backbone",
        "not_P": "the animal is not a mammal",
    },
]

DISJUNCTIVE_SYLLOGISM_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "not_P": "A is not true", "Q_conclusion": "B is true",
    },
    {
        "id": "transport",
        "P": "she took the bus", "Q": "she walked",
        "not_P": "She did not take the bus", "Q_conclusion": "she walked",
    },
    {
        "id": "meal",
        "P": "we eat pizza", "Q": "we eat pasta",
        "not_P": "We are not eating pizza", "Q_conclusion": "we eat pasta",
    },
    {
        "id": "season",
        "P": "it is summer", "Q": "it is winter",
        "not_P": "It is not summer", "Q_conclusion": "it is winter",
    },
]

HYPOTHETICAL_SYLLOGISM_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true", "R": "C is true",
    },
    {
        "id": "weather",
        "P": "it rains", "Q": "the streets flood", "R": "traffic stops",
    },
    {
        "id": "economy",
        "P": "interest rates rise", "Q": "borrowing decreases",
        "R": "economic growth slows",
    },
    {
        "id": "health",
        "P": "you exercise regularly", "Q": "your cardiovascular health improves",
        "R": "your lifespan increases",
    },
]

CHAIN_CONTRAPOSITIVE_TEMPLATES = [
    {
        "id": "abstract",
        "A": "X is true", "B": "Y is true", "C": "Z is true",
        "not_C": "Z is not true", "not_A": "X is not true", "not_B": "Y is not true",
    },
    {
        "id": "weather",
        "A": "it rains", "B": "the ground gets wet", "C": "plants grow",
        "not_C": "Plants are not growing",
        "not_A": "it is not raining", "not_B": "the ground is not wet",
    },
    {
        "id": "academic",
        "A": "the student attends class", "B": "the student learns the material",
        "C": "the student passes the test",
        "not_C": "The student did not pass the test",
        "not_A": "the student did not attend class",
        "not_B": "the student did not learn the material",
    },
]

CONJUNCTION_ELIMINATION_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "P_alone": "A is true", "Q_alone": "B is true",
    },
    {
        "id": "weather",
        "P": "it is cold", "Q": "it is windy",
        "P_alone": "it is cold", "Q_alone": "it is windy",
    },
    {
        "id": "person",
        "P": "the cat is black", "Q": "the cat is small",
        "P_alone": "the cat is black", "Q_alone": "the cat is small",
    },
]

AFFIRMING_CONSEQUENT_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "Q_true": "B is true", "P_conclusion": "A is true",
    },
    {
        "id": "rain",
        "P": "it is raining", "Q": "the ground is wet",
        "Q_true": "The ground is wet",
        "P_conclusion": "it is raining",
    },
    {
        "id": "study",
        "P": "you study hard", "Q": "you pass the exam",
        "Q_true": "You passed the exam",
        "P_conclusion": "you studied hard",
    },
    {
        "id": "fire",
        "P": "there is fire", "Q": "there is smoke",
        "Q_true": "There is smoke",
        "P_conclusion": "there is fire",
    },
]

DENYING_ANTECEDENT_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "not_P": "A is not true", "not_Q": "B is not true",
    },
    {
        "id": "rain",
        "P": "it is raining", "Q": "the ground is wet",
        "not_P": "It is not raining", "not_Q": "the ground is not wet",
    },
    {
        "id": "exercise",
        "P": "you exercise daily", "Q": "you are healthy",
        "not_P": "You do not exercise daily", "not_Q": "you are not healthy",
    },
]

BICONDITIONAL_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "P_true": "A is true", "Q_conclusion": "B is true",
    },
    {
        "id": "geometry",
        "P": "a shape is a square", "Q": "it has four equal sides and four right angles",
        "P_true": "The shape is a square",
        "Q_conclusion": "it has four equal sides and four right angles",
    },
    {
        "id": "legal",
        "P": "someone is eligible to vote", "Q": "they are a citizen over 18",
        "P_true": "John is eligible to vote",
        "Q_conclusion": "John is a citizen over 18",
    },
]

DISJUNCTION_INTRO_TEMPLATES = [
    {
        "id": "abstract",
        "P": "A is true", "Q": "B is true",
        "P_true": "A is true",
    },
    {
        "id": "weather",
        "P": "it is sunny", "Q": "it is cloudy",
        "P_true": "It is sunny",
    },
    {
        "id": "animal",
        "P": "the animal is a dog", "Q": "the animal is a cat",
        "P_true": "The animal is a dog",
    },
]


# =============================================================================
# Logic form registry
# =============================================================================

LOGIC_FORMS: list[tuple[str, callable, list[dict]]] = [
    ("modus_ponens", _modus_ponens, MODUS_PONENS_TEMPLATES),
    ("modus_tollens", _modus_tollens, MODUS_TOLLENS_TEMPLATES),
    ("disjunctive_syllogism", _disjunctive_syllogism, DISJUNCTIVE_SYLLOGISM_TEMPLATES),
    ("hypothetical_syllogism", _hypothetical_syllogism, HYPOTHETICAL_SYLLOGISM_TEMPLATES),
    ("chain_contrapositive", _chain_contrapositive, CHAIN_CONTRAPOSITIVE_TEMPLATES),
    ("conjunction_elimination", _conjunction_elimination, CONJUNCTION_ELIMINATION_TEMPLATES),
    ("affirming_consequent_fallacy", _affirming_consequent_fallacy, AFFIRMING_CONSEQUENT_TEMPLATES),
    ("denying_antecedent_fallacy", _denying_antecedent_fallacy, DENYING_ANTECEDENT_TEMPLATES),
    ("biconditional", _biconditional, BICONDITIONAL_TEMPLATES),
    ("disjunction_intro", _disjunction_intro, DISJUNCTION_INTRO_TEMPLATES),
]


def generate_samples() -> list[PairedSample]:
    """Generate all paired samples from logic forms × templates."""
    samples: list[PairedSample] = []

    # For each logic form, generate one sample per template
    for logic_id, fn, templates in LOGIC_FORMS:
        for t in templates:
            text, answer = fn(t)
            template_id = t["id"]

            samples.append(PairedSample(
                text=text,
                answer_start=answer,
                logic_id=logic_id,
                template_id=template_id,
                pair_type="anchor",
            ))

    # Now assign pair_types based on relationships.
    # For invariance: same logic_id, different template_id
    # For counterfactual: same template_id, different logic_id
    # We mark additional copies with appropriate pair_type.
    #
    # Strategy: for each anchor, find its invariance partners (same logic, diff template)
    # and counterfactual partners (same template, diff logic).
    # The anchor itself stays "anchor". Partners get "invariance" or "counterfactual".
    # Since every sample is already generated, we just need to label them correctly.
    #
    # Actually — every sample serves as anchor for its own (logic, template) combination,
    # AND as invariance/counterfactual partner for other combinations. The pair_type
    # field is about how the sample relates within a batch. The batch sampler
    # will use logic_id and template_id to construct pairs dynamically.
    #
    # So we keep all samples as "anchor" — the pairing is implicit in
    # (logic_id, template_id) and the batch sampler constructs pairs.

    return samples


def split_train_val(
    samples: list[PairedSample], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[PairedSample], list[PairedSample]]:
    """Split by template_id, stratified by logic_id.

    All templates of a given ID go to either train or val, not split across.
    This ensures the model can't memorize template-specific patterns from training.

    Additionally, every logic_id that appears in the dataset will have at
    least one sample in BOTH train and val (logic stratification). This
    ensures val can measure all constraint types the model trains on.
    """
    rng = random.Random(seed)

    # Collect unique template IDs
    template_ids = sorted(set(s.template_id for s in samples))
    rng.shuffle(template_ids)

    n_val = max(1, int(len(template_ids) * val_fraction))
    val_templates = set(template_ids[:n_val])

    train = [s for s in samples if s.template_id not in val_templates]
    val = [s for s in samples if s.template_id in val_templates]

    # Check logic_id coverage: every logic_id should appear in both splits
    train_logic = set(s.logic_id for s in train)
    val_logic = set(s.logic_id for s in val)
    all_logic = train_logic | val_logic
    missing_in_val = train_logic - val_logic
    missing_in_train = val_logic - train_logic

    # If any logic_id is missing from val, move one sample per missing
    # logic_id from train to val (pick the shortest to minimize data loss)
    if missing_in_val:
        for lid in missing_in_val:
            candidates = [s for s in train if s.logic_id == lid]
            if candidates:
                # Move shortest sample to preserve training data
                donor = min(candidates, key=lambda s: len(s.text))
                train.remove(donor)
                val.append(donor)

    # If any logic_id is missing from train, move one sample back
    if missing_in_train:
        for lid in missing_in_train:
            candidates = [s for s in val if s.logic_id == lid]
            if candidates:
                donor = min(candidates, key=lambda s: len(s.text))
                val.remove(donor)
                train.append(donor)

    return train, val


def write_jsonl(samples: list[PairedSample], path: Path) -> None:
    """Write samples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def print_stats(samples: list[PairedSample], label: str) -> None:
    """Print dataset statistics."""
    logic_ids = set(s.logic_id for s in samples)
    template_ids = set(s.template_id for s in samples)

    # Count invariance pairs (same logic, different template)
    from collections import Counter
    logic_counts = Counter(s.logic_id for s in samples)
    template_counts = Counter(s.template_id for s in samples)

    n_inv_pairs = sum(c * (c - 1) // 2 for c in logic_counts.values())
    n_cf_pairs = sum(c * (c - 1) // 2 for c in template_counts.values())

    print(f"\n{label}:")
    print(f"  Samples: {len(samples)}")
    print(f"  Logic forms: {len(logic_ids)}")
    print(f"  Templates: {len(template_ids)}")
    print(f"  Invariance pairs (same logic): {n_inv_pairs}")
    print(f"  Counterfactual pairs (same template): {n_cf_pairs}")

    # Show sample
    if samples:
        s = samples[0]
        print(f"\n  Example:")
        print(f"    text: {s.text[:80]}...")
        print(f"    answer_start: {s.answer_start[:40]}")
        print(f"    logic_id: {s.logic_id}")
        print(f"    template_id: {s.template_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate paired reasoning data")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for training JSONL",
    )
    parser.add_argument(
        "--val-output", required=True,
        help="Output path for validation JSONL",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of templates held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = generate_samples()
    train, val = split_train_val(samples, args.val_fraction, args.seed)

    write_jsonl(train, Path(args.output))
    write_jsonl(val, Path(args.val_output))

    print_stats(train, "Training set")
    print_stats(val, "Validation set")
    print(f"\nWritten to:")
    print(f"  {args.output}")
    print(f"  {args.val_output}")


if __name__ == "__main__":
    main()
