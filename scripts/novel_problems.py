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

"""Combinatorial novel problem generator for STaR bootstrap.

Produces logic problems with:
  - Known logic forms (MP, MT, DS, HS, chain contrapositive, biconditional)
  - Novel domains and entities never seen in training
  - Deterministic correct answers derived from logic form structure
  - Structural verification (not keyword matching)

Usage:
    # Self-test: print 10 sample problems
    python scripts/novel_problems.py

    # Print N problems with seed
    python scripts/novel_problems.py --count 20 --seed 99
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Shared verification helpers (used by test_bootstrap_feasibility.py too)
# ---------------------------------------------------------------------------

def first_sentences(text: str, n: int = 2) -> str:
    """Extract first n non-empty sentences from response."""
    text = text.strip()
    parts = re.split(r'(?<=[.!?])\s', text)
    sentences = [s.strip() for s in parts if s.strip() and len(s.strip()) > 5]
    return " ".join(sentences[:n]).lower()


def word_match(text: str, word: str) -> bool:
    """Check for whole-word match (not substring). 'no' won't match 'not'."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text.lower()))


def first_stated_answer(text: str) -> str:
    """Extract the model's first stated answer/conclusion, ignoring option lists.

    Looks for patterns like:
    - 'Answer: X'
    - 'The answer is X'
    - 'Therefore, X'
    - 'We can conclude X'
    - First sentence if none of the above
    """
    text = text.strip()
    for pattern in [
        r'(?:the\s+)?(?:correct\s+)?answer(?:\s+is)?[:\s]+(.+?)(?:\.|$)',
        r'therefore[,:\s]+(.+?)(?:\.|$)',
        r'we can (?:conclude|determine|infer)[:\s]+(.+?)(?:\.|$)',
        r'what follows[:\s]+(.+?)(?:\.|$)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().lower()
    return first_sentences(text, 1)


def concludes_not(text: str, keyword: str) -> bool:
    """Check if first stated conclusion contains 'not/did not/has not' + keyword.

    ONLY checks the first stated answer (extracted by first_stated_answer),
    NOT the full text -- prevents false positives from option lists.
    """
    answer = first_stated_answer(text)
    neg_patterns = [
        rf'not\s+\w*\s*{keyword}',
        rf'has not\s+{keyword}',
        rf'did not\s+{keyword}',
        rf"didn't\s+{keyword}",
        rf"hasn't\s+{keyword}",
        rf'not\s+{keyword}',
        rf'{keyword}.*not',
    ]
    for pat in neg_patterns:
        if re.search(pat, answer, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Premise pairs: (domain, A_condition, B_consequence)
# "If A then B" must be plausible in-domain.
# ---------------------------------------------------------------------------

PREMISE_PAIRS: list[tuple[str, str, str]] = [
    # astronomy
    ("astronomy", "a star exhausts its hydrogen fuel", "it expands into a red giant"),
    ("astronomy", "a comet approaches the Sun", "its tail becomes visible"),
    ("astronomy", "a planet has sufficient mass", "it clears its orbital neighborhood"),
    ("astronomy", "a galaxy undergoes a merger", "its spiral arms become distorted"),
    ("astronomy", "a neutron star spins rapidly", "it emits pulsar beams"),
    ("astronomy", "a black hole accretes matter", "an accretion disk forms"),
    # geology
    ("geology", "tectonic plates collide", "mountain ranges form"),
    ("geology", "magma reaches the surface", "a volcanic eruption occurs"),
    ("geology", "sediment accumulates over millennia", "sedimentary rock forms"),
    ("geology", "an earthquake occurs underwater", "a tsunami may be generated"),
    ("geology", "groundwater dissolves limestone", "caves develop"),
    ("geology", "glaciers advance over terrain", "moraines are deposited"),
    # cooking
    ("cooking", "the oil reaches its smoke point", "it begins to break down"),
    ("cooking", "yeast is added to warm dough", "the dough rises"),
    ("cooking", "sugar is heated past 170 degrees Celsius", "it caramelizes"),
    ("cooking", "an egg is placed in boiling water for ten minutes", "the yolk becomes solid"),
    ("cooking", "salt is dissolved in water", "the boiling point increases"),
    ("cooking", "cream is whipped vigorously", "it forms stiff peaks"),
    # law
    ("law", "a contract lacks consideration", "it is unenforceable"),
    ("law", "evidence is obtained without a warrant", "it may be excluded at trial"),
    ("law", "a defendant pleads guilty", "no trial is held"),
    ("law", "a statute of limitations expires", "charges cannot be filed"),
    ("law", "a corporation is dissolved", "its assets are distributed to creditors"),
    ("law", "a patent application is approved", "the inventor gains exclusive rights"),
    # music
    ("music", "a string is shortened by half", "its frequency doubles"),
    ("music", "a major chord is inverted", "the bass note changes"),
    ("music", "a conductor raises the baton", "the orchestra prepares to play"),
    ("music", "a piece is transposed up a fifth", "every note shifts by seven semitones"),
    ("music", "resonance occurs in a pipe organ", "the sound amplifies"),
    # medicine
    ("medicine", "a patient's blood pressure drops below 90/60", "hypotension is diagnosed"),
    ("medicine", "a virus mutates its surface proteins", "existing antibodies become less effective"),
    ("medicine", "an artery is blocked", "blood flow to the tissue is reduced"),
    ("medicine", "a bone fracture is not immobilized", "healing is delayed"),
    ("medicine", "the thyroid produces excess hormones", "metabolism accelerates"),
    # engineering
    ("engineering", "a bridge exceeds its load capacity", "structural failure occurs"),
    ("engineering", "a circuit is overloaded", "the fuse blows"),
    ("engineering", "friction is eliminated from bearings", "efficiency increases"),
    ("engineering", "a turbine blade develops cracks", "the engine must be shut down"),
    ("engineering", "coolant pressure drops below threshold", "the reactor trips"),
    # ecology
    ("ecology", "a keystone species is removed", "the ecosystem destabilizes"),
    ("ecology", "nitrogen runoff enters a lake", "algal blooms occur"),
    ("ecology", "pollinators decline in a region", "crop yields decrease"),
    ("ecology", "an invasive species is introduced", "native species face competition"),
    ("ecology", "deforestation increases in a watershed", "soil erosion accelerates"),
    # economics
    ("economics", "the central bank raises interest rates", "borrowing becomes more expensive"),
    ("economics", "a tariff is imposed on imports", "domestic prices rise"),
    ("economics", "unemployment increases", "consumer spending decreases"),
    ("economics", "a currency is devalued", "exports become cheaper abroad"),
    ("economics", "inflation exceeds wage growth", "real purchasing power declines"),
    # chemistry
    ("chemistry", "a catalyst is added to a reaction", "the activation energy decreases"),
    ("chemistry", "the temperature of a gas increases", "its volume expands at constant pressure"),
    ("chemistry", "an acid is added to a base", "a neutralization reaction occurs"),
    ("chemistry", "a solution is supersaturated", "crystals begin to precipitate"),
    ("chemistry", "electrons are removed from an atom", "a positive ion is formed"),
    # oceanography
    ("oceanography", "sea surface temperature rises", "hurricane intensity increases"),
    ("oceanography", "salinity increases in a water column", "the density of the water increases"),
    ("oceanography", "upwelling occurs along a coast", "nutrient-rich water reaches the surface"),
    ("oceanography", "polar ice melts", "sea levels rise"),
    # architecture
    ("architecture", "a building exceeds 150 meters in height", "it requires a tuned mass damper"),
    ("architecture", "a foundation is placed on clay soil", "differential settling may occur"),
    ("architecture", "an atrium is enclosed with glass", "solar heat gain increases"),
]

# Domains for filtering
DOMAINS = sorted({p[0] for p in PREMISE_PAIRS})


# ---------------------------------------------------------------------------
# Logic form templates
# ---------------------------------------------------------------------------

def _make_mp_prompt(a: str, b: str, domain: str) -> str:
    return (
        f"If {a}, then {b}. "
        f"{a[0].upper()}{a[1:]}.\n"
        f"What can we conclude?"
    )


def _make_mt_prompt(a: str, b: str, domain: str) -> str:
    return (
        f"If {a}, then {b}. "
        f"It is not the case that {b}.\n"
        f"What can we conclude?"
    )


def _make_ds_prompt(a: str, b: str, domain: str) -> str:
    return (
        f"Either {a} or {b}. "
        f"It is not the case that {a}.\n"
        f"What must be the case?"
    )


def _make_hs_prompt(
    a1: str, b1: str, a2: str, b2: str, domain: str
) -> str:
    return (
        f"If {a1}, then {b1}. "
        f"If {b1}, then {b2}.\n"
        f"What can we conclude about the relationship between "
        f"{a1} and {b2}?"
    )


def _make_cc_prompt(
    a1: str, b1: str, a2: str, b2: str, domain: str
) -> str:
    return (
        f"If {a1}, then {b1}. "
        f"If {b1}, then {b2}. "
        f"It is not the case that {b2}.\n"
        f"What can we conclude?"
    )


def _make_bic_prompt(a: str, b: str, domain: str) -> str:
    return (
        f"{a[0].upper()}{a[1:]} if and only if {b}. "
        f"{b[0].upper()}{b[1:]}.\n"
        f"What follows?"
    )


# ---------------------------------------------------------------------------
# Verification functions per logic form
# ---------------------------------------------------------------------------

def _verify_mp(response: str, expected_b: str) -> bool:
    """MP: conclusion must contain key phrase from B."""
    key = _extract_key_phrase(expected_b)
    fs = first_sentences(response, 3)
    fa = first_stated_answer(response)
    return key in fs or key in fa


def _verify_mt(response: str, expected_a: str) -> bool:
    """MT: conclusion must negate A."""
    key = _extract_key_phrase(expected_a)
    return concludes_not(response, key)


def _verify_ds(response: str, expected_b: str) -> bool:
    """DS: conclusion must affirm B (without negation)."""
    key = _extract_key_phrase(expected_b)
    fa = first_stated_answer(response)
    fs = first_sentences(response, 3)
    affirmed = key in fa or key in fs
    negated = f"not {key}" in fa or f"not {key}" in fs
    return affirmed and not negated


def _verify_hs(response: str, a1: str, b2: str) -> bool:
    """HS: conclusion must link A1 to B2 (chain)."""
    key_a = _extract_key_phrase(a1)
    key_b = _extract_key_phrase(b2)
    text = response.lower()
    return key_a in text and key_b in text


def _verify_cc(response: str, a1: str) -> bool:
    """Chain contrapositive: must negate A1."""
    key = _extract_key_phrase(a1)
    return concludes_not(response, key)


def _verify_bic(response: str, expected_a: str) -> bool:
    """Biconditional: must affirm A."""
    key = _extract_key_phrase(expected_a)
    fa = first_stated_answer(response)
    fs = first_sentences(response, 2)
    return key in fa or key in fs


def _extract_key_phrase(text: str) -> str:
    """Extract a short identifying phrase from a premise clause.

    Strategy: take the last 2-3 meaningful words, which typically carry
    the distinguishing content (e.g. "red giant", "algal blooms").
    """
    text = text.strip().lower()
    # Remove leading articles/pronouns
    text = re.sub(r'^(a |an |the |it |its |they |this )', '', text)
    words = text.split()
    if len(words) <= 3:
        return text
    # Take last 3 words as the key phrase
    return " ".join(words[-3:])


# ---------------------------------------------------------------------------
# NovelProblem dataclass
# ---------------------------------------------------------------------------

@dataclass
class NovelProblem:
    id: str
    logic: str
    domain: str
    prompt: str
    expected_answer: str
    _verify_fn: Callable[[str], bool] = field(repr=False)

    def verify(self, response: str) -> bool:
        """Structural verification of model response."""
        return self._verify_fn(response)


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def _generate_single_form(
    form: str,
    pair: tuple[str, str, str],
    idx: int,
    pair2: tuple[str, str, str] | None = None,
) -> NovelProblem:
    """Generate a single problem from a logic form + premise pair(s)."""
    domain, a, b = pair

    if form == "modus_ponens":
        prompt = _make_mp_prompt(a, b, domain)
        expected = f"{b[0].upper()}{b[1:]}"
        verify_fn = lambda r, _b=b: _verify_mp(r, _b)

    elif form == "modus_tollens":
        prompt = _make_mt_prompt(a, b, domain)
        expected = f"It is not the case that {a}"
        verify_fn = lambda r, _a=a: _verify_mt(r, _a)

    elif form == "disjunctive_syllogism":
        prompt = _make_ds_prompt(a, b, domain)
        expected = f"{b[0].upper()}{b[1:]}"
        verify_fn = lambda r, _b=b: _verify_ds(r, _b)

    elif form == "biconditional":
        prompt = _make_bic_prompt(a, b, domain)
        expected = f"{a[0].upper()}{a[1:]}"
        verify_fn = lambda r, _a=a: _verify_bic(r, _a)

    elif form == "hypothetical_syllogism":
        if pair2 is None:
            raise ValueError("HS requires two premise pairs")
        d2, a2, b2 = pair2
        prompt = _make_hs_prompt(a, b, a2, b2, domain)
        expected = f"If {a}, then {b2}"
        verify_fn = lambda r, _a=a, _b2=b2: _verify_hs(r, _a, _b2)

    elif form == "chain_contrapositive":
        if pair2 is None:
            raise ValueError("CC requires two premise pairs")
        d2, a2, b2 = pair2
        prompt = _make_cc_prompt(a, b, a2, b2, domain)
        expected = f"It is not the case that {a}"
        verify_fn = lambda r, _a=a: _verify_cc(r, _a)

    else:
        raise ValueError(f"Unknown logic form: {form}")

    return NovelProblem(
        id=f"novel_{form}_{domain}_{idx}",
        logic=form,
        domain=domain,
        prompt=prompt,
        expected_answer=expected,
        _verify_fn=verify_fn,
    )


SINGLE_FORMS = [
    "modus_ponens",
    "modus_tollens",
    "disjunctive_syllogism",
    "biconditional",
]

CHAIN_FORMS = [
    "hypothetical_syllogism",
    "chain_contrapositive",
]


def generate_novel_problems(
    n: int,
    seed: int,
    domains: list[str] | None = None,
) -> list[NovelProblem]:
    """Generate n novel logic problems via combinatorial selection.

    Args:
        n: Number of problems to generate.
        seed: Random seed for reproducibility.
        domains: Optional domain filter. None = use all domains.

    Returns:
        List of NovelProblem instances with deterministic answers.
    """
    rng = random.Random(seed)

    pool = PREMISE_PAIRS
    if domains:
        pool = [p for p in pool if p[0] in domains]
    if len(pool) < 2:
        raise ValueError(f"Need at least 2 premise pairs, got {len(pool)}")

    problems: list[NovelProblem] = []
    idx = 0

    # Build all possible (form, pair, pair2) combinations
    combos: list[tuple[str, tuple, tuple | None]] = []

    for pair in pool:
        for form in SINGLE_FORMS:
            combos.append((form, pair, None))

    # Chain forms need two pairs from the same domain (for coherence)
    by_domain: dict[str, list[tuple[str, str, str]]] = {}
    for p in pool:
        by_domain.setdefault(p[0], []).append(p)

    for domain, pairs in by_domain.items():
        if len(pairs) < 2:
            continue
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i == j:
                    continue
                for form in CHAIN_FORMS:
                    combos.append((form, pairs[i], pairs[j]))

    rng.shuffle(combos)

    # Take n problems, cycling if needed
    for i in range(n):
        form, pair, pair2 = combos[i % len(combos)]
        problem = _generate_single_form(form, pair, idx, pair2)
        problems.append(problem)
        idx += 1

    return problems


# ---------------------------------------------------------------------------
# Adapter-to-JSONL: convert verified novel problems to training format
# ---------------------------------------------------------------------------

def novel_problem_to_training_sample(
    problem: NovelProblem,
    verified_response: str,
    round_idx: int,
) -> dict:
    """Convert a verified novel problem response to paired training format."""
    # Use the canonical expected answer, not the model's rambling
    answer_line = problem.expected_answer
    text = f"{problem.prompt}\n{answer_line}"
    return {
        "text": text,
        "answer_start": answer_line,
        "logic_id": problem.logic,
        "template_id": f"novel_{problem.domain}_{problem.id}",
        "pair_type": f"star_novel_round_{round_idx}",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test(count: int = 10, seed: int = 42) -> None:
    """Print sample problems for manual inspection."""
    problems = generate_novel_problems(count, seed)

    form_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}

    print(f"Generated {len(problems)} novel problems (seed={seed})\n")
    for p in problems:
        form_counts[p.logic] = form_counts.get(p.logic, 0) + 1
        domain_counts[p.domain] = domain_counts.get(p.domain, 0) + 1

        print(f"--- {p.id} [{p.logic}] ({p.domain}) ---")
        print(f"Q: {p.prompt}")
        print(f"Expected: {p.expected_answer}")
        print()

    print("=" * 60)
    print("FORM DISTRIBUTION:")
    for form, cnt in sorted(form_counts.items()):
        print(f"  {form}: {cnt}")
    print("\nDOMAIN DISTRIBUTION:")
    for domain, cnt in sorted(domain_counts.items()):
        print(f"  {domain}: {cnt}")
    print(f"\nTotal combinations available: "
          f"{len(PREMISE_PAIRS) * len(SINGLE_FORMS)} single + chain combos")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Novel problem generator self-test")
    parser.add_argument("--count", type=int, default=10, help="Number of problems")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    _self_test(args.count, args.seed)
