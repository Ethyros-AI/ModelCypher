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

Each logical form gets multiple surface templates from diverse domains.
This creates:
- Invariance pairs: same logic, different template
- Counterfactual pairs: same template, different logic

Concepts are defined as rich variable sets that map into any logic form.
Target: 10 logic forms x 50 concepts = 500 samples.

Usage:
    python scripts/generate_paired_data.py \
        --output data/training/paired_reasoning_train.jsonl \
        --val-output data/training/paired_reasoning_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PairedSample:
    text: str
    answer_start: str
    logic_id: str
    template_id: str
    pair_type: str  # anchor | invariance | counterfactual


@dataclass
class Concept:
    """A rich variable set that can fill any logic form template.

    Each concept represents a plausible causal/logical relationship:
    P → Q (→ R for chains). All forms (positive, negative, standalone)
    are provided so the concept can be mapped to any logic form.
    """
    domain: str   # e.g. "cooking_v0"
    # Two-step: P → Q
    P: str           # "the oven is preheated"
    Q: str           # "the bread rises properly"
    P_true: str      # "The oven is preheated"
    P_alone: str     # "the oven is preheated"
    P_conclusion: str  # "the oven is preheated"
    not_P: str       # "The oven is not preheated"
    Q_true: str      # "The bread rises properly"
    Q_alone: str     # "the bread rises properly"
    Q_conclusion: str  # "the bread rises properly"
    not_Q: str       # "The bread does not rise properly"
    # Three-step extension: Q → R
    R: str           # "the meal is ready on time"
    not_R: str       # "The meal is not ready on time"


# =============================================================================
# Logic form functions (unchanged from original)
# =============================================================================


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
# Concept-to-template mapping
# =============================================================================

def concept_to_template(c: Concept, logic_id: str) -> dict:
    """Convert a Concept into the template dict for a given logic form."""
    if logic_id == "chain_contrapositive":
        return {
            "id": c.domain,
            "A": c.P, "B": c.Q, "C": c.R,
            "not_C": c.not_R,
            "not_A": c.not_P.lower() if c.not_P[0].isupper() else c.not_P,
            "not_B": c.not_Q.lower() if c.not_Q[0].isupper() else c.not_Q,
        }
    # All other forms use P/Q/R keys directly
    return {
        "id": c.domain,
        "P": c.P, "Q": c.Q, "R": c.R,
        "P_true": c.P_true,
        "P_alone": c.P_alone,
        "P_conclusion": c.P_conclusion,
        "not_P": c.not_P,
        "Q_true": c.Q_true,
        "Q_alone": c.Q_alone,
        "Q_conclusion": c.Q_conclusion,
        "not_Q": c.not_Q,
    }


# =============================================================================
# Concept definitions (~50 concepts across 12+ domains)
# =============================================================================

CONCEPTS: list[Concept] = [
    # --- Abstract (3 variants) ---
    Concept(
        domain="abstract_v0",
        P="A is true", Q="B is true",
        P_true="A is true", P_alone="A is true", P_conclusion="A is true",
        not_P="A is not true",
        Q_true="B is true", Q_alone="B is true", Q_conclusion="B is true",
        not_Q="B is not true",
        R="C is true", not_R="C is not true",
    ),
    Concept(
        domain="abstract_v1",
        P="X holds", Q="Y holds",
        P_true="X holds", P_alone="X holds", P_conclusion="X holds",
        not_P="X does not hold",
        Q_true="Y holds", Q_alone="Y holds", Q_conclusion="Y holds",
        not_Q="Y does not hold",
        R="Z holds", not_R="Z does not hold",
    ),
    Concept(
        domain="abstract_v2",
        P="condition alpha is met", Q="condition beta is met",
        P_true="Condition alpha is met", P_alone="condition alpha is met",
        P_conclusion="condition alpha is met",
        not_P="Condition alpha is not met",
        Q_true="Condition beta is met", Q_alone="condition beta is met",
        Q_conclusion="condition beta is met",
        not_Q="Condition beta is not met",
        R="condition gamma is met", not_R="Condition gamma is not met",
    ),
    # --- Weather (4 variants) ---
    Concept(
        domain="weather_v0",
        P="it is raining", Q="the ground is wet",
        P_true="It is raining", P_alone="it is raining", P_conclusion="it is raining",
        not_P="It is not raining",
        Q_true="The ground is wet", Q_alone="the ground is wet", Q_conclusion="the ground is wet",
        not_Q="The ground is not wet",
        R="plants grow faster", not_R="Plants are not growing faster",
    ),
    Concept(
        domain="weather_v1",
        P="the temperature drops below freezing", Q="the roads become icy",
        P_true="The temperature has dropped below freezing",
        P_alone="the temperature drops below freezing",
        P_conclusion="the temperature dropped below freezing",
        not_P="The temperature has not dropped below freezing",
        Q_true="The roads have become icy", Q_alone="the roads become icy",
        Q_conclusion="the roads become icy",
        not_Q="The roads have not become icy",
        R="accidents increase", not_R="Accidents have not increased",
    ),
    Concept(
        domain="weather_v2",
        P="a hurricane is approaching", Q="evacuation orders are issued",
        P_true="A hurricane is approaching",
        P_alone="a hurricane is approaching",
        P_conclusion="a hurricane is approaching",
        not_P="No hurricane is approaching",
        Q_true="Evacuation orders have been issued",
        Q_alone="evacuation orders are issued",
        Q_conclusion="evacuation orders are issued",
        not_Q="Evacuation orders have not been issued",
        R="residents leave the area", not_R="Residents have not left the area",
    ),
    Concept(
        domain="weather_v3",
        P="it is sunny", Q="people go to the beach",
        P_true="It is sunny", P_alone="it is sunny",
        P_conclusion="it is sunny",
        not_P="It is not sunny",
        Q_true="People are going to the beach", Q_alone="people go to the beach",
        Q_conclusion="people go to the beach",
        not_Q="People are not going to the beach",
        R="sunscreen sales increase", not_R="Sunscreen sales have not increased",
    ),
    # --- Cooking (4 variants) ---
    Concept(
        domain="cooking_v0",
        P="the oven is preheated", Q="the bread rises properly",
        P_true="The oven is preheated", P_alone="the oven is preheated",
        P_conclusion="the oven is preheated",
        not_P="The oven is not preheated",
        Q_true="The bread rises properly", Q_alone="the bread rises properly",
        Q_conclusion="the bread rises properly",
        not_Q="The bread does not rise properly",
        R="the crust turns golden", not_R="The crust has not turned golden",
    ),
    Concept(
        domain="cooking_v1",
        P="salt is added to the water", Q="the pasta is flavorful",
        P_true="Salt was added to the water", P_alone="salt is added to the water",
        P_conclusion="salt was added to the water",
        not_P="Salt was not added to the water",
        Q_true="The pasta is flavorful", Q_alone="the pasta is flavorful",
        Q_conclusion="the pasta is flavorful",
        not_Q="The pasta is not flavorful",
        R="the dish receives good reviews", not_R="The dish has not received good reviews",
    ),
    Concept(
        domain="cooking_v2",
        P="the butter is melted", Q="the sauce thickens",
        P_true="The butter is melted", P_alone="the butter is melted",
        P_conclusion="the butter is melted",
        not_P="The butter is not melted",
        Q_true="The sauce has thickened", Q_alone="the sauce thickens",
        Q_conclusion="the sauce thickens",
        not_Q="The sauce has not thickened",
        R="the flavor becomes rich", not_R="The flavor has not become rich",
    ),
    Concept(
        domain="cooking_v3",
        P="the dough is kneaded for ten minutes", Q="the gluten develops properly",
        P_true="The dough was kneaded for ten minutes",
        P_alone="the dough is kneaded for ten minutes",
        P_conclusion="the dough was kneaded for ten minutes",
        not_P="The dough was not kneaded for ten minutes",
        Q_true="The gluten has developed properly",
        Q_alone="the gluten develops properly",
        Q_conclusion="the gluten develops properly",
        not_Q="The gluten has not developed properly",
        R="the bread has good texture", not_R="The bread does not have good texture",
    ),
    # --- Medicine (4 variants) ---
    Concept(
        domain="medicine_v0",
        P="the patient takes the antibiotic", Q="the infection clears",
        P_true="The patient takes the antibiotic",
        P_alone="the patient takes the antibiotic",
        P_conclusion="the patient takes the antibiotic",
        not_P="The patient does not take the antibiotic",
        Q_true="The infection has cleared", Q_alone="the infection clears",
        Q_conclusion="the infection clears",
        not_Q="The infection has not cleared",
        R="the patient recovers fully", not_R="The patient has not recovered fully",
    ),
    Concept(
        domain="medicine_v1",
        P="blood pressure is elevated", Q="the doctor prescribes medication",
        P_true="Blood pressure is elevated",
        P_alone="blood pressure is elevated",
        P_conclusion="blood pressure is elevated",
        not_P="Blood pressure is not elevated",
        Q_true="The doctor has prescribed medication",
        Q_alone="the doctor prescribes medication",
        Q_conclusion="the doctor prescribes medication",
        not_Q="The doctor has not prescribed medication",
        R="the patient's risk of stroke decreases",
        not_R="The patient's risk of stroke has not decreased",
    ),
    Concept(
        domain="medicine_v2",
        P="the vaccine is administered", Q="antibodies develop",
        P_true="The vaccine was administered",
        P_alone="the vaccine is administered",
        P_conclusion="the vaccine was administered",
        not_P="The vaccine was not administered",
        Q_true="Antibodies have developed", Q_alone="antibodies develop",
        Q_conclusion="antibodies develop",
        not_Q="Antibodies have not developed",
        R="the patient gains immunity", not_R="The patient has not gained immunity",
    ),
    Concept(
        domain="medicine_v3",
        P="the wound is cleaned properly", Q="the risk of infection decreases",
        P_true="The wound was cleaned properly",
        P_alone="the wound is cleaned properly",
        P_conclusion="the wound was cleaned properly",
        not_P="The wound was not cleaned properly",
        Q_true="The risk of infection has decreased",
        Q_alone="the risk of infection decreases",
        Q_conclusion="the risk of infection decreases",
        not_Q="The risk of infection has not decreased",
        R="healing progresses normally", not_R="Healing has not progressed normally",
    ),
    # --- Sports (4 variants) ---
    Concept(
        domain="sports_v0",
        P="the team practices every day", Q="their performance improves",
        P_true="The team practices every day",
        P_alone="the team practices every day",
        P_conclusion="the team practices every day",
        not_P="The team does not practice every day",
        Q_true="Their performance has improved",
        Q_alone="their performance improves",
        Q_conclusion="their performance improves",
        not_Q="Their performance has not improved",
        R="they win the championship", not_R="They have not won the championship",
    ),
    Concept(
        domain="sports_v1",
        P="the runner stretches before the race", Q="muscle injuries are prevented",
        P_true="The runner stretched before the race",
        P_alone="the runner stretches before the race",
        P_conclusion="the runner stretched before the race",
        not_P="The runner did not stretch before the race",
        Q_true="Muscle injuries were prevented",
        Q_alone="muscle injuries are prevented",
        Q_conclusion="muscle injuries are prevented",
        not_Q="Muscle injuries were not prevented",
        R="the runner finishes the race", not_R="The runner did not finish the race",
    ),
    Concept(
        domain="sports_v2",
        P="the goalkeeper is positioned correctly", Q="the shot is saved",
        P_true="The goalkeeper is positioned correctly",
        P_alone="the goalkeeper is positioned correctly",
        P_conclusion="the goalkeeper is positioned correctly",
        not_P="The goalkeeper is not positioned correctly",
        Q_true="The shot was saved", Q_alone="the shot is saved",
        Q_conclusion="the shot is saved",
        not_Q="The shot was not saved",
        R="the team stays in the lead", not_R="The team is no longer in the lead",
    ),
    Concept(
        domain="sports_v3",
        P="the swimmer trains at altitude", Q="their lung capacity increases",
        P_true="The swimmer trains at altitude",
        P_alone="the swimmer trains at altitude",
        P_conclusion="the swimmer trains at altitude",
        not_P="The swimmer does not train at altitude",
        Q_true="Their lung capacity has increased",
        Q_alone="their lung capacity increases",
        Q_conclusion="their lung capacity increases",
        not_Q="Their lung capacity has not increased",
        R="race times improve", not_R="Race times have not improved",
    ),
    # --- Technology (5 variants) ---
    Concept(
        domain="technology_v0",
        P="the battery is charged", Q="the device turns on",
        P_true="The battery is charged", P_alone="the battery is charged",
        P_conclusion="the battery is charged",
        not_P="The battery is not charged",
        Q_true="The device turns on", Q_alone="the device turns on",
        Q_conclusion="the device turns on",
        not_Q="The device does not turn on",
        R="the user can make calls", not_R="The user cannot make calls",
    ),
    Concept(
        domain="technology_v1",
        P="the software is updated", Q="the security vulnerabilities are patched",
        P_true="The software is updated",
        P_alone="the software is updated",
        P_conclusion="the software is updated",
        not_P="The software is not updated",
        Q_true="The security vulnerabilities are patched",
        Q_alone="the security vulnerabilities are patched",
        Q_conclusion="the security vulnerabilities are patched",
        not_Q="The security vulnerabilities are not patched",
        R="the system remains secure", not_R="The system does not remain secure",
    ),
    Concept(
        domain="technology_v2",
        P="the server is overloaded", Q="response times increase",
        P_true="The server is overloaded",
        P_alone="the server is overloaded",
        P_conclusion="the server is overloaded",
        not_P="The server is not overloaded",
        Q_true="Response times have increased",
        Q_alone="response times increase",
        Q_conclusion="response times increase",
        not_Q="Response times have not increased",
        R="users experience timeouts", not_R="Users are not experiencing timeouts",
    ),
    Concept(
        domain="technology_v3",
        P="the code is compiled without errors", Q="the program runs successfully",
        P_true="The code compiled without errors",
        P_alone="the code is compiled without errors",
        P_conclusion="the code was compiled without errors",
        not_P="The code did not compile without errors",
        Q_true="The program runs successfully",
        Q_alone="the program runs successfully",
        Q_conclusion="the program runs successfully",
        not_Q="The program does not run successfully",
        R="the tests pass", not_R="The tests do not pass",
    ),
    Concept(
        domain="technology_v4",
        P="the Wi-Fi signal is strong", Q="the download speed is fast",
        P_true="The Wi-Fi signal is strong",
        P_alone="the Wi-Fi signal is strong",
        P_conclusion="the Wi-Fi signal is strong",
        not_P="The Wi-Fi signal is not strong",
        Q_true="The download speed is fast",
        Q_alone="the download speed is fast",
        Q_conclusion="the download speed is fast",
        not_Q="The download speed is not fast",
        R="the video streams without buffering",
        not_R="The video does not stream without buffering",
    ),
    # --- Education (4 variants) ---
    Concept(
        domain="education_v0",
        P="you study hard", Q="you will pass the exam",
        P_true="You studied hard", P_alone="you study hard",
        P_conclusion="you studied hard",
        not_P="You did not study hard",
        Q_true="You passed the exam", Q_alone="you will pass the exam",
        Q_conclusion="you will pass the exam",
        not_Q="You did not pass the exam",
        R="you graduate on time", not_R="You do not graduate on time",
    ),
    Concept(
        domain="education_v1",
        P="the student attends every lecture", Q="the student understands the material",
        P_true="The student attended every lecture",
        P_alone="the student attends every lecture",
        P_conclusion="the student attended every lecture",
        not_P="The student did not attend every lecture",
        Q_true="The student understands the material",
        Q_alone="the student understands the material",
        Q_conclusion="the student understands the material",
        not_Q="The student does not understand the material",
        R="the student writes a strong thesis",
        not_R="The student does not write a strong thesis",
    ),
    Concept(
        domain="education_v2",
        P="the library is open", Q="students can access research materials",
        P_true="The library is open", P_alone="the library is open",
        P_conclusion="the library is open",
        not_P="The library is not open",
        Q_true="Students can access research materials",
        Q_alone="students can access research materials",
        Q_conclusion="students can access research materials",
        not_Q="Students cannot access research materials",
        R="research projects proceed on schedule",
        not_R="Research projects do not proceed on schedule",
    ),
    Concept(
        domain="education_v3",
        P="the teacher explains clearly", Q="the students learn effectively",
        P_true="The teacher explains clearly",
        P_alone="the teacher explains clearly",
        P_conclusion="the teacher explains clearly",
        not_P="The teacher does not explain clearly",
        Q_true="The students learn effectively",
        Q_alone="the students learn effectively",
        Q_conclusion="the students learn effectively",
        not_Q="The students do not learn effectively",
        R="test scores improve", not_R="Test scores do not improve",
    ),
    # --- Biology (4 variants) ---
    Concept(
        domain="biology_v0",
        P="an animal is a mammal", Q="it has a backbone",
        P_true="The animal is a mammal", P_alone="the animal is a mammal",
        P_conclusion="the animal is a mammal",
        not_P="The animal is not a mammal",
        Q_true="The animal has a backbone", Q_alone="it has a backbone",
        Q_conclusion="it has a backbone",
        not_Q="The animal does not have a backbone",
        R="it can regulate its body temperature",
        not_R="It cannot regulate its body temperature",
    ),
    Concept(
        domain="biology_v1",
        P="a plant receives sunlight", Q="photosynthesis occurs",
        P_true="The plant receives sunlight",
        P_alone="the plant receives sunlight",
        P_conclusion="the plant receives sunlight",
        not_P="The plant does not receive sunlight",
        Q_true="Photosynthesis occurs", Q_alone="photosynthesis occurs",
        Q_conclusion="photosynthesis occurs",
        not_Q="Photosynthesis does not occur",
        R="the plant produces oxygen", not_R="The plant does not produce oxygen",
    ),
    Concept(
        domain="biology_v2",
        P="the cell has a nucleus", Q="it is a eukaryotic cell",
        P_true="The cell has a nucleus", P_alone="the cell has a nucleus",
        P_conclusion="the cell has a nucleus",
        not_P="The cell does not have a nucleus",
        Q_true="It is a eukaryotic cell", Q_alone="it is a eukaryotic cell",
        Q_conclusion="it is a eukaryotic cell",
        not_Q="It is not a eukaryotic cell",
        R="it can undergo mitosis", not_R="It cannot undergo mitosis",
    ),
    Concept(
        domain="biology_v3",
        P="the predator population increases", Q="the prey population decreases",
        P_true="The predator population has increased",
        P_alone="the predator population increases",
        P_conclusion="the predator population has increased",
        not_P="The predator population has not increased",
        Q_true="The prey population has decreased",
        Q_alone="the prey population decreases",
        Q_conclusion="the prey population decreases",
        not_Q="The prey population has not decreased",
        R="the ecosystem becomes unbalanced",
        not_R="The ecosystem has not become unbalanced",
    ),
    # --- Law (4 variants) ---
    Concept(
        domain="law_v0",
        P="someone is born in France", Q="they are a French citizen",
        P_true="Marie was born in France",
        P_alone="someone is born in France",
        P_conclusion="Marie was born in France",
        not_P="Marie was not born in France",
        Q_true="Marie is a French citizen",
        Q_alone="they are a French citizen",
        Q_conclusion="Marie is a French citizen",
        not_Q="Marie is not a French citizen",
        R="they can vote in French elections",
        not_R="They cannot vote in French elections",
    ),
    Concept(
        domain="law_v1",
        P="someone is eligible to vote", Q="they are a citizen over 18",
        P_true="John is eligible to vote",
        P_alone="someone is eligible to vote",
        P_conclusion="John is eligible to vote",
        not_P="John is not eligible to vote",
        Q_true="John is a citizen over 18",
        Q_alone="they are a citizen over 18",
        Q_conclusion="John is a citizen over 18",
        not_Q="John is not a citizen over 18",
        R="they can serve on a jury", not_R="They cannot serve on a jury",
    ),
    Concept(
        domain="law_v2",
        P="a contract is signed by both parties", Q="the agreement is legally binding",
        P_true="The contract was signed by both parties",
        P_alone="the contract is signed by both parties",
        P_conclusion="the contract was signed by both parties",
        not_P="The contract was not signed by both parties",
        Q_true="The agreement is legally binding",
        Q_alone="the agreement is legally binding",
        Q_conclusion="the agreement is legally binding",
        not_Q="The agreement is not legally binding",
        R="either party can be sued for breach",
        not_R="Neither party can be sued for breach",
    ),
    Concept(
        domain="law_v3",
        P="the defendant is found guilty", Q="a sentence is imposed",
        P_true="The defendant was found guilty",
        P_alone="the defendant is found guilty",
        P_conclusion="the defendant was found guilty",
        not_P="The defendant was not found guilty",
        Q_true="A sentence was imposed",
        Q_alone="a sentence is imposed",
        Q_conclusion="a sentence is imposed",
        not_Q="A sentence was not imposed",
        R="the defendant serves time", not_R="The defendant does not serve time",
    ),
    # --- Economics (4 variants) ---
    Concept(
        domain="economics_v0",
        P="interest rates rise", Q="borrowing decreases",
        P_true="Interest rates have risen",
        P_alone="interest rates rise",
        P_conclusion="interest rates have risen",
        not_P="Interest rates have not risen",
        Q_true="Borrowing has decreased", Q_alone="borrowing decreases",
        Q_conclusion="borrowing decreases",
        not_Q="Borrowing has not decreased",
        R="economic growth slows", not_R="Economic growth has not slowed",
    ),
    Concept(
        domain="economics_v1",
        P="demand exceeds supply", Q="prices increase",
        P_true="Demand exceeds supply",
        P_alone="demand exceeds supply",
        P_conclusion="demand exceeds supply",
        not_P="Demand does not exceed supply",
        Q_true="Prices have increased", Q_alone="prices increase",
        Q_conclusion="prices increase",
        not_Q="Prices have not increased",
        R="consumers reduce spending", not_R="Consumers have not reduced spending",
    ),
    Concept(
        domain="economics_v2",
        P="the government increases taxes", Q="disposable income decreases",
        P_true="The government increased taxes",
        P_alone="the government increases taxes",
        P_conclusion="the government increased taxes",
        not_P="The government did not increase taxes",
        Q_true="Disposable income has decreased",
        Q_alone="disposable income decreases",
        Q_conclusion="disposable income decreases",
        not_Q="Disposable income has not decreased",
        R="consumer spending drops", not_R="Consumer spending has not dropped",
    ),
    Concept(
        domain="economics_v3",
        P="unemployment rises", Q="consumer confidence falls",
        P_true="Unemployment has risen",
        P_alone="unemployment rises",
        P_conclusion="unemployment has risen",
        not_P="Unemployment has not risen",
        Q_true="Consumer confidence has fallen",
        Q_alone="consumer confidence falls",
        Q_conclusion="consumer confidence falls",
        not_Q="Consumer confidence has not fallen",
        R="retail sales decline", not_R="Retail sales have not declined",
    ),
    # --- Geography (4 variants) ---
    Concept(
        domain="geography_v0",
        P="a region is near the equator", Q="the climate is tropical",
        P_true="The region is near the equator",
        P_alone="the region is near the equator",
        P_conclusion="the region is near the equator",
        not_P="The region is not near the equator",
        Q_true="The climate is tropical", Q_alone="the climate is tropical",
        Q_conclusion="the climate is tropical",
        not_Q="The climate is not tropical",
        R="biodiversity is high", not_R="Biodiversity is not high",
    ),
    Concept(
        domain="geography_v1",
        P="the river floods", Q="the surrounding farmland is irrigated",
        P_true="The river has flooded",
        P_alone="the river floods",
        P_conclusion="the river has flooded",
        not_P="The river has not flooded",
        Q_true="The surrounding farmland is irrigated",
        Q_alone="the surrounding farmland is irrigated",
        Q_conclusion="the surrounding farmland is irrigated",
        not_Q="The surrounding farmland is not irrigated",
        R="crop yields increase", not_R="Crop yields have not increased",
    ),
    Concept(
        domain="geography_v2",
        P="tectonic plates collide", Q="mountains form",
        P_true="Tectonic plates are colliding",
        P_alone="tectonic plates collide",
        P_conclusion="tectonic plates are colliding",
        not_P="Tectonic plates are not colliding",
        Q_true="Mountains are forming", Q_alone="mountains form",
        Q_conclusion="mountains form",
        not_Q="Mountains are not forming",
        R="earthquakes occur in the region", not_R="Earthquakes are not occurring in the region",
    ),
    Concept(
        domain="geography_v3",
        P="deforestation occurs", Q="soil erosion increases",
        P_true="Deforestation is occurring",
        P_alone="deforestation occurs",
        P_conclusion="deforestation is occurring",
        not_P="Deforestation is not occurring",
        Q_true="Soil erosion has increased", Q_alone="soil erosion increases",
        Q_conclusion="soil erosion increases",
        not_Q="Soil erosion has not increased",
        R="water quality in nearby rivers declines",
        not_R="Water quality in nearby rivers has not declined",
    ),
    # --- Music (4 variants) ---
    Concept(
        domain="music_v0",
        P="the musician practices daily", Q="their technique improves",
        P_true="The musician practices daily",
        P_alone="the musician practices daily",
        P_conclusion="the musician practices daily",
        not_P="The musician does not practice daily",
        Q_true="Their technique has improved",
        Q_alone="their technique improves",
        Q_conclusion="their technique improves",
        not_Q="Their technique has not improved",
        R="they are accepted into the orchestra",
        not_R="They are not accepted into the orchestra",
    ),
    Concept(
        domain="music_v1",
        P="the guitar is in tune", Q="the chords sound harmonious",
        P_true="The guitar is in tune", P_alone="the guitar is in tune",
        P_conclusion="the guitar is in tune",
        not_P="The guitar is not in tune",
        Q_true="The chords sound harmonious",
        Q_alone="the chords sound harmonious",
        Q_conclusion="the chords sound harmonious",
        not_Q="The chords do not sound harmonious",
        R="the audience enjoys the performance",
        not_R="The audience does not enjoy the performance",
    ),
    Concept(
        domain="music_v2",
        P="the conductor signals a tempo change", Q="the orchestra adjusts speed",
        P_true="The conductor signaled a tempo change",
        P_alone="the conductor signals a tempo change",
        P_conclusion="the conductor signaled a tempo change",
        not_P="The conductor did not signal a tempo change",
        Q_true="The orchestra adjusted speed",
        Q_alone="the orchestra adjusts speed",
        Q_conclusion="the orchestra adjusts speed",
        not_Q="The orchestra did not adjust speed",
        R="the piece reaches its climax on cue",
        not_R="The piece did not reach its climax on cue",
    ),
    Concept(
        domain="music_v3",
        P="the studio has good acoustics", Q="the recording sounds clear",
        P_true="The studio has good acoustics",
        P_alone="the studio has good acoustics",
        P_conclusion="the studio has good acoustics",
        not_P="The studio does not have good acoustics",
        Q_true="The recording sounds clear",
        Q_alone="the recording sounds clear",
        Q_conclusion="the recording sounds clear",
        not_Q="The recording does not sound clear",
        R="the album receives positive reviews",
        not_R="The album does not receive positive reviews",
    ),
    # --- History (4 variants) ---
    Concept(
        domain="history_v0",
        P="a civilization develops agriculture", Q="permanent settlements form",
        P_true="The civilization developed agriculture",
        P_alone="the civilization develops agriculture",
        P_conclusion="the civilization developed agriculture",
        not_P="The civilization did not develop agriculture",
        Q_true="Permanent settlements have formed",
        Q_alone="permanent settlements form",
        Q_conclusion="permanent settlements form",
        not_Q="Permanent settlements have not formed",
        R="trade networks emerge", not_R="Trade networks have not emerged",
    ),
    Concept(
        domain="history_v1",
        P="the printing press is invented", Q="literacy rates increase",
        P_true="The printing press was invented",
        P_alone="the printing press is invented",
        P_conclusion="the printing press was invented",
        not_P="The printing press was not invented",
        Q_true="Literacy rates have increased",
        Q_alone="literacy rates increase",
        Q_conclusion="literacy rates increase",
        not_Q="Literacy rates have not increased",
        R="scientific knowledge spreads rapidly",
        not_R="Scientific knowledge has not spread rapidly",
    ),
    Concept(
        domain="history_v2",
        P="the empire overextends its borders", Q="military resources are stretched thin",
        P_true="The empire has overextended its borders",
        P_alone="the empire overextends its borders",
        P_conclusion="the empire has overextended its borders",
        not_P="The empire has not overextended its borders",
        Q_true="Military resources are stretched thin",
        Q_alone="military resources are stretched thin",
        Q_conclusion="military resources are stretched thin",
        not_Q="Military resources are not stretched thin",
        R="the empire declines", not_R="The empire has not declined",
    ),
    Concept(
        domain="history_v3",
        P="a revolution occurs", Q="the existing government is overthrown",
        P_true="A revolution has occurred",
        P_alone="a revolution occurs",
        P_conclusion="a revolution has occurred",
        not_P="A revolution has not occurred",
        Q_true="The existing government has been overthrown",
        Q_alone="the existing government is overthrown",
        Q_conclusion="the existing government is overthrown",
        not_Q="The existing government has not been overthrown",
        R="a new constitution is drafted",
        not_R="A new constitution has not been drafted",
    ),
    # --- Transport (3 variants) ---
    Concept(
        domain="transport_v0",
        P="the traffic light turns red", Q="cars stop",
        P_true="The traffic light turned red",
        P_alone="the traffic light turns red",
        P_conclusion="the traffic light turned red",
        not_P="The traffic light did not turn red",
        Q_true="Cars have stopped", Q_alone="cars stop",
        Q_conclusion="cars stop",
        not_Q="Cars have not stopped",
        R="pedestrians cross safely", not_R="Pedestrians do not cross safely",
    ),
    Concept(
        domain="transport_v1",
        P="the train arrives on schedule", Q="commuters reach work on time",
        P_true="The train arrived on schedule",
        P_alone="the train arrives on schedule",
        P_conclusion="the train arrived on schedule",
        not_P="The train did not arrive on schedule",
        Q_true="Commuters reached work on time",
        Q_alone="commuters reach work on time",
        Q_conclusion="commuters reach work on time",
        not_Q="Commuters did not reach work on time",
        R="productivity is maintained", not_R="Productivity is not maintained",
    ),
    Concept(
        domain="transport_v2",
        P="the bridge is closed for repairs", Q="drivers must take a detour",
        P_true="The bridge is closed for repairs",
        P_alone="the bridge is closed for repairs",
        P_conclusion="the bridge is closed for repairs",
        not_P="The bridge is not closed for repairs",
        Q_true="Drivers must take a detour",
        Q_alone="drivers must take a detour",
        Q_conclusion="drivers must take a detour",
        not_Q="Drivers do not need to take a detour",
        R="commute times increase", not_R="Commute times have not increased",
    ),
]


# =============================================================================
# Logic form registry
# =============================================================================

LOGIC_FORMS: list[tuple[str, callable, list[str]]] = [
    ("modus_ponens", _modus_ponens, []),
    ("modus_tollens", _modus_tollens, []),
    ("disjunctive_syllogism", _disjunctive_syllogism, []),
    ("hypothetical_syllogism", _hypothetical_syllogism, []),
    ("chain_contrapositive", _chain_contrapositive, []),
    ("conjunction_elimination", _conjunction_elimination, []),
    ("affirming_consequent_fallacy", _affirming_consequent_fallacy, []),
    ("denying_antecedent_fallacy", _denying_antecedent_fallacy, []),
    ("biconditional", _biconditional, []),
    ("disjunction_intro", _disjunction_intro, []),
]


def generate_samples() -> list[PairedSample]:
    """Generate all paired samples: logic forms x concepts."""
    samples: list[PairedSample] = []

    for logic_id, fn, _ in LOGIC_FORMS:
        for concept in CONCEPTS:
            template = concept_to_template(concept, logic_id)
            text, answer = fn(template)

            samples.append(PairedSample(
                text=text,
                answer_start=answer,
                logic_id=logic_id,
                template_id=concept.domain,
                pair_type="anchor",
            ))

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

    missing_in_val = train_logic - val_logic
    missing_in_train = val_logic - train_logic

    # If any logic_id is missing from val, move one sample per missing
    # logic_id from train to val (pick the shortest to minimize data loss)
    if missing_in_val:
        for lid in missing_in_val:
            candidates = [s for s in train if s.logic_id == lid]
            if candidates:
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

    if samples:
        s = samples[0]
        print("\n  Example:")
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
    print("\nWritten to:")
    print(f"  {args.output}")
    print(f"  {args.val_output}")


if __name__ == "__main__":
    main()
