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

"""Phase B training patterns derived from real benchmark failures.

Focus areas (from failures):
- BoolQ yes/no polarity
- ARC MCQ option elimination
- GSM8K rate/percent/total/comparison/remaining
"""

from __future__ import annotations

from modelcypher.adapters.training.mlx.self_reflection import SelfReflectionExample


def _gsm8k_rate_examples() -> list[SelfReflectionExample]:
    examples = []
    params = [
        (12, 5, 3, 2),  # rate, hours, extra, lost
        (8, 6, 10, 4),
        (15, 4, 0, 3),
        (20, 2, 5, 0),
        (9, 7, 6, 1),
        (14, 3, 12, 2),
        (6, 10, 0, 5),
        (11, 8, 7, 3),
        (18, 4, 9, 2),
        (5, 12, 4, 1),
        (16, 3, 6, 0),
        (7, 9, 8, 2),
        (13, 5, 5, 1),
        (10, 6, 3, 2),
        (4, 15, 2, 0),
    ]
    for rate, hours, extra, lost in params:
        total = rate * hours + extra - lost
        examples.append(
            SelfReflectionExample(
                input_question=(
                    f"A worker makes {rate} units per hour for {hours} hours. "
                    f"They make {extra} extra units and lose {lost} units. "
                    "How many units are made in total?"
                ),
                core_question=f"{rate}*{hours} + {extra} - {lost} = ?",
                reasoning=(
                    f"Base production: {rate} * {hours} = {rate * hours}\n"
                    f"Add extra: {rate * hours} + {extra} = {rate * hours + extra}\n"
                    f"Subtract lost: {rate * hours + extra} - {lost} = {total}"
                ),
                answer=str(total),
            )
        )
    return examples


def _gsm8k_percent_examples() -> list[SelfReflectionExample]:
    examples = []
    params = [
        (80, 20, 10),
        (120, 15, 8),
        (50, 30, 5),
        (200, 25, 12),
        (75, 10, 7),
        (90, 40, 5),
        (60, 15, 15),
        (150, 35, 10),
        (110, 20, 6),
        (95, 25, 9),
        (130, 30, 7),
        (55, 10, 12),
        (170, 15, 5),
        (65, 20, 10),
        (140, 25, 8),
    ]
    for price, discount, tax in params:
        discounted = price * (1 - discount / 100)
        final = discounted * (1 + tax / 100)
        examples.append(
            SelfReflectionExample(
                input_question=(
                    f"An item costs ${price}. It is discounted by {discount}%. "
                    f"Then a {tax}% tax is applied. What is the final price?"
                ),
                core_question=f"{price}*(1-{discount}/100)*(1+{tax}/100) = ?",
                reasoning=(
                    f"Discounted price: {price} * (1 - {discount}/100) = {discounted}\n"
                    f"After tax: {discounted} * (1 + {tax}/100) = {final}"
                ),
                answer=str(round(final, 2)),
            )
        )
    return examples


def _gsm8k_total_examples() -> list[SelfReflectionExample]:
    examples = []
    params = [
        (4, 12, 3, 5),
        (6, 8, 2, 9),
        (3, 15, 5, 4),
        (7, 6, 4, 10),
        (5, 9, 3, 7),
        (8, 5, 6, 3),
        (2, 20, 4, 6),
        (9, 4, 3, 11),
        (10, 3, 5, 8),
        (12, 2, 7, 6),
        (5, 14, 2, 12),
        (3, 18, 4, 7),
        (6, 11, 3, 9),
        (4, 16, 5, 6),
        (7, 9, 2, 13),
    ]
    for a_qty, a_price, b_qty, b_price in params:
        total = a_qty * a_price + b_qty * b_price
        examples.append(
            SelfReflectionExample(
                input_question=(
                    f"A person buys {a_qty} items at ${a_price} each and {b_qty} items at "
                    f"${b_price} each. What is the total cost?"
                ),
                core_question=f"{a_qty}*{a_price} + {b_qty}*{b_price} = ?",
                reasoning=(
                    f"First items: {a_qty} * {a_price} = {a_qty * a_price}\n"
                    f"Second items: {b_qty} * {b_price} = {b_qty * b_price}\n"
                    f"Total: {a_qty * a_price} + {b_qty * b_price} = {total}"
                ),
                answer=str(total),
            )
        )
    return examples


def _gsm8k_comparison_examples() -> list[SelfReflectionExample]:
    examples = []
    params = [
        (2, 3, 10),
        (3, 2, 8),
        (4, 2, 6),
        (2, 5, 7),
        (5, 3, 4),
        (3, 4, 9),
        (2, 6, 5),
        (4, 3, 11),
        (3, 5, 6),
        (5, 2, 12),
        (2, 4, 13),
        (4, 5, 7),
        (3, 6, 8),
        (5, 4, 9),
        (2, 7, 6),
    ]
    for a_mult, b_mult, base in params:
        mid = b_mult * base
        total = a_mult * mid
        examples.append(
            SelfReflectionExample(
                input_question=(
                    f"Alex has {a_mult} times as many marbles as Ben. "
                    f"Ben has {b_mult} times as many marbles as Cara. "
                    f"Cara has {base} marbles. How many marbles does Alex have?"
                ),
                core_question=f"{a_mult} * ({b_mult} * {base}) = ?",
                reasoning=(
                    f"Cara: {base}\n"
                    f"Ben: {b_mult} * {base} = {mid}\n"
                    f"Alex: {a_mult} * {mid} = {total}"
                ),
                answer=str(total),
            )
        )
    return examples


def _gsm8k_remaining_examples() -> list[SelfReflectionExample]:
    examples = []
    params = [
        (50, 12, 8),
        (80, 25, 10),
        (100, 30, 15),
        (60, 14, 9),
        (90, 20, 7),
        (75, 18, 12),
        (120, 35, 20),
        (40, 10, 5),
        (110, 28, 14),
        (55, 16, 6),
        (95, 22, 11),
        (70, 15, 10),
        (85, 24, 13),
        (65, 12, 9),
        (105, 26, 18),
    ]
    for start, sold, given in params:
        left = start - sold - given
        examples.append(
            SelfReflectionExample(
                input_question=(
                    f"A shop has {start} items. It sells {sold} items and gives away {given}. "
                    "How many items are left?"
                ),
                core_question=f"{start} - {sold} - {given} = ?",
                reasoning=(
                    f"After selling: {start} - {sold} = {start - sold}\n"
                    f"After giving away: {start - sold} - {given} = {left}"
                ),
                answer=str(left),
            )
        )
    return examples


def _boolq_examples() -> list[SelfReflectionExample]:
    samples = [
        ("Mars is the fourth planet from the Sun. It is called the Red Planet.", "Is Mars the fourth planet from the Sun?", "yes"),
        ("Water freezes at 0 degrees Celsius at sea level.", "Does water freeze at 0 degrees Celsius?", "yes"),
        ("The Eiffel Tower is in Paris, France.", "Is the Eiffel Tower in London?", "no"),
        ("A triangle has three sides.", "Does a triangle have four sides?", "no"),
        ("The Pacific Ocean is the largest ocean on Earth.", "Is the Pacific Ocean the largest ocean?", "yes"),
        ("The Sun is a star.", "Is the Sun a planet?", "no"),
        ("An octagon has eight sides.", "Does an octagon have eight sides?", "yes"),
        ("Penguins are birds that cannot fly.", "Can penguins fly?", "no"),
        ("The Great Wall of China is in China.", "Is the Great Wall in China?", "yes"),
        ("Lightning is a form of electricity.", "Is lightning a form of electricity?", "yes"),
        ("Sharks are fish.", "Are sharks mammals?", "no"),
        ("The human heart has four chambers.", "Does the human heart have four chambers?", "yes"),
        ("Saturn has rings.", "Does Saturn have rings?", "yes"),
        ("A square has four equal sides.", "Does a square have three sides?", "no"),
        ("The Amazon is a river in South America.", "Is the Amazon in Africa?", "no"),
        ("Venus is closer to the Sun than Earth.", "Is Venus closer to the Sun than Earth?", "yes"),
        ("Bats are mammals.", "Are bats reptiles?", "no"),
        ("A decade is ten years.", "Is a decade ten years?", "yes"),
        ("The Nile is a river in Africa.", "Is the Nile in Africa?", "yes"),
        ("Gold is a metal.", "Is gold a gas?", "no"),
    ]
    examples = []
    for passage, question, answer in samples:
        examples.append(
            SelfReflectionExample(
                input_question=f"{passage}\n\nQuestion: {question}",
                core_question="Is the question supported by the passage?",
                reasoning=(
                    "Identify the factual claim in the question.\n"
                    "Check the passage for that claim.\n"
                    f"If it matches, answer yes; otherwise no."
                ),
                answer=answer,
            )
        )
    return examples


def _arc_mcq_examples() -> list[SelfReflectionExample]:
    samples = [
        (
            "Which object produces light by itself?\nA. Moon\nB. Sun\nC. Mirror\nD. Book",
            "Sun",
        ),
        (
            "Which tool is best for seeing cells?\nA. Telescope\nB. Microscope\nC. Thermometer\nD. Compass",
            "Microscope",
        ),
        (
            "Which material is a good conductor of electricity?\nA. Rubber\nB. Glass\nC. Copper\nD. Plastic",
            "Copper",
        ),
        (
            "What is the main source of energy for the water cycle?\nA. Wind\nB. The Sun\nC. Volcanoes\nD. The Moon",
            "The Sun",
        ),
        (
            "Plants make food using sunlight in a process called?\nA. Digestion\nB. Photosynthesis\nC. Respiration\nD. Fermentation",
            "Photosynthesis",
        ),
        (
            "Which planet is closest to the Sun?\nA. Venus\nB. Mars\nC. Mercury\nD. Jupiter",
            "Mercury",
        ),
        (
            "What gas do humans need to breathe?\nA. Carbon dioxide\nB. Oxygen\nC. Nitrogen\nD. Helium",
            "Oxygen",
        ),
        (
            "Which part of the plant absorbs water?\nA. Leaves\nB. Roots\nC. Stem\nD. Flower",
            "Roots",
        ),
        (
            "Which force pulls objects toward Earth?\nA. Magnetism\nB. Gravity\nC. Friction\nD. Electricity",
            "Gravity",
        ),
        (
            "What is the state of water at room temperature?\nA. Solid\nB. Liquid\nC. Gas\nD. Plasma",
            "Liquid",
        ),
        (
            "Which is an example of a renewable resource?\nA. Coal\nB. Oil\nC. Wind\nD. Natural gas",
            "Wind",
        ),
        (
            "Which organ pumps blood through the body?\nA. Lung\nB. Brain\nC. Heart\nD. Liver",
            "Heart",
        ),
        (
            "What do bees collect from flowers?\nA. Sand\nB. Pollen\nC. Rocks\nD. Ice",
            "Pollen",
        ),
        (
            "Which is a mammal?\nA. Shark\nB. Dolphin\nC. Trout\nD. Lizard",
            "Dolphin",
        ),
        (
            "Which is the largest planet?\nA. Earth\nB. Mars\nC. Jupiter\nD. Venus",
            "Jupiter",
        ),
        (
            "What causes day and night on Earth?\nA. The Moon\nB. Earth's rotation\nC. Earth's orbit\nD. The Sun's rotation",
            "Earth's rotation",
        ),
        (
            "Which layer of the Earth is liquid?\nA. Crust\nB. Mantle\nC. Outer core\nD. Inner core",
            "Outer core",
        ),
        (
            "What type of energy is stored in food?\nA. Thermal\nB. Chemical\nC. Nuclear\nD. Light",
            "Chemical",
        ),
        (
            "Which organ is part of the respiratory system?\nA. Heart\nB. Lung\nC. Kidney\nD. Stomach",
            "Lung",
        ),
        (
            "Which process changes liquid water into vapor?\nA. Condensation\nB. Evaporation\nC. Freezing\nD. Melting",
            "Evaporation",
        ),
    ]
    examples = []
    for prompt, answer in samples:
        examples.append(
            SelfReflectionExample(
                input_question=prompt,
                core_question="Which option best answers the question?",
                reasoning=(
                    "Evaluate each choice against known facts.\n"
                    "Select the option that directly matches the correct fact."
                ),
                answer=answer,
            )
        )
    return examples


def get_phase_b_examples() -> list[SelfReflectionExample]:
    """Phase B examples based on dominant failure buckets."""
    examples = []
    examples.extend(_gsm8k_rate_examples())
    examples.extend(_gsm8k_percent_examples())
    examples.extend(_gsm8k_total_examples())
    examples.extend(_gsm8k_comparison_examples())
    examples.extend(_gsm8k_remaining_examples())
    examples.extend(_boolq_examples())
    examples.extend(_arc_mcq_examples())
    return examples
