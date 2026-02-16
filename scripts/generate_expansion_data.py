#!/usr/bin/env python3
"""Generate dimensional-expansion training data.

Each sample requires progressive loop closure — holding multiple constraints
simultaneously, discovering which interact, failing at lower-dimensional
approaches, and resolving through higher-dimensional reasoning.

No rule labels. No template answers. The reasoning discovers structure.

Usage:
  python scripts/generate_expansion_data.py \
    --n-train 2000 --n-eval 500 --output-dir data/training
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path


# =============================================================================
# Generator 1: Multi-Constraint Logic
# =============================================================================

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Kim", "Leo",
]
_COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
_NUMBERS = list(range(1, 10))


def _gen_constraint_logic(rng: random.Random, difficulty: int) -> dict | None:
    """Generate a constraint satisfaction problem over entities and properties.

    difficulty = number of constraints (2–5).
    Returns {"text": "..."} or None if generation fails.
    """
    n_entities = min(difficulty + 1, 5)
    names = rng.sample(_NAMES, n_entities)
    colors = rng.sample(_COLORS, n_entities)
    numbers = rng.sample(_NUMBERS, n_entities)

    # Build the correct assignment
    assignment = {}
    for i, name in enumerate(names):
        assignment[name] = {"color": colors[i], "number": numbers[i]}

    # Generate constraints from the assignment
    constraints = []
    constraint_texts = []

    # Type 1: Direct assignment (easy, always include one)
    i = rng.randrange(n_entities)
    constraints.append(("direct_color", names[i], colors[i]))
    constraint_texts.append(f"{names[i]} picks {colors[i]}.")

    # Type 2: Relational (number comparison)
    if n_entities >= 2 and difficulty >= 2:
        i, j = rng.sample(range(n_entities), 2)
        if numbers[i] < numbers[j]:
            constraints.append(("less_than", names[i], names[j]))
            constraint_texts.append(
                f"{names[i]}'s number is less than {names[j]}'s number."
            )
        else:
            constraints.append(("greater_than", names[i], names[j]))
            constraint_texts.append(
                f"{names[i]}'s number is greater than {names[j]}'s number."
            )

    # Type 3: Exclusion
    if difficulty >= 3:
        i = rng.randrange(n_entities)
        wrong_color = rng.choice([c for c in _COLORS if c != colors[i]])
        constraints.append(("not_color", names[i], wrong_color))
        constraint_texts.append(f"{names[i]} does not pick {wrong_color}.")

    # Type 4: Conditional (if X picks color Y, then X's number is Z)
    if difficulty >= 4:
        i = rng.randrange(n_entities)
        constraints.append(("conditional", names[i], colors[i], numbers[i]))
        constraint_texts.append(
            f"If someone picks {colors[i]}, their number is {numbers[i]}."
        )

    # Type 5: Sum constraint
    if difficulty >= 5 and n_entities >= 3:
        i, j = rng.sample(range(n_entities), 2)
        total = numbers[i] + numbers[j]
        constraints.append(("sum", names[i], names[j], total))
        constraint_texts.append(
            f"{names[i]}'s number plus {names[j]}'s number equals {total}."
        )

    # Build the problem text
    avail_colors = ", ".join(colors)
    avail_numbers = ", ".join(str(n) for n in sorted(numbers))

    problem = (
        f"{n_entities} people — {', '.join(names)} — each pick a different color "
        f"from {{{avail_colors}}} and a different number from {{{avail_numbers}}}.\n\n"
        f"Constraints:\n"
    )
    for i, ct in enumerate(constraint_texts, 1):
        problem += f"{i}. {ct}\n"
    problem += "\nWho picks what?\n\n"

    # Build reasoning that shows attempted assignment, conflict, resolution
    reasoning = ""

    # Greedy attempt (intentionally incomplete — only uses first constraint)
    greedy_name = constraints[0][1]
    greedy_color = constraints[0][2] if constraints[0][0] == "direct_color" else "unknown"

    reasoning += (
        f"Start with what's known: {greedy_name} picks {greedy_color} "
        f"(from constraint 1).\n\n"
    )

    if len(constraints) >= 2:
        reasoning += (
            f"Try assigning the remaining people greedily.\n"
        )
        # Show partial assignment attempt
        remaining_names = [n for n in names if n != greedy_name]
        remaining_colors = [c for c in colors if c != greedy_color]
        remaining_numbers = list(sorted(numbers))

        # Attempt that hits a constraint
        attempted = rng.choice(remaining_names)
        attempted_color = rng.choice(remaining_colors)
        reasoning += f"Try: {attempted} picks {attempted_color}.\n"

        # Check against constraints
        actual_color = assignment[attempted]["color"]
        if attempted_color != actual_color:
            # Find which constraint is violated
            for ci, c in enumerate(constraints):
                if c[0] == "not_color" and c[1] == attempted and c[2] == attempted_color:
                    reasoning += (
                        f"Check constraint {ci + 1}: {constraint_texts[ci]}\n"
                        f"Conflict! {attempted} cannot pick {attempted_color}.\n\n"
                    )
                    break
                if c[0] == "conditional" and c[1] == attempted:
                    reasoning += (
                        f"Check constraint {ci + 1}: {constraint_texts[ci]}\n"
                        f"This constrains {attempted}'s options.\n\n"
                    )
                    break
            else:
                reasoning += (
                    f"This doesn't immediately violate a constraint, "
                    f"but let's check if it's consistent with all constraints together.\n\n"
                )

        reasoning += (
            "Need to consider all constraints simultaneously.\n\n"
        )

    # Correct solution with verification
    reasoning += "Working through all constraints together:\n\n"
    for name in names:
        c = assignment[name]["color"]
        n = assignment[name]["number"]
        reasoning += f"  {name}: color={c}, number={n}\n"

    reasoning += "\nVerification:\n"
    for i, ct in enumerate(constraint_texts, 1):
        reasoning += f"  Constraint {i} ({ct}): "
        c = constraints[i - 1]
        if c[0] == "direct_color":
            reasoning += f"{c[1]} has {c[2]}. Satisfied.\n"
        elif c[0] in ("less_than", "greater_than"):
            n1 = assignment[c[1]]["number"]
            n2 = assignment[c[2]]["number"]
            if c[0] == "less_than":
                reasoning += f"{n1} < {n2}. Satisfied.\n"
            else:
                reasoning += f"{n1} > {n2}. Satisfied.\n"
        elif c[0] == "not_color":
            actual = assignment[c[1]]["color"]
            reasoning += f"{c[1]} has {actual}, not {c[2]}. Satisfied.\n"
        elif c[0] == "conditional":
            who_has_color = [n for n in names if assignment[n]["color"] == c[2]]
            if who_has_color:
                reasoning += (
                    f"{who_has_color[0]} has {c[2]} and number {assignment[who_has_color[0]]['number']}={c[3]}. Satisfied.\n"
                )
            else:
                reasoning += "No one has that color. Vacuously satisfied.\n"
        elif c[0] == "sum":
            n1 = assignment[c[1]]["number"]
            n2 = assignment[c[2]]["number"]
            reasoning += f"{n1} + {n2} = {n1 + n2} = {c[3]}. Satisfied.\n"

    reasoning += "\nAll constraints satisfied."

    text = problem + reasoning
    return {"text": text}


# =============================================================================
# Generator 2: Rate Problems with Traps
# =============================================================================


def _gen_rate_trap(rng: random.Random, difficulty: int) -> dict | None:
    """Generate rate/work problems where intuitive answer is wrong.

    difficulty controls the number of interacting rates.
    """
    trap_type = rng.choice(["parallel_work", "meeting_point", "pool_fill"])

    if trap_type == "parallel_work":
        # N machines make N widgets in T minutes. How long for M machines to make M?
        n = rng.randint(3, 10)
        t = rng.choice([3, 4, 5, 6, 8, 10])
        m = n * rng.randint(2, 5)
        wrong_answer = t * m // n  # intuitive (wrong: linear scaling)
        correct_answer = t  # each machine makes 1 widget in T minutes

        items = rng.choice([
            ("machines", "widgets"),
            ("printers", "pages"),
            ("ovens", "cakes"),
            ("workers", "reports"),
        ])

        problem = (
            f"{n} {items[0]} can produce {n} {items[1]} in {t} minutes. "
            f"How long would it take {m} {items[0]} to produce {m} {items[1]}?\n\n"
        )

        reasoning = (
            f"Intuitive approach: {m} is {m//n}× more than {n}, so multiply time by {m//n}.\n"
            f"That gives {wrong_answer} minutes.\n\n"
            f"Wait, let me check that. If {n} {items[0]} make {n} {items[1]} in {t} minutes, "
            f"that means each {items[0][:-1] if items[0].endswith('s') else items[0]} makes "
            f"1 {items[1][:-1] if items[1].endswith('s') else items[1]} in {t} minutes.\n\n"
            f"So {m} {items[0]} would each make 1 {items[1][:-1] if items[1].endswith('s') else items[1]} "
            f"in {t} minutes — producing {m} {items[1]} total in {t} minutes.\n\n"
            f"The intuitive answer ({wrong_answer} minutes) was wrong because it treated "
            f"the work as sequential when the {items[0]} work in parallel.\n\n"
            f"Verification: Rate per {items[0][:-1] if items[0].endswith('s') else items[0]} = "
            f"1/{t} {items[1][:-1] if items[1].endswith('s') else items[1]}/minute. "
            f"{m} {items[0]} × {t} minutes × 1/{t} = {m} {items[1]}. Correct.\n\n"
            f"Answer: {correct_answer} minutes."
        )

    elif trap_type == "meeting_point":
        # Two travelers, different speeds, one starts later
        d = rng.choice([60, 80, 100, 120, 150, 200])
        v1 = rng.choice([10, 15, 20, 25, 30])
        v2 = rng.choice([v for v in [10, 15, 20, 25, 30, 40, 50] if v != v1])
        delay = rng.choice([1, 2, 3])

        # They travel toward each other, one starts `delay` hours late
        # Person 1 travels for t hours, person 2 for (t - delay) hours
        # v1*t + v2*(t - delay) = d
        # t = (d + v2*delay) / (v1 + v2)
        t_exact_num = d + v2 * delay
        t_exact_den = v1 + v2

        if t_exact_num % t_exact_den != 0:
            # Pick values that divide evenly
            return None

        t = t_exact_num // t_exact_den
        if t <= delay:
            return None

        wrong_t = d // (v1 + v2)  # ignoring the delay

        name1 = rng.choice(["Alex", "Sam", "Jordan", "Taylor"])
        name2 = rng.choice([n for n in ["Morgan", "Casey", "Riley", "Pat"] if n != name1])

        problem = (
            f"{name1} and {name2} are {d} km apart and travel toward each other. "
            f"{name1} travels at {v1} km/h. {name2} travels at {v2} km/h but starts "
            f"{delay} hour{'s' if delay > 1 else ''} later. "
            f"When do they meet (hours after {name1} starts)?\n\n"
        )

        reasoning = (
            f"Quick estimate: Total closing speed = {v1} + {v2} = {v1 + v2} km/h. "
            f"Time = {d}/{v1 + v2} = {wrong_t} hours.\n\n"
            f"But wait — {name2} starts {delay} hour{'s' if delay > 1 else ''} late. "
            f"In that time, {name1} covers {v1}×{delay} = {v1 * delay} km.\n\n"
            f"Remaining distance when {name2} starts: {d} - {v1 * delay} = {d - v1 * delay} km.\n"
            f"Combined speed: {v1 + v2} km/h.\n"
            f"Time after {name2} starts: {d - v1 * delay}/{v1 + v2} = "
            f"{(d - v1 * delay) // (v1 + v2)} hours.\n\n"
            f"Total time from {name1}'s start: {delay} + {(d - v1 * delay) // (v1 + v2)} "
            f"= {t} hours.\n\n"
            f"Verification:\n"
            f"  {name1} travels {v1}×{t} = {v1 * t} km.\n"
            f"  {name2} travels {v2}×{t - delay} = {v2 * (t - delay)} km.\n"
            f"  Total: {v1 * t} + {v2 * (t - delay)} = {v1 * t + v2 * (t - delay)} km "
            f"= {d} km. Correct.\n\n"
            f"Answer: {t} hours after {name1} starts."
        )

    elif trap_type == "pool_fill":
        # Pool with inlet and drain, or multiple inlets at different rates
        # Inlet fills in A hours, drain empties in B hours
        a = rng.choice([2, 3, 4, 5, 6])
        b = rng.choice([b for b in [3, 4, 5, 6, 8, 10, 12] if b > a])

        # Net rate = 1/a - 1/b = (b-a)/(ab)
        # Time = ab/(b-a)
        net_num = a * b
        net_den = b - a

        if net_num % net_den != 0:
            return None

        correct_time = net_num // net_den
        wrong_time = a  # intuitive: just use the inlet time

        problem = (
            f"A tank has an inlet pipe that fills it in {a} hours and a drain that "
            f"empties it in {b} hours. If both are open, how long to fill the tank?\n\n"
        )

        reasoning = (
            f"Intuitive answer: The inlet fills in {a} hours, so {wrong_time} hours.\n\n"
            f"But the drain is also open! Need to account for both.\n\n"
            f"Inlet rate: 1/{a} tank/hour\n"
            f"Drain rate: 1/{b} tank/hour (removing water)\n"
            f"Net rate: 1/{a} - 1/{b} = {b}/({a}×{b}) - {a}/({a}×{b}) "
            f"= ({b}-{a})/({a}×{b}) = {b - a}/{a * b} tank/hour\n\n"
            f"Time to fill: {a * b}/{b - a} = {correct_time} hours.\n\n"
            f"Verification: In {correct_time} hours:\n"
            f"  Inlet adds: {correct_time}/{a} = {correct_time / a:.2f} tanks\n"
            f"  Drain removes: {correct_time}/{b} = {correct_time / b:.2f} tanks\n"
            f"  Net: {correct_time / a:.2f} - {correct_time / b:.2f} = "
            f"{correct_time / a - correct_time / b:.2f} tank. "
            f"{'Correct.' if abs(correct_time / a - correct_time / b - 1.0) < 0.01 else 'Check needed.'}\n\n"
            f"Answer: {correct_time} hours."
        )

    if difficulty >= 3:
        # Add follow-up constraint
        reasoning += (
            f"\n\nFollow-up: If the tank needs to be at least half full within "
            f"the first {correct_time // 2 if correct_time > 1 else 1} hours, "
            f"is this achievable?\n"
            f"At t={correct_time // 2 if correct_time > 1 else 1}: "
            f"net filled = {(correct_time // 2 if correct_time > 1 else 1) * (b - a) / (a * b):.2f} tanks. "
            f"{'Yes' if (correct_time // 2 if correct_time > 1 else 1) * (b - a) / (a * b) >= 0.5 else 'No'}, "
            f"{'that is' if (correct_time // 2 if correct_time > 1 else 1) * (b - a) / (a * b) >= 0.5 else 'that is not'} "
            f"at least half."
        )

    text = problem + reasoning
    return {"text": text}


# =============================================================================
# Generator 3: Multi-Step Math with Verification
# =============================================================================


def _gen_verified_math(rng: random.Random, difficulty: int) -> dict | None:
    """Generate multi-step arithmetic with intermediate verification checkpoints."""
    problem_type = rng.choice(["store_profit", "distance_fuel", "budget_allocation"])

    if problem_type == "store_profit":
        n_items = rng.randint(20, 100)
        cost_each = rng.choice([5, 8, 10, 12, 15, 20, 25])
        total_cost = n_items * cost_each

        # Sell portions at different prices
        portions = []
        remaining = n_items
        for i in range(min(difficulty, 4)):
            if i == min(difficulty, 4) - 1:
                count = remaining
            else:
                count = rng.randint(1, remaining - (min(difficulty, 4) - i - 1))
                remaining -= count
            markup = rng.choice([10, 15, 20, 25, 30, 40, 50])
            if rng.random() < 0.3 and i > 0:
                # Some at a loss
                markup = -rng.choice([5, 10, 15])
            sell_price = round(cost_each * (1 + markup / 100))
            portions.append((count, sell_price, markup))

        total_revenue = sum(count * price for count, price, _ in portions)
        profit = total_revenue - total_cost

        problem = (
            f"A store buys {n_items} items at ${cost_each} each (total investment: ${total_cost}).\n"
        )
        for i, (count, price, markup) in enumerate(portions):
            if markup >= 0:
                problem += f"  - Sells {count} items at ${price} each ({markup}% markup)\n"
            else:
                problem += f"  - Sells {count} items at ${price} each ({abs(markup)}% discount)\n"
        problem += f"\nWhat is the total profit or loss?\n\n"

        reasoning = "Step by step:\n\n"
        running_revenue = 0
        for i, (count, price, markup) in enumerate(portions):
            batch_revenue = count * price
            running_revenue += batch_revenue
            reasoning += (
                f"Batch {i + 1}: {count} × ${price} = ${batch_revenue}\n"
            )
            if difficulty >= 3 and i == len(portions) // 2:
                reasoning += (
                    f"\nMidpoint check: Revenue so far = ${running_revenue}. "
                    f"Investment = ${total_cost}. "
                    f"{'Ahead' if running_revenue > total_cost / 2 else 'Behind'} "
                    f"pace for profit.\n\n"
                )

        reasoning += (
            f"\nTotal revenue: ${running_revenue}\n"
            f"Total cost: ${total_cost}\n"
            f"{'Profit' if profit >= 0 else 'Loss'}: "
            f"${running_revenue} - ${total_cost} = ${profit}\n\n"
        )

        # Verification
        reasoning += (
            f"Verification:\n"
            f"  Items sold: {' + '.join(str(c) for c, _, _ in portions)} "
            f"= {sum(c for c, _, _ in portions)} "
            f"{'= ' + str(n_items) + '. All accounted for.' if sum(c for c, _, _ in portions) == n_items else '≠ ' + str(n_items) + '. ERROR!'}\n"
            f"  Revenue: {' + '.join(f'${c * p}' for c, p, _ in portions)} = ${total_revenue}\n"
            f"  Profit check: ${total_revenue} - ${total_cost} = ${profit}. Consistent.\n\n"
            f"Answer: {'Profit' if profit >= 0 else 'Loss'} of ${abs(profit)}."
        )

    elif problem_type == "distance_fuel":
        legs = []
        total_distance = 0
        fuel_rate_base = rng.choice([25, 30, 35, 40])  # km per liter

        for i in range(difficulty):
            dist = rng.randint(50, 200)
            # Different terrain affects fuel rate
            terrain = rng.choice(["highway", "city", "mountain"])
            if terrain == "highway":
                rate = fuel_rate_base
            elif terrain == "city":
                rate = int(fuel_rate_base * 0.7)
            else:
                rate = int(fuel_rate_base * 0.5)
            legs.append((dist, terrain, rate))
            total_distance += dist

        problem = (
            f"A car's base fuel efficiency is {fuel_rate_base} km/liter.\n"
            f"  Highway: {fuel_rate_base} km/L\n"
            f"  City: {int(fuel_rate_base * 0.7)} km/L (30% worse)\n"
            f"  Mountain: {int(fuel_rate_base * 0.5)} km/L (50% worse)\n\n"
            f"Trip legs:\n"
        )
        for i, (dist, terrain, _) in enumerate(legs):
            problem += f"  Leg {i + 1}: {dist} km on {terrain}\n"

        fuel_price = rng.choice([1.20, 1.50, 1.80, 2.00])
        problem += f"\nFuel costs ${fuel_price:.2f}/liter. What's the total fuel cost?\n\n"

        reasoning = "Calculate fuel for each leg:\n\n"
        total_fuel = 0
        for i, (dist, terrain, rate) in enumerate(legs):
            fuel = dist / rate
            total_fuel += fuel
            reasoning += (
                f"Leg {i + 1} ({terrain}): {dist} km ÷ {rate} km/L = {fuel:.2f} L\n"
            )

        total_cost = total_fuel * fuel_price
        reasoning += (
            f"\nTotal fuel: {total_fuel:.2f} L\n"
            f"Total cost: {total_fuel:.2f} × ${fuel_price:.2f} = ${total_cost:.2f}\n\n"
            f"Verification:\n"
            f"  Total distance: {' + '.join(str(d) for d, _, _ in legs)} = {total_distance} km\n"
            f"  Average rate: {total_distance / total_fuel:.1f} km/L "
            f"(should be between {int(fuel_rate_base * 0.5)} and {fuel_rate_base})\n"
            f"  {'Consistent.' if int(fuel_rate_base * 0.5) <= total_distance / total_fuel <= fuel_rate_base else 'ERROR: outside expected range!'}\n\n"
            f"Answer: ${total_cost:.2f}"
        )

    elif problem_type == "budget_allocation":
        total_budget = rng.choice([1000, 2000, 5000, 10000])
        categories = rng.sample(
            ["rent", "food", "transport", "savings", "utilities", "entertainment"],
            min(difficulty + 1, 5),
        )

        # Assign percentages that must sum to 100
        pcts = []
        remaining_pct = 100
        for i, cat in enumerate(categories):
            if i == len(categories) - 1:
                pcts.append(remaining_pct)
            else:
                p = rng.randint(10, remaining_pct - 10 * (len(categories) - i - 1))
                pcts.append(p)
                remaining_pct -= p

        allocations = [(cat, pct, total_budget * pct // 100) for cat, pct in zip(categories, pcts)]

        problem = f"Budget: ${total_budget}\nAllocate as follows:\n"
        for cat, pct, _ in allocations:
            problem += f"  {cat}: {pct}%\n"

        # Add a constraint that makes naive allocation wrong
        constraint_cat = rng.choice(categories[:2])
        min_amount = rng.choice([200, 300, 400, 500])
        constraint_allocation = next(a for c, _, a in allocations if c == constraint_cat)

        if constraint_allocation < min_amount:
            problem += (
                f"\nConstraint: {constraint_cat} must be at least ${min_amount}.\n"
                f"If needed, reduce the largest category to compensate.\n\n"
            )
            # Need to adjust
            deficit = min_amount - constraint_allocation
            largest_cat = max(allocations, key=lambda x: x[2])

            reasoning = (
                f"Initial allocation:\n"
            )
            for cat, pct, amt in allocations:
                reasoning += f"  {cat}: {pct}% = ${amt}\n"

            reasoning += (
                f"\nCheck constraint: {constraint_cat} = ${constraint_allocation} "
                f"< ${min_amount}. Constraint violated!\n\n"
                f"Need ${deficit} more for {constraint_cat}.\n"
                f"Largest category: {largest_cat[0]} at ${largest_cat[2]}.\n"
                f"Reduce {largest_cat[0]} by ${deficit}: "
                f"${largest_cat[2]} → ${largest_cat[2] - deficit}.\n"
                f"Increase {constraint_cat}: ${constraint_allocation} → ${min_amount}.\n\n"
                f"Adjusted allocation:\n"
            )
            for cat, pct, amt in allocations:
                if cat == constraint_cat:
                    reasoning += f"  {cat}: ${min_amount}\n"
                elif cat == largest_cat[0]:
                    reasoning += f"  {cat}: ${largest_cat[2] - deficit}\n"
                else:
                    reasoning += f"  {cat}: ${amt}\n"

            # Verify sum
            adjusted_total = sum(
                min_amount if c == constraint_cat
                else (largest_cat[2] - deficit if c == largest_cat[0] else a)
                for c, _, a in allocations
            )
            reasoning += (
                f"\nVerification: Total = ${adjusted_total} "
                f"{'= $' + str(total_budget) + '. Correct.' if adjusted_total == total_budget else '≠ $' + str(total_budget) + '. ERROR!'}\n"
                f"Constraint: {constraint_cat} = ${min_amount} ≥ ${min_amount}. Satisfied.\n\n"
                f"Answer: {constraint_cat} gets ${min_amount}, "
                f"{largest_cat[0]} reduced to ${largest_cat[2] - deficit}."
            )
        else:
            # Constraint already satisfied, add a different wrinkle
            problem += (
                f"\nConstraint: {constraint_cat} must be at least ${min_amount}.\n"
                f"What is the maximum that can go to savings?\n\n"
            )
            savings_amt = next((a for c, _, a in allocations if c == "savings"), 0)
            reasoning = (
                f"Initial allocation:\n"
            )
            for cat, pct, amt in allocations:
                reasoning += f"  {cat}: {pct}% = ${amt}\n"
            reasoning += (
                f"\nConstraint check: {constraint_cat} = ${constraint_allocation} ≥ ${min_amount}. Satisfied.\n"
                f"{'savings' if savings_amt else 'No savings category.'}\n\n"
                f"Answer: Savings gets ${savings_amt}."
            )

    text = problem + reasoning
    return {"text": text}


# =============================================================================
# Generator 4: Scheduling with Conflicts
# =============================================================================


def _gen_scheduling(rng: random.Random, difficulty: int) -> dict | None:
    """Generate task-to-worker scheduling with conflicting constraints."""
    n_tasks = difficulty + 1
    n_workers = difficulty

    worker_names = rng.sample(["Alex", "Blake", "Casey", "Drew", "Ellis", "Fran"], n_workers)
    task_names = [f"Task {chr(65 + i)}" for i in range(n_tasks)]

    # Each task has a time requirement
    task_times = {t: rng.choice([1, 2, 3, 4]) for t in task_names}

    # Workers have limited hours
    worker_hours = {w: rng.randint(3, 6) for w in worker_names}

    # Skills: each worker can do some tasks
    worker_skills = {}
    for w in worker_names:
        n_skills = rng.randint(2, min(n_tasks, 4))
        worker_skills[w] = rng.sample(task_names, n_skills)

    # Find a valid assignment (brute force for small sizes)
    valid_assignment = None
    for perm in itertools.permutations(range(n_tasks)):
        assignment = {}
        worker_load = {w: 0 for w in worker_names}
        ok = True
        for task_idx in perm:
            task = task_names[task_idx]
            assigned = False
            for w in worker_names:
                if (
                    task in worker_skills[w]
                    and worker_load[w] + task_times[task] <= worker_hours[w]
                ):
                    assignment[task] = w
                    worker_load[w] += task_times[task]
                    assigned = True
                    break
            if not assigned:
                ok = False
                break
        if ok and len(assignment) == n_tasks:
            valid_assignment = assignment
            break

    if valid_assignment is None:
        return None

    # Build problem
    problem = f"Assign {n_tasks} tasks to {n_workers} workers.\n\n"
    problem += "Tasks and time required:\n"
    for t in task_names:
        problem += f"  {t}: {task_times[t]} hours\n"
    problem += "\nWorker availability and skills:\n"
    for w in worker_names:
        problem += f"  {w}: {worker_hours[w]} hours available, can do {', '.join(worker_skills[w])}\n"
    problem += "\nAssign all tasks. Each task to exactly one worker. No worker exceeds their hours.\n\n"

    # Reasoning with attempted assignment that fails
    reasoning = "Try assigning greedily (first available worker for each task):\n\n"
    greedy_assignment = {}
    greedy_load = {w: 0 for w in worker_names}
    failed_task = None

    for task in task_names:
        assigned = False
        for w in worker_names:
            if task in worker_skills[w]:
                if greedy_load[w] + task_times[task] <= worker_hours[w]:
                    greedy_assignment[task] = w
                    greedy_load[w] += task_times[task]
                    reasoning += f"  {task} → {w} ({greedy_load[w]}/{worker_hours[w]} hours used)\n"
                    assigned = True
                    break
                else:
                    reasoning += (
                        f"  {task} → {w}? No, {w} has {greedy_load[w]}/{worker_hours[w]} hours, "
                        f"needs {task_times[task]} more.\n"
                    )
        if not assigned:
            failed_task = task
            reasoning += f"  {task} → No worker available! Greedy assignment fails.\n\n"
            break

    if failed_task:
        reasoning += (
            f"The greedy approach failed because {failed_task} couldn't be assigned.\n"
            f"Need to reconsider earlier assignments to make room.\n\n"
        )

    reasoning += "Correct assignment (considering all constraints together):\n\n"
    final_load = {w: 0 for w in worker_names}
    for task in task_names:
        w = valid_assignment[task]
        final_load[w] += task_times[task]
        reasoning += f"  {task} → {w}\n"

    reasoning += "\nVerification:\n"
    for w in worker_names:
        tasks_for_w = [t for t, assigned_w in valid_assignment.items() if assigned_w == w]
        total_time = sum(task_times[t] for t in tasks_for_w)
        reasoning += (
            f"  {w}: {', '.join(tasks_for_w) if tasks_for_w else 'no tasks'} "
            f"= {total_time}/{worker_hours[w]} hours. "
            f"{'OK' if total_time <= worker_hours[w] else 'OVERLOADED!'}\n"
        )
        for t in tasks_for_w:
            if t not in worker_skills[w]:
                reasoning += f"    ERROR: {w} cannot do {t}!\n"

    all_assigned = set(valid_assignment.keys())
    missing = set(task_names) - all_assigned
    reasoning += f"  All tasks assigned: {'Yes' if not missing else 'No — missing: ' + ', '.join(missing)}\n"
    reasoning += "\nAll constraints satisfied."

    text = problem + reasoning
    return {"text": text}


# =============================================================================
# Generator 5: Self-Correction Chains
# =============================================================================


def _gen_self_correction(rng: random.Random, difficulty: int) -> dict | None:
    """Generate problems where an initial approach is wrong and must be corrected."""
    correction_type = rng.choice(["algebra_check", "logic_fallacy", "estimation_catch"])

    if correction_type == "algebra_check":
        # Classic bat-and-ball variants
        total = rng.choice([110, 220, 330, 150, 250])
        diff = rng.choice([100, 200, 50, 150])

        if total <= diff:
            return None

        correct_small = (total - diff) / 2
        if correct_small != int(correct_small):
            return None
        correct_small = int(correct_small)
        correct_large = correct_small + diff
        wrong_small = total - diff  # common mistake

        item_large = rng.choice(["bat", "jacket", "laptop", "book", "guitar"])
        item_small = rng.choice(["ball", "hat", "charger", "bookmark", "pick"])

        total_dollars = total / 100
        diff_dollars = diff / 100
        correct_small_dollars = correct_small / 100
        correct_large_dollars = correct_large / 100
        wrong_small_dollars = wrong_small / 100

        problem = (
            f"A {item_large} and a {item_small} cost ${total_dollars:.2f} in total. "
            f"The {item_large} costs ${diff_dollars:.2f} more than the {item_small}. "
            f"How much does the {item_small} cost?\n\n"
        )

        reasoning = (
            f"First instinct: The {item_large} costs ${diff_dollars:.2f} more, "
            f"so the {item_small} costs ${total_dollars:.2f} - ${diff_dollars:.2f} = "
            f"${wrong_small_dollars:.2f}.\n\n"
            f"Wait, let me check: if the {item_small} = ${wrong_small_dollars:.2f}, "
            f"then the {item_large} = ${wrong_small_dollars:.2f} + ${diff_dollars:.2f} = "
            f"${wrong_small_dollars + diff_dollars:.2f}.\n"
            f"Total: ${wrong_small_dollars:.2f} + ${wrong_small_dollars + diff_dollars:.2f} = "
            f"${wrong_small_dollars + wrong_small_dollars + diff_dollars:.2f}.\n"
            f"But the total should be ${total_dollars:.2f}. "
            f"${wrong_small_dollars + wrong_small_dollars + diff_dollars:.2f} ≠ ${total_dollars:.2f}. "
            f"The first instinct was WRONG.\n\n"
            f"Correct approach: Let {item_small} = x.\n"
            f"Then {item_large} = x + ${diff_dollars:.2f}.\n"
            f"Total: x + (x + ${diff_dollars:.2f}) = ${total_dollars:.2f}\n"
            f"2x + ${diff_dollars:.2f} = ${total_dollars:.2f}\n"
            f"2x = ${total_dollars - diff_dollars:.2f}\n"
            f"x = ${correct_small_dollars:.2f}\n\n"
            f"Verification: {item_small} = ${correct_small_dollars:.2f}, "
            f"{item_large} = ${correct_large_dollars:.2f}.\n"
            f"Difference: ${correct_large_dollars:.2f} - ${correct_small_dollars:.2f} = ${diff_dollars:.2f}. Correct.\n"
            f"Sum: ${correct_small_dollars:.2f} + ${correct_large_dollars:.2f} = ${total_dollars:.2f}. Correct.\n\n"
            f"Answer: ${correct_small_dollars:.2f}"
        )

    elif correction_type == "logic_fallacy":
        # Affirming the consequent or denying the antecedent
        fallacy = rng.choice(["affirming_consequent", "denying_antecedent"])

        if fallacy == "affirming_consequent":
            premises = rng.choice([
                ("it rains", "the ground is wet", "the sprinkler was on"),
                ("studying hard", "getting good grades", "the test was easy"),
                ("the alarm rings", "people evacuate", "it was a fire drill"),
                ("exercising", "being healthy", "good genetics"),
            ])
            p, q, alt = premises

            problem = (
                f"If {p}, then {q}.\n"
                f"We observe: {q}.\n"
                f"Can we conclude {p}?\n\n"
            )

            reasoning = (
                f"Initial thought: {q} is true, and {p} causes {q}, so {p} must be true.\n\n"
                f"Wait — that's affirming the consequent (a logical fallacy).\n"
                f"The rule says: {p} → {q}.\n"
                f"But {q} can be true for OTHER reasons. For example: {alt}.\n\n"
                f"The valid inference from {q} is:\n"
                f"  If NOT {q}, then NOT {p} (contrapositive — valid).\n"
                f"  If {q}, then MAYBE {p} (NOT valid — other causes exist).\n\n"
                f"Verification with counterexample:\n"
                f"  Scenario: {alt}. Then {q} is true but {p} is false.\n"
                f"  This proves we CANNOT conclude {p} from {q} alone.\n\n"
                f"Answer: No, we cannot conclude {p}. "
                f"This would be affirming the consequent (invalid reasoning). "
                f"{q} could be caused by {alt} instead."
            )

        else:  # denying_antecedent
            premises = rng.choice([
                ("it rains", "the picnic is canceled", "if there's a thunderstorm warning"),
                ("you study", "you pass", "if the test is very easy"),
                ("the store is open", "you can buy groceries", "if you order online"),
            ])
            p, q, alt = premises

            problem = (
                f"If {p}, then {q}.\n"
                f"We observe: NOT {p}.\n"
                f"Can we conclude NOT {q}?\n\n"
            )

            reasoning = (
                f"Initial thought: {p} is false, and {p} leads to {q}, so {q} must be false.\n\n"
                f"Wait — that's denying the antecedent (a logical fallacy).\n"
                f"The rule says: {p} → {q}.\n"
                f"NOT {p} does NOT mean NOT {q}. {q} could still happen another way.\n\n"
                f"Example: {alt} — then {q} even without {p}.\n\n"
                f"The valid inferences are:\n"
                f"  {p} → {q} (given)\n"
                f"  NOT {q} → NOT {p} (contrapositive — valid)\n"
                f"  NOT {p} → ??? (NOTHING follows — invalid)\n\n"
                f"Answer: No. Denying the antecedent is invalid. "
                f"{q} might still occur {alt}."
            )

    elif correction_type == "estimation_catch":
        # Calculation where rounding early gives a very different answer
        base = rng.randint(10, 50)
        pct1 = rng.choice([10, 15, 20, 25])
        pct2 = rng.choice([10, 15, 20, 25, 30])

        # Successive percentage changes
        after_first = base * (1 + pct1 / 100)
        after_second = after_first * (1 - pct2 / 100)
        naive_net = pct1 - pct2  # common wrong answer: just subtract percentages

        problem = (
            f"A price starts at ${base}. It increases by {pct1}%, "
            f"then decreases by {pct2}%. What's the final price?\n\n"
        )

        reasoning = (
            f"Quick estimate: +{pct1}% then -{pct2}% ≈ net {'+' if naive_net > 0 else ''}{naive_net}%.\n"
            f"That gives ${base} × {1 + naive_net / 100:.2f} = ${base * (1 + naive_net / 100):.2f}.\n\n"
            f"But wait — percentages don't add linearly when applied sequentially!\n"
            f"The {pct2}% decrease applies to the INCREASED price, not the original.\n\n"
            f"Step 1: ${base} × {1 + pct1 / 100:.2f} = ${after_first:.2f}\n"
            f"Step 2: ${after_first:.2f} × {1 - pct2 / 100:.2f} = ${after_second:.2f}\n\n"
            f"The naive estimate was ${base * (1 + naive_net / 100):.2f}, "
            f"but the correct answer is ${after_second:.2f}.\n"
            f"Difference: ${abs(after_second - base * (1 + naive_net / 100)):.2f}.\n\n"
            f"Verification: ${after_second:.2f} / ${base} = {after_second / base:.4f} "
            f"= {(after_second / base - 1) * 100:+.2f}% net change.\n"
            f"Expected: (1 + {pct1 / 100:.2f})(1 - {pct2 / 100:.2f}) = "
            f"{(1 + pct1 / 100) * (1 - pct2 / 100):.4f} = "
            f"{((1 + pct1 / 100) * (1 - pct2 / 100) - 1) * 100:+.2f}% "
            f"(not {'+' if naive_net > 0 else ''}{naive_net}%). Correct.\n\n"
            f"Answer: ${after_second:.2f}"
        )

    text = problem + reasoning
    return {"text": text}


# =============================================================================
# Main
# =============================================================================

GENERATORS = {
    "constraint_logic": _gen_constraint_logic,
    "rate_trap": _gen_rate_trap,
    "verified_math": _gen_verified_math,
    "scheduling": _gen_scheduling,
    "self_correction": _gen_self_correction,
}


def _verify_sample(sample: dict) -> bool:
    """Basic structural verification of a generated sample."""
    if not sample or "text" not in sample:
        return False
    text = sample["text"]
    if len(text) < 100:
        return False
    # Must contain verification/checking
    if not any(kw in text.lower() for kw in ["verif", "check", "correct", "wrong", "error", "wait"]):
        return False
    # Must NOT contain rule labels
    if any(label in text for label in ["Rule: Modus", "Rule: Disjunct", "Pattern: If P"]):
        return False
    return True


def generate_dataset(
    n_samples: int,
    output_path: str,
    seed: int = 42,
    min_difficulty: int = 2,
    max_difficulty: int = 5,
    category_weights: dict[str, float] | None = None,
) -> int:
    """Generate expansion-inducing training data.

    Returns number of samples actually generated.
    """
    rng = random.Random(seed)
    weights = category_weights or {k: 1.0 for k in GENERATORS}
    categories = list(weights.keys())
    cat_weights = [weights[k] for k in categories]

    samples = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        cat = rng.choices(categories, weights=cat_weights, k=1)[0]
        difficulty = rng.randint(min_difficulty, max_difficulty)
        gen = GENERATORS[cat]

        try:
            sample = gen(rng, difficulty)
        except Exception:
            continue

        if sample and _verify_sample(sample):
            samples.append(sample)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dimensional-expansion training data",
    )
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-difficulty", type=int, default=2)
    parser.add_argument("--max-difficulty", type=int, default=5)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print 5 samples without writing",
    )
    args = parser.parse_args()

    if args.dry_run:
        rng = random.Random(args.seed)
        for i, (name, gen) in enumerate(GENERATORS.items()):
            difficulty = rng.randint(args.min_difficulty, args.max_difficulty)
            sample = gen(rng, difficulty)
            if sample:
                print(f"=== {name} (difficulty={difficulty}) ===")
                print(sample["text"][:500])
                print("...\n")
        return

    out_dir = Path(args.output_dir)

    print(f"Generating {args.n_train} training samples...")
    n_train = generate_dataset(
        args.n_train,
        str(out_dir / "expansion_train.jsonl"),
        seed=args.seed,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    print(f"  Generated {n_train} training samples → {out_dir / 'expansion_train.jsonl'}")

    print(f"Generating {args.n_eval} eval samples...")
    n_eval = generate_dataset(
        args.n_eval,
        str(out_dir / "expansion_eval.jsonl"),
        seed=args.seed + 12345,  # different seed
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    print(f"  Generated {n_eval} eval samples → {out_dir / 'expansion_eval.jsonl'}")


if __name__ == "__main__":
    main()
