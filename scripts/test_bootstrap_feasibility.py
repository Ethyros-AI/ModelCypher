#!/usr/bin/env python3
"""Bootstrap feasibility test: can the adapted model reason on NOVEL problems?

Generates logic problems with:
1. Novel domains NOT in training data (astronomy, architecture, gaming, etc.)
2. Novel variable names/entities never seen during training
3. Compositional problems (2-step chains) not in any training template

Runs base model and adapted model, verifies correctness programmatically.
If the adapted model generates correct reasoning on novel problems,
we have bootstrapping material for STaR-style iterative self-training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from novel_problems import (
    concludes_not as _concludes_not,
    first_sentences as _first_sentences,
    first_stated_answer as _first_stated_answer,
    word_match as _word_match,
)


NOVEL_PROBLEMS = [
    # --- NOVEL DOMAINS (astronomy, architecture, gaming, philosophy, oceanography) ---
    # Modus ponens - astronomy
    {
        "id": "novel_mp_1",
        "logic": "modus_ponens",
        "prompt": "If a star exhausts its hydrogen fuel, then it expands into a red giant. "
                  "The star Betelgeuse has exhausted its hydrogen fuel.\nWhat can we conclude?",
        # Strict: first answer must say "red giant" or "expands"
        "verify": lambda r: "red giant" in _first_sentences(r, 3) or "expands" in _first_sentences(r, 3),
    },
    # Modus ponens - architecture
    {
        "id": "novel_mp_2",
        "logic": "modus_ponens",
        "prompt": "If a building exceeds 150 meters in height, then it requires a tuned mass damper. "
                  "The Pinnacle Tower exceeds 150 meters in height.\nWhat follows from this?",
        # Strict: first stated answer must mention tuned mass damper
        "verify": lambda r: "tuned mass damper" in _first_stated_answer(r),
    },
    # Modus tollens - gaming
    {
        "id": "novel_mt_1",
        "logic": "modus_tollens",
        "prompt": "If a player completes all side quests, then they unlock the secret ending. "
                  "The secret ending was not unlocked.\nWhat can we conclude?",
        # Strict: first stated answer must conclude "did not complete" (not just keywords present)
        "verify": lambda r: _concludes_not(r, "complete"),
    },
    # Modus tollens - philosophy
    {
        "id": "novel_mt_2",
        "logic": "modus_tollens",
        "prompt": "If an argument is sound, then its conclusion must be true. "
                  "The conclusion of this argument is not true.\nWhat follows?",
        # Strict: first answer must say "not sound"
        "verify": lambda r: "not sound" in _first_stated_answer(r) or "not sound" in _first_sentences(r, 2),
    },
    # Disjunctive syllogism - oceanography
    {
        "id": "novel_ds_1",
        "logic": "disjunctive_syllogism",
        "prompt": "Either the deep-sea current flows northward or it flows southward. "
                  "The current does not flow northward.\nWhat must be the case?",
        # Strict: must conclude southward
        "verify": lambda r: "southward" in _first_stated_answer(r) or "southward" in _first_sentences(r, 3),
    },
    # Disjunctive syllogism - novel
    {
        "id": "novel_ds_2",
        "logic": "disjunctive_syllogism",
        "prompt": "Either the artifact is made of bronze or it is made of iron. "
                  "Analysis shows the artifact is not made of bronze.\nWhat can we determine?",
        # Strict: first answer must say iron AND not contradict with "not iron"
        "verify": lambda r: ("iron" in _first_stated_answer(r)
                             and "not made of iron" not in _first_stated_answer(r)
                             and "not iron" not in _first_stated_answer(r)),
    },
    # Hypothetical syllogism - novel
    {
        "id": "novel_hs_1",
        "logic": "hypothetical_syllogism",
        "prompt": "If the glacier melts, then the river level rises. "
                  "If the river level rises, then the downstream village floods.\n"
                  "What can we conclude from these two conditionals?",
        # Strict: must state glacier → flood chain, WITHOUT negation ("does not flood" = FAIL)
        "verify": lambda r: (
            ("glacier" in r.lower() and "flood" in r.lower())
            and "does not flood" not in _first_stated_answer(r)
            and "not flood" not in _first_stated_answer(r)
        ),
    },
    # Chain contrapositive - novel
    {
        "id": "novel_cc_1",
        "logic": "chain_contrapositive",
        "prompt": "If the reactor overheats, then the coolant system activates. "
                  "If the coolant system activates, then the alarm sounds. "
                  "The alarm did not sound.\nWhat can we conclude?",
        # Strict: first answer must conclude "not overheating/did not overheat"
        "verify": lambda r: _concludes_not(r, "overheat"),
    },
    # Biconditional - novel
    {
        "id": "novel_bi_1",
        "logic": "biconditional",
        "prompt": "A compound is an acid if and only if it donates protons. "
                  "Hydrochloric acid donates protons.\nWhat follows?",
        # Strict: first answer must say "acid" (as conclusion, not just in prompt echo)
        "verify": lambda r: ("is an acid" in _first_stated_answer(r)
                             or "is an acid" in _first_sentences(r, 2)),
    },
    # Affirming consequent fallacy - novel
    {
        "id": "novel_ac_1",
        "logic": "affirming_consequent_fallacy",
        "prompt": "If a substance is gold, then it is a metal. "
                  "This substance is a metal.\nCan we conclude it is gold?",
        # Strict: must reject the inference — "cannot", "no" (word boundary), "fallacy"
        "verify": lambda r: (
            "cannot" in _first_stated_answer(r)
            or _word_match(_first_stated_answer(r), "no")
            or "fallacy" in r.lower()
            or "not necessarily" in r.lower()
            or "not valid" in r.lower()
        ),
    },
    # Denying antecedent fallacy - novel
    {
        "id": "novel_da_1",
        "logic": "denying_antecedent_fallacy",
        "prompt": "If an animal is a dog, then it is a mammal. "
                  "This animal is not a dog.\nCan we conclude it is not a mammal?",
        # Strict: must reject — "cannot conclude", "no" (whole word), "fallacy", "not necessarily"
        # "not a mammal" WITHOUT "cannot" = committing the fallacy = FAIL
        "verify": lambda r: (
            "cannot" in _first_sentences(r, 2)
            or _word_match(_first_sentences(r, 1), "no")
            or "fallacy" in r.lower()
            or "not necessarily" in r.lower()
            or "not valid" in r.lower()
        ),
    },
    # --- COMPOSITIONAL (combine forms not seen together in training) ---
    # MP + MT chain
    {
        "id": "comp_1",
        "logic": "compositional_mp_mt",
        "prompt": "If the satellite loses power, then it stops transmitting. "
                  "If the satellite stops transmitting, then ground control loses contact. "
                  "Ground control has not lost contact.\n"
                  "What can we conclude about the satellite's power?",
        # Strict: must say "has not lost power" or "not lost power"
        "verify": lambda r: (
            "has not lost power" in _first_stated_answer(r)
            or "not lost power" in _first_sentences(r, 2)
            or "has not lost power" in _first_sentences(r, 2)
            or _concludes_not(r, "lost power")
        ),
    },
    # DS + MP chain
    {
        "id": "comp_2",
        "logic": "compositional_ds_mp",
        "prompt": "Either the encryption uses RSA or it uses elliptic curves. "
                  "The encryption does not use RSA. "
                  "If the encryption uses elliptic curves, then it requires a 256-bit key.\n"
                  "What key size is required?",
        "verify": lambda r: "256" in _first_stated_answer(r) or "256" in _first_sentences(r, 3),
    },
    # 3-step chain (length generalization)
    {
        "id": "comp_3",
        "logic": "compositional_3step",
        "prompt": "If the volcano erupts, then ash clouds form. "
                  "If ash clouds form, then flights are grounded. "
                  "If flights are grounded, then tourists are stranded. "
                  "The volcano erupted.\nWhat happens to the tourists?",
        "verify": lambda r: "strand" in _first_stated_answer(r) or "strand" in _first_sentences(r, 2),
    },
    # Biconditional + MT
    {
        "id": "comp_4",
        "logic": "compositional_bic_mt",
        "prompt": "A material is superconducting if and only if its resistance is zero. "
                  "If a material is superconducting, then it expels magnetic fields. "
                  "This material does not expel magnetic fields.\n"
                  "What is its resistance?",
        # Strict: must conclude resistance is NOT zero (or "not superconducting")
        "verify": lambda r: (
            "not zero" in _first_stated_answer(r)
            or "not superconducting" in _first_stated_answer(r)
            or "not superconducting" in _first_sentences(r, 3)
            # Must NOT say "resistance is zero" as the conclusion
            or (_word_match(_first_stated_answer(r), "resistance")
                and "zero" not in _first_stated_answer(r))
        ),
    },
    # --- NOVEL VARIABLE BINDING (same logic, completely new entities) ---
    {
        "id": "bind_mp_1",
        "logic": "modus_ponens",
        "prompt": "If the quasar emits gamma rays, then the detector triggers an alert. "
                  "The quasar is emitting gamma rays.\nWhat happens?",
        # Strict: must say "triggers an alert" or "detector triggers" in first answer
        "verify": lambda r: (
            "triggers an alert" in _first_sentences(r, 2)
            or "trigger" in _first_stated_answer(r)
            or "alert" in _first_stated_answer(r)
        ),
    },
    {
        "id": "bind_mt_1",
        "logic": "modus_tollens",
        "prompt": "If the submarine dives below 500 meters, then the hull pressure exceeds safety limits. "
                  "The hull pressure has not exceeded safety limits.\nWhat can we infer?",
        # Strict: must conclude submarine did NOT dive below 500m
        "verify": lambda r: (
            _concludes_not(r, "dive")
            or _concludes_not(r, "below 500")
            or "has not dived" in _first_sentences(r, 2)
            or "did not dive" in _first_sentences(r, 2)
            or "not dive" in _first_stated_answer(r)
        ),
    },
    {
        "id": "bind_ds_1",
        "logic": "disjunctive_syllogism",
        "prompt": "Either the manuscript was written in Latin or it was written in Greek. "
                  "Linguistic analysis confirms it was not written in Latin.\n"
                  "In what language was the manuscript written?",
        # Strict: must say "greek" in first answer, not empty
        "verify": lambda r: len(r.strip()) > 3 and "greek" in _first_sentences(r, 2),
    },
    {
        "id": "bind_hs_1",
        "logic": "hypothetical_syllogism",
        "prompt": "If the compiler finds a syntax error, then it halts compilation. "
                  "If compilation halts, then no executable is produced.\n"
                  "What is the relationship between syntax errors and executables?",
        # Strict: must link syntax errors to no/prevented executable production
        "verify": lambda r: (
            ("syntax error" in r.lower() and "no executable" in r.lower())
            or ("error" in _first_stated_answer(r) and "not" in _first_stated_answer(r) and "produc" in _first_stated_answer(r))
            or ("error" in _first_stated_answer(r) and "prevent" in _first_stated_answer(r))
            or ("syntax error" in _first_sentences(r, 2) and "prevent" in _first_sentences(r, 2))
        ),
    },
    {
        "id": "bind_cc_1",
        "logic": "chain_contrapositive",
        "prompt": "If the patient has condition X, then biomarker Y is elevated. "
                  "If biomarker Y is elevated, then the blood test is positive. "
                  "The blood test is negative.\nDoes the patient have condition X?",
        # Strict: first stated answer must say "no" (whole word) or "does not have condition X"
        # Self-contradicting (says both yes and no) = FAIL
        "verify": lambda r: (
            ("does not have condition" in _first_stated_answer(r)
             or "does not have condition" in _first_sentences(r, 2))
            and "has condition x" not in _first_sentences(r, 3)  # no self-contradiction in first 3
        ),
    },
]


def run_test():
    import mlx.core as mx
    from mlx_lm import generate, load as mlx_load

    MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    # Use the adapter from the validated n10-nomon run (seed 42)
    ADAPTER_PATH = "/Volumes/CodeCypher/experiments/ablation-350m-A-seeded-n10-nomon/arm_A_seed_42/adapter"

    if not Path(ADAPTER_PATH).exists():
        # Fall back to v2
        ADAPTER_PATH = "/Volumes/CodeCypher/experiments/ablation-350m-v2/arm_A_seed_456/adapter"
        print(f"Using fallback adapter: {ADAPTER_PATH}")

    print(f"Model: {MODEL_PATH}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Novel problems: {len(NOVEL_PROBLEMS)}")
    print()

    # Load base model
    print("Loading base model...")
    base_model, tokenizer = mlx_load(MODEL_PATH)

    base_results = []
    for p in NOVEL_PROBLEMS:
        response = generate(base_model, tokenizer, prompt=p["prompt"], max_tokens=150)
        correct = p["verify"](response)
        base_results.append({"id": p["id"], "logic": p["logic"], "correct": correct, "response": response})

    del base_model
    mx.clear_cache()

    # Load adapted model
    print("Loading adapted model...")
    adapted_model, tokenizer = mlx_load(MODEL_PATH, adapter_path=ADAPTER_PATH)

    adapted_results = []
    for p in NOVEL_PROBLEMS:
        response = generate(adapted_model, tokenizer, prompt=p["prompt"], max_tokens=150)
        correct = p["verify"](response)
        adapted_results.append({"id": p["id"], "logic": p["logic"], "correct": correct, "response": response})

    del adapted_model
    mx.clear_cache()

    # Report
    base_correct = sum(1 for r in base_results if r["correct"])
    adapted_correct = sum(1 for r in adapted_results if r["correct"])

    print(f"\n{'='*60}")
    print(f"BOOTSTRAP FEASIBILITY TEST")
    print(f"{'='*60}")
    print(f"Base model:    {base_correct}/{len(NOVEL_PROBLEMS)} correct")
    print(f"Adapted model: {adapted_correct}/{len(NOVEL_PROBLEMS)} correct")
    print(f"Delta:         {adapted_correct - base_correct:+d}")
    print(f"{'='*60}\n")

    # Per-problem comparison
    new_wins = []
    regressions = []
    print(f"{'ID':<20} {'Logic':<30} {'Base':>5} {'Adapt':>5}")
    print("-" * 65)
    for b, a in zip(base_results, adapted_results):
        b_mark = "Y" if b["correct"] else "."
        a_mark = "Y" if a["correct"] else "."
        tag = ""
        if a["correct"] and not b["correct"]:
            tag = " <- NEW"
            new_wins.append(a)
        elif b["correct"] and not a["correct"]:
            tag = " <- LOST"
            regressions.append(b)
        print(f"{b['id']:<20} {b['logic']:<30} {b_mark:>5} {a_mark:>5}{tag}")

    # Show new wins (these are bootstrap candidates)
    if new_wins:
        print(f"\n{'='*60}")
        print(f"NEW WINS (bootstrap candidates): {len(new_wins)}")
        print(f"{'='*60}")
        for w in new_wins:
            print(f"\n--- {w['id']} ({w['logic']}) ---")
            # Find the prompt
            prompt = next(p["prompt"] for p in NOVEL_PROBLEMS if p["id"] == w["id"])
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {w['response'][:200]}")

    # Show adapted correct responses (ALL bootstrap material)
    bootstrap_count = sum(1 for r in adapted_results if r["correct"])
    print(f"\n{'='*60}")
    print(f"TOTAL BOOTSTRAP MATERIAL: {bootstrap_count} correct responses on novel problems")
    print(f"{'='*60}")

    # Show a few adapted correct responses
    for r in adapted_results:
        if r["correct"]:
            prompt = next(p["prompt"] for p in NOVEL_PROBLEMS if p["id"] == r["id"])
            print(f"\n--- {r['id']} ({r['logic']}) ---")
            print(f"Q: {prompt}")
            print(f"A: {r['response'][:300]}")

    # Save full results
    out = {
        "base_correct": base_correct,
        "adapted_correct": adapted_correct,
        "n_problems": len(NOVEL_PROBLEMS),
        "base_results": base_results,
        "adapted_results": adapted_results,
        "adapter_path": ADAPTER_PATH,
    }
    out_path = Path("/Volumes/CodeCypher/experiments/bootstrap_feasibility.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull results saved to {out_path}")


def recount():
    """Re-evaluate saved results with tightened verification (no model inference)."""
    results_path = Path("/Volumes/CodeCypher/experiments/bootstrap_feasibility.json")
    if not results_path.exists():
        print(f"No saved results at {results_path}")
        sys.exit(1)

    saved = json.loads(results_path.read_text())
    problems_by_id = {p["id"]: p for p in NOVEL_PROBLEMS}

    print(f"Re-evaluating {saved['n_problems']} problems with strict verification\n")
    print(f"Original scores: base={saved['base_correct']}/{saved['n_problems']}, "
          f"adapted={saved['adapted_correct']}/{saved['n_problems']}")
    print()

    base_results = []
    adapted_results = []

    for b_saved, a_saved in zip(saved["base_results"], saved["adapted_results"]):
        pid = b_saved["id"]
        p = problems_by_id[pid]

        b_correct = p["verify"](b_saved["response"])
        a_correct = p["verify"](a_saved["response"])

        base_results.append({**b_saved, "correct": b_correct, "old_correct": b_saved["correct"]})
        adapted_results.append({**a_saved, "correct": a_correct, "old_correct": a_saved["correct"]})

    base_correct = sum(1 for r in base_results if r["correct"])
    adapted_correct = sum(1 for r in adapted_results if r["correct"])

    print(f"{'='*70}")
    print(f"STRICT RECOUNT")
    print(f"{'='*70}")
    print(f"Base model:    {base_correct}/{len(NOVEL_PROBLEMS)} (was {saved['base_correct']})")
    print(f"Adapted model: {adapted_correct}/{len(NOVEL_PROBLEMS)} (was {saved['adapted_correct']})")
    print(f"Delta:         {adapted_correct - base_correct:+d}")
    print(f"{'='*70}\n")

    # Per-problem comparison
    new_wins = []
    regressions = []
    flipped = []  # changed from old verify
    print(f"{'ID':<20} {'Logic':<30} {'Base':>5} {'Adapt':>5}  {'Change'}")
    print("-" * 80)
    for b, a in zip(base_results, adapted_results):
        b_mark = "Y" if b["correct"] else "."
        a_mark = "Y" if a["correct"] else "."
        tag = ""
        if a["correct"] and not b["correct"]:
            tag = " <- NEW WIN"
            new_wins.append(a)
        elif b["correct"] and not a["correct"]:
            tag = " <- REGRESSION"
            regressions.append(b)

        # Note changes from old verification
        b_changed = b["correct"] != b["old_correct"]
        a_changed = a["correct"] != a["old_correct"]
        flip_note = ""
        if b_changed:
            flip_note += f"  [base: {'Y' if b['old_correct'] else '.'}->{'Y' if b['correct'] else '.'}]"
        if a_changed:
            flip_note += f"  [adapt: {'Y' if a['old_correct'] else '.'}->{'Y' if a['correct'] else '.'}]"

        print(f"{b['id']:<20} {b['logic']:<30} {b_mark:>5} {a_mark:>5}{tag}{flip_note}")

    # Show new wins detail
    if new_wins:
        print(f"\n{'='*70}")
        print(f"NEW WINS (strict verified): {len(new_wins)}")
        print(f"{'='*70}")
        for w in new_wins:
            p = problems_by_id[w["id"]]
            print(f"\n--- {w['id']} ({w['logic']}) ---")
            print(f"Q: {p['prompt'][:120]}...")
            ans = _first_stated_answer(w["response"])
            print(f"First answer: {ans}")
            print(f"Full: {w['response'][:200]}")

    if regressions:
        print(f"\n{'='*70}")
        print(f"REGRESSIONS: {len(regressions)}")
        print(f"{'='*70}")
        for reg in regressions:
            p = problems_by_id[reg["id"]]
            print(f"\n--- {reg['id']} ({reg['logic']}) ---")
            # Show what adapted said
            a = next(a for a in adapted_results if a["id"] == reg["id"])
            print(f"Base first answer: {_first_stated_answer(reg['response'])}")
            print(f"Adapted first answer: {_first_stated_answer(a['response'])}")

    # Summary
    print(f"\n{'='*70}")
    print(f"STRICT BOOTSTRAP MATERIAL: {adapted_correct} correct on novel problems")
    print(f"Genuine new capabilities: {len(new_wins)}")
    print(f"Regressions: {len(regressions)}")
    print(f"Net delta: {adapted_correct - base_correct:+d}")
    print(f"{'='*70}")


if __name__ == "__main__":
    if "--recount" in sys.argv:
        recount()
    else:
        run_test()
