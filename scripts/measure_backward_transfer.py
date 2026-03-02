#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Measure backward transfer: does training skill B cause forgetting of skill A?

Used to answer open experiment #2 in docs/curriculum/skill_dag.md:
  "Is 'reinforce' regime sufficient for safe advancement?
   Measure backward transfer after training MT on model that has 'mastered' MP."

Two-phase workflow
------------------

Phase 1 — save baseline predictions before training:

  poetry run python scripts/measure_backward_transfer.py \\
    --save-baseline \\
    --model /path/to/base-model \\
    --eval-skill modus_ponens \\
    --eval-file data/eval/modus_ponens_eval.jsonl \\
    --out results/mp_baseline.json

Phase 2 — compare against baseline after training:

  # (train on modus_tollens first)
  poetry run python scripts/measure_backward_transfer.py \\
    --compare \\
    --model /path/to/trained-model \\
    --eval-skill modus_ponens \\
    --eval-file data/eval/modus_ponens_eval.jsonl \\
    --baseline results/mp_baseline.json \\
    --out results/backward_transfer_mp_mt.json

Output (JSON):
  {
    "skill": "modus_ponens",
    "base_model": "...",
    "trained_model": "...",
    "n_total": 100,
    "pre_accuracy": 0.45,
    "post_accuracy": 0.43,
    "delta_accuracy": -0.02,
    "n_correct_both": 37,      -- stable correct (neither lost nor gained)
    "n_lost": 6,               -- correct before, wrong after (forgetting)
    "n_gained": 4,             -- wrong before, correct after (improvement)
    "n_wrong_both": 53,
    "mcnemar_chi2": 0.1,       -- (|n_lost - n_gained| - 1)^2 / (n_lost + n_gained)
    "mcnemar_p": 0.75,         -- p-value under H0: no systematic change
    "significant_forgetting": false,
    "verdict": "No significant backward transfer (p=0.75)."
  }

Interpretation:
  - mcnemar_p < 0.05 AND n_lost > n_gained → significant forgetting
  - mcnemar_p < 0.05 AND n_gained > n_lost → significant improvement (transfer!)
  - mcnemar_p >= 0.05 → no systematic change (reinforce gate likely safe)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-item inference
# ---------------------------------------------------------------------------

def _run_item_inference(
    model_path: str,
    eval_file: Path,
) -> list[dict]:
    """Run inference on each eval item and return per-item results.

    Returns list of dicts: {prompt, expected, predicted, correct}.
    Uses the same prompt-extraction logic as evaluate_skill_mastery.
    """
    from modelcypher.core.domain._backend import get_default_backend

    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend
        set_default_backend(get_backend(detect_default_backend_type()))

    from modelcypher.adapters.inference_engine import get_inference_engine
    engine = get_inference_engine()

    problems = []
    with eval_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    results = []
    for i, item in enumerate(problems):
        text = item["text"]
        answer_start = item.get("answer_start")

        if answer_start is not None:
            prompt = text[:answer_start]
            expected = text[answer_start:].strip().lower()
        else:
            parts = text.rsplit("Answer:", 1)
            if len(parts) == 2:
                prompt = parts[0] + "Answer:"
                expected = parts[1].strip().lower()
            else:
                tokens = text.split()
                prompt = " ".join(tokens[:-1])
                expected = tokens[-1].strip().lower()

        predicted = ""
        correct = False
        try:
            result = engine.run(model=model_path, prompt=prompt, max_tokens=None)
            predicted = result.response.strip().lower()
            correct = bool(expected and expected in predicted)
        except Exception:
            logger.debug("Inference failed on item %d", i, exc_info=True)

        results.append({
            "idx": i,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        })

        if (i + 1) % 10 == 0:
            n_correct = sum(r["correct"] for r in results)
            logger.info("Progress: %d/%d, accuracy so far: %.3f",
                        i + 1, len(problems), n_correct / (i + 1))

    return results


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def _mcnemar(pre_correct: list[bool], post_correct: list[bool]) -> dict:
    """Compute McNemar's test for paired before/after correctness.

    McNemar (1947): tests whether the marginal frequencies of a 2×2 table are
    equal — i.e., whether P(correct before, wrong after) = P(wrong before, correct after).
    A significant result means training caused a SYSTEMATIC change (forgetting or improvement).

    Chi-squared with continuity correction (Fleiss et al. 2003):
        chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)

    Reference: McNemar Q (1947). "Note on the sampling error of the difference
    between correlated proportions or percentages." Psychometrika 12(2):153-157.
    """
    assert len(pre_correct) == len(post_correct), "Pre/post must have same length"
    n = len(pre_correct)

    n_both_correct = sum(1 for p, q in zip(pre_correct, post_correct) if p and q)
    n_lost = sum(1 for p, q in zip(pre_correct, post_correct) if p and not q)
    n_gained = sum(1 for p, q in zip(pre_correct, post_correct) if not p and q)
    n_both_wrong = sum(1 for p, q in zip(pre_correct, post_correct) if not p and not q)

    discordant = n_lost + n_gained
    if discordant == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(n_lost - n_gained) - 1) ** 2 / discordant
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=1))

    significant_forgetting = (p < 0.05) and (n_lost > n_gained)
    significant_improvement = (p < 0.05) and (n_gained > n_lost)

    if significant_forgetting:
        verdict = f"Significant forgetting (p={p:.4f}): {n_lost} items lost, {n_gained} gained. reinforce gate may be too permissive."
    elif significant_improvement:
        verdict = f"Significant improvement / forward transfer (p={p:.4f}): {n_gained} items gained, {n_lost} lost."
    else:
        verdict = f"No significant change (p={p:.4f}): {n_lost} items lost, {n_gained} gained. reinforce gate appears safe for this transition."

    return {
        "n_total": n,
        "n_correct_both": n_both_correct,
        "n_lost": n_lost,
        "n_gained": n_gained,
        "n_wrong_both": n_both_wrong,
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p": round(p, 4),
        "significant_forgetting": significant_forgetting,
        "significant_improvement": significant_improvement,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save-baseline", action="store_true",
                      help="Phase 1: run inference and save per-item results as baseline")
    mode.add_argument("--compare", action="store_true",
                      help="Phase 2: run inference and compare against saved baseline")

    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--eval-skill", required=True,
                        help="Skill name to evaluate (e.g. modus_ponens)")
    parser.add_argument("--eval-file", required=True, help="Path to eval JSONL file")
    parser.add_argument("--out", required=True, help="Path for output JSON")
    parser.add_argument("--baseline", help="(--compare only) Path to baseline JSON from phase 1")

    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"ERROR: eval file not found: {eval_file}", file=sys.stderr)
        sys.exit(1)

    if args.compare and not args.baseline:
        print("ERROR: --compare requires --baseline", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {args.eval_skill} with {args.model}...")
    item_results = _run_item_inference(args.model, eval_file)
    correct_flags = [r["correct"] for r in item_results]
    n_correct = sum(correct_flags)
    accuracy = n_correct / len(correct_flags)
    print(f"  n={len(correct_flags)}, accuracy={accuracy:.3f}, n_correct={n_correct}")

    if args.save_baseline:
        output = {
            "phase": "baseline",
            "skill": args.eval_skill,
            "model": args.model,
            "eval_file": str(eval_file),
            "n_total": len(item_results),
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "items": item_results,
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nBaseline saved → {out_path}")
        print("Next: train on the dependent skill, then re-run with --compare --baseline {out_path}")

    else:  # --compare
        baseline_data = json.loads(Path(args.baseline).read_text())
        pre_correct = [r["correct"] for r in baseline_data["items"]]

        if len(pre_correct) != len(correct_flags):
            print(
                f"ERROR: baseline has {len(pre_correct)} items, current eval has {len(correct_flags)}. "
                "Must use the same eval file.",
                file=sys.stderr,
            )
            sys.exit(1)

        stats = _mcnemar(pre_correct, correct_flags)

        output = {
            "phase": "comparison",
            "skill": args.eval_skill,
            "base_model": baseline_data["model"],
            "trained_model": args.model,
            "eval_file": str(eval_file),
            "pre_accuracy": baseline_data["accuracy"],
            "post_accuracy": round(accuracy, 4),
            "delta_accuracy": round(accuracy - baseline_data["accuracy"], 4),
            **stats,
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n{'='*60}")
        print(f"BACKWARD TRANSFER RESULT: {args.eval_skill}")
        print(f"{'='*60}")
        print(f"  pre_accuracy:  {output['pre_accuracy']:.3f}")
        print(f"  post_accuracy: {output['post_accuracy']:.3f}")
        print(f"  delta:         {output['delta_accuracy']:+.3f}")
        print(f"  n_lost:        {output['n_lost']}")
        print(f"  n_gained:      {output['n_gained']}")
        print(f"  McNemar p:     {output['mcnemar_p']:.4f}")
        print(f"\nVerdict: {output['verdict']}")
        print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
