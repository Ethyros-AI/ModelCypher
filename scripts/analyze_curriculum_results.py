#!/usr/bin/env python3
"""Item-level flip analysis for curriculum training experiment.

Runs baseline and post-training inference on the same eval items,
computes per-item correctness flips, and classifies remaining misses.

Usage:
    poetry run python scripts/analyze_curriculum_results.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --adapter /Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_adapter \
        --eval-data data/eval/modus_ponens_eval.jsonl \
        --output /Volumes/CodeCypher/experiments/curriculum_e2e_v3/modus_ponens_flip_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

# Determiners, pronouns, copulas — stripped for semantic content-word comparison.
STOP_WORDS = frozenset({
    "a", "an", "the", "its", "his", "her", "their", "this", "that",
    "is", "are", "was", "were", "be", "been", "it", "they", "we",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_backend():
    from modelcypher.core.domain._backend import get_default_backend
    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend
        set_default_backend(get_backend(detect_default_backend_type()))


def _split_prompt_expected(item: dict) -> tuple[str, str]:
    """Split a JSONL item into (prompt, expected_answer)."""
    text = item["text"]
    answer_start = item.get("answer_start")
    if answer_start is not None:
        return text[:answer_start], text[answer_start:].strip()
    parts = text.rsplit("Answer:", 1)
    if len(parts) == 2:
        return parts[0] + "Answer:", parts[1].strip()
    tokens = text.split()
    return " ".join(tokens[:-1]), tokens[-1].strip()


def _extract_answer_span(predicted: str) -> str:
    """Extract the direct-answer span (first line) from model output."""
    return predicted.split("\n")[0].strip()


def _exact_match(expected: str, predicted: str) -> bool:
    """Case-insensitive substring match on the answer span (first line only).

    Matches the evaluator in curriculum_eval_adapter.py: only the direct
    answer counts, not incidental mentions in explanation text.
    """
    answer_span = _extract_answer_span(predicted)
    return bool(expected) and expected.lower() in answer_span.lower()


def _explanation_match(expected: str, predicted: str) -> bool:
    """True if expected appears in full output but NOT in the answer span.

    Diagnostic only — these are not counted as correct by the primary metric.
    """
    if not expected:
        return False
    answer_span = _extract_answer_span(predicted)
    return (
        expected.lower() not in answer_span.lower()
        and expected.lower() in predicted.lower()
    )


def _content_words(text: str) -> set[str]:
    """Extract content words: lowercase, strip punctuation and stop words."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def _semantic_match(expected: str, predicted: str) -> bool:
    """True if all content words from expected appear in predicted."""
    expected_words = _content_words(expected)
    predicted_words = _content_words(predicted)
    if not expected_words:
        return False
    return expected_words.issubset(predicted_words)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_all_items(engine, model_path: str, items: list[dict],
                   adapter_path: str | None = None) -> list[dict]:
    """Run inference on all items, return list of per-item results."""
    results = []
    for i, item in enumerate(items):
        prompt, expected = _split_prompt_expected(item)
        try:
            result = engine.run(
                model=model_path, prompt=prompt,
                adapter=adapter_path, max_tokens=256,
            )
            predicted = result.response.strip()
            results.append({
                "idx": i,
                "prompt": prompt[:200],
                "expected": expected,
                "predicted": predicted,
                "exact_match": _exact_match(expected, predicted),
                "explanation_match": _explanation_match(expected, predicted),
                "error": None,
            })
        except Exception as e:
            results.append({
                "idx": i,
                "prompt": prompt[:200],
                "expected": expected,
                "predicted": "",
                "exact_match": False,
                "explanation_match": False,
                "error": str(e),
            })
        if (i + 1) % 10 == 0:
            n_correct = sum(1 for r in results if r["exact_match"])
            print(f"  {i + 1}/{len(items)} complete ({n_correct} correct so far)")
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _classify_flips(baseline: list[dict], adapter: list[dict]) -> list[dict]:
    """For each item, compute the flip category and semantic classification."""
    items = []
    for b, a in zip(baseline, adapter):
        b_correct = b["exact_match"]
        a_correct = a["exact_match"]

        if not b_correct and a_correct:
            flip = "wrong_to_right"
        elif b_correct and not a_correct:
            flip = "right_to_wrong"
        elif b_correct and a_correct:
            flip = "stayed_right"
        else:
            flip = "stayed_wrong"

        miss_type = None
        if not a_correct:
            if a["error"]:
                miss_type = "inference_error"
            elif a.get("explanation_match"):
                miss_type = "explanation_only"
            elif _semantic_match(a["expected"], a["predicted"]):
                miss_type = "semantic_correct_exact_miss"
            else:
                miss_type = "actually_wrong"

        items.append({
            "idx": b["idx"],
            "expected": b["expected"],
            "baseline_predicted": b["predicted"][:300],
            "adapter_predicted": a["predicted"][:300],
            "baseline_correct": b_correct,
            "baseline_explanation_only": b.get("explanation_match", False),
            "adapter_correct": a_correct,
            "adapter_explanation_only": a.get("explanation_match", False),
            "flip": flip,
            "miss_type": miss_type,
        })
    return items


def _compute_summary(analysis: list[dict]) -> dict:
    n = len(analysis)
    flips = {"wrong_to_right": 0, "right_to_wrong": 0, "stayed_right": 0, "stayed_wrong": 0}
    miss_types = {
        "explanation_only": 0,
        "semantic_correct_exact_miss": 0,
        "actually_wrong": 0,
        "inference_error": 0,
    }

    baseline_explanation_only = 0
    adapter_explanation_only = 0

    for item in analysis:
        flips[item["flip"]] += 1
        if item["miss_type"]:
            miss_types[item["miss_type"]] += 1
        if item.get("baseline_explanation_only"):
            baseline_explanation_only += 1
        if item.get("adapter_explanation_only"):
            adapter_explanation_only += 1

    baseline_correct = flips["stayed_right"] + flips["right_to_wrong"]
    adapter_correct = flips["stayed_right"] + flips["wrong_to_right"]

    return {
        "n_total": n,
        "baseline_correct": baseline_correct,
        "adapter_correct": adapter_correct,
        "baseline_accuracy": baseline_correct / n,
        "adapter_accuracy": adapter_correct / n,
        "accuracy_delta": (adapter_correct - baseline_correct) / n,
        "flips": flips,
        "post_training_misses": n - adapter_correct,
        "miss_classification": miss_types,
        "baseline_explanation_only": baseline_explanation_only,
        "adapter_explanation_only": adapter_explanation_only,
        "net_improvement": flips["wrong_to_right"] - flips["right_to_wrong"],
        "mcnemar_b": flips["right_to_wrong"],
        "mcnemar_c": flips["wrong_to_right"],
    }


def _print_summary(summary: dict, analysis: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("CURRICULUM FLIP ANALYSIS: modus_ponens")
    print("=" * 60)

    print(f"\nBaseline: {summary['baseline_correct']}/{summary['n_total']} "
          f"({summary['baseline_accuracy']:.1%})")
    print(f"Adapter:  {summary['adapter_correct']}/{summary['n_total']} "
          f"({summary['adapter_accuracy']:.1%})")
    print(f"Delta:    {summary['accuracy_delta']:+.1%}")

    f = summary["flips"]
    print(f"\n--- Flip Analysis (paired) ---")
    print(f"  Stayed right:    {f['stayed_right']}")
    print(f"  Wrong -> Right:  {f['wrong_to_right']}")
    print(f"  Right -> Wrong:  {f['right_to_wrong']}")
    print(f"  Stayed wrong:    {f['stayed_wrong']}")
    print(f"  Net improvement: {summary['net_improvement']} items")
    print(f"  McNemar discordant: b={summary['mcnemar_b']}, c={summary['mcnemar_c']}")

    print(f"\n--- Explanation-Only Matches (diagnostic, NOT counted as correct) ---")
    print(f"  Baseline: {summary['baseline_explanation_only']}")
    print(f"  Adapter:  {summary['adapter_explanation_only']}")

    m = summary["miss_classification"]
    print(f"\n--- Post-Training Miss Classification ---")
    print(f"  Explanation only (not correct):  {m['explanation_only']}")
    print(f"  Semantic correct (exact miss):   {m['semantic_correct_exact_miss']}")
    print(f"  Actually wrong:                  {m['actually_wrong']}")
    print(f"  Inference errors:                {m['inference_error']}")

    for category, label in [
        ("right_to_wrong", "Right -> Wrong"),
        ("explanation_only", "Explanation Only (not counted)"),
        ("semantic_correct_exact_miss", "Semantic Correct, Exact Miss"),
    ]:
        items_in_cat = [
            a for a in analysis
            if a["flip"] == category or a.get("miss_type") == category
        ]
        if items_in_cat:
            print(f"\n--- Examples: {label} ({len(items_in_cat)} total) ---")
            for item in items_in_cat[:3]:
                print(f"  [{item['idx']}] Expected: {item['expected']}")
                print(f"       Baseline: {item['baseline_predicted'][:120]}")
                print(f"       Adapter:  {item['adapter_predicted'][:120]}")
                print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Item-level flip analysis for curriculum experiment."
    )
    parser.add_argument("--model", required=True, help="Path to base model")
    parser.add_argument("--adapter", required=True, help="Path to trained adapter")
    parser.add_argument("--eval-data", required=True, help="Path to eval JSONL")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    # GPU safety check — look for actual model/training processes
    _safe_patterns = {
        "analyze_curriculum_results", "pet server", "pylance",
        "Code Helper", "uvicorn", "resource_tracker", "spawn_main",
        "multiprocessing", "pgrep",
    }
    gpu_check = subprocess.run(
        ["pgrep", "-af", "python|mlx"], capture_output=True, text=True
    )
    active = []
    for line in gpu_check.stdout.strip().splitlines():
        if not line:
            continue
        if any(p in line for p in _safe_patterns):
            continue
        # Lines that are just a PID (no command) come from pgrep matching itself
        if line.strip().isdigit():
            continue
        active.append(line)
    if active:
        print("WARNING: GPU-using processes detected:")
        for line in active:
            print(f"  {line}")
        print("Aborting. Kill conflicting processes or verify they're safe.")
        sys.exit(1)

    # Load eval items
    items = []
    with open(args.eval_data) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Loaded {len(items)} eval items from {args.eval_data}")

    _ensure_backend()
    from modelcypher.adapters.inference_engine import get_inference_engine
    engine = get_inference_engine()

    # Phase 1: Baseline
    print(f"\n--- Phase 1: Baseline Inference ({len(items)} items) ---")
    t0 = time.time()
    baseline_results = _run_all_items(engine, args.model, items)
    baseline_time = time.time() - t0
    n_base = sum(1 for r in baseline_results if r["exact_match"])
    print(f"  Done: {n_base}/{len(items)} correct in {baseline_time:.1f}s")

    engine.clear_cache()

    # Phase 2: Adapter
    print(f"\n--- Phase 2: Adapter Inference ({len(items)} items) ---")
    t0 = time.time()
    adapter_results = _run_all_items(engine, args.model, items, adapter_path=args.adapter)
    adapter_time = time.time() - t0
    n_adapt = sum(1 for r in adapter_results if r["exact_match"])
    print(f"  Done: {n_adapt}/{len(items)} correct in {adapter_time:.1f}s")

    # Phase 3: Analysis
    print("\n--- Phase 3: Flip Analysis ---")
    analysis = _classify_flips(baseline_results, adapter_results)
    summary = _compute_summary(analysis)

    output = {
        "metadata": {
            "model": args.model,
            "adapter": args.adapter,
            "eval_data": args.eval_data,
            "n_items": len(items),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "baseline_inference_seconds": round(baseline_time, 1),
            "adapter_inference_seconds": round(adapter_time, 1),
        },
        "summary": summary,
        "items": analysis,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {args.output}")

    _print_summary(summary, analysis)


if __name__ == "__main__":
    main()
