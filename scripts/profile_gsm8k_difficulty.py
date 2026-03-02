"""Split GSM8K into difficulty tiers based on reasoning step count.

Difficulty metric: number of reasoning steps in the answer, counted as
non-empty lines before the '####' final-answer marker.

Derivation: Each reasoning step in a GSM8K answer represents one inference
operation (arithmetic, substitution, or logical conclusion). A 3-step answer
requires 3 rule applications; a 7-step answer requires 7. This is the
logical structure of the problem — not a heuristic, not a model-predicted
score. More steps = strictly more rule applications required.

Outputs (all JSONL, {"text": "prompt answer"} format):
  data/training/gsm8k_easy_train.jsonl    (bottom third by steps)
  data/training/gsm8k_medium_train.jsonl  (middle third)
  data/training/gsm8k_hard_train.jsonl    (top third)
  data/eval/gsm8k_easy_eval.jsonl         (held-out eval, 100 samples per tier)
  data/eval/gsm8k_medium_eval.jsonl
  data/eval/gsm8k_hard_eval.jsonl

Usage:
  poetry run python scripts/profile_gsm8k_difficulty.py
  poetry run python scripts/profile_gsm8k_difficulty.py --eval-size 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def count_reasoning_steps(answer_text: str) -> int:
    """Count reasoning steps in a GSM8K answer.

    Steps are non-empty lines before the '####' final-answer marker.
    The '####' line itself is not a reasoning step.

    Examples:
      "John has 5 apples.\nHe buys 3 more.\n5 + 3 = 8\n#### 8" -> 3 steps
      "#### 42" -> 0 steps (answer only, trivial)
    """
    if "####" in answer_text:
        reasoning_part = answer_text.split("####")[0]
    else:
        reasoning_part = answer_text

    steps = [line.strip() for line in reasoning_part.split("\n") if line.strip()]
    return len(steps)


def load_gsm8k_raw(split: str) -> list[dict]:
    """Load raw GSM8K from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading GSM8K ({split} split) from HuggingFace...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        return list(ds)
    except Exception as e:
        print(f"ERROR: Could not load GSM8K: {e}")
        sys.exit(1)


def format_as_text_continuation(question: str, answer: str) -> dict:
    """Convert to {"text": "prompt answer"} training format.

    The full answer (including reasoning chain) is kept. The prompt ends at
    "Answer:" so the model must generate the full chain.
    """
    # Extract final numerical answer
    if "####" in answer:
        final = answer.split("####")[-1].strip()
        reasoning = answer.split("####")[0].strip()
    else:
        final = answer.strip()
        reasoning = ""

    # Format: question + reasoning chain + final answer on last line
    if reasoning:
        text = f"{question}\nAnswer:\n{reasoning}\n#### {final}"
    else:
        text = f"{question}\nAnswer: {final}"

    return {"text": text}


def tertile_split(items: list, eval_size: int) -> tuple[list, list, list, list, list, list]:
    """Split items into easy/medium/hard tertiles with held-out eval sets.

    Returns: (easy_train, easy_eval, medium_train, medium_eval, hard_train, hard_eval)

    Eval sets are taken from the END of each tertile (most representative examples).
    """
    n = len(items)
    t1 = n // 3
    t2 = 2 * n // 3

    easy = items[:t1]
    medium = items[t1:t2]
    hard = items[t2:]

    def split_eval(tier: list, size: int) -> tuple[list, list]:
        if len(tier) <= size:
            # Not enough for separate eval; use first half as train, second as eval
            mid = len(tier) // 2
            return tier[:mid], tier[mid:]
        return tier[:-size], tier[-size:]

    easy_train, easy_eval = split_eval(easy, eval_size)
    medium_train, medium_eval = split_eval(medium, eval_size)
    hard_train, hard_eval = split_eval(hard, eval_size)

    return easy_train, easy_eval, medium_train, medium_eval, hard_train, hard_eval


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"  Wrote {len(items):,} samples → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-size", type=int, default=100,
        help="Number of held-out eval samples per tier (default: 100)"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="GSM8K split to use (default: train; use 'test' for smaller set)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "data" / "training"
    eval_dir = output_dir / "data" / "eval"

    # ── Load ──────────────────────────────────────────────────────────────
    raw = load_gsm8k_raw(args.split)
    print(f"Loaded {len(raw):,} GSM8K problems from {args.split} split.")

    # ── Score ─────────────────────────────────────────────────────────────
    scored = []
    for item in raw:
        steps = count_reasoning_steps(item["answer"])
        scored.append((steps, item))

    scored.sort(key=lambda x: x[0])

    step_counts = [s for s, _ in scored]
    print(f"\nReasoning step distribution:")
    print(f"  Min:    {min(step_counts)} steps")
    print(f"  Median: {sorted(step_counts)[len(step_counts)//2]} steps")
    print(f"  Max:    {max(step_counts)} steps")
    print(f"  Mean:   {sum(step_counts)/len(step_counts):.1f} steps")

    easy_cutoff = step_counts[len(step_counts) // 3]
    hard_cutoff = step_counts[2 * len(step_counts) // 3]
    print(f"\nTertile boundaries:")
    print(f"  Easy:   ≤ {easy_cutoff} steps")
    print(f"  Medium: {easy_cutoff+1}–{hard_cutoff} steps")
    print(f"  Hard:   > {hard_cutoff} steps")

    # ── Convert to training format ─────────────────────────────────────────
    formatted = [
        format_as_text_continuation(item["question"], item["answer"])
        for _, item in scored
    ]

    # ── Split ─────────────────────────────────────────────────────────────
    easy_train, easy_eval, medium_train, medium_eval, hard_train, hard_eval = (
        tertile_split(formatted, args.eval_size)
    )

    print(f"\nSplits:")
    print(f"  Easy:   {len(easy_train):,} train + {len(easy_eval):,} eval")
    print(f"  Medium: {len(medium_train):,} train + {len(medium_eval):,} eval")
    print(f"  Hard:   {len(hard_train):,} train + {len(hard_eval):,} eval")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving to {output_dir}/...")
    save_jsonl(easy_train, train_dir / "gsm8k_easy_train.jsonl")
    save_jsonl(easy_eval, eval_dir / "gsm8k_easy_eval.jsonl")
    save_jsonl(medium_train, train_dir / "gsm8k_medium_train.jsonl")
    save_jsonl(medium_eval, eval_dir / "gsm8k_medium_eval.jsonl")
    save_jsonl(hard_train, train_dir / "gsm8k_hard_train.jsonl")
    save_jsonl(hard_eval, eval_dir / "gsm8k_hard_eval.jsonl")

    # ── Save step-count metadata ───────────────────────────────────────────
    meta = {
        "split": args.split,
        "total_problems": len(raw),
        "step_distribution": {
            "min": min(step_counts),
            "median": sorted(step_counts)[len(step_counts)//2],
            "max": max(step_counts),
            "mean": round(sum(step_counts)/len(step_counts), 2),
        },
        "tertile_boundaries": {
            "easy_max_steps": easy_cutoff,
            "medium_max_steps": hard_cutoff,
        },
        "split_sizes": {
            "easy_train": len(easy_train),
            "easy_eval": len(easy_eval),
            "medium_train": len(medium_train),
            "medium_eval": len(medium_eval),
            "hard_train": len(hard_train),
            "hard_eval": len(hard_eval),
        },
        "dag_node_mapping": {
            "gsm8k_easy → word_problem_1step": "single-operation problems",
            "gsm8k_medium → word_problem_multi": "multi-step problems (medium chain)",
            "gsm8k_hard → algebra_linear": "complex multi-step (long chain)",
        },
    }
    meta_path = eval_dir / "gsm8k_difficulty_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata → {meta_path}")

    print("\nDone. Next steps:")
    print("  1. Run 'mc curriculum dag' to see where these fit in the training order")
    print("  2. Update skill_dag.py: add gsm8k files to word_problem_1step/multi nodes")
    print("  3. Run 'mc curriculum next --model <path>' to start training")


if __name__ == "__main__":
    main()
