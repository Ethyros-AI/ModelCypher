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

"""R2 Format-Contract Ablation: Does explicit output contracting rescue benchmark accuracy?

Phase D showed 100% step-0 divergence. The corpus audit showed the training data
teaches bare numbers (GSM8K), content words (ARC), and yes/no (BoolQ).

This script tests whether prepending explicit output contracts rescues accuracy:
- "Respond with ONLY the final numeric answer."
- "Respond with ONLY the letter A, B, C, or D."
- "Respond with ONLY yes or no."

Three conditions per model:
  1. Baseline: unmodified prompts
  2. Contracted: prompts with explicit output contracts
  3. Base model variants for comparison

Usage:
    poetry run python scripts/r2_format_contract_ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
ADAPTER_PATH = "/Volumes/CodeCypher/models/adapters/350m-geometric-lora-r1"
OUTPUT_DIR = Path("results/r2_format_contract_ablation")

N_PER_TASK = 10
MAX_TOKENS = 64  # Short answers only — we want first-token behavior


# Output contracts per task
CONTRACTS = {
    "gsm8k": "\nRespond with ONLY the final numeric answer.",
    "arc_easy": "\nRespond with ONLY the letter A, B, C, or D.",
    "boolq": "\nRespond with ONLY yes or no.",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R2 format-contract ablation.",
    )
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-per-task", type=int, default=N_PER_TASK)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    return parser.parse_args()


def check_answer(response: str, expected: str, choices: list[str] | None) -> bool:
    """Replicate BenchmarkService._check_answer logic."""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in response_lower:
        return True

    if choices:
        for i, choice in enumerate(choices):
            if choice.lower() == expected_lower:
                letter = chr(65 + i)
                if letter.lower() in response_lower[:5]:
                    return True

    return False


def run_evaluation(
    model: Any,
    tokenizer: Any,
    prompts: list[dict],
    backend: Any,
    condition: str,
    contract: bool,
    max_tokens: int,
    log: logging.Logger,
) -> list[dict]:
    """Run evaluation on prompts, optionally with output contracts."""
    results = []

    for p in prompts:
        prompt = p["prompt"]
        if contract:
            task_contract = CONTRACTS.get(p["task"], "")
            prompt = prompt + task_contract

        response = backend.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        )

        correct = check_answer(response, p["answer"], p.get("choices"))

        # Extract first generated token (strip prompt from response if echoed)
        gen_text = response
        first_token = gen_text.split()[0] if gen_text.split() else ""

        results.append({
            "task": p["task"],
            "idx": p["idx"],
            "condition": condition,
            "contract": contract,
            "correct": correct,
            "expected": p["answer"],
            "response_preview": response[:100],
            "first_token": first_token,
        })

        log.info(
            "  %s[%d] %s %s: %s | expected=%r got=%r",
            p["task"], p["idx"], condition,
            "contracted" if contract else "baseline",
            "CORRECT" if correct else "WRONG",
            p["answer"][:30],
            response[:50],
        )

    return results


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="w"),
        ],
    )
    log = logging.getLogger("r2_format_contract")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.adapter_path).exists():
        print(f"ERROR: Adapter not found: {args.adapter_path}", file=sys.stderr)
        sys.exit(2)

    log.info("=" * 70)
    log.info("R2 Format-Contract Ablation")
    log.info("  Model:   %s", MODEL_PATH)
    log.info("  Adapter: %s", args.adapter_path)
    log.info("=" * 70)

    t_start = time.time()

    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()
    loader = ModelLoader(backend)

    # Load benchmark prompts
    from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader
    bench_loader = BenchmarkLoader()
    prompts: list[dict] = []
    for task_name in ["gsm8k", "arc_easy", "boolq"]:
        try:
            bench = bench_loader.load(task_name, split="test", limit=args.n_per_task)
            for i, sample in enumerate(bench.samples):
                prompts.append({
                    "task": task_name,
                    "prompt": sample.prompt,
                    "answer": sample.answer,
                    "choices": sample.choices,
                    "idx": i,
                })
            log.info("  Loaded %d %s prompts", len(bench.samples), task_name)
        except Exception as e:
            log.warning("  Failed to load %s: %s", task_name, e)

    if not prompts:
        log.error("No prompts loaded.")
        sys.exit(1)

    all_results: list[dict] = []

    # --- Base model ---
    log.info("Loading base model...")
    model_base, tokenizer = loader.load_model(MODEL_PATH)

    log.info("=== Base model, baseline ===")
    all_results.extend(run_evaluation(
        model_base, tokenizer, prompts, backend,
        "base", False, args.max_tokens, log,
    ))

    log.info("=== Base model, contracted ===")
    all_results.extend(run_evaluation(
        model_base, tokenizer, prompts, backend,
        "base", True, args.max_tokens, log,
    ))

    del model_base

    # --- Adapted model ---
    log.info("Loading adapted model...")
    model_adapted, _ = loader.load_model(MODEL_PATH, adapter_path=args.adapter_path)

    log.info("=== Adapted model, baseline ===")
    all_results.extend(run_evaluation(
        model_adapted, tokenizer, prompts, backend,
        "adapted", False, args.max_tokens, log,
    ))

    log.info("=== Adapted model, contracted ===")
    all_results.extend(run_evaluation(
        model_adapted, tokenizer, prompts, backend,
        "adapted", True, args.max_tokens, log,
    ))

    del model_adapted

    elapsed = time.time() - t_start

    # Write results
    json_path = output_dir / "ablation_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info("Wrote %s", json_path)

    # Build analysis
    analysis = build_analysis(all_results, elapsed)
    analysis_path = output_dir / "ANALYSIS.md"
    analysis_path.write_text(analysis, encoding="utf-8")
    log.info("Wrote %s", analysis_path)

    print()
    print(analysis)
    print(f"\nCompleted in {elapsed:.1f}s")


def build_analysis(results: list[dict], elapsed: float) -> str:
    lines = [
        "# R2 Format-Contract Ablation Results",
        "",
    ]

    # Summary table
    conditions = [
        ("base", False, "Base baseline"),
        ("base", True, "Base contracted"),
        ("adapted", False, "Adapted baseline"),
        ("adapted", True, "Adapted contracted"),
    ]

    tasks = sorted(set(r["task"] for r in results))

    lines.extend([
        "## Accuracy Summary",
        "",
        "| Condition |",
    ])

    # Build header
    header = "| Condition |"
    sep = "|-----------|"
    for task in tasks:
        header += f" {task} |"
        sep += "--------:|"
    header += " Overall |"
    sep += "--------:|"
    lines = lines[:-1]  # Remove partial header
    lines.extend([header, sep])

    for model_name, contract, label in conditions:
        row = f"| {label} |"
        total_correct = 0
        total_count = 0
        for task in tasks:
            task_results = [r for r in results
                           if r["task"] == task
                           and r["condition"] == model_name
                           and r["contract"] == contract]
            correct = sum(1 for r in task_results if r["correct"])
            n = len(task_results)
            total_correct += correct
            total_count += n
            pct = correct / n * 100 if n > 0 else 0
            row += f" {correct}/{n} ({pct:.0f}%) |"
        overall = total_correct / total_count * 100 if total_count > 0 else 0
        row += f" {total_correct}/{total_count} ({overall:.0f}%) |"
        lines.append(row)

    lines.append("")

    # Contract delta
    lines.extend(["## Contract Effect (adapted model)", ""])
    lines.append("| Task | Baseline | Contracted | Delta |")
    lines.append("|------|--------:|----------:|------:|")
    for task in tasks:
        baseline = [r for r in results if r["task"] == task and r["condition"] == "adapted" and not r["contract"]]
        contracted = [r for r in results if r["task"] == task and r["condition"] == "adapted" and r["contract"]]
        b_correct = sum(1 for r in baseline if r["correct"])
        c_correct = sum(1 for r in contracted if r["correct"])
        b_n = len(baseline)
        c_n = len(contracted)
        b_pct = b_correct / b_n * 100 if b_n else 0
        c_pct = c_correct / c_n * 100 if c_n else 0
        delta = c_pct - b_pct
        lines.append(f"| {task} | {b_correct}/{b_n} ({b_pct:.0f}%) | {c_correct}/{c_n} ({c_pct:.0f}%) | {delta:+.0f}pp |")
    lines.append("")

    # Per-prompt detail for adapted contracted (show responses)
    lines.extend(["## Adapted Contracted Responses (per prompt)", ""])
    for task in tasks:
        lines.append(f"### {task}")
        lines.append("")
        task_results = [r for r in results
                        if r["task"] == task
                        and r["condition"] == "adapted"
                        and r["contract"]]
        for r in task_results:
            status = "CORRECT" if r["correct"] else "WRONG"
            lines.append(
                f"- [{status}] expected=`{r['expected'][:40]}` "
                f"got=`{r['response_preview'][:60]}`"
            )
        lines.append("")

    # Interpretation
    lines.extend(["## Interpretation", ""])

    # Check if contracts helped
    adapted_baseline = [r for r in results if r["condition"] == "adapted" and not r["contract"]]
    adapted_contracted = [r for r in results if r["condition"] == "adapted" and r["contract"]]
    b_overall = sum(1 for r in adapted_baseline if r["correct"]) / len(adapted_baseline) * 100 if adapted_baseline else 0
    c_overall = sum(1 for r in adapted_contracted if r["correct"]) / len(adapted_contracted) * 100 if adapted_contracted else 0
    delta_overall = c_overall - b_overall

    if delta_overall > 10:
        lines.append(f"**Contracts rescued accuracy by {delta_overall:+.0f}pp overall.**")
        lines.append("The adapter has the capability but was producing answers in the wrong format.")
        lines.append("Next step: format-aware training or decode control.")
    elif delta_overall > 0:
        lines.append(f"**Contracts had modest effect ({delta_overall:+.0f}pp overall).**")
        lines.append("Some improvement, but the adapter is still selecting wrong answers on many prompts.")
    else:
        lines.append(f"**Contracts did not help ({delta_overall:+.0f}pp overall).**")
        lines.append("The adapter is genuinely producing wrong answers, not just wrong format.")
        lines.append("Next step: investigate training data quality or logit calibration.")

    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
