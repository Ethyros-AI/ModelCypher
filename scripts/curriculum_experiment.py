#!/usr/bin/env python3
"""End-to-end curriculum training experiment.

Tests whether curriculum-driven training actually improves a model on a
targeted skill. Runs baseline eval, trains, runs post-training eval, and
diagnoses any failures.

Usage:
    poetry run python scripts/curriculum_experiment.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --skill modus_ponens \
        --output-dir /Volumes/CodeCypher/experiments/curriculum_e2e

    # Dry run (no training, just baseline eval):
    poetry run python scripts/curriculum_experiment.py \
        --model /path/to/model \
        --skill modus_ponens \
        --output-dir /tmp/curriculum_exp \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))


def _find_skill_node(skill_name: str):
    """Look up a skill in the existing CURRICULUM_DAG."""
    from modelcypher.core.use_cases.curriculum.skill_dag import CURRICULUM_DAG

    try:
        return CURRICULUM_DAG.get(skill_name)
    except KeyError:
        available = [n.name for n in CURRICULUM_DAG.nodes]
        print(f"ERROR: Skill '{skill_name}' not found in CURRICULUM_DAG.")
        print(f"Available skills: {', '.join(sorted(available))}")
        sys.exit(1)


def _resolve_data_path(rel_path: str) -> Path:
    """Resolve a path relative to the project root."""
    full = _project_root / rel_path
    if not full.exists():
        print(f"ERROR: Data file not found: {full}")
        sys.exit(1)
    return full


def _prepare_training_data(skill) -> Path:
    """Merge prompt/completion format into text format if needed.

    The existing curriculum data uses {"prompt": ..., "completion": ...} but
    the training pipeline expects {"text": ...}. This converts on the fly.

    Returns path to a temp JSONL file with {"text": prompt + completion} format.
    """
    all_samples = []
    for rel_path in skill.train_files:
        p = _resolve_data_path(rel_path)
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "text" in item:
                    all_samples.append(item)
                elif "prompt" in item and "completion" in item:
                    # Merge prompt + completion into text
                    text = item["prompt"] + item["completion"]
                    converted = {"text": text}
                    if "answer_start" in item:
                        converted["answer_start"] = item["answer_start"]
                    elif "prompt" in item:
                        converted["answer_start"] = len(item["prompt"])
                    all_samples.append(converted)
                else:
                    print(f"WARNING: Skipping sample with unknown format: {list(item.keys())}")

    if not all_samples:
        print(f"ERROR: No training samples found for skill '{skill.name}'")
        sys.exit(1)

    # Write to temp file
    tmp = Path(tempfile.mktemp(suffix=f"_{skill.name}_train.jsonl"))
    with open(tmp, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Prepared {len(all_samples)} training samples -> {tmp}")
    return tmp


def _run_mastery_eval(model_path: str, skill, adapter_path: str | None = None):
    """Run mastery evaluation on a skill's held-out eval set.

    Returns (n_correct, n_total, accuracy, regime, sample_outputs).
    """
    from modelcypher.adapters.curriculum_eval_adapter import evaluate_skill_mastery

    eval_path = _resolve_data_path(skill.eval_files[0])
    print(f"  Eval file: {eval_path}")

    effective_model = adapter_path if adapter_path else model_path
    record = evaluate_skill_mastery(
        model_path=effective_model,
        skill=skill,
        eval_jsonl_path=eval_path,
    )

    return record


def _run_sample_inference(model_path: str, skill, n_samples: int = 5, adapter_path: str | None = None):
    """Run inference on a few eval samples and print actual outputs for diagnosis."""
    eval_path = _resolve_data_path(skill.eval_files[0])
    with open(eval_path) as f:
        items = [json.loads(line.strip()) for line in f if line.strip()][:n_samples]

    from modelcypher.adapters.inference_engine import get_inference_engine
    from modelcypher.core.domain._backend import get_default_backend

    try:
        get_default_backend()
    except RuntimeError:
        from modelcypher.backends import detect_default_backend_type, get_backend
        from modelcypher.core.domain._backend import set_default_backend
        set_default_backend(get_backend(detect_default_backend_type()))

    engine = get_inference_engine()
    effective_model = adapter_path if adapter_path else model_path

    print(f"\n  === Sample Outputs ({n_samples} examples) ===")
    for i, item in enumerate(items):
        text = item["text"]
        parts = text.rsplit("Answer:", 1)
        if len(parts) == 2:
            prompt = parts[0] + "Answer:"
            expected = parts[1].strip()
        else:
            prompt = text
            expected = "(unknown)"

        result = engine.run(model=effective_model, prompt=prompt, max_tokens=256)
        predicted = result.response.strip()

        print(f"\n  [{i+1}] Prompt: {prompt[:100]}...")
        print(f"      Expected: {expected}")
        print(f"      Got:      {predicted[:200]}")
        correct = expected.lower() in predicted.lower() if expected != "(unknown)" else "?"
        print(f"      Correct:  {correct}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end curriculum training experiment."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--skill", type=str, required=True, help="Skill name from CURRICULUM_DAG")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for adapter and results")
    parser.add_argument("--dry-run", action="store_true", help="Only run baseline eval, skip training")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample outputs to show for diagnosis")
    args = parser.parse_args()

    skill = _find_skill_node(args.skill)
    print(f"=== Curriculum Experiment: {skill.name} ===")
    print(f"  Formal statement: {skill.formal_statement}")
    print(f"  Answer mode: {skill.answer_mode}")
    print(f"  Branch: {skill.branch}")
    print(f"  Prerequisites: {skill.prerequisites or '(none)'}")
    print(f"  Train files: {skill.train_files}")
    print(f"  Eval files: {skill.eval_files}")
    print()

    # Step 1: Baseline eval
    print("--- Step 1: Baseline Evaluation (no adapter) ---")
    baseline = _run_mastery_eval(args.model, skill)
    print(f"  Accuracy: {baseline.accuracy:.1%} ({baseline.n_correct}/{baseline.n_total})")
    print(f"  CI: [{baseline.ci_lower:.3f}, {baseline.ci_upper:.3f}]")
    print(f"  Regime: {baseline.regime}")

    # Show sample outputs for diagnosis
    print("\n--- Baseline Sample Outputs ---")
    _run_sample_inference(args.model, skill, n_samples=args.samples)

    if args.dry_run:
        print("\n--- Dry run: skipping training ---")
        return

    # Step 2: Prepare training data
    print("\n--- Step 2: Prepare Training Data ---")
    train_path = _prepare_training_data(skill)

    # Step 3: Train
    print("\n--- Step 3: Training ---")
    adapter_path = args.output_dir / f"{skill.name}_adapter"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_data_path = _resolve_data_path(skill.eval_files[0])

    cmd = [
        "poetry", "run", "mc", "train", "run",
        "--model", args.model,
        "--data", str(train_path),
        "--eval-data", str(eval_data_path),
        "--output", str(adapter_path),
        "--explain",
        "--json",
    ]
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_project_root))

    if result.returncode != 0:
        print(f"  TRAINING FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[:2000]}")
        print(f"  stdout: {result.stdout[:2000]}")

        # Save error output
        error_path = args.output_dir / f"{skill.name}_training_error.txt"
        with open(error_path, "w") as f:
            f.write(f"exit_code: {result.returncode}\n\n")
            f.write("=== STDERR ===\n")
            f.write(result.stderr)
            f.write("\n\n=== STDOUT ===\n")
            f.write(result.stdout)
        print(f"  Full output saved to: {error_path}")
        return

    # Parse training result
    training_result = None
    try:
        training_result = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("  WARNING: Could not parse training JSON output")
        print(f"  stdout: {result.stdout[:2000]}")

    if training_result:
        print(f"  Training completed:")
        print(f"    Iterations: {training_result.get('train_iters', '?')}")
        print(f"    Baseline loss: {training_result.get('baseline_loss', '?')}")
        print(f"    Final loss: {training_result.get('final_loss', '?')}")
        print(f"    Post-val loss: {training_result.get('post_loss', '?')}")
        print(f"    CKA: {training_result.get('min_cka', '?')}")
        print(f"    Spectral bounds OK: {training_result.get('spectral_bounds_ok', '?')}")
        print(f"    Pipeline gate: {training_result.get('pipeline_gate_passed', '?')}")
        print(f"    Adapter: {training_result.get('adapter_path', '?')}")

        # Save training result
        result_path = args.output_dir / f"{skill.name}_training_result.json"
        with open(result_path, "w") as f:
            json.dump(training_result, f, indent=2)

    # Step 4: Post-training eval
    print("\n--- Step 4: Post-Training Evaluation ---")
    resolved_adapter = training_result.get("adapter_path", str(adapter_path)) if training_result else str(adapter_path)

    if not Path(resolved_adapter).exists():
        print(f"  ERROR: Adapter not found at {resolved_adapter}")
        print("  Training may have failed the pipeline gate (--no-save behavior)")
        return

    post = _run_mastery_eval(args.model, skill, adapter_path=resolved_adapter)
    print(f"  Accuracy: {post.accuracy:.1%} ({post.n_correct}/{post.n_total})")
    print(f"  CI: [{post.ci_lower:.3f}, {post.ci_upper:.3f}]")
    print(f"  Regime: {post.regime}")

    # Show sample outputs with adapter
    print("\n--- Post-Training Sample Outputs ---")
    _run_sample_inference(args.model, skill, n_samples=args.samples, adapter_path=resolved_adapter)

    # Step 5: Comparison
    print("\n--- Step 5: Results Comparison ---")
    delta = post.accuracy - baseline.accuracy
    print(f"  Baseline: {baseline.accuracy:.1%} ({baseline.n_correct}/{baseline.n_total})")
    print(f"  Post:     {post.accuracy:.1%} ({post.n_correct}/{post.n_total})")
    print(f"  Delta:    {delta:+.1%}")

    if delta > 0:
        print(f"\n  RESULT: Accuracy IMPROVED by {delta:.1%}")
    elif delta == 0:
        print(f"\n  RESULT: No change in accuracy")
    else:
        print(f"\n  RESULT: Accuracy DEGRADED by {abs(delta):.1%}")

    if training_result:
        loss_improved = (
            training_result.get("final_loss", float("inf"))
            < training_result.get("baseline_loss", float("inf"))
        )
        print(f"  Loss decreased: {loss_improved}")
        if loss_improved and delta <= 0:
            print("\n  DIAGNOSIS: Loss decreased but accuracy didn't improve.")
            print("  Possible causes:")
            print("    1. Format mismatch: training data format differs from eval expectations")
            print("    2. Overfitting to training distribution without generalization")
            print("    3. Model capacity: adapter too small to encode the skill")
            print("    4. Answer extraction: model outputs correct reasoning but wrong format")
            print("  Check the sample outputs above to diagnose which case applies.")

    # Save summary
    summary = {
        "skill": skill.name,
        "model": args.model,
        "adapter": resolved_adapter,
        "baseline_accuracy": baseline.accuracy,
        "baseline_n_correct": baseline.n_correct,
        "baseline_n_total": baseline.n_total,
        "baseline_regime": baseline.regime,
        "post_accuracy": post.accuracy,
        "post_n_correct": post.n_correct,
        "post_n_total": post.n_total,
        "post_regime": post.regime,
        "accuracy_delta": delta,
        "training_result": training_result,
    }
    summary_path = args.output_dir / f"{skill.name}_experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    # Cleanup temp training file
    print()


if __name__ == "__main__":
    main()
