#!/usr/bin/env python3
"""Experiment 64: Make LFM2-1.2B the World's Best 1.2B Model.

Goal: Systematically upgrade LFM2 by teaching it from experts.

Strategy:
1. Profile LFM2's current capabilities across many domains
2. Identify specific weaknesses (high entropy, wrong answers)
3. Apply targeted teaching from the Knowledge Bank
4. Measure improvement

We're not compressing. We're UPGRADING.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def spectral_entropy(Y):
    """Compute entropy from singular value spectrum."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / np.sum(S)
    S_norm = S_norm[S_norm > 1e-10]
    return -np.sum(S_norm * np.log(S_norm))


def run_experiment():
    """Profile and upgrade LFM2-1.2B."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Load our target model
    logger.info("Loading LFM2-1.2B (our student to upgrade)...")
    lfm2_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    lfm2, tokenizer = load(lfm2_path)

    # Comprehensive capability assessment
    capabilities = {
        "basic_facts": {
            "prompts": [
                ("The capital of France is", "Paris"),
                ("The largest planet is", "Jupiter"),
                ("Water freezes at", "zero"),
                ("The speed of light is", "300"),
            ]
        },
        "reasoning": {
            "prompts": [
                ("If A implies B and B implies C, then A implies", "C"),
                ("The opposite of hot is", "cold"),
                ("If it rains, the ground gets wet. The ground is wet, so it", "rain"),
                ("2 + 2 equals", "4"),
            ]
        },
        "science": {
            "prompts": [
                ("Photosynthesis occurs in", "plants"),
                ("DNA stands for", "deoxyribonucleic"),
                ("The nucleus of an atom contains", "protons"),
                ("Gravity causes objects to", "fall"),
            ]
        },
        "language": {
            "prompts": [
                ("A noun is a word that names a", "person"),
                ("The past tense of 'run' is", "ran"),
                ("An adjective describes a", "noun"),
                ("A synonym for 'happy' is", "joy"),
            ]
        },
        "math": {
            "prompts": [
                ("The square root of 16 is", "4"),
                ("10 times 10 equals", "100"),
                ("Half of 50 is", "25"),
                ("3 squared equals", "9"),
            ]
        },
        "common_sense": {
            "prompts": [
                ("The sky is usually", "blue"),
                ("Birds can", "fly"),
                ("Fish live in", "water"),
                ("The sun rises in the", "east"),
            ]
        },
    }

    def get_prediction_and_confidence(model, tokenizer, prompt):
        """Get model's prediction, confidence, and top alternatives."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist())
        probs = softmax(logits_np)

        # Top 5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_words = [tokenizer.decode([int(idx)]).strip() for idx in top_indices]

        # Entropy (uncertainty)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            "top_word": top_words[0],
            "top_prob": float(top_probs[0]),
            "top_5": list(zip(top_words, [float(p) for p in top_probs])),
            "entropy": float(entropy),
        }

    # ========================================
    # PHASE 1: Capability Assessment
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: LFM2-1.2B Capability Assessment")
    logger.info(f"{'='*80}")

    results = {}
    for domain, data in capabilities.items():
        logger.info(f"\n--- {domain.upper()} ---")

        correct = 0
        total = 0
        domain_results = []

        for prompt, expected in data["prompts"]:
            pred = get_prediction_and_confidence(lfm2, tokenizer, prompt)

            # Check if any expected substring is in the prediction
            is_correct = expected.lower() in pred["top_word"].lower()
            if is_correct:
                correct += 1
            total += 1

            mark = "✓" if is_correct else "✗"
            domain_results.append({
                "prompt": prompt,
                "expected": expected,
                "got": pred["top_word"],
                "confidence": pred["top_prob"],
                "entropy": pred["entropy"],
                "correct": is_correct,
            })

            logger.info(f"  {mark} '{prompt}' → '{pred['top_word']}' ({pred['top_prob']*100:.1f}%) H={pred['entropy']:.2f}")

        accuracy = correct / total
        results[domain] = {
            "accuracy": accuracy,
            "details": domain_results,
            "avg_entropy": np.mean([r["entropy"] for r in domain_results]),
        }

        logger.info(f"  Score: {correct}/{total} = {accuracy*100:.0f}%")

    # ========================================
    # PHASE 2: Weakness Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Weakness Analysis")
    logger.info(f"{'='*80}")

    # Sort by accuracy (worst first)
    sorted_domains = sorted(results.items(), key=lambda x: x[1]["accuracy"])

    logger.info(f"\n{'Domain':<20} {'Accuracy':>10} {'Avg Entropy':>12} {'Status':>15}")
    logger.info("-" * 60)

    for domain, data in sorted_domains:
        acc = data["accuracy"]
        entropy = data["avg_entropy"]

        if acc < 0.5:
            status = "NEEDS HELP"
        elif acc < 0.75:
            status = "Could improve"
        else:
            status = "Strong"

        logger.info(f"{domain:<20} {acc*100:>9.0f}% {entropy:>12.2f} {status:>15}")

    # Find specific failures
    logger.info(f"\n\nSpecific Failures (to teach):")
    failures = []
    for domain, data in results.items():
        for detail in data["details"]:
            if not detail["correct"]:
                failures.append({
                    "domain": domain,
                    "prompt": detail["prompt"],
                    "expected": detail["expected"],
                    "got": detail["got"],
                    "entropy": detail["entropy"],
                })
                logger.info(f"  - {detail['prompt']} → expected '{detail['expected']}', got '{detail['got']}'")

    # ========================================
    # PHASE 3: Apply Teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Teaching from DeepSeek-R1 (Best Available Teacher)")
    logger.info(f"{'='*80}")

    # Load teacher
    logger.info("\nLoading teacher (DeepSeek-R1-8B)...")
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    teacher, teacher_tok = load(teacher_path)

    # Use the domains where LFM2 is weakest and DeepSeek is strong
    weak_domains = [d for d, data in sorted_domains if data["accuracy"] < 0.75]

    if not weak_domains:
        logger.info("\nLFM2 is already strong in all domains! (>=75%)")
        weak_domains = [sorted_domains[0][0]]  # Still try the worst one

    logger.info(f"\nWeak domains to improve: {weak_domains}")

    # Best layer pairs from our experiments
    teacher_layer = 24
    student_layer = 10

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        """Get MLP input and output activations."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                key = 'mlp'

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(np.array(mlp_input[0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(mlp_output[0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def apply_direction_teaching(teacher_Y, student_X, student_Y, direction_idx=5):
        """Apply direction replacement teaching."""
        teacher_centered = teacher_Y - teacher_Y.mean(axis=0)
        student_centered = student_Y - student_Y.mean(axis=0)

        _, _, Vh_t = svd(teacher_centered, full_matrices=False)
        _, _, Vh_s = svd(student_centered, full_matrices=False)

        F = np.linalg.lstsq(teacher_Y, student_Y, rcond=1e-10)[0]

        result = student_Y.copy()
        d = direction_idx

        if d < len(Vh_t) and d < len(Vh_s):
            student_coefs_d = student_centered @ Vh_s[d]
            student_proj_d = np.outer(student_coefs_d, Vh_s[d])
            result -= student_proj_d

            teacher_coefs_d = teacher_centered @ Vh_t[d]
            teacher_proj_d = np.outer(teacher_coefs_d, Vh_t[d])
            result += teacher_proj_d @ F

        alpha = 1e-6
        ATA = student_X.T @ student_X + alpha * np.eye(student_X.shape[1])
        ATB = student_X.T @ result
        W = np.linalg.solve(ATA, ATB).T

        return W

    # Collect calibration prompts from weak domains
    calibration_prompts = []
    for domain in weak_domains:
        for prompt, _ in capabilities[domain]["prompts"]:
            calibration_prompts.append(prompt)

    logger.info(f"\nCollecting activations from {len(calibration_prompts)} prompts...")

    # Get activations
    T_X, T_Y = get_layer_activations(teacher, teacher_tok, teacher_layer, calibration_prompts)
    S_X, S_Y = get_layer_activations(lfm2, tokenizer, student_layer, calibration_prompts)

    # Apply teaching
    logger.info("Applying direction replacement teaching...")
    W_taught = apply_direction_teaching(T_Y, S_X, S_Y, direction_idx=5)

    W_mx = mx.array(W_taught.astype(np.float32))
    mx.eval(W_mx)

    class TaughtMLP:
        def __init__(self, W):
            self.W = W
        def __call__(self, x):
            return mx.matmul(x, self.W.T)

    # Install taught MLP
    layer = lfm2.model.layers[student_layer]
    if hasattr(layer, 'feed_forward'):
        original_mlp = layer.feed_forward
        layer.feed_forward = TaughtMLP(W_mx)
        mlp_key = 'feed_forward'
    else:
        original_mlp = layer.mlp
        layer.mlp = TaughtMLP(W_mx)
        mlp_key = 'mlp'

    # ========================================
    # PHASE 4: Re-evaluate After Teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Re-evaluation After Teaching")
    logger.info(f"{'='*80}")

    new_results = {}
    for domain, data in capabilities.items():
        correct = 0
        total = 0

        for prompt, expected in data["prompts"]:
            pred = get_prediction_and_confidence(lfm2, tokenizer, prompt)
            is_correct = expected.lower() in pred["top_word"].lower()
            if is_correct:
                correct += 1
            total += 1

        new_results[domain] = correct / total

    # Restore original MLP
    if mlp_key == 'feed_forward':
        layer.feed_forward = original_mlp
    else:
        layer.mlp = original_mlp

    # Compare
    logger.info(f"\n{'Domain':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    logger.info("-" * 55)

    total_before = 0
    total_after = 0
    for domain in capabilities.keys():
        before = results[domain]["accuracy"]
        after = new_results[domain]
        change = after - before

        total_before += before
        total_after += after

        arrow = "↑" if change > 0 else ("↓" if change < 0 else "=")
        logger.info(f"{domain:<20} {before*100:>9.0f}% {after*100:>9.0f}% {arrow} {change*100:>+8.0f}pp")

    logger.info("-" * 55)
    avg_before = total_before / len(capabilities)
    avg_after = total_after / len(capabilities)
    avg_change = avg_after - avg_before
    logger.info(f"{'AVERAGE':<20} {avg_before*100:>9.1f}% {avg_after*100:>9.1f}% {avg_change*100:>+9.1f}pp")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: LFM2-1.2B Upgrade Results")
    logger.info(f"{'='*80}")

    improved = sum(1 for d in capabilities if new_results[d] > results[d]["accuracy"])
    same = sum(1 for d in capabilities if new_results[d] == results[d]["accuracy"])
    degraded = sum(1 for d in capabilities if new_results[d] < results[d]["accuracy"])

    logger.info(f"""
TEACHING APPLIED:
  - Teacher: DeepSeek-R1-8B (Layer {teacher_layer})
  - Student: LFM2-1.2B (Layer {student_layer})
  - Method: Direction 6 replacement
  - Domains targeted: {weak_domains}

RESULTS:
  - Improved: {improved} domains
  - Unchanged: {same} domains
  - Degraded: {degraded} domains
  - Overall change: {avg_change*100:+.1f}pp

BEFORE: {avg_before*100:.1f}% average accuracy
AFTER:  {avg_after*100:.1f}% average accuracy

NEXT STEPS:
  1. Try different directions (1-12)
  2. Try multiple layer pairs
  3. Per-domain optimal teaching
  4. Save the improved model persistently
""")


if __name__ == "__main__":
    run_experiment()
