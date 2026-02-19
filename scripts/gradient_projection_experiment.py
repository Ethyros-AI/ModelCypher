#!/usr/bin/env python3
"""Gradient Projection Causal Experiment

Three tests to establish causality of the format-conditioning account:

1. INTERVENTION: Narrow-format data, but project out the top format eigen-
   component from each gradient step. If MT improves, conditioning is causal.

2. REINJECTION: Augmented data, but add back format-component energy.
   MT should degrade monotonically as reinjection strength grows.

3. THRESHOLD: Track eigenvalue crossing of logic-vs-format gradient
   covariance during sample growth; predict n* and compare to ~990.

Usage:
  python scripts/gradient_projection_experiment.py --arm intervention
  python scripts/gradient_projection_experiment.py --arm reinjection --alpha 0.5
  python scripts/gradient_projection_experiment.py --arm baseline-narrow
  python scripts/gradient_projection_experiment.py --arm baseline-augmented
  python scripts/gradient_projection_experiment.py --arm threshold
  python scripts/gradient_projection_experiment.py --arm all

Results go to /Volumes/CodeCypher/experiments/gradient-projection-causal/
"""
import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import default_loss, iterate_batches
from mlx.utils import tree_flatten as mlx_flatten, tree_unflatten

from modelcypher.adapters.mlx_training_adapter import MLXTrainingAdapter
from modelcypher.backends.mlx_backend import MLXBackend
from modelcypher.core.domain.training.geometric_lora import (
    analyze_weight_geometries,
    select_target_modules,
)
from modelcypher.core.domain.training.geometric_optimizer import (
    derive_optimizer_geometry_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
NARROW_DATA = "data/training/ce_reasoning_traces_train.jsonl"
NARROW_EVAL = "data/training/ce_reasoning_traces_val.jsonl"
AUGMENTED_DATA = "data/training/format_augmented_train.jsonl"
AUGMENTED_EVAL = "data/training/format_augmented_val.jsonl"

# MT evaluation cases
MT_CASES = [
    ("mammals",
     "Apply logical reasoning:\nIf an animal is a mammal, then it is warm-blooded. "
     "This animal is not warm-blooded. What can we conclude?",
     "mammal", False),
    ("rain",
     "Apply logical reasoning:\nIf it rains, the streets get wet. "
     "The streets are not wet. What can we conclude?",
     "rain", False),
    ("diff",
     "Apply logical reasoning:\nIf a function is differentiable at a point, "
     "then it is continuous at that point. Function f is not continuous at x=3. "
     "What can we conclude?",
     "differentiable", False),
    ("cert",
     "Apply logical reasoning:\nEvery employee who passed the certification "
     "received a bonus. Maria did not receive a bonus. "
     "What can we conclude about Maria's certification?",
     "pass", False),
    ("birds",
     "Apply logical reasoning:\nAll birds have feathers. An animal does not "
     "have feathers. Is it a bird?",
     "bird", True),
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_dataset(samples, tokenizer):
    """Tokenize to mlx-lm format: list of (token_array, 0)."""
    dataset = []
    for s in samples:
        text = s.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        dataset.append((mx.array(tokens, dtype=mx.int32), 0))
    return dataset


def first_stated_answer(response):
    """Extract the first sentence/clause that states the answer."""
    text = response.strip()
    # Skip reasoning preamble: "We know that...", "The premise states..."
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that just restate the premise
        if any(line.lower().startswith(p) for p in [
            "we know", "the premise", "premise:", "given:", "recall",
        ]):
            continue
        return line.lower()
    return text[:200].lower()


def check_mt_correct(response, keyword, yes_no_format):
    """Check if response correctly negates the keyword."""
    answer = first_stated_answer(response)
    if yes_no_format and answer.strip().startswith("no"):
        return True
    neg_patterns = [
        rf'not\s+.*\b{keyword}\b',
        rf'cannot\s+.*\b{keyword}\b',
        rf"doesn't\s+.*\b{keyword}\b",
        rf"isn't\s+.*\b{keyword}\b",
        rf'not\s+the\s+case\s+that\s+.*\b{keyword}\b',
    ]
    for pat in neg_patterns:
        if re.search(pat, answer, re.IGNORECASE):
            return True
    # Check first 2 lines
    first_2 = ' '.join(response.strip().split('\n')[:2]).lower()[:300]
    for pat in neg_patterns:
        if re.search(pat, first_2, re.IGNORECASE):
            return True
    return False


def evaluate_mt(model, tokenizer):
    """Evaluate MT accuracy on 5 test cases. Returns score and details."""
    results = []
    for domain, prompt, keyword, yn_fmt in MT_CASES:
        resp = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        passed = check_mt_correct(resp, keyword, yn_fmt)
        results.append({
            "domain": domain,
            "passed": passed,
            "first_answer": first_stated_answer(resp)[:100],
        })
    score = sum(r["passed"] for r in results)
    return score, results


# =====================================================================
# EXPANDED EVALUATION SUITES
# =====================================================================

def evaluate_novel_problems(model, tokenizer, n=60, seed=42):
    """Evaluate on combinatorially-generated novel logic problems.

    Returns: {"total": n, "correct": int, "by_form": {form: {"n": int, "correct": int}}}
    """
    from novel_problems import generate_novel_problems

    problems = generate_novel_problems(n, seed)
    by_form = {}
    correct = 0

    for p in problems:
        resp = generate(model, tokenizer, prompt=p.prompt, max_tokens=200)
        passed = p.verify(resp)
        if passed:
            correct += 1
        entry = by_form.setdefault(p.logic, {"n": 0, "correct": 0})
        entry["n"] += 1
        if passed:
            entry["correct"] += 1

    logger.info("Novel problems: %d/%d correct", correct, n)
    return {"total": n, "correct": correct, "by_form": by_form}


def evaluate_inference_suite(model, tokenizer):
    """Evaluate on the 20-problem inference test suite.

    Returns: {"total": int, "correct": int, "by_category": {cat: {"n": int, "correct": int}}}
    """
    suite_path = Path(__file__).parent.parent / "data" / "eval_prompts" / "nblora_inference_tests.jsonl"
    problems = load_jsonl(str(suite_path))

    by_category = {}
    correct = 0

    for p in problems:
        resp = generate(model, tokenizer, prompt=p["prompt"], max_tokens=200)
        expected = p["expected"].lower()
        resp_lower = resp.strip().lower()

        # Check: does the response contain the key expected content?
        passed = False
        # For math: check numeric answers with word boundaries
        # e.g. "5" must match "5 minutes" but not "15" or "45"
        numbers_expected = re.findall(r'[\$]?([\d.]+)', expected)
        if numbers_expected:
            for num in numbers_expected:
                # Word-boundary match: \b5\b matches "5" but not "15"
                if re.search(r'\b' + re.escape(num) + r'\b', resp_lower):
                    passed = True
                    break
        # For logic: check key phrases (>8 chars to avoid trivial matches)
        if not passed:
            # Extract first clause of expected, split on period or parenthetical
            key_phrases = [
                s.strip().lower()
                for s in re.split(r'[.(]', expected)
                if len(s.strip()) > 8
            ]
            for phrase in key_phrases[:2]:
                # Require phrase to appear as coherent substring (already >8 chars,
                # so accidental substring matches are unlikely)
                if phrase in resp_lower:
                    passed = True
                    break

        if passed:
            correct += 1

        cat = p.get("category", "unknown")
        entry = by_category.setdefault(cat, {"n": 0, "correct": 0})
        entry["n"] += 1
        if passed:
            entry["correct"] += 1

    total = len(problems)
    logger.info("Inference suite: %d/%d correct", correct, total)
    return {"total": total, "correct": correct, "by_category": by_category}


def evaluate_feasibility_suite(model, tokenizer):
    """Evaluate on the 20-problem feasibility suite from test_bootstrap_feasibility.py.

    Returns: {"total": int, "correct": int, "by_logic": {logic: {"n": int, "correct": int}}}
    """
    from test_bootstrap_feasibility import NOVEL_PROBLEMS

    by_logic = {}
    correct = 0

    for p in NOVEL_PROBLEMS:
        resp = generate(model, tokenizer, prompt=p["prompt"], max_tokens=200)
        passed = p["verify"](resp)
        if passed:
            correct += 1

        logic = p.get("logic", "unknown")
        entry = by_logic.setdefault(logic, {"n": 0, "correct": 0})
        entry["n"] += 1
        if passed:
            entry["correct"] += 1

    total = len(NOVEL_PROBLEMS)
    logger.info("Feasibility suite: %d/%d correct", correct, total)
    return {"total": total, "correct": correct, "by_logic": by_logic}


def evaluate_all(model, tokenizer):
    """Run all evaluation suites. Returns combined dict."""
    mt_score, mt_details = evaluate_mt(model, tokenizer)
    novel = evaluate_novel_problems(model, tokenizer, n=60, seed=42)
    inference = evaluate_inference_suite(model, tokenizer)
    feasibility = evaluate_feasibility_suite(model, tokenizer)

    return {
        "mt": {"score": mt_score, "total": 5, "details": mt_details},
        "novel": novel,
        "inference": inference,
        "feasibility": feasibility,
    }


# =====================================================================
# FORMAT BIAS DECOMPOSITION (geometrically derived)
# =====================================================================
#
# Theory:
#   μ_narrow   = μ_invariant + μ_format     (signal + format bias)
#   μ_augmented ≈ μ_invariant                (format cancels under group avg)
#   μ_format   = μ_narrow - μ_augmented      (derivable from data)
#   α_crit     = ‖μ_invariant‖ / ‖μ_format‖  (bias = signal threshold)
#
# Intervention: project out v_format = μ_format / ‖μ_format‖ from each grad
# Reinjection:  add μ_format to each augmented grad (fixed bias, not amplification)
#

def _compute_mean_gradient(model, tokenizer, samples, n_samples=40):
    """Compute mean gradient direction over samples. Returns float32 numpy vector."""
    loss_vg = nn.value_and_grad(model, default_loss)
    dataset = tokenize_dataset(samples[:n_samples], tokenizer)

    sum_g = None
    count = 0

    for i, (tokens, _) in enumerate(dataset):
        batch = tokens.reshape(1, -1)
        lengths = mx.array([[0, batch.shape[1]]])
        (loss, ntoks), grad = loss_vg(model, batch, lengths)
        mx.eval(loss)

        flat = []
        for name, arr in mlx_flatten(grad):
            if 'A_tilde' in name or 'B_tilde' in name or 'lora_a' in name or 'lora_b' in name:
                flat.append(arr.reshape(-1).astype(mx.float32))
        if flat:
            g = mx.concatenate(flat)
            mx.eval(g)
            g_np = np.array(g.tolist(), dtype=np.float64)
            if sum_g is None:
                sum_g = g_np
            else:
                sum_g += g_np
            count += 1

        if (i + 1) % 20 == 0:
            logger.info("  Mean gradient: %d/%d samples", i + 1, len(dataset))

    if count == 0:
        raise RuntimeError("No valid gradients computed")
    return (sum_g / count).astype(np.float32), count


def compute_format_bias(model, tokenizer, narrow_samples, augmented_samples, n_samples=40):
    """Derive the format bias vector and critical reinjection strength.

    Returns:
        mu_format: float32 numpy [d] — the format bias direction (unnormalized)
        v_format:  float32 numpy [d] — unit format bias direction
        alpha_crit: float — ‖μ_invariant‖ / ‖μ_format‖
        mu_narrow:  float32 numpy [d] — mean narrow gradient
        mu_aug:     float32 numpy [d] — mean augmented gradient (≈ invariant)
    """
    logger.info("Computing mean gradient on %d narrow samples...", n_samples)
    mu_narrow, n_narrow = _compute_mean_gradient(model, tokenizer, narrow_samples, n_samples)

    logger.info("Computing mean gradient on %d augmented samples...", n_samples)
    mu_aug, n_aug = _compute_mean_gradient(model, tokenizer, augmented_samples, n_samples)

    # Format bias: the difference
    mu_format = mu_narrow - mu_aug

    norm_format = np.linalg.norm(mu_format.astype(np.float64))
    norm_invariant = np.linalg.norm(mu_aug.astype(np.float64))  # μ_aug ≈ μ_invariant
    norm_narrow = np.linalg.norm(mu_narrow.astype(np.float64))

    # Unit format direction
    if norm_format > 1e-20:
        v_format = (mu_format / norm_format).astype(np.float32)
    else:
        v_format = np.zeros_like(mu_format)

    # Critical alpha: where injected bias equals signal strength
    alpha_crit = float(norm_invariant / max(norm_format, 1e-20))

    # Verification: cosine between narrow and augmented mean gradients
    cos_narrow_aug = float(np.dot(
        mu_narrow.astype(np.float64), mu_aug.astype(np.float64)
    ) / max(norm_narrow * norm_invariant, 1e-20))

    # Format fraction of narrow gradient: ||μ_format||² / ||μ_narrow||²
    format_frac = float(norm_format**2 / max(norm_narrow**2, 1e-20))

    logger.info("Format bias decomposition:")
    logger.info("  ‖μ_narrow‖    = %.6f  (n=%d)", norm_narrow, n_narrow)
    logger.info("  ‖μ_augmented‖ = %.6f  (n=%d, ≈ μ_invariant)", norm_invariant, n_aug)
    logger.info("  ‖μ_format‖    = %.6f  (bias = μ_narrow - μ_aug)", norm_format)
    logger.info("  cos(μ_narrow, μ_aug) = %.4f", cos_narrow_aug)
    logger.info("  format fraction of narrow grad = %.4f", format_frac)
    logger.info("  α_crit = ‖μ_invariant‖/‖μ_format‖ = %.4f", alpha_crit)

    return mu_format, v_format, alpha_crit, mu_narrow, mu_aug


def project_out_bias(grad, v_format_mx, param_keys):
    """Remove format bias direction from gradient.

    g_clean = g - (v · g) v   where v = μ_format / ‖μ_format‖
    """
    flat = dict(mlx_flatten(grad))
    pieces = []
    for key in param_keys:
        if key in flat:
            pieces.append(flat[key].reshape(-1).astype(mx.float32))
    if not pieces:
        return grad

    g_vec = mx.concatenate(pieces)  # [d]
    mx.eval(g_vec)

    # Project out: g_clean = g - (v^T g) v
    coeff = mx.sum(v_format_mx * g_vec)  # scalar dot product
    g_clean = g_vec - coeff * v_format_mx
    mx.eval(g_clean)

    # Unflatten back
    offset = 0
    for key in param_keys:
        if key in flat:
            size = flat[key].size
            shape = flat[key].shape
            flat[key] = g_clean[offset:offset + size].reshape(shape)
            offset += size

    return tree_unflatten(flat)


def reinject_bias(grad, mu_format_mx, param_keys):
    """Add fixed format bias to gradient (simulates narrow-format training).

    g_contaminated = g + μ_format
    The bias is fixed (not scaled by grad content) — this is what
    narrow-format training actually does: every sample shares the same
    format bias component in its gradient.
    """
    flat = dict(mlx_flatten(grad))
    pieces = []
    for key in param_keys:
        if key in flat:
            pieces.append(flat[key].reshape(-1).astype(mx.float32))
    if not pieces:
        return grad

    g_vec = mx.concatenate(pieces)
    mx.eval(g_vec)

    # Add fixed bias: g_contaminated = g + μ_format
    g_contaminated = g_vec + mu_format_mx
    mx.eval(g_contaminated)

    offset = 0
    for key in param_keys:
        if key in flat:
            size = flat[key].size
            shape = flat[key].shape
            flat[key] = g_contaminated[offset:offset + size].reshape(shape)
            offset += size

    return tree_unflatten(flat)


# =====================================================================
# TRAINING LOOP WITH GRADIENT HOOK
# =====================================================================

def train_with_projection(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    batch_size,
    seq_length,
    max_epochs=10,
    sigma_max=1.0,
    seed=42,
    gradient_hook=None,
    hook_label="none",
):
    """Simplified training loop with optional gradient projection hook.

    gradient_hook: callable(grad) -> grad, applied after backward but before
                   optimizer update. None = standard training.
    """
    backend = MLXBackend()
    adapter = MLXTrainingAdapter(backend)
    loss_fn = default_loss
    loss_vg = nn.value_and_grad(model, loss_fn)

    # Measure Lipschitz for learning rate
    L = adapter._measure_lipschitz_robust(
        model, train_dataset, batch_size, seq_length,
        loss_fn, n_batches=3, n_iters=10, seed=seed,
    )
    if L is not None and L > 0:
        eta = 1.0 / L
    else:
        eta = 1.0 / sigma_max
    logger.info("LR = %.4e (L=%.4f)", eta, L if L else 1.0/eta)

    L_current = L if (L and L > 0) else 1.0 / eta
    optimizer = opt.SGD(learning_rate=eta)

    # Val loss tracking for best checkpoint
    best_val_loss = float('inf')
    best_params = None

    epoch_losses = []
    epoch_val_losses = []

    for epoch in range(max_epochs):
        batch_iter = iterate_batches(
            dataset=train_dataset,
            batch_size=batch_size,
            max_seq_length=seq_length,
        )

        epoch_loss_sum = 0
        epoch_steps = 0

        for batch, lengths in batch_iter:
            (loss, ntoks), grad = loss_vg(model, batch, lengths)
            mx.eval(loss)

            # Cayley-Riemannian preconditioning
            grad, precond_metrics = adapter._apply_cayley_preconditioner(model, grad)
            lambda_max_P = precond_metrics.get("precond_lambda_max", 1.0)
            eps_mach = math.ldexp(1.0, -23)
            eta_max = 2.0 / (L_current * lambda_max_P + eps_mach)
            eta_step = min(eta, eta_max)

            # === GRADIENT HOOK: insert projection here ===
            if gradient_hook is not None:
                grad = gradient_hook(grad)

            optimizer.learning_rate = mx.array(eta_step)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)

            # Clamp NB-LoRA scales
            adapter._clamp_all_scales(model)

            epoch_loss_sum += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_losses.append(avg_loss)

        # Eval
        val_loss, val_ppl = adapter.evaluate_loss(
            model=model, dataset=eval_dataset, tokenizer=tokenizer,
            batch_size=max(1, min(4, len(eval_dataset) // 10)),
            seq_length=seq_length, n_batches=10,
        )
        epoch_val_losses.append(val_loss)

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                k: mx.array(v)
                for k, v in dict(mlx_flatten(model.trainable_parameters())).items()
            }
            mx.eval(*best_params.values())

        logger.info(
            "Epoch %d [%s]: train_loss=%.4f, val_loss=%.4f (best=%.4f)",
            epoch, hook_label, avg_loss, val_loss, best_val_loss,
        )

        # Val-loss stopping: 3 epochs non-improving
        if len(epoch_val_losses) >= 4:
            recent = epoch_val_losses[-3:]
            if all(r >= best_val_loss * 1.01 for r in recent):
                logger.info("Val-loss stopping: 3 epochs non-improving")
                break

    # Restore best checkpoint
    if best_params is not None:
        model.load_weights(list(best_params.items()), strict=False)
        mx.eval(model.parameters())
        logger.info("Restored best checkpoint (val_loss=%.4f)", best_val_loss)

    return epoch_losses, epoch_val_losses, best_val_loss


# =====================================================================
# SETUP: Model + LoRA injection
# =====================================================================

def setup_model_with_lora(model_path, seed=42):
    """Load model, analyze geometry, inject NB-LoRA, freeze base."""
    backend = MLXBackend()
    adapter = MLXTrainingAdapter(backend)

    backend.random_seed(seed)
    model, tokenizer = backend.load_model(model_path)

    weights = adapter.extract_weight_matrices(model)
    geometries = analyze_weight_geometries(weights, backend)
    target_modules = select_target_modules(geometries)

    adapter.inject_nb_lora(model, geometries, target_modules, safety_margin=0.9)
    adapter.freeze_and_apply_lora(model)

    sigma_max = max(g.sigma_max for g in geometries.values() if g.sigma_max > 0)

    # Discover LoRA gradient keys by running a test gradient
    # (keys in gradient pytree may differ from trainable_parameters keys)
    loss_vg = nn.value_and_grad(model, default_loss)
    dummy_tokens = tokenizer.encode("test")
    dummy_batch = mx.array([dummy_tokens[:10]], dtype=mx.int32)
    dummy_lengths = mx.array([[0, dummy_batch.shape[1]]])
    (_, _), test_grad = loss_vg(model, dummy_batch, dummy_lengths)

    param_keys = []
    for name, arr in mlx_flatten(test_grad):
        if ('A_tilde' in name or 'B_tilde' in name or 'lora_a' in name or 'lora_b' in name) and arr.size > 0:
            param_keys.append(name)
    del test_grad

    n_params = sum(p.size for _, p in mlx_flatten(model.trainable_parameters()))
    logger.info("Model setup: %d LoRA params, %d grad keys, sigma_max=%.4f",
                n_params, len(param_keys), sigma_max)

    return model, tokenizer, sigma_max, param_keys


# =====================================================================
# EXPERIMENT ARMS
# =====================================================================

def run_arm(arm, output_dir=None):
    """Run one experiment arm."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = Path(f"/Volumes/CodeCypher/experiments/gradient-projection-causal/{arm}-{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ARM: %s", arm)
    logger.info("=" * 60)

    # Choose data
    if arm in ("baseline-narrow", "intervention"):
        train_path, eval_path = NARROW_DATA, NARROW_EVAL
    else:
        train_path, eval_path = AUGMENTED_DATA, AUGMENTED_EVAL

    train_samples = load_jsonl(train_path)
    eval_samples = load_jsonl(eval_path)

    # Setup model
    model, tokenizer, sigma_max, param_keys = setup_model_with_lora(MODEL_PATH)
    train_dataset = tokenize_dataset(train_samples, tokenizer)
    eval_dataset = tokenize_dataset(eval_samples, tokenizer)

    batch_size = MLXTrainingAdapter(MLXBackend()).derive_critical_batch_size(
        model, train_dataset, seq_length=256,
    )
    logger.info("Batch size: %d, Train samples: %d", batch_size, len(train_dataset))

    # Compute format bias decomposition (needed for intervention and reinjection)
    bias_info = None
    if arm in ("intervention", "reinjection"):
        narrow_samples = load_jsonl(NARROW_DATA)
        augmented_samples = load_jsonl(AUGMENTED_DATA)
        mu_format, v_format, alpha_crit, mu_narrow, mu_aug = compute_format_bias(
            model, tokenizer, narrow_samples, augmented_samples, n_samples=40,
        )
        bias_info = {
            "alpha_crit": alpha_crit,
            "norm_format": float(np.linalg.norm(mu_format)),
            "norm_invariant": float(np.linalg.norm(mu_aug)),
        }

    # Define gradient hook
    if arm == "intervention":
        # Project out the format bias direction from each gradient
        v_format_mx = mx.array(v_format)
        mx.eval(v_format_mx)
        def gradient_hook(grad):
            return project_out_bias(grad, v_format_mx, param_keys)
        hook_label = "project-out-bias"
    elif arm == "reinjection":
        # Add the fixed format bias to each augmented gradient
        # This simulates narrow-format training: every sample gets the same bias
        mu_format_mx = mx.array(mu_format)
        mx.eval(mu_format_mx)
        def gradient_hook(grad):
            return reinject_bias(grad, mu_format_mx, param_keys)
        hook_label = f"reinject-bias (α_crit={alpha_crit:.4f})"
    else:
        gradient_hook = None
        hook_label = "none"

    # Pre-training full eval
    logger.info("Running pre-training evaluation...")
    eval_pre = evaluate_all(model, tokenizer)
    mt_score_pre = eval_pre["mt"]["score"]
    logger.info("Pre-training MT: %d/5", mt_score_pre)

    # Train
    train_losses, val_losses, best_val = train_with_projection(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_length=256,
        max_epochs=10,
        sigma_max=sigma_max,
        gradient_hook=gradient_hook,
        hook_label=hook_label,
    )

    # Post-training full eval
    logger.info("Running post-training evaluation...")
    eval_post = evaluate_all(model, tokenizer)
    mt_score_post = eval_post["mt"]["score"]
    logger.info("Post-training MT: %d/5", mt_score_post)

    # Save results
    results = {
        "arm": arm,
        "train_data": train_path,
        "train_samples": len(train_samples),
        "batch_size": batch_size,
        "hook_label": hook_label,
        "bias_info": bias_info,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "mt_score_pre": mt_score_pre,
        "mt_score_post": mt_score_post,
        "mt_details_post": eval_post["mt"]["details"],
        "eval_pre": eval_pre,
        "eval_post": eval_post,
        "timestamp": timestamp,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


def run_threshold_test(output_dir=None):
    """Test 3: Track eigenvalue crossing during sample growth.

    For n = 50, 100, 200, 400, 600, 990:
    - Compute gradient covariance on n augmented samples
    - Decompose into format-aligned and format-orthogonal components
    - Track when the invariant component dominates
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = Path(f"/Volumes/CodeCypher/experiments/gradient-projection-causal/threshold-{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  THRESHOLD TEST: Eigenvalue crossing during sample growth")
    logger.info("=" * 60)

    model, tokenizer, sigma_max, param_keys = setup_model_with_lora(MODEL_PATH)

    # Pre-compute format bias direction from narrow vs augmented data
    narrow_samples = load_jsonl(NARROW_DATA)
    augmented_all = load_jsonl(AUGMENTED_DATA)
    mu_format, v_format, alpha_crit, mu_narrow, mu_aug = compute_format_bias(
        model, tokenizer, narrow_samples, augmented_all, n_samples=40,
    )
    # Use v_format (unit bias direction) as the 1D format subspace for projection
    V_format = v_format.reshape(1, -1)  # [1, d]

    # Load all augmented samples
    aug_samples = load_jsonl(AUGMENTED_DATA)
    random.seed(42)
    random.shuffle(aug_samples)

    loss_vg = nn.value_and_grad(model, default_loss)

    def compute_grads(samples):
        """Compute gradients for all given samples (float32)."""
        grads = []
        dataset = tokenize_dataset(samples, tokenizer)
        for i, (tokens, _) in enumerate(dataset):
            batch = tokens.reshape(1, -1)
            lengths = mx.array([[0, batch.shape[1]]])
            (loss, ntoks), grad = loss_vg(model, batch, lengths)
            mx.eval(loss)
            flat = []
            for name, arr in mlx_flatten(grad):
                if 'A_tilde' in name or 'B_tilde' in name or 'lora_a' in name or 'lora_b' in name:
                    flat.append(arr.reshape(-1).astype(mx.float32))
            if flat:
                g = mx.concatenate(flat)
                mx.eval(g)
                grads.append(np.array(g.tolist(), dtype=np.float32))
            if (i + 1) % 20 == 0:
                logger.info("  Threshold grads: %d/%d", i + 1, len(dataset))
        return np.stack(grads) if grads else np.zeros((0, 0), dtype=np.float32)

    # For each n, subsample GRADS_PER_N gradients from the pool of n.
    # As n grows, the pool becomes more format-diverse, so even a fixed
    # subsample size reflects the changing distribution.
    GRADS_PER_N = 60
    sample_counts = [20, 50, 100, 200, 400, 600, 990]
    results = []

    print(f"\n  {'n':>5s} {'n_grad':>6s} {'κ':>8s} {'eff_rank':>10s} {'format_frac':>12s} {'invariant_frac':>14s} {'cos_mean_v':>10s}")
    print(f"  {'-'*72}")

    for n in sample_counts:
        if n > len(aug_samples):
            break
        subset = aug_samples[:n]

        # Subsample: at each n, draw GRADS_PER_N random samples from the pool
        # Use n-dependent seed so each pool size gives a different subsample
        rng = np.random.RandomState(42 + n)
        n_grads = min(GRADS_PER_N, n)
        indices = rng.choice(n, size=n_grads, replace=False)
        grad_samples = [subset[i] for i in sorted(indices)]

        logger.info("n=%d: computing %d gradients...", n, n_grads)
        G = compute_grads(grad_samples)
        n_actual = G.shape[0]

        # Gram matrix eigenspectrum (use float64 for numerical stability)
        G64 = G.astype(np.float64)
        gram = G64 @ G64.T / n_actual
        eigvals = np.linalg.eigvalsh(gram)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = eigvals[eigvals > 1e-20]

        p = eigvals / eigvals.sum()
        p = p[p > 1e-15]
        eff_rank = np.exp(-np.sum(p * np.log(p)))

        positive = eigvals[eigvals > 1e-10]
        cond = positive[0] / positive[-1] if len(positive) > 1 else float('inf')

        # Project gradients onto format subspace
        # format_energy = ||V @ g||^2 / ||g||^2 for each sample
        V64 = V_format.astype(np.float64)
        format_fracs = []
        for i in range(n_actual):
            g = G64[i]
            g_norm_sq = np.dot(g, g)
            if g_norm_sq < 1e-20:
                continue
            proj = V64 @ g  # [k]
            format_energy = np.dot(proj, proj)
            format_fracs.append(format_energy / g_norm_sq)

        mean_format_frac = np.mean(format_fracs) if format_fracs else 0.0
        invariant_frac = 1 - mean_format_frac

        # Cosine of mean gradient with format subspace
        mean_g = G64.mean(axis=0)
        mean_norm = np.linalg.norm(mean_g)
        if mean_norm > 1e-10:
            proj_mean = V64 @ mean_g
            cos_mean_v = np.linalg.norm(proj_mean) / mean_norm
        else:
            cos_mean_v = 0

        del G64  # free memory

        print(f"  {n:5d} {n_actual:6d} {cond:8.1f} {eff_rank:10.2f} {mean_format_frac:12.4f} "
              f"{invariant_frac:14.4f} {cos_mean_v:10.4f}")

        results.append({
            "n": n,
            "n_actual_grads": n_actual,
            "cond": float(cond),
            "eff_rank": float(eff_rank),
            "format_frac": float(mean_format_frac),
            "invariant_frac": float(invariant_frac),
            "cos_mean_format": float(cos_mean_v),
        })

    # Save
    with open(output_dir / "threshold_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Threshold results saved to %s", output_dir)

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_eval_only(adapter_dir, model_path=None, output_dir=None):
    """Load a saved adapter and run full evaluation (no training)."""
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Resolve base model: explicit arg > adapter metadata > script default
    if model_path is None:
        adapter_config = adapter_dir / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                cfg = json.load(f)
            model_path = cfg.get("base_model_path", MODEL_PATH)
            logger.info("Base model from adapter_config.json: %s", model_path)
        else:
            model_path = MODEL_PATH
            logger.warning(
                "No --model and no adapter_config.json; using default %s",
                MODEL_PATH,
            )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = adapter_dir / f"eval-{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  EVAL-ONLY: %s", adapter_dir)
    logger.info("  BASE MODEL: %s", model_path)
    logger.info("=" * 60)

    # Load model with adapter
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_path, adapter_path=str(adapter_dir))

    # Run full eval
    eval_results = evaluate_all(model, tokenizer)

    # Print summary
    mt = eval_results["mt"]
    novel = eval_results["novel"]
    inference = eval_results["inference"]
    feasibility = eval_results["feasibility"]

    print(f"\n{'='*60}")
    print(f"  EVAL-ONLY RESULTS: {adapter_dir.name}")
    print(f"{'='*60}")
    print(f"  MT:          {mt['score']}/{mt['total']}")
    print(f"  Novel:       {novel['correct']}/{novel['total']}")
    print(f"  Inference:   {inference['correct']}/{inference['total']}")
    print(f"  Feasibility: {feasibility['correct']}/{feasibility['total']}")

    # Per-form breakdown for novel
    print(f"\n  Novel by form:")
    for form, stats in sorted(novel["by_form"].items()):
        print(f"    {form:>25s}: {stats['correct']}/{stats['n']}")

    # Per-category breakdown for inference
    print(f"\n  Inference by category:")
    for cat, stats in sorted(inference["by_category"].items()):
        print(f"    {cat:>25s}: {stats['correct']}/{stats['n']}")

    results = {
        "adapter_dir": str(adapter_dir),
        "eval": eval_results,
        "timestamp": timestamp,
    }
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Eval results saved to %s", results_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Gradient projection causal experiment")
    parser.add_argument("--arm",
                        choices=["baseline-narrow", "baseline-augmented",
                                 "intervention", "reinjection", "threshold", "all"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to adapter dir — run full eval without training")
    parser.add_argument("--model", type=str, default=None,
                        help="Base model path (default: reads adapter_config.json or uses script default)")
    args = parser.parse_args()

    if args.eval_only:
        run_eval_only(args.eval_only, model_path=args.model, output_dir=args.output)
        return

    if args.arm is None:
        parser.error("--arm is required (unless using --eval-only)")

    if args.arm == "all":
        print("\n" + "="*60)
        print("  RUNNING ALL ARMS")
        print("="*60)

        all_results = {}

        # 1. Baselines
        all_results["baseline-narrow"] = run_arm("baseline-narrow")
        all_results["baseline-augmented"] = run_arm("baseline-augmented")

        # 2. Intervention: project out derived format bias direction
        all_results["intervention"] = run_arm("intervention")

        # 3. Reinjection: add derived format bias to augmented gradients
        all_results["reinjection"] = run_arm("reinjection")

        # 4. Threshold
        all_results["threshold"] = run_threshold_test()

        # Summary table
        print("\n" + "="*60)
        print("  SUMMARY TABLE")
        print("="*60)
        print(f"\n  {'Arm':>30s} {'Data':>8s} {'Hook':>20s} {'MT pre':>7s} {'MT post':>8s} {'best_val':>9s}")
        print(f"  {'-'*85}")
        for key, r in all_results.items():
            if isinstance(r, dict) and "mt_score_post" in r:
                print(f"  {key:>30s} {r.get('train_samples', ''):>8} "
                      f"{r.get('hook_label', ''):>20s} "
                      f"{r['mt_score_pre']:>7d} {r['mt_score_post']:>8d} "
                      f"{r['best_val_loss']:>9.4f}")

    elif args.arm == "threshold":
        run_threshold_test(output_dir=args.output)
    else:
        run_arm(args.arm, output_dir=args.output)


if __name__ == "__main__":
    main()
