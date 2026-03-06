#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Verify REINFORCE orthogonality through the actual NB-LoRA outcome path:
# greedy StarProblem rollouts, deterministic verifiers, response-only masking,
# and per-completion log-probability gradients.
"""Derive REINFORCE-vs-CE geometry from the real outcome-loss operator."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("reinforce_orth")

DEFAULT_MODEL = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
DEFAULT_OUTPUT = "results/reinforce_orthogonality_derivation"
DEFAULT_PROBLEM_SEED = 3475334679
DEFAULT_N_PROBLEMS = 10
DEFAULT_MAX_GENERATION_TOKENS = 320


def _flatten_grad_tree(grads) -> "mx.array":
    """Flatten a trainable-parameter gradient tree into one float32 vector."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    arrays = []
    for _, value in tree_flatten(grads):
        if value is not None and hasattr(value, "reshape") and value.size > 0:
            arrays.append(value.reshape(-1).astype(mx.float32))
    if not arrays:
        return mx.zeros((1,), dtype=mx.float32)
    return mx.concatenate(arrays, axis=0)


def _load_model_and_inject_adapters(model_path: str):
    import mlx.nn as nn

    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.domain.training.geometric_lora import select_target_modules

    backend = MLXBackend()
    log.info("Loading model: %s", model_path)
    model, tokenizer = backend.load_model(model_path)
    adapter = MLXTrainingAdapter(backend)

    log.info("Analyzing weight geometry...")
    geometries = adapter.analyze_model_geometry_streaming(
        model,
        use_randomized=True,
    )
    target_modules = select_target_modules(geometries)
    log.info("Targetable modules: %d", len(target_modules))

    n_injected = adapter.inject_nb_lora(
        model,
        geometries,
        target_modules,
        safety_margin=None,
    )
    adapter.freeze_and_apply_lora(model)

    n_trainable = sum(
        p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters())
    )
    log.info(
        "Injected %d NB-LoRA layers (%d trainable adapter parameters)",
        n_injected,
        n_trainable,
    )
    return backend, adapter, model, tokenizer, geometries, target_modules, n_injected, n_trainable


def _collect_outcome_completions(
    backend,
    model,
    tokenizer,
    *,
    n_problems: int,
    problem_seed: int,
    max_generation_tokens: int,
):
    from modelcypher.core.domain.star.prompting import default_few_shot_examples
    from modelcypher.core.domain.training.online_eval import create_eval_problem_set
    from modelcypher.core.domain.training.outcome_objective import collect_outcomes

    problems = create_eval_problem_set(n_problems=n_problems, seed=problem_seed)
    n_variants = len(default_few_shot_examples())
    log.info(
        "Created %d StarProblems (seed=%d) with %d prompt variants each",
        len(problems),
        problem_seed,
        n_variants,
    )

    def _generate(prompt: str, max_tokens: int) -> str:
        return backend.generate(model, tokenizer, prompt, max_tokens=max_tokens)

    def _tokenize(text: str) -> list[int]:
        return tokenizer.encode(text)

    result = collect_outcomes(
        problems=problems,
        generate_fn=_generate,
        tokenize_fn=_tokenize,
        n_variants=n_variants,
        max_tokens=max_generation_tokens,
    )
    return problems, result


def _single_completion_gradient(
    model,
    completion_tokens: list[int],
    response_start: int,
    seq_length: int,
):
    import mlx.core as mx
    import mlx.nn as nn

    from modelcypher.backends.mlx_training_adapter_core import (
        make_outcome_loss,
        prepare_outcome_batches,
    )

    # With advantage = -1 and batch size 1:
    #   loss = -mean((-1) * seq_log_prob) = seq_log_prob
    # so the returned gradient is ∇ log π(response | prompt).
    loss_fn = make_outcome_loss()
    logprob_vg = nn.value_and_grad(model, loss_fn)
    (batch, lengths, advantages, response_starts) = prepare_outcome_batches(
        [(completion_tokens, -1.0, response_start)],
        batch_size=1,
        seq_length=seq_length,
    )[0]
    (seq_log_prob, ntoks), grads = logprob_vg(
        model,
        batch,
        lengths,
        advantages,
        response_starts,
    )
    mx.eval(seq_log_prob, ntoks, grads)
    grad_flat = _flatten_grad_tree(grads)
    mx.eval(grad_flat)
    return float(seq_log_prob.item()), int(ntoks.item()), grad_flat


def _direct_weighted_outcome_gradient(model, completions, seq_length: int):
    import mlx.core as mx
    import mlx.nn as nn

    from modelcypher.backends.mlx_training_adapter_core import (
        make_outcome_loss,
        prepare_outcome_batches,
    )

    loss_fn = make_outcome_loss()
    outcome_vg = nn.value_and_grad(model, loss_fn)
    (batch, lengths, advantages, response_starts) = prepare_outcome_batches(
        [
            (comp["tokens"], comp["advantage"], comp["response_start"])
            for comp in completions
        ],
        batch_size=len(completions),
        seq_length=seq_length,
    )[0]
    (loss_val, ntoks), grads = outcome_vg(
        model,
        batch,
        lengths,
        advantages,
        response_starts,
    )
    mx.eval(loss_val, ntoks, grads)
    grad_flat = _flatten_grad_tree(grads)
    mx.eval(grad_flat)
    return float(loss_val.item()), int(ntoks.item()), grad_flat


def _cosine(a, b) -> float:
    import mlx.core as mx

    dot = mx.sum(a * b)
    a_norm = mx.sqrt(mx.sum(a * a))
    b_norm = mx.sqrt(mx.sum(b * b))
    mx.eval(dot, a_norm, b_norm)
    a_val = float(a_norm.item())
    b_val = float(b_norm.item())
    if a_val <= 0.0 or b_val <= 0.0:
        return 0.0
    return float(dot.item()) / (a_val * b_val)


def _norm(arr) -> float:
    import mlx.core as mx

    value = mx.sqrt(mx.sum(arr * arr))
    mx.eval(value)
    return float(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify REINFORCE orthogonality through the real NB-LoRA outcome path.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-problems", type=int, default=DEFAULT_N_PROBLEMS)
    parser.add_argument("--problem-seed", type=int, default=DEFAULT_PROBLEM_SEED)
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=DEFAULT_MAX_GENERATION_TOKENS,
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    import mlx.core as mx
    from scipy.stats import ks_2samp, spearmanr

    (
        backend,
        _adapter,
        model,
        tokenizer,
        geometries,
        target_modules,
        n_injected,
        n_trainable,
    ) = _load_model_and_inject_adapters(str(model_path))

    problems, outcome_result = _collect_outcome_completions(
        backend,
        model,
        tokenizer,
        n_problems=args.n_problems,
        problem_seed=args.problem_seed,
        max_generation_tokens=args.max_generation_tokens,
    )

    completions: list[dict[str, Any]] = []
    for completion in outcome_result.completions:
        completions.append({
            "problem_id": completion.problem_id,
            "prompt_variant": completion.prompt_variant,
            "tokens": completion.tokens,
            "correct": completion.correct,
            "reward": 1.0 if completion.correct else 0.0,
            "advantage": completion.advantage,
            "response_start": completion.response_start,
        })

    n_completions = len(completions)
    n_correct = int(sum(comp["reward"] for comp in completions))
    n_incorrect = n_completions - n_correct
    n_active = sum(1 for comp in completions if comp["advantage"] != 0.0)
    accuracy = n_correct / n_completions if n_completions > 0 else 0.0
    if n_completions == 0 or n_correct == 0 or n_incorrect == 0:
        log.error(
            "Need mixed correct/incorrect completions. Got correct=%d incorrect=%d",
            n_correct,
            n_incorrect,
        )
        sys.exit(1)

    log.info(
        "Outcome collection: %d completions, %d active, accuracy=%.1f%%, signal_density=%.1f%%",
        n_completions,
        n_active,
        accuracy * 100.0,
        outcome_result.signal_density * 100.0,
    )

    analysis_seq_length = max(len(comp["tokens"]) for comp in completions)
    log.info(
        "Analysis sequence length derived from observed rollouts: %d tokens",
        analysis_seq_length,
    )

    log.info("Computing per-completion log-prob gradients...")
    grad_vectors = []
    seq_log_probs = []
    response_token_counts = []
    grad_norms = []

    for idx, comp in enumerate(completions, start=1):
        seq_log_prob, ntoks, grad_flat = _single_completion_gradient(
            model,
            comp["tokens"],
            comp["response_start"],
            analysis_seq_length,
        )
        grad_norm = _norm(grad_flat)
        grad_vectors.append(grad_flat)
        seq_log_probs.append(seq_log_prob)
        response_token_counts.append(ntoks)
        grad_norms.append(grad_norm)

        comp["seq_log_prob"] = seq_log_prob
        comp["response_tokens"] = ntoks
        comp["grad_norm"] = grad_norm
        comp["token_count"] = len(comp["tokens"])

        if idx <= 3 or idx % 5 == 0:
            log.info(
                "  gradient %d/%d  problem=%s v%d correct=%s adv=%+.3f logp=%.4f grad_norm=%.4e",
                idx,
                n_completions,
                comp["problem_id"],
                comp["prompt_variant"],
                comp["correct"],
                comp["advantage"],
                seq_log_prob,
                grad_norm,
            )

    G = mx.stack(grad_vectors, axis=0)
    mx.eval(G)
    k_samples, n_params = G.shape
    log.info("Gradient matrix: K=%d completions × D=%d adapter parameters", k_samples, n_params)

    g_mean = G.mean(axis=0)
    advantages = [float(comp["advantage"]) for comp in completions]
    rewards = [float(comp["reward"]) for comp in completions]
    advantages_arr = mx.array(advantages, dtype=mx.float32)[:, None]
    g_reinforce = (advantages_arr * G).mean(axis=0)
    mx.eval(g_mean, g_reinforce)

    problem_groups: dict[str, list[int]] = {}
    for idx, comp in enumerate(completions):
        problem_groups.setdefault(comp["problem_id"], []).append(idx)

    g_reinforce_residual_sum = mx.zeros_like(g_reinforce)
    group_mean_norms: list[float] = []
    residual_sample_norms: list[float] = []
    common_energy = 0.0
    residual_energy = 0.0
    for indices in problem_groups.values():
        group_idx = mx.array(indices)
        group_grads = mx.take(G, group_idx, axis=0)
        group_mean = group_grads.mean(axis=0)
        group_residuals = group_grads - group_mean[None, :]
        group_advantages = mx.array(
            [advantages[i] for i in indices],
            dtype=mx.float32,
        )[:, None]
        group_residual_contrib = mx.sum(group_advantages * group_residuals, axis=0)
        mean_norm_sq = mx.sum(group_mean * group_mean)
        residual_norm_sq = mx.sum(group_residuals * group_residuals)
        residual_norms_sq = mx.sum(group_residuals * group_residuals, axis=1)
        mx.eval(
            group_residual_contrib,
            mean_norm_sq,
            residual_norm_sq,
            residual_norms_sq,
        )

        g_reinforce_residual_sum = g_reinforce_residual_sum + group_residual_contrib
        group_mean_norms.append(math.sqrt(float(mean_norm_sq.item())))
        residual_sample_norms.extend(
            math.sqrt(float(value)) for value in residual_norms_sq.tolist()
        )
        common_energy += len(indices) * float(mean_norm_sq.item())
        residual_energy += float(residual_norm_sq.item())

    g_reinforce_from_residuals = g_reinforce_residual_sum / float(k_samples)
    mx.eval(g_reinforce_from_residuals)
    residualization_cosine = _cosine(g_reinforce, g_reinforce_from_residuals)
    total_group_decomp_energy = common_energy + residual_energy
    common_energy_fraction = (
        common_energy / total_group_decomp_energy
        if total_group_decomp_energy > 0.0
        else float("nan")
    )
    mean_group_mean_norm = (
        sum(group_mean_norms) / len(group_mean_norms)
        if group_mean_norms
        else float("nan")
    )
    mean_residual_sample_norm = (
        sum(residual_sample_norms) / len(residual_sample_norms)
        if residual_sample_norms
        else float("nan")
    )

    g_mean_norm = _norm(g_mean)
    g_reinforce_norm = _norm(g_reinforce)
    dot_val = float(mx.sum(g_mean * g_reinforce).item()) if (g_mean_norm > 0 and g_reinforce_norm > 0) else 0.0
    cos_sim = (
        dot_val / (g_mean_norm * g_reinforce_norm)
        if g_mean_norm > 0.0 and g_reinforce_norm > 0.0
        else 0.0
    )
    if g_mean_norm > 0.0 and g_reinforce_norm > 0.0:
        parallel_norm = abs(dot_val) / g_mean_norm
        orth_sq = max(0.0, g_reinforce_norm * g_reinforce_norm - parallel_norm * parallel_norm)
        orth_fraction = math.sqrt(orth_sq) / g_reinforce_norm
    else:
        orth_fraction = 1.0

    alignments_arr = (
        mx.sum(G * g_mean[None, :], axis=1) / g_mean_norm
        if g_mean_norm > 0.0
        else mx.zeros((k_samples,), dtype=mx.float32)
    )
    mx.eval(alignments_arr)
    alignments = [float(value) for value in alignments_arr.tolist()]

    for comp, alignment in zip(completions, alignments):
        comp["alignment_to_mean"] = alignment

    spearman_r, spearman_p = spearmanr(advantages, alignments)
    active_completions = [comp for comp in completions if comp["advantage"] != 0.0]
    active_advantages = [float(comp["advantage"]) for comp in active_completions]
    active_alignments = [float(comp["alignment_to_mean"]) for comp in active_completions]
    if len(active_completions) >= 2:
        active_spearman_r, active_spearman_p = spearmanr(
            active_advantages,
            active_alignments,
        )
    else:
        active_spearman_r = float("nan")
        active_spearman_p = float("nan")

    correct_alignments = [
        alignments[i] for i, reward in enumerate(rewards) if reward > 0.5
    ]
    incorrect_alignments = [
        alignments[i] for i, reward in enumerate(rewards) if reward <= 0.5
    ]
    ks_stat, ks_p = ks_2samp(correct_alignments, incorrect_alignments)
    pos_adv_alignments = [value for value, adv in zip(alignments, advantages) if adv > 0.0]
    neg_adv_alignments = [value for value, adv in zip(alignments, advantages) if adv < 0.0]
    if pos_adv_alignments and neg_adv_alignments:
        active_ks_stat, active_ks_p = ks_2samp(pos_adv_alignments, neg_adv_alignments)
    else:
        active_ks_stat = float("nan")
        active_ks_p = float("nan")

    direct_loss, direct_ntoks, direct_outcome_grad = _direct_weighted_outcome_gradient(
        model,
        completions,
        analysis_seq_length,
    )
    direct_outcome_norm = _norm(direct_outcome_grad)
    decomposition_vs_direct_cosine = _cosine(g_reinforce, -direct_outcome_grad)
    direct_mean_check = float("nan")
    if direct_outcome_norm > 0.0:
        direct_mean_check = decomposition_vs_direct_cosine

    sigma_ratios = {}
    for key in sorted(target_modules):
        geom = geometries[key]
        if geom.sigma_max > 0.0 and geom.sigma_k > 0.0:
            sigma_ratios[key] = float(geom.sigma_k / geom.sigma_max)

    problem_type_counts = Counter(problem.problem_type for problem in problems)
    elapsed = time.monotonic() - t_start

    results = {
        "model": str(model_path),
        "problem_seed": args.problem_seed,
        "max_generation_tokens": args.max_generation_tokens,
        "analysis_seq_length": analysis_seq_length,
        "n_problems": len(problems),
        "n_variants": int(n_completions / len(problems)) if problems else 0,
        "n_completions": n_completions,
        "n_active_completions": n_active,
        "signal_density": outcome_result.signal_density,
        "completion_accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_injected_layers": n_injected,
        "n_adapter_params": int(n_trainable),
        "g_mean_norm": g_mean_norm,
        "g_reinforce_norm": g_reinforce_norm,
        "cos_g_mean_g_reinforce": cos_sim,
        "orth_fraction": orth_fraction,
        "group_mean_component_energy_fraction": common_energy_fraction,
        "mean_group_mean_norm": mean_group_mean_norm,
        "mean_group_residual_sample_norm": mean_residual_sample_norm,
        "residualization_cosine": residualization_cosine,
        "P1_spearman_r": float(spearman_r),
        "P1_spearman_p": float(spearman_p),
        "P1_pass": bool(spearman_p > 0.05),
        "P1_active_spearman_r": float(active_spearman_r),
        "P1_active_spearman_p": float(active_spearman_p),
        "P1_active_pass": bool(active_spearman_p > 0.05) if not math.isnan(active_spearman_p) else False,
        "P3_ks_statistic": float(ks_stat),
        "P3_ks_p": float(ks_p),
        "P3_pass": bool(ks_p > 0.05),
        "P3_active_adv_sign_ks_statistic": float(active_ks_stat),
        "P3_active_adv_sign_ks_p": float(active_ks_p),
        "P3_active_adv_sign_pass": bool(active_ks_p > 0.05) if not math.isnan(active_ks_p) else False,
        "direct_outcome_loss": direct_loss,
        "direct_outcome_response_tokens": direct_ntoks,
        "direct_outcome_grad_norm": direct_outcome_norm,
        "decomposition_vs_direct_outcome_cosine": direct_mean_check,
        "sigma_k_over_sigma_1": sigma_ratios,
        "problem_type_counts": dict(problem_type_counts),
        "problems": [problem.to_problem_record() for problem in problems],
        "per_completion": [
            {
                "problem_id": comp["problem_id"],
                "prompt_variant": comp["prompt_variant"],
                "correct": comp["correct"],
                "reward": comp["reward"],
                "advantage": comp["advantage"],
                "response_start": comp["response_start"],
                "token_count": comp["token_count"],
                "response_tokens": comp["response_tokens"],
                "seq_log_prob": comp["seq_log_prob"],
                "grad_norm": comp["grad_norm"],
                "alignment_to_mean": comp["alignment_to_mean"],
            }
            for comp in completions
        ],
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved results to %s", out_path)

    print("\n" + "=" * 64)
    print("REINFORCE Orthogonality Derivation")
    print("=" * 64)
    print(f"Model:                {model_path.name}")
    print(f"Problems:             {len(problems)}  (seed={args.problem_seed})")
    print(f"Max gen tokens:       {args.max_generation_tokens}")
    print(f"Analysis seq length:  {analysis_seq_length}")
    print(f"Completions:          {n_completions}  ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"Active completions:   {n_active}  signal_density={outcome_result.signal_density:.1%}")
    print(f"Adapter params:       {n_trainable:,}")
    print(f"Completion accuracy:  {accuracy:.1%}")
    print()
    print(f"cos(g_mean, g_RE):    {cos_sim:.6f}")
    print(f"orth_fraction:        {orth_fraction:.4f}")
    print(f"||g_mean||:           {g_mean_norm:.6e}")
    print(f"||g_RE||:             {g_reinforce_norm:.6e}")
    print(f"group-mean energy:    {common_energy_fraction:.4f}")
    print(f"residualization cos:  {residualization_cosine:.6f}")
    print()
    p1_verdict = "PASS" if spearman_p > 0.05 else "FAIL"
    print(
        f"P1 Spearman(A, alignment) = {spearman_r:+.4f}  p={spearman_p:.4f}  [{p1_verdict}]",
    )
    if not math.isnan(active_spearman_p):
        p1_active_verdict = "PASS" if active_spearman_p > 0.05 else "FAIL"
        print(
            "P1 active-only               "
            f"= {active_spearman_r:+.4f}  p={active_spearman_p:.4f}  [{p1_active_verdict}]",
        )
    p3_verdict = "PASS" if ks_p > 0.05 else "FAIL"
    print(
        f"P3 KS(correct, incorrect) = {ks_stat:.4f}  p={ks_p:.4f}  [{p3_verdict}]",
    )
    if not math.isnan(active_ks_p):
        p3_active_verdict = "PASS" if active_ks_p > 0.05 else "FAIL"
        print(
            "P3 active KS(pos_adv, neg_adv) = "
            f"{active_ks_stat:.4f}  p={active_ks_p:.4f}  [{p3_active_verdict}]",
        )
    print(
        "decomp check cosine:  "
        f"{direct_mean_check:.6f}  (mean(A_k g_k) vs direct outcome gradient)",
    )
    if sigma_ratios:
        values = list(sigma_ratios.values())
        print(
            f"σ_k/σ_1 range:        [{min(values):.4e}, {max(values):.4e}]  "
            f"mean={sum(values) / len(values):.4e}",
        )
    print(f"Elapsed:              {elapsed:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
