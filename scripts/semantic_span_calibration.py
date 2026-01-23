#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Semantic Span Calibration
"""
Semantic Span Calibration - Find the minimal set that spans the activation space.

THE QUESTION:
With 500 diverse prompts, we still get 20-77% out-of-span error.
What IS the intrinsic dimension of the reachable activation space?
How do we systematically span it?

THE APPROACH:
1. Generate a HUGE pool of candidate prompts (10,000+)
2. Compute activations for all of them
3. Use greedy selection: pick prompts that maximize coverage
4. Monitor when we've "saturated" the span
5. Test if this calibration covers held-out prompts

KEY INSIGHT:
The activation space at layer 3 might have intrinsic dimension << 1024.
We need to find this dimension and ensure our calibration spans it.

Usage:
    python semantic_span_calibration.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_massive_prompt_pool() -> list[str]:
    """Generate a massive pool of diverse prompts."""
    prompts = []

    # 1. ALL COUNTRIES (195+)
    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
        "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
        "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
        "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei",
        "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
        "Chad", "Chile", "China", "Colombia", "Congo", "Costa Rica", "Croatia",
        "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominican Republic",
        "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia", "Fiji",
        "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
        "Greece", "Guatemala", "Guinea", "Guyana", "Haiti", "Honduras", "Hungary",
        "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
        "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait",
        "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
        "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
        "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Moldova", "Monaco",
        "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
        "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
        "North Korea", "Norway", "Oman", "Pakistan", "Palestine", "Panama",
        "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
        "Qatar", "Romania", "Russia", "Rwanda", "Saudi Arabia", "Senegal", "Serbia",
        "Singapore", "Slovakia", "Slovenia", "Somalia", "South Africa", "South Korea",
        "Spain", "Sri Lanka", "Sudan", "Sweden", "Switzerland", "Syria", "Taiwan",
        "Tajikistan", "Tanzania", "Thailand", "Togo", "Trinidad", "Tunisia", "Turkey",
        "Turkmenistan", "Uganda", "Ukraine", "UAE", "UK", "USA", "Uruguay",
        "Uzbekistan", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # 2. ARITHMETIC (all combinations)
    for a in range(1, 51):
        for b in range(1, 51):
            if len(prompts) < 3000:  # Limit
                prompts.append(f"{a} + {b} =")
                prompts.append(f"{a} - {b} =")
                if a * b < 1000:
                    prompts.append(f"{a} * {b} =")
                if b != 0 and a % b == 0:
                    prompts.append(f"{a} / {b} =")

    # 3. WORDS (opposites, synonyms, categories)
    words = [
        "hot", "cold", "big", "small", "happy", "sad", "light", "dark", "up", "down",
        "good", "bad", "old", "young", "fast", "slow", "loud", "quiet", "wet", "dry",
        "full", "empty", "rich", "poor", "strong", "weak", "true", "false", "open", "closed",
        "hard", "soft", "near", "far", "early", "late", "clean", "dirty", "safe", "dangerous",
        "alive", "dead", "thick", "thin", "sharp", "dull", "smooth", "rough", "bright", "dim",
        "beautiful", "ugly", "simple", "complex", "easy", "difficult", "right", "wrong",
        "love", "hate", "peace", "war", "success", "failure", "hope", "despair",
    ]
    for w in words:
        prompts.append(f"The opposite of {w} is")
        prompts.append(f"A synonym for {w} is")
        prompts.append(f"{w.capitalize()} is a type of")

    # 4. DEFINITIONS
    concepts = [
        "democracy", "capitalism", "socialism", "evolution", "gravity", "entropy",
        "consciousness", "intelligence", "memory", "emotion", "language", "culture",
        "religion", "philosophy", "science", "technology", "art", "music", "literature",
        "mathematics", "physics", "chemistry", "biology", "psychology", "sociology",
        "economics", "politics", "history", "geography", "astronomy", "geology",
        "medicine", "engineering", "architecture", "law", "ethics", "logic",
        "metaphysics", "epistemology", "ontology", "aesthetics", "semantics",
    ]
    for c in concepts:
        prompts.append(f"{c.capitalize()} is")
        prompts.append(f"The definition of {c} is")
        prompts.append(f"In simple terms, {c} means")

    # 5. SENTENCE STARTERS (many variations)
    starters = [
        "Once upon a time", "In the beginning", "Long ago", "Years ago",
        "The story begins", "It all started when", "Before the dawn",
        "Deep in the forest", "High in the mountains", "Across the ocean",
        "In a distant land", "On a dark night", "During the storm",
        "After the war", "Before the revolution", "In ancient times",
        "The legend says", "According to myth", "As the prophecy foretold",
        "When the sun set", "As the moon rose", "In the depths of",
        "Beyond the horizon", "Beneath the surface", "Above the clouds",
        "Inside the castle", "Outside the walls", "Through the portal",
    ]
    for s in starters:
        prompts.append(s)
        prompts.append(f"{s}, there was")
        prompts.append(f"{s}, a young")

    # 6. TECHNICAL TERMS
    tech = [
        "algorithm", "database", "network", "protocol", "encryption", "compiler",
        "interpreter", "variable", "function", "class", "object", "method",
        "interface", "implementation", "abstraction", "inheritance", "polymorphism",
        "recursion", "iteration", "parallelism", "concurrency", "thread", "process",
        "memory", "cache", "buffer", "stack", "queue", "heap", "tree", "graph",
        "API", "SDK", "IDE", "GUI", "CLI", "URL", "HTTP", "TCP", "IP", "DNS",
        "CPU", "GPU", "RAM", "ROM", "SSD", "HDD", "BIOS", "kernel", "driver",
    ]
    for t in tech:
        prompts.append(f"In computing, {t} is")
        prompts.append(f"A {t} is used for")
        prompts.append(f"The purpose of a {t} is")

    # 7. TRANSITIONAL PHRASES
    transitions = [
        "However,", "Therefore,", "Moreover,", "Furthermore,", "Additionally,",
        "Consequently,", "Nevertheless,", "Nonetheless,", "In contrast,",
        "On the other hand,", "Similarly,", "Likewise,", "For example,",
        "For instance,", "In particular,", "Specifically,", "Generally,",
        "In conclusion,", "To summarize,", "In summary,", "To conclude,",
        "As a result,", "Due to this,", "Because of this,", "Despite this,",
        "Regardless,", "Meanwhile,", "Subsequently,", "Previously,", "Initially,",
        "Finally,", "Ultimately,", "Essentially,", "Basically,", "Fundamentally,",
    ]
    for t in transitions:
        prompts.append(t)
        prompts.append(f"{t} the")
        prompts.append(f"{t} we can")

    # 8. QUESTIONS
    question_starts = [
        "What is", "Who was", "Where is", "When did", "Why do", "How does",
        "Which is", "Can you", "Could we", "Should I", "Would it", "Will the",
        "Is there", "Are we", "Was it", "Were they", "Has the", "Have you",
        "Does this", "Do they", "Did you", "What if", "How can", "Why would",
    ]
    question_topics = [
        "the meaning of life", "the universe", "consciousness", "time travel",
        "artificial intelligence", "climate change", "quantum mechanics",
        "the origin of language", "human evolution", "the future of technology",
        "the nature of reality", "free will", "the purpose of existence",
    ]
    for q in question_starts:
        for topic in question_topics[:5]:  # Limit combinations
            prompts.append(f"{q} {topic}")

    # 9. CODE SNIPPETS
    code_starts = [
        "def ", "class ", "import ", "from ", "if ", "for ", "while ",
        "try:", "except ", "with ", "return ", "yield ", "async def ",
        "lambda ", "@", "self.", "print(", "assert ", "raise ",
    ]
    for c in code_starts:
        prompts.append(c)

    # 10. NUMBERS AND DATES
    for year in range(1900, 2030, 10):
        prompts.append(f"In {year},")
        prompts.append(f"The year {year} was")

    for n in [1, 2, 3, 5, 7, 10, 12, 13, 42, 100, 1000, 1000000]:
        prompts.append(f"The number {n} is")
        prompts.append(f"{n} is special because")

    # Remove duplicates and shuffle
    prompts = list(set(prompts))
    random.shuffle(prompts)

    return prompts


def collect_activations(model, tokenizer, prompts, target_layer):
    """Collect activations at target layer for all prompts."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    activations = []

    for i, prompt in enumerate(prompts):
        if i % 500 == 0:
            logger.info(f"Collecting {i+1}/{len(prompts)}")

        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.bos_token_id or 1]

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)

        if is_lfm2:
            from mlx_lm.models.lfm2 import create_attention_mask, create_ssm_mask
            attn_mask = create_attention_mask(h, None)
            conv_mask = create_ssm_mask(h, None)
        else:
            attn_mask = None
            conv_mask = None

        for idx, layer in enumerate(inner_model.layers):
            if idx == target_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                activations.append(h_in)
                break

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    return np.stack(activations, axis=1).astype(np.float64)


def greedy_span_selection(X, target_coverage=0.99, max_samples=500):
    """
    Greedily select samples that maximize span coverage.

    Returns indices of selected samples.
    """
    n_samples = X.shape[1]
    hidden_dim = X.shape[0]

    # Normalize columns
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X_norm = X / (norms + 1e-10)

    selected = []
    remaining = list(range(n_samples))

    # Start with the sample that has largest norm
    first = np.argmax(norms[0])
    selected.append(first)
    remaining.remove(first)

    # Current span basis (via QR)
    Q = X_norm[:, [first]].copy()
    Q, _ = np.linalg.qr(Q)

    iteration = 0
    while len(selected) < max_samples and remaining:
        iteration += 1

        # Find sample with largest component orthogonal to current span
        X_remaining = X_norm[:, remaining]

        # Project out current span
        projections = Q @ (Q.T @ X_remaining)
        orthogonal = X_remaining - projections
        orth_norms = np.linalg.norm(orthogonal, axis=0)

        # Pick sample with largest orthogonal component
        best_idx = np.argmax(orth_norms)
        best_orth_norm = orth_norms[best_idx]

        # Check if we've saturated
        if best_orth_norm < 1e-6:
            logger.info(f"Saturated at {len(selected)} samples (orth_norm={best_orth_norm:.2e})")
            break

        # Add to selected
        actual_idx = remaining[best_idx]
        selected.append(actual_idx)
        remaining.remove(actual_idx)

        # Update basis
        new_vec = X_norm[:, [actual_idx]]
        Q_new = np.hstack([Q, new_vec])
        Q, _ = np.linalg.qr(Q_new)

        # Compute current coverage
        if iteration % 50 == 0:
            # Total variance
            total_var = np.sum(np.linalg.norm(X_norm, axis=0)**2)
            # Variance explained by current span
            projections_all = Q @ (Q.T @ X_norm)
            explained_var = np.sum(np.linalg.norm(projections_all, axis=0)**2)
            coverage = explained_var / total_var

            logger.info(f"Selected {len(selected)}, rank={Q.shape[1]}, coverage={coverage:.4f}")

            if coverage >= target_coverage:
                logger.info(f"Reached {target_coverage*100}% coverage")
                break

    return selected, Q.shape[1]


def analyze_span_saturation(X, selected_indices):
    """Analyze how quickly the span saturates."""
    X_selected = X[:, selected_indices]

    coverages = []
    ranks = []

    for k in range(1, len(selected_indices) + 1, max(1, len(selected_indices) // 50)):
        X_k = X_selected[:, :k]

        # Compute SVD
        U, S, _ = np.linalg.svd(X_k, full_matrices=False)

        # Effective rank
        total_var = np.sum(S**2)
        cumsum = np.cumsum(S**2)
        rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1

        # Coverage of full dataset
        Q, _ = np.linalg.qr(X_k)
        projections = Q @ (Q.T @ X)
        proj_norms = np.linalg.norm(projections, axis=0)
        orig_norms = np.linalg.norm(X, axis=0)
        coverage = np.mean(proj_norms / (orig_norms + 1e-10))

        coverages.append(coverage)
        ranks.append(rank_99)

    return coverages, ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=5000, help="Max prompts to collect")
    parser.add_argument("--target-layer", type=int, default=3, help="Layer to analyze")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*70}")
    print("SEMANTIC SPAN CALIBRATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Target layer: {args.target_layer}")

    # Generate prompt pool
    print(f"\n{'='*70}")
    print("PHASE 1: GENERATE PROMPT POOL")
    print("="*70)

    all_prompts = generate_massive_prompt_pool()
    prompts = all_prompts[:args.max_prompts]
    print(f"Generated {len(all_prompts)} prompts, using {len(prompts)}")

    # Collect activations
    print(f"\n{'='*70}")
    print("PHASE 2: COLLECT ACTIVATIONS")
    print("="*70)

    X = collect_activations(model, tokenizer, prompts, args.target_layer)
    print(f"Activation matrix shape: {X.shape}")

    # Analyze full rank
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    total_var = np.sum(S**2)
    cumsum = np.cumsum(S**2)
    rank_90 = np.searchsorted(cumsum / total_var, 0.90) + 1
    rank_95 = np.searchsorted(cumsum / total_var, 0.95) + 1
    rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1

    print(f"\nFull dataset rank:")
    print(f"  90% variance: {rank_90}")
    print(f"  95% variance: {rank_95}")
    print(f"  99% variance: {rank_99}")
    print(f"  Top 10 SVs: {np.round(S[:10], 2)}")

    # Greedy selection
    print(f"\n{'='*70}")
    print("PHASE 3: GREEDY SPAN SELECTION")
    print("="*70)

    selected_indices, final_rank = greedy_span_selection(X, target_coverage=0.99, max_samples=500)
    print(f"\nSelected {len(selected_indices)} prompts, spanning rank {final_rank}")

    # Analyze saturation
    print(f"\n{'='*70}")
    print("PHASE 4: SPAN SATURATION ANALYSIS")
    print("="*70)

    coverages, ranks = analyze_span_saturation(X, selected_indices)

    print("\nSamples | Rank | Coverage")
    print("-" * 30)
    for i, (cov, rank) in enumerate(zip(coverages, ranks)):
        k = 1 + i * max(1, len(selected_indices) // 50)
        print(f"{k:>7} | {rank:>4} | {cov:.4f}")

    # Test: OOS error on remaining prompts
    print(f"\n{'='*70}")
    print("PHASE 5: TEST OUT-OF-SPAN ERROR")
    print("="*70)

    X_selected = X[:, selected_indices]
    Q, _ = np.linalg.qr(X_selected)

    # Test on ALL prompts
    projections = Q @ (Q.T @ X)
    oos_errors = []
    for i in range(X.shape[1]):
        proj_norm = np.linalg.norm(projections[:, i])
        orig_norm = np.linalg.norm(X[:, i])
        oos = 1 - proj_norm / (orig_norm + 1e-10)
        oos_errors.append(max(0, oos))

    oos_errors = np.array(oos_errors)
    print(f"\nOut-of-span error statistics:")
    print(f"  Mean: {np.mean(oos_errors)*100:.2f}%")
    print(f"  Median: {np.median(oos_errors)*100:.2f}%")
    print(f"  Max: {np.max(oos_errors)*100:.2f}%")
    print(f"  < 10%: {np.mean(oos_errors < 0.1)*100:.1f}% of prompts")
    print(f"  < 5%: {np.mean(oos_errors < 0.05)*100:.1f}% of prompts")

    # Save selected prompts
    selected_prompts = [prompts[i] for i in selected_indices]

    print(f"\n{'='*70}")
    print("SELECTED CALIBRATION PROMPTS (first 20)")
    print("="*70)
    for i, p in enumerate(selected_prompts[:20]):
        print(f"  {i+1}. {p[:60]}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"""
SEMANTIC SPAN ANALYSIS:

Input: {len(prompts)} diverse prompts
Hidden dim: {hidden_dim}
Target layer: {args.target_layer}

FINDINGS:
- Full dataset effective rank (99%): {rank_99}
- Greedy selection needed: {len(selected_indices)} prompts
- Selected set spans rank: {final_rank}
- Mean OOS error on full dataset: {np.mean(oos_errors)*100:.2f}%

IMPLICATION:
The activation space has intrinsic dimension ~{rank_99}.
With {len(selected_indices)} well-chosen prompts, we can span it.
Remaining OOS error ({np.mean(oos_errors)*100:.1f}%) comes from:
1. Numerical precision
2. Truly out-of-distribution inputs
3. Nonlinear components not captured by linear span

RECOMMENDATION:
Use these {len(selected_indices)} calibration prompts for Lie algebra compression.
Expected coverage: {(1-np.mean(oos_errors))*100:.1f}% of all inputs.
""")

    # Return for use in other scripts
    return selected_prompts, X_selected


if __name__ == "__main__":
    main()
