#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Piecewise Linear Lie Algebra Compression
"""
FINDING: A single linear T doesn't generalize - it memorizes.

INSIGHT: The model's transformation is nonlinear. A single linear map
can't capture it. But a PIECEWISE linear map can approximate any
continuous function.

APPROACH:
1. Cluster calibration inputs into K groups (by semantic similarity)
2. Compute a separate T_k for each cluster
3. At inference, find nearest cluster and use that T_k

This is essentially a soft Mixture of Linear Experts.

Usage:
    python lie_algebra_piecewise.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Diverse calibration with semantic categories
CALIBRATION_PROMPTS = {
    "capitals": [f"The capital of {c} is" for c in [
        "France", "Japan", "Germany", "Italy", "Spain", "UK", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt",
    ]],
    "math": [f"{a} + {b} =" for a in range(1, 8) for b in range(1, 8)],
    "opposites": [f"The opposite of {w} is" for w in [
        "hot", "big", "happy", "light", "up", "good", "old", "fast", "slow", "wet",
    ]],
    "physical": [
        "Water freezes at", "Ice melts at", "Steam forms at", "Iron melts at",
        "Gold melts at", "Mercury boils at", "Nitrogen boils at",
    ],
    "astronomical": [
        "The moon orbits", "The Earth orbits", "Mars orbits", "Jupiter orbits",
        "Stars are made of", "The sun is made of", "Planets are made of",
    ],
    "conversational": [
        "Well, actually", "Actually,", "In fact,", "To be honest,",
        "If you think about it,", "When you consider,", "Looking at it,",
    ],
    "answers": [
        "The answer is", "The solution is", "The result is", "The conclusion is",
        "In summary,", "Therefore,", "Thus,",
    ],
}

HELD_OUT_PROMPTS = [
    ("capitals", "The capital of Egypt is"),
    ("math", "9 + 9 ="),
    ("opposites", "The opposite of slow is"),
    ("physical", "Silver melts at"),
    ("astronomical", "Saturn orbits"),
    ("conversational", "Honestly,"),
    ("answers", "The outcome is"),
]


def collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X and Y in float64."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    inputs = []
    outputs = []

    for prompt in prompts:
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
            if idx == start_layer:
                h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
                outputs.append(h_out)

    X = np.stack(inputs, axis=1)
    Y = np.stack(outputs, axis=1)
    return X, Y


def test_generation(model, tokenizer, prompt, T, start_layer, end_layer):
    """Test generation with T."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    # Factored
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
        if idx == start_layer:
            h_in = np.array(h[0, -1, :].astype(mx.float32)).astype(np.float64)
            h_out = T @ h_in
            h_np = np.array(h.astype(mx.float32))
            h_np[0, -1, :] = h_out.astype(np.float32)
            h = mx.array(h_np).astype(h.dtype)
            mx.eval(h)
        elif start_layer < idx <= end_layer:
            pass
        else:
            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

    if is_lfm2:
        h = inner_model.embedding_norm(h)
    else:
        h = inner_model.norm(h)
    logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)
    fact_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))

    return normal_token == fact_token, tokenizer.decode([normal_token]), tokenizer.decode([fact_token])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    start_layer = 3
    end_layer = n_layers - 2

    print(f"\n{'='*70}")
    print("PIECEWISE LINEAR COMPRESSION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Collect data per category
    category_data = {}
    all_prompts = []
    all_categories = []

    for cat, prompts in CALIBRATION_PROMPTS.items():
        X_cat, Y_cat = collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer)
        category_data[cat] = (X_cat, Y_cat, prompts)
        all_prompts.extend(prompts)
        all_categories.extend([cat] * len(prompts))
        print(f"Category '{cat}': {len(prompts)} prompts")

    # TEST 1: Single global T (baseline)
    print(f"\n{'='*70}")
    print("TEST 1: Single global T")
    print("="*70)

    all_X = np.hstack([d[0] for d in category_data.values()])
    all_Y = np.hstack([d[1] for d in category_data.values()])
    T_global = all_Y @ np.linalg.pinv(all_X)

    global_matches = 0
    print("\nHeld-out results:")
    for cat, prompt in HELD_OUT_PROMPTS:
        match, normal, factored = test_generation(model, tokenizer, prompt, T_global, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        print(f"  [{cat}] {status}: '{prompt[:25]}...' | {normal[:10]} vs {factored[:10]}")
        if match:
            global_matches += 1
    print(f"\nGlobal T: {global_matches}/{len(HELD_OUT_PROMPTS)}")

    # TEST 2: Category-specific T
    print(f"\n{'='*70}")
    print("TEST 2: Category-specific T")
    print("="*70)

    T_per_category = {}
    for cat, (X_cat, Y_cat, _) in category_data.items():
        T_per_category[cat] = Y_cat @ np.linalg.pinv(X_cat)

    category_matches = 0
    print("\nHeld-out results (using matching category T):")
    for cat, prompt in HELD_OUT_PROMPTS:
        T_cat = T_per_category[cat]
        match, normal, factored = test_generation(model, tokenizer, prompt, T_cat, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        print(f"  [{cat}] {status}: '{prompt[:25]}...' | {normal[:10]} vs {factored[:10]}")
        if match:
            category_matches += 1
    print(f"\nCategory-specific T: {category_matches}/{len(HELD_OUT_PROMPTS)}")

    # TEST 3: Nearest-neighbor T selection
    print(f"\n{'='*70}")
    print("TEST 3: Nearest-neighbor T selection")
    print("="*70)

    # For each held-out prompt, find nearest calibration prompt and use that category's T
    X_held, _ = collect_endpoint_data(model, tokenizer, [p for _, p in HELD_OUT_PROMPTS], start_layer, end_layer)

    nn_matches = 0
    print("\nHeld-out results (using nearest neighbor's T):")
    for i, (true_cat, prompt) in enumerate(HELD_OUT_PROMPTS):
        x_held = X_held[:, i]

        # Find nearest calibration sample
        min_dist = float('inf')
        best_cat = None
        for cat, (X_cat, _, prompts) in category_data.items():
            for j in range(X_cat.shape[1]):
                dist = np.linalg.norm(x_held - X_cat[:, j])
                if dist < min_dist:
                    min_dist = dist
                    best_cat = cat

        T_nn = T_per_category[best_cat]
        match, normal, factored = test_generation(model, tokenizer, prompt, T_nn, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        cat_match = "SAME" if best_cat == true_cat else f"DIFF({best_cat})"
        print(f"  [{true_cat}→{best_cat}] {status}: '{prompt[:20]}...' | {normal[:10]} vs {factored[:10]}")
        if match:
            nn_matches += 1
    print(f"\nNearest-neighbor T: {nn_matches}/{len(HELD_OUT_PROMPTS)}")

    # TEST 4: Include one example of held-out category in calibration
    print(f"\n{'='*70}")
    print("TEST 4: Add one anchor per held-out category")
    print("="*70)

    # For each held-out prompt, add IT to the category and recompute T
    anchor_matches = 0
    print("\nHeld-out results (with self as anchor):")
    for true_cat, prompt in HELD_OUT_PROMPTS:
        # Get category data
        X_cat, Y_cat, _ = category_data[true_cat]

        # Add this prompt to category
        X_anchor, Y_anchor = collect_endpoint_data(model, tokenizer, [prompt], start_layer, end_layer)
        X_aug = np.hstack([X_cat, X_anchor])
        Y_aug = np.hstack([Y_cat, Y_anchor])

        T_aug = Y_aug @ np.linalg.pinv(X_aug)
        match, normal, factored = test_generation(model, tokenizer, prompt, T_aug, start_layer, end_layer)
        status = "OK" if match else "FAIL"
        print(f"  [{true_cat}] {status}: '{prompt[:25]}...' | {normal[:10]} vs {factored[:10]}")
        if match:
            anchor_matches += 1
    print(f"\nWith self-anchor: {anchor_matches}/{len(HELD_OUT_PROMPTS)}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"""
Method                     | Accuracy
---------------------------|----------
Global T                   | {global_matches}/{len(HELD_OUT_PROMPTS)} ({100*global_matches/len(HELD_OUT_PROMPTS):.0f}%)
Category-specific T        | {category_matches}/{len(HELD_OUT_PROMPTS)} ({100*category_matches/len(HELD_OUT_PROMPTS):.0f}%)
Nearest-neighbor T         | {nn_matches}/{len(HELD_OUT_PROMPTS)} ({100*nn_matches/len(HELD_OUT_PROMPTS):.0f}%)
With self-anchor           | {anchor_matches}/{len(HELD_OUT_PROMPTS)} ({100*anchor_matches/len(HELD_OUT_PROMPTS):.0f}%)

INTERPRETATION:
- Global T: Tries to fit everything with one linear map
- Category T: Specialized for semantic category
- NN T: Uses nearest calibration sample's category
- Self-anchor: Cheating (includes test sample)

If category T > global T: Semantic specialization helps
If self-anchor = 100%: Problem is coverage, not algorithm
""")


if __name__ == "__main__":
    main()
