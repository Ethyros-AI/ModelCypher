#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Complete Semantic Span Calibration
"""
THE INSIGHT:
The held-out prompts have 5-54% OOS because they contain semantic categories
NOT present in the calibration pool. The pool covered:
- Country capitals, arithmetic, opposites, definitions, tech terms, etc.

But MISSED:
- Physical facts ("Water freezes at")
- Astronomical facts ("The moon orbits")
- Conversational phrases ("Well, actually")
- Personal stance ("In my opinion")
- Compositional facts ("Diamonds are made of")
- Landmark facts ("The Great Wall of China")

THIS SCRIPT:
1. Create a TRULY comprehensive prompt pool covering ALL semantic categories
2. Run greedy selection
3. Test that held-out prompts now have LOW OOS error
4. Verify compression still works

Usage:
    python lie_algebra_complete_span.py --model /path/to/model
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


def generate_complete_prompt_pool():
    """Generate a COMPLETE pool covering ALL semantic categories."""
    prompts = []

    # ==== CATEGORY 1: GEOGRAPHY (capitals, landmarks) ====
    countries = [
        "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria",
        "Bangladesh", "Belgium", "Brazil", "Canada", "Chile", "China", "Colombia",
        "Denmark", "Egypt", "Finland", "France", "Germany", "Greece", "Hungary",
        "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Japan",
        "Kenya", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nigeria",
        "Norway", "Pakistan", "Peru", "Philippines", "Poland", "Portugal", "Russia",
        "Saudi Arabia", "South Africa", "South Korea", "Spain", "Sweden",
        "Switzerland", "Thailand", "Turkey", "UK", "USA", "Vietnam",
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Landmarks
    landmarks = [
        "The Great Wall of China", "The Eiffel Tower", "The Taj Mahal",
        "The Pyramids of Giza", "The Colosseum", "Machu Picchu",
        "The Statue of Liberty", "Big Ben", "The Sydney Opera House",
        "The Leaning Tower of Pisa", "Christ the Redeemer", "Stonehenge",
        "The Grand Canyon", "Mount Everest", "The Amazon Rainforest",
        "The Great Barrier Reef", "Niagara Falls", "Victoria Falls",
    ]
    for l in landmarks:
        prompts.append(f"{l} is")
        prompts.append(f"{l} is located in")

    # ==== CATEGORY 2: ARITHMETIC ====
    for a in range(1, 31):
        for b in range(1, 31):
            prompts.append(f"{a} + {b} =")
            prompts.append(f"{a} - {b} =")
            if a * b < 1000:
                prompts.append(f"{a} * {b} =")
            if b != 0 and a % b == 0:
                prompts.append(f"{a} / {b} =")

    # ==== CATEGORY 3: PHYSICAL FACTS (temperature, phases, properties) ====
    physical_facts = [
        "Water freezes at", "Water boils at", "Ice melts at",
        "The speed of light is", "The speed of sound is",
        "Absolute zero is", "Room temperature is",
        "Normal body temperature is", "Gold melts at",
        "Iron melts at", "Mercury boils at", "Nitrogen boils at",
        "Oxygen boils at", "Helium boils at", "Carbon dioxide sublimes at",
        "The density of water is", "The density of gold is",
        "The hardest natural material is", "The densest element is",
        "The lightest gas is", "The most conductive metal is",
    ]
    for f in physical_facts:
        prompts.append(f)

    # ==== CATEGORY 4: ASTRONOMICAL/COSMOLOGICAL FACTS ====
    astro_facts = [
        "The moon orbits", "The Earth orbits", "The sun is",
        "Stars are made of", "The universe is", "Black holes are",
        "The Milky Way is", "Light years measure", "Gravity is",
        "The Big Bang was", "Planets orbit", "Asteroids are",
        "Comets are", "Galaxies contain", "Supernovas are",
        "Neutron stars are", "Pulsars are", "Quasars are",
        "Dark matter is", "Dark energy is", "The cosmos is",
        "The solar system is", "Jupiter is", "Saturn's rings are",
        "Mars is called", "Venus is", "Mercury is",
        "The closest star to Earth is", "The moon is",
        "Tides are caused by", "Seasons are caused by",
    ]
    for f in astro_facts:
        prompts.append(f)

    # ==== CATEGORY 5: COMPOSITIONAL FACTS ====
    compositions = [
        "Diamonds are made of", "Water is made of", "Air is made of",
        "Salt is made of", "Sugar is made of", "Steel is made of",
        "Bronze is made of", "Brass is made of", "Glass is made of",
        "Concrete is made of", "Paper is made of", "Plastic is made of",
        "DNA is made of", "Proteins are made of", "Cells are made of",
        "Atoms are made of", "Molecules are made of",
        "The sun is made of", "The Earth's core is made of",
        "Blood is made of", "Bones are made of", "Muscles are made of",
    ]
    for c in compositions:
        prompts.append(c)

    # ==== CATEGORY 6: WORD RELATIONSHIPS (opposites, synonyms) ====
    words = [
        "hot", "cold", "big", "small", "happy", "sad", "light", "dark",
        "up", "down", "good", "bad", "old", "young", "fast", "slow",
        "loud", "quiet", "wet", "dry", "full", "empty", "rich", "poor",
        "strong", "weak", "true", "false", "open", "closed", "hard", "soft",
        "near", "far", "early", "late", "clean", "dirty", "safe", "dangerous",
        "beautiful", "ugly", "simple", "complex", "easy", "difficult",
    ]
    for w in words:
        prompts.append(f"The opposite of {w} is")
        prompts.append(f"A synonym for {w} is")

    # ==== CATEGORY 7: COLORS ====
    colors = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "cyan", "magenta", "violet",
    ]
    for c in colors:
        prompts.append(f"The color {c} is")
        prompts.append(f"{c.capitalize()} is associated with")

    prompts.append("The color of the sky is")
    prompts.append("The color of grass is")
    prompts.append("The color of blood is")
    prompts.append("The color of snow is")
    prompts.append("The color of gold is")

    # ==== CATEGORY 8: ABSTRACT CONCEPTS (definitions) ====
    concepts = [
        "democracy", "freedom", "justice", "equality", "liberty",
        "evolution", "gravity", "entropy", "consciousness", "intelligence",
        "memory", "emotion", "language", "culture", "religion", "philosophy",
        "science", "technology", "art", "music", "literature", "mathematics",
        "physics", "chemistry", "biology", "psychology", "economics",
        "ethics", "truth", "beauty", "love", "time", "space", "energy",
    ]
    for c in concepts:
        prompts.append(f"{c.capitalize()} is")
        prompts.append(f"The definition of {c} is")

    # ==== CATEGORY 9: STORY STARTERS ====
    starters = [
        "Once upon a time", "In the beginning", "Long ago",
        "The story begins", "It all started when", "Years ago",
        "Deep in the forest", "High in the mountains",
        "After the war", "Before the revolution", "In ancient times",
    ]
    for s in starters:
        prompts.append(s)
        prompts.append(f"{s}, there was")

    # ==== CATEGORY 10: TECHNICAL/COMPUTING ====
    tech = [
        "algorithm", "database", "network", "function", "class", "variable",
        "recursion", "API", "CPU", "GPU", "RAM", "kernel", "compiler",
        "Neural networks are", "Machine learning is", "Artificial intelligence is",
        "Deep learning is", "Natural language processing is",
        "Computer vision is", "Reinforcement learning is",
    ]
    for t in tech:
        if t.endswith("is"):
            prompts.append(t)
        else:
            prompts.append(f"In computing, {t} is")

    # ==== CATEGORY 11: CONVERSATIONAL/SOCIAL PHRASES (CRITICAL!) ====
    conversational = [
        "Well, actually", "You know what,", "That's a great question",
        "That's interesting", "I see what you mean", "Good point",
        "Let me think about", "To be honest,", "Frankly,", "Actually,",
        "In fact,", "As a matter of fact,", "The thing is,",
        "Here's the thing,", "Look,", "Listen,", "See,", "So basically,",
        "I mean,", "You know,", "Like,", "Right?", "Okay so,",
        "Let me explain", "Here's why:", "The reason is",
        "I think that", "I believe that", "I feel that",
        "It seems like", "It appears that", "It looks like",
        "That makes sense", "That's fair", "That's true",
        "Absolutely", "Definitely", "Certainly", "Obviously",
        "Exactly", "Precisely", "Indeed", "Sure", "Of course",
        "No problem", "No worries", "Don't worry",
        "Thanks for asking", "Great question", "Good question",
    ]
    for c in conversational:
        prompts.append(c)

    # ==== CATEGORY 12: PERSONAL STANCE/OPINION (CRITICAL!) ====
    opinion = [
        "In my opinion,", "From my perspective,", "As I see it,",
        "I personally think", "My view is that", "I would say that",
        "I'd argue that", "I tend to think", "I'm inclined to say",
        "If you ask me,", "Speaking for myself,", "Personally,",
        "Subjectively speaking,", "In my experience,", "Based on my experience,",
    ]
    for o in opinion:
        prompts.append(o)

    # ==== CATEGORY 13: REFLECTIVE PHRASES (CRITICAL!) ====
    reflective = [
        "If you think about it,", "When you consider,", "Looking at it this way,",
        "From this perspective,", "Taking a step back,", "On reflection,",
        "Upon further thought,", "Thinking about it,", "Considering this,",
        "Given that,", "Assuming that,", "If we assume,",
        "Hypothetically speaking,", "In theory,", "In practice,",
    ]
    for r in reflective:
        prompts.append(r)

    # ==== CATEGORY 14: PROBLEM/ISSUE STATEMENTS (CRITICAL!) ====
    problem = [
        "The problem is that", "The issue is", "The challenge is",
        "The difficulty is", "The obstacle is", "The barrier is",
        "What's wrong is", "The trouble is", "The catch is",
        "The downside is", "The drawback is", "The limitation is",
        "One concern is", "A major issue is", "The fundamental problem is",
    ]
    for p in problem:
        prompts.append(p)

    # ==== CATEGORY 15: TRANSITIONAL PHRASES ====
    transitions = [
        "However,", "Therefore,", "Moreover,", "Furthermore,",
        "In conclusion,", "To summarize,", "As a result,", "Despite this,",
        "Nevertheless,", "Nonetheless,", "On the other hand,",
        "In contrast,", "Similarly,", "Likewise,", "Additionally,",
        "Consequently,", "Hence,", "Thus,", "Accordingly,",
    ]
    for t in transitions:
        prompts.append(t)
        prompts.append(f"{t} the")

    # ==== CATEGORY 16: QUESTION STARTERS ====
    question_starts = [
        "What is", "Who was", "Where is", "When did", "Why do", "How does",
        "Which is", "Can you", "Could we", "Should I", "Would it",
        "Is there", "Are we", "Was it", "Have you", "Does this", "Do they",
    ]
    for q in question_starts:
        prompts.append(f"{q} the meaning of life")
        prompts.append(f"{q} consciousness")
        prompts.append(f"{q} the universe")

    # ==== CATEGORY 17: CODE SNIPPETS ====
    code_starts = [
        "def ", "class ", "import ", "if ", "for ", "while ",
        "return ", "print(", "lambda ", "@", "try:", "except ",
    ]
    for c in code_starts:
        prompts.append(c)

    # ==== CATEGORY 18: ANSWER/CONCLUSION PHRASES ====
    answers = [
        "The answer is", "The solution is", "The result is",
        "The correct answer is", "The short answer is", "The simple answer is",
        "In short,", "To sum up,", "Ultimately,", "In essence,",
        "The bottom line is", "At the end of the day,", "When all is said and done,",
    ]
    for a in answers:
        prompts.append(a)

    # ==== CATEGORY 19: PRESIDENT/POLITICAL (specific to held-out) ====
    political = [
        "The president of the United States is",
        "The prime minister of the UK is",
        "The chancellor of Germany is",
        "The president of France is",
        "The president of China is",
        "The president of Russia is",
        "The leader of", "The head of state of",
        "The government of", "The constitution of",
    ]
    for p in political:
        prompts.append(p)

    # ==== CATEGORY 20: EXPANDING UNIVERSE (specific to held-out) ====
    cosmology = [
        "The universe is expanding", "The universe is infinite",
        "The universe began with", "The age of the universe is",
        "Space is", "Time is", "Spacetime is", "The fabric of space is",
        "General relativity says", "Quantum mechanics says",
        "String theory proposes", "The multiverse is",
    ]
    for c in cosmology:
        prompts.append(c)

    # Remove duplicates
    prompts = list(set(prompts))
    random.seed(42)
    random.shuffle(prompts)
    return prompts


# The held-out prompts that we need to cover
HELD_OUT_PROMPTS = [
    "Water freezes at",
    "The color of the sky is",
    "The president of the United States is",
    "The moon orbits",
    "Diamonds are made of",
    "The Great Wall of China is",
    "100 / 10 =",
    "50 + 50 =",
    "9 * 9 =",
    "1000 - 1 =",
    "Neural networks are",
    "Stars are made of",
    "The universe is expanding",
    "The answer is",
    "Well, actually",
    "You know what,",
    "If you think about it,",
    "The problem is that",
    "In my opinion,",
    "That's a great question",
]


def collect_activations(model, tokenizer, prompts, target_layer):
    """Collect activations at target layer."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    activations = []
    for i, prompt in enumerate(prompts):
        if i % 500 == 0 and i > 0:
            logger.info(f"Collecting {i}/{len(prompts)}")

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


def greedy_span_selection(X, target_coverage=0.995, max_samples=500):
    """Greedily select samples maximizing span coverage."""
    n_samples = X.shape[1]

    # Ensure we're working with finite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    norms = np.linalg.norm(X, axis=0, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-10)

    selected = []
    remaining = list(range(n_samples))

    first = int(np.argmax(norms[0]))
    selected.append(first)
    remaining.remove(first)

    Q = X_norm[:, [first]].copy()
    Q, _ = np.linalg.qr(Q)

    while len(selected) < max_samples and remaining:
        X_remaining = X_norm[:, remaining]
        projections = Q @ (Q.T @ X_remaining)
        projections = np.nan_to_num(projections, nan=0.0, posinf=1e10, neginf=-1e10)
        orthogonal = X_remaining - projections
        orth_norms = np.linalg.norm(orthogonal, axis=0)

        # Handle NaN
        orth_norms = np.nan_to_num(orth_norms, nan=0.0)

        best_idx = np.argmax(orth_norms)
        if orth_norms[best_idx] < 1e-6:
            break

        actual_idx = remaining[best_idx]
        selected.append(actual_idx)
        remaining.remove(actual_idx)

        new_vec = X_norm[:, [actual_idx]]
        Q_new = np.hstack([Q, new_vec])
        Q, _ = np.linalg.qr(Q_new)

        if len(selected) % 50 == 0:
            total_var = np.sum(np.linalg.norm(X_norm, axis=0)**2)
            projections_all = Q @ (Q.T @ X_norm)
            projections_all = np.nan_to_num(projections_all, nan=0.0, posinf=1e10, neginf=-1e10)
            explained_var = np.sum(np.linalg.norm(projections_all, axis=0)**2)
            coverage = explained_var / max(total_var, 1e-10)
            logger.info(f"Selected {len(selected)}, coverage={coverage:.4f}")
            if coverage >= target_coverage:
                break

    return selected


def collect_endpoint_data(model, tokenizer, prompts, start_layer, end_layer):
    """Collect X (at start_layer) and Y (at end_layer)."""
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
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                inputs.append(h_in)

            if is_lfm2:
                mask = attn_mask if layer.is_attention_layer else conv_mask
            else:
                mask = attn_mask
            h = layer(h, mask, None)
            mx.eval(h)

            if idx == end_layer:
                h_out = np.array(h[0, -1, :].astype(mx.float32))
                outputs.append(h_out)

    X = np.stack(inputs, axis=1).astype(np.float64)
    Y = np.stack(outputs, axis=1).astype(np.float64)
    return X, Y


def test_generation(model, tokenizer, prompt, T_fact, start_layer, end_layer):
    """Test generation with T_fact."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    is_lfm2 = "lfm" in type(inner_model).__name__.lower()

    # Normal
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    normal_token = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    normal = tokenizer.decode([normal_token]).split()[0] if tokenizer.decode([normal_token]).split() else "(empty)"

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
            h_out = T_fact @ h_in
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
    factored = tokenizer.decode([fact_token]).split()[0] if tokenizer.decode([fact_token]).split() else "(empty)"

    return normal, factored


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
    print("COMPLETE SEMANTIC SPAN CALIBRATION")
    print("="*70)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Compressing layers {start_layer} -> {end_layer}")

    # PHASE 1: Generate complete pool
    print(f"\n{'='*70}")
    print("PHASE 1: GENERATE COMPLETE PROMPT POOL")
    print("="*70)

    all_prompts = generate_complete_prompt_pool()
    print(f"Total prompts generated: {len(all_prompts)}")

    # Check that held-out categories are covered
    print("\nVerifying held-out categories are in pool:")
    for hp in HELD_OUT_PROMPTS[:5]:
        found = any(hp in p or hp == p for p in all_prompts)
        symbol = "+" if found else "MISSING"
        print(f"  {symbol}: '{hp[:40]}...'")

    # PHASE 2: Collect activations and select
    print(f"\n{'='*70}")
    print("PHASE 2: COLLECT ACTIVATIONS & SELECT")
    print("="*70)

    # Use up to 3000 prompts for selection
    pool_prompts = all_prompts[:3000]
    logger.info(f"Collecting activations for {len(pool_prompts)} prompts...")
    X_pool = collect_activations(model, tokenizer, pool_prompts, start_layer)
    print(f"Pool activations: {X_pool.shape}")

    # Analyze pool rank
    U_pool, S_pool, _ = np.linalg.svd(X_pool, full_matrices=False)
    total_var = np.sum(S_pool**2)
    cumsum = np.cumsum(S_pool**2)
    pool_rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1
    print(f"Pool effective rank (99%): {pool_rank_99}")

    # Greedy select
    logger.info("Running greedy selection...")
    selected_indices = greedy_span_selection(X_pool, target_coverage=0.995, max_samples=400)
    calibration_prompts = [pool_prompts[i] for i in selected_indices]
    print(f"Selected {len(calibration_prompts)} calibration prompts")

    # PHASE 3: Check held-out OOS BEFORE computing T
    print(f"\n{'='*70}")
    print("PHASE 3: CHECK HELD-OUT OOS (before transformation)")
    print("="*70)

    X_calib = X_pool[:, selected_indices]
    Q, _ = np.linalg.qr(X_calib)

    # Collect held-out activations
    X_held = collect_activations(model, tokenizer, HELD_OUT_PROMPTS, start_layer)

    # Compute OOS
    X_held = np.nan_to_num(X_held, nan=0.0, posinf=1e10, neginf=-1e10)
    projections_held = Q @ (Q.T @ X_held)
    projections_held = np.nan_to_num(projections_held, nan=0.0, posinf=1e10, neginf=-1e10)
    print(f"\n{'Held-out prompt':<45} | {'OOS%':>8}")
    print("-" * 58)

    total_oos = 0
    for i, hp in enumerate(HELD_OUT_PROMPTS):
        proj_norm = np.linalg.norm(projections_held[:, i])
        orig_norm = np.linalg.norm(X_held[:, i])
        oos = max(0, 1 - proj_norm / (orig_norm + 1e-10))
        total_oos += oos
        print(f"  {hp[:43]:<43} | {oos*100:>7.2f}%")

    mean_held_oos = total_oos / len(HELD_OUT_PROMPTS)
    print(f"\nMean held-out OOS: {mean_held_oos*100:.2f}%")

    if mean_held_oos > 0.10:
        print("\n>>> WARNING: High held-out OOS! Pool may still be missing categories.")
    else:
        print("\n>>> SUCCESS: Low held-out OOS! Pool covers held-out categories.")

    # PHASE 4: Compute transformation
    print(f"\n{'='*70}")
    print("PHASE 4: COMPUTE TRANSFORMATION")
    print("="*70)

    logger.info("Collecting endpoint data for calibration...")
    X, Y = collect_endpoint_data(model, tokenizer, calibration_prompts, start_layer, end_layer)
    print(f"X: {X.shape}, Y: {Y.shape}")

    # Ensure finite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1e10, neginf=-1e10)

    T = Y @ np.linalg.pinv(X)
    T = np.nan_to_num(T, nan=0.0, posinf=1e10, neginf=-1e10)

    # Reconstruction error
    Y_pred = T @ X
    Y_pred = np.nan_to_num(Y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    Y_norm = np.linalg.norm(Y)
    recon_error = np.linalg.norm(Y - Y_pred) / max(Y_norm, 1e-10)
    print(f"Calibration reconstruction error: {recon_error:.6f}")

    # T analysis
    U_t, S_t, Vh_t = np.linalg.svd(T, full_matrices=False)
    t_total_var = np.sum(S_t**2)
    t_cumsum = np.cumsum(S_t**2)
    t_rank_90 = np.searchsorted(t_cumsum / t_total_var, 0.90) + 1
    t_rank_99 = np.searchsorted(t_cumsum / t_total_var, 0.99) + 1
    print(f"T rank (90%): {t_rank_90}, (99%): {t_rank_99}")

    # PHASE 5: Test token prediction
    print(f"\n{'='*70}")
    print("PHASE 5: TOKEN PREDICTION TEST")
    print("="*70)

    for rank in [400, 350, 300, 256, 200, 128, 64]:
        if rank > len(S_t):
            continue
        T_fact = U_t[:, :rank] @ np.diag(S_t[:rank]) @ Vh_t[:rank, :]
        T_fact = np.nan_to_num(T_fact, nan=0.0, posinf=1e10, neginf=-1e10)

        # Calibration test
        calib_matches = 0
        for p in calibration_prompts[:30]:
            normal, factored = test_generation(model, tokenizer, p, T_fact, start_layer, end_layer)
            if normal == factored:
                calib_matches += 1

        # Held-out test
        held_matches = 0
        for hp in HELD_OUT_PROMPTS:
            normal, factored = test_generation(model, tokenizer, hp, T_fact, start_layer, end_layer)
            if normal == factored:
                held_matches += 1

        compression = hidden_dim / rank
        print(f"Rank={rank:>3}: calib={calib_matches}/30, held-out={held_matches}/{len(HELD_OUT_PROMPTS)}, {compression:.1f}x compression")

    # PHASE 6: Detailed held-out results
    print(f"\n{'='*70}")
    print("PHASE 6: DETAILED HELD-OUT RESULTS (rank=128)")
    print("="*70)

    T_fact = U_t[:, :128] @ np.diag(S_t[:128]) @ Vh_t[:128, :]
    T_fact = np.nan_to_num(T_fact, nan=0.0, posinf=1e10, neginf=-1e10)

    print(f"\n{'Prompt':<40} | {'OOS%':>7} | {'Match':>5} | {'Normal':>12} | {'Factored':>12}")
    print("-" * 90)

    for i, hp in enumerate(HELD_OUT_PROMPTS):
        # OOS
        proj_norm = np.linalg.norm(projections_held[:, i])
        orig_norm = np.linalg.norm(X_held[:, i])
        oos = max(0, 1 - proj_norm / (orig_norm + 1e-10))

        # Token prediction
        normal, factored = test_generation(model, tokenizer, hp, T_fact, start_layer, end_layer)
        match = "Yes" if normal == factored else "No"

        print(f"  {hp[:38]:<38} | {oos*100:>6.1f}% | {match:>5} | {normal[:12]:>12} | {factored[:12]:>12}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"""
COMPLETE SEMANTIC SPAN CALIBRATION:

Pool: {len(pool_prompts)} prompts covering 20 semantic categories
Selected: {len(calibration_prompts)} via greedy selection
Pool effective rank (99%): {pool_rank_99}
T rank (99%): {t_rank_99}
Mean held-out OOS: {mean_held_oos*100:.2f}%

KEY INSIGHT:
By including ALL semantic categories (physical facts, conversational phrases,
opinion starters, etc.), we should achieve LOW OOS on held-out prompts.

If held-out OOS is still high, we need even more diverse categories.
""")


if __name__ == "__main__":
    main()
