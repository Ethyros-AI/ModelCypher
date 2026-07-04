#!/usr/bin/env python3
"""MP vs MT Trajectory Comparison.

Core question: Why can the 350M model do modus ponens but not modus tollens?
The architecture is identical — same operations, same forward pass.
If there's no structural barrier, the answer is in the learned geometry.

Approach: Run the FULL PROMPT through the model, capture hidden states at
every layer, then compare MP vs MT at the geometric level. The difference
is in how the model processes the premise + observation, not in generation.

With --attention: Also captures attention weight matrices at every attention
layer by manually computing softmax(Q @ K^T * scale). Shows where each head
attends at the last prompt position — the critical readout position.

Usage:
    poetry run python scripts/mp_vs_mt_trajectory.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16

    # With attention weight analysis:
    poetry run python scripts/mp_vs_mt_trajectory.py \
        --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 \
        --attention
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Matched pairs: SAME premise, different reasoning direction
MATCHED_PAIRS = [
    {
        "domain": "mammals",
        "premise": "If an animal is a mammal, then it is warm-blooded.",
        "mp": {
            "prompt": "Apply logical reasoning:\nIf an animal is a mammal, then it is warm-blooded. This animal is a mammal. What can we conclude?",
            "expected": "warm-blooded",
        },
        "mt": {
            "prompt": "Apply logical reasoning:\nIf an animal is a mammal, then it is warm-blooded. This animal is not warm-blooded. What can we conclude?",
            "expected": "not a mammal",
        },
    },
    {
        "domain": "rain",
        "premise": "If it rains, the streets get wet.",
        "mp": {
            "prompt": "Apply logical reasoning:\nIf it rains, the streets get wet. It is raining. What can we conclude?",
            "expected": "streets get wet",
        },
        "mt": {
            "prompt": "Apply logical reasoning:\nIf it rains, the streets get wet. The streets are not wet. What can we conclude?",
            "expected": "not raining",
        },
    },
    {
        "domain": "differentiable",
        "premise": "If a function is differentiable at a point, then it is continuous at that point.",
        "mp": {
            "prompt": "Apply logical reasoning:\nIf a function is differentiable at a point, then it is continuous at that point. Function f is differentiable at x=3. What can we conclude?",
            "expected": "continuous at x=3",
        },
        "mt": {
            "prompt": "Apply logical reasoning:\nIf a function is differentiable at a point, then it is continuous at that point. Function f is not continuous at x=3. What can we conclude?",
            "expected": "not differentiable",
        },
    },
    {
        "domain": "certification",
        "premise": "Every employee who passed the certification received a bonus.",
        "mp": {
            "prompt": "Apply logical reasoning:\nEvery employee who passed the certification received a bonus. Maria passed the certification. What can we conclude?",
            "expected": "received a bonus",
        },
        "mt": {
            "prompt": "Apply logical reasoning:\nEvery employee who passed the certification received a bonus. Maria did not receive a bonus. What can we conclude about Maria's certification?",
            "expected": "did not pass",
        },
    },
    {
        "domain": "birds",
        "premise": "All birds have feathers.",
        "mp": {
            "prompt": "Apply logical reasoning:\nAll birds have feathers. A robin is a bird. What can we conclude?",
            "expected": "has feathers",
        },
        "mt": {
            "prompt": "Apply logical reasoning:\nAll birds have feathers. An animal does not have feathers. Is it a bird?",
            "expected": "not a bird",
        },
    },
]


def capture_attention_weights(model, tokenizer, backend, prompt: str) -> dict:
    """Run prompt through model, capture attention weights at every attention layer.

    Manually computes softmax(Q @ K^T * scale) since the fused MLX kernel
    doesn't expose weights. Handles GQA by repeating KV heads.

    Returns: {
        "tokens": list[str],  # tokenized prompt
        "token_ids": list[int],
        "attention": {layer_idx: weights},  # weights: [n_heads, seq_len, seq_len]
        "hidden_states": {layer_idx: hidden_state_at_last_position},
        "logits": last_logits
    }
    """
    import mlx.core as mx

    b = backend
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)

    token_ids = tokenizer.encode(prompt)
    if not isinstance(token_ids, list):
        token_ids = b.tolist(token_ids)

    # Decode individual tokens for labeling
    tokens = []
    for tid in token_ids:
        try:
            tokens.append(tokenizer.decode([tid]))
        except Exception:
            tokens.append(f"<{tid}>")

    input_ids = b.array([token_ids])

    captured_hidden: dict[int, Any] = {}
    captured_attn: dict[int, Any] = {}

    class AttentionCaptureWrapper:
        """Wraps an attention layer to capture Q, K after RoPE and compute weights."""

        def __init__(self, layer, layer_idx):
            self._layer = layer
            self._layer_idx = layer_idx

        def __call__(self, x, mask=None, cache=None):
            if not self._layer.is_attention_layer:
                output = self._get_operator()(x, mask=mask, cache=cache)
                captured_hidden[self._layer_idx] = output
                return output

            # For attention layers: intercept self_attn to capture weights
            attn = self._layer.self_attn
            x_normed = self._layer.operator_norm(x)
            B, L, D = x_normed.shape

            # QKV projections (same as Attention.__call__)
            queries = attn.q_proj(x_normed)
            keys = attn.k_proj(x_normed)
            values = attn.v_proj(x_normed)

            queries = attn.q_layernorm(
                queries.reshape(B, L, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)   # [B, n_heads, L, head_dim]
            keys = attn.k_layernorm(
                keys.reshape(B, L, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)   # [B, n_kv_heads, L, head_dim]
            values = values.reshape(
                B, L, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)   # [B, n_kv_heads, L, head_dim]

            # RoPE (no cache during full-prompt forward pass)
            if cache is not None:
                queries = attn.rope(queries, offset=cache.offset)
                keys = attn.rope(keys, offset=cache.offset)
            else:
                queries = attn.rope(queries)
                keys = attn.rope(keys)

            # GQA: expand KV heads to match query heads
            # n_heads=16, n_kv_heads=8 → repeat each KV head 2x
            n_rep = attn.n_heads // attn.n_kv_heads
            if n_rep > 1:
                keys_expanded = mx.repeat(keys, n_rep, axis=1)    # [B, 16, L, 64]
            else:
                keys_expanded = keys

            # Manually compute attention weights: softmax(Q @ K^T * scale)
            scores = mx.matmul(queries, keys_expanded.transpose(0, 1, 3, 2)) * attn.scale
            # [B, n_heads, L, L]

            # Apply causal mask
            causal = mx.triu(mx.full((L, L), float("-inf")), k=1)
            scores = scores + causal

            weights = mx.softmax(scores.astype(mx.float32), axis=-1)
            mx.eval(weights)
            captured_attn[self._layer_idx] = weights[0]  # drop batch dim: [n_heads, L, L]

            # Now run the ACTUAL layer forward (so output is numerically identical)
            output = self._layer(x, mask=mask, cache=cache)
            captured_hidden[self._layer_idx] = output
            return output

        def _get_operator(self):
            """Get the actual operator (conv or attn) for non-attention layers."""
            return self._layer

        def __getattr__(self, name):
            return getattr(self._layer, name)

    original_layers = list(layers)
    try:
        for i in range(len(layers)):
            layers[i] = AttentionCaptureWrapper(original_layers[i], i)
        logits = model(input_ids)
        b.eval(logits)
    finally:
        for i, layer in enumerate(original_layers):
            layers[i] = layer

    # Extract last-position hidden state per layer
    hidden_states = {}
    for layer_idx, hidden in captured_hidden.items():
        if hidden.ndim == 3:
            h = hidden[0, -1, :]
        else:
            h = hidden[-1, :]
        b.eval(h)
        hidden_states[layer_idx] = h

    if logits.ndim == 3:
        last_logits = logits[0, -1, :]
    else:
        last_logits = logits[-1, :]
    b.eval(last_logits)

    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "attention": captured_attn,
        "hidden_states": hidden_states,
        "logits": last_logits,
    }


def capture_prompt_hidden_states(model, tokenizer, backend, prompt: str) -> dict[int, Any]:
    """Run prompt through model, capture hidden states at every layer.

    Returns: {layer_idx: hidden_state_at_last_position} for all layers.
    """
    b = backend
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)

    tokens = tokenizer.encode(prompt)
    if not isinstance(tokens, list):
        tokens = b.tolist(tokens)
    input_ids = b.array([tokens])

    captured: dict[int, Any] = {}

    class CaptureWrapper:
        def __init__(self, layer, layer_idx):
            self._layer = layer
            self._layer_idx = layer_idx

        def __call__(self, *args, **kwargs):
            output = self._layer(*args, **kwargs)
            if isinstance(output, tuple):
                captured[self._layer_idx] = output[0]
            else:
                captured[self._layer_idx] = output
            return output

        def __getattr__(self, name):
            return getattr(self._layer, name)

    original_layers = list(layers)
    try:
        for i in range(len(layers)):
            layers[i] = CaptureWrapper(original_layers[i], i)
        logits = model(input_ids)
        b.eval(logits)
    finally:
        for i, layer in enumerate(original_layers):
            layers[i] = layer

    # Extract last-position hidden state per layer
    result = {}
    for layer_idx, hidden in captured.items():
        if hidden.ndim == 3:
            h = hidden[0, -1, :]  # [batch, seq, dim] -> last position
        else:
            h = hidden[-1, :]
        b.eval(h)
        result[layer_idx] = h

    # Also get logits at last position
    if logits.ndim == 3:
        last_logits = logits[0, -1, :]
    else:
        last_logits = logits[-1, :]
    b.eval(last_logits)
    result["logits"] = last_logits

    return result


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 96) -> str:
    """Generate response using mlx_lm.generate (proper KV cache handling)."""
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


def cosine_similarity(a, b, backend):
    """Compute cosine similarity between two vectors."""
    bk = backend
    dot = bk.to_scalar(bk.sum(a * b))
    norm_a = bk.to_scalar(bk.sqrt(bk.sum(a * a)))
    norm_b = bk.to_scalar(bk.sqrt(bk.sum(b * b)))
    return dot / max(norm_a * norm_b, 1e-10)


def print_attention_analysis(mp_result: dict, mt_result: dict, domain: str, layers_info: list):
    """Compare attention weights between MP and MT at the last prompt position.

    For each attention layer, shows:
    1. Where the last position attends (top-5 tokens per head)
    2. How MP and MT attention patterns differ
    3. Which heads show the largest divergence
    """

    mp_tokens = mp_result["tokens"]
    mt_tokens = mt_result["tokens"]
    mp_attn = mp_result["attention"]
    mt_attn = mt_result["attention"]

    attn_layers = sorted(mp_attn.keys())

    for layer_idx in attn_layers:
        mp_w = mp_attn[layer_idx]  # [n_heads, L_mp, L_mp]
        mt_w = mt_attn[layer_idx]  # [n_heads, L_mt, L_mt]
        n_heads = mp_w.shape[0]
        # Attention from the LAST position (the one that generates the first answer token)
        mp_last = mp_w[:, -1, :]  # [n_heads, L_mp]
        mt_last = mt_w[:, -1, :]  # [n_heads, L_mt]

        print(f"\n  LAYER {layer_idx} ATTENTION (last position → all positions):")
        print(f"  {'─' * 70}")

        # Per-head analysis
        head_divergences = []
        for head in range(n_heads):
            mp_head = [float(x) for x in mp_last[head].tolist()]
            mt_head = [float(x) for x in mt_last[head].tolist()]

            # Top-5 attended positions for MP
            mp_top5 = sorted(enumerate(mp_head), key=lambda x: -x[1])[:5]
            mt_top5 = sorted(enumerate(mt_head), key=lambda x: -x[1])[:5]

            # Compute max attention weight (concentration measure)
            mp_max = max(mp_head)
            mt_max = max(mt_head)

            # Entropy of attention distribution (how spread out)
            mp_ent = -sum(p * math.log(p + 1e-30) for p in mp_head)
            mt_ent = -sum(p * math.log(p + 1e-30) for p in mt_head)

            head_divergences.append({
                "head": head,
                "mp_top5": mp_top5,
                "mt_top5": mt_top5,
                "mp_max": mp_max,
                "mt_max": mt_max,
                "mp_entropy": mp_ent,
                "mt_entropy": mt_ent,
            })

        # Show summary: heads sorted by entropy difference
        print("  Head  MP_maxw  MT_maxw  MP_ent   MT_ent   Δent")
        print(f"  {'─' * 55}")
        for hd in sorted(head_divergences, key=lambda x: abs(x["mp_entropy"] - x["mt_entropy"]), reverse=True):
            delta_ent = hd["mt_entropy"] - hd["mp_entropy"]
            marker = ""
            if abs(delta_ent) > 0.5:
                marker = " ←← DIFFERENT"
            elif abs(delta_ent) > 0.2:
                marker = " ← diverging"
            print(f"  {hd['head']:>4}  {hd['mp_max']:>7.4f}  {hd['mt_max']:>7.4f}  "
                  f"{hd['mp_entropy']:>7.3f}  {hd['mt_entropy']:>7.3f}  "
                  f"{delta_ent:>+7.3f}{marker}")

        # Show top-3 most divergent heads in detail
        most_divergent = sorted(
            head_divergences,
            key=lambda x: abs(x["mp_entropy"] - x["mt_entropy"]),
            reverse=True,
        )[:3]

        for hd in most_divergent:
            head = hd["head"]
            print(f"\n  Head {head} detail (Δent = {hd['mt_entropy'] - hd['mp_entropy']:+.3f}):")

            print("    MP attends to:")
            for pos, weight in hd["mp_top5"]:
                tok = mp_tokens[pos] if pos < len(mp_tokens) else "?"
                print(f"      pos {pos:3d} ({weight:.4f}): {repr(tok)}")

            print("    MT attends to:")
            for pos, weight in hd["mt_top5"]:
                tok = mt_tokens[pos] if pos < len(mt_tokens) else "?"
                print(f"      pos {pos:3d} ({weight:.4f}): {repr(tok)}")


def main():
    parser = argparse.ArgumentParser(description="MP vs MT trajectory comparison")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--adapter", default=None, help="Optional adapter path")
    parser.add_argument("--max-tokens", type=int, default=96, help="Max tokens per generation")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--attention", action="store_true", help="Capture and analyze attention weights")
    args = parser.parse_args()

    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()
    b = backend

    if args.adapter:
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(args.model, adapter_path=args.adapter)
    else:
        model, tokenizer = backend.load_model(args.model)

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    num_layers = len(layers)
    logger.info(f"Model loaded: {num_layers} layers")

    # Check which layers are attention vs conv
    for i, layer in enumerate(layers):
        is_attn = getattr(layer, "is_attention_layer", None)
        logger.info(f"  Layer {i}: {'attention' if is_attn else 'conv'}")

    for pair in MATCHED_PAIRS:
        domain = pair["domain"]
        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain}")
        print(f"Premise: {pair['premise']}")
        print(f"{'='*80}")

        for form in ["mp", "mt"]:
            info = pair[form]
            prompt = info["prompt"]

            if args.attention:
                # Full capture: hidden states + attention weights
                result = capture_attention_weights(model, tokenizer, backend, prompt)
                hidden_states = result["hidden_states"]
                hidden_states["logits"] = result["logits"]
            else:
                # Just hidden states (faster)
                hidden_states = capture_prompt_hidden_states(model, tokenizer, backend, prompt)
                result = None

            # Generate actual response (with proper KV cache)
            response = generate_response(model, tokenizer, prompt, max_tokens=args.max_tokens)
            correct = info["expected"].lower() in response.lower()

            # Analyze hidden states
            norms = []
            for l in range(num_layers):
                if l in hidden_states:
                    h = hidden_states[l]
                    norm = float(b.to_scalar(b.sqrt(b.sum(h * h))))
                    norms.append(norm)
                else:
                    norms.append(0.0)

            # Get logit statistics
            logits = hidden_states["logits"]
            logits_list = b.tolist(logits)
            max_logit = max(logits_list)
            exp_logits = [math.exp(l - max_logit) for l in logits_list]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            entropy = -sum(p * math.log(p + 1e-30) for p in probs)

            # Top 10 tokens
            indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:10]
            top_tokens = []
            for idx, prob in indexed:
                try:
                    tok_str = tokenizer.decode([idx])
                except Exception:
                    tok_str = f"<{idx}>"
                top_tokens.append((tok_str, prob))

            label = "MP (forward)" if form == "mp" else "MT (backward)"
            status = "CORRECT" if correct else "WRONG"
            print(f"\n  {label}: {status}")
            print(f"    Response: {response.strip()[:200]}")
            print(f"    Logit entropy: {entropy:.3f}")
            print("    Top 10 next tokens:")
            for tok, prob in top_tokens:
                print(f"      {repr(tok):20s} {prob:.4f}")

            pair.setdefault("results", {})[form] = {
                "response": response,
                "correct": correct,
                "norms": norms,
                "entropy": entropy,
                "top_tokens": top_tokens,
                "hidden_states": hidden_states,
                "attn_result": result,
            }

        # Attention weight analysis
        if args.attention:
            mp_attn_result = pair["results"]["mp"]["attn_result"]
            mt_attn_result = pair["results"]["mt"]["attn_result"]
            if mp_attn_result and mt_attn_result:
                print_attention_analysis(mp_attn_result, mt_attn_result, domain, layers)

        # Compare MP vs MT hidden states directly
        mp_hs = pair["results"]["mp"]["hidden_states"]
        mt_hs = pair["results"]["mt"]["hidden_states"]

        print("\n  PER-LAYER COMPARISON (hidden state at last prompt position):")
        print(f"  {'Layer':>6} {'Type':>6}  {'MP norm':>10}  {'MT norm':>10}  {'Cos(MP,MT)':>12}  {'L2 dist':>10}")
        print(f"  {'-'*6} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

        for l in range(num_layers):
            if l not in mp_hs or l not in mt_hs:
                continue
            mp_h = mp_hs[l]
            mt_h = mt_hs[l]

            mp_norm = float(b.to_scalar(b.sqrt(b.sum(mp_h * mp_h))))
            mt_norm = float(b.to_scalar(b.sqrt(b.sum(mt_h * mt_h))))
            cos_sim = cosine_similarity(mp_h, mt_h, backend)

            diff = mp_h - mt_h
            b.eval(diff)
            l2_dist = float(b.to_scalar(b.sqrt(b.sum(diff * diff))))

            is_attn = getattr(layers[l], "is_attention_layer", False)
            layer_type = "attn" if is_attn else "conv"

            marker = ""
            if cos_sim < 0.95:
                marker = " <-- DIVERGENT"
            elif cos_sim < 0.99:
                marker = " <-- diverging"

            print(f"  {l:>6} {layer_type:>6}  {mp_norm:>10.4f}  {mt_norm:>10.4f}  {cos_sim:>12.6f}  {l2_dist:>10.4f}{marker}")

    # Cross-pair aggregate
    print(f"\n{'='*80}")
    print("AGGREGATE ACROSS ALL PAIRS")
    print(f"{'='*80}")

    mp_correct = sum(1 for p in MATCHED_PAIRS if p["results"]["mp"]["correct"])
    mt_correct = sum(1 for p in MATCHED_PAIRS if p["results"]["mt"]["correct"])
    print(f"\n  MP correct: {mp_correct}/{len(MATCHED_PAIRS)}")
    print(f"  MT correct: {mt_correct}/{len(MATCHED_PAIRS)}")

    print("\n  Mean cosine similarity MP↔MT by layer:")
    print(f"  {'Layer':>6} {'Type':>6}  {'Mean cos':>10}  {'Min cos':>10}  {'Mean L2':>10}")
    print(f"  {'-'*6} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    for l in range(num_layers):
        cos_vals = []
        l2_vals = []
        for pair in MATCHED_PAIRS:
            if l not in pair["results"]["mp"]["hidden_states"]:
                continue
            mp_h = pair["results"]["mp"]["hidden_states"][l]
            mt_h = pair["results"]["mt"]["hidden_states"][l]
            cos_vals.append(cosine_similarity(mp_h, mt_h, backend))
            diff = mp_h - mt_h
            b.eval(diff)
            l2_vals.append(float(b.to_scalar(b.sqrt(b.sum(diff * diff)))))

        is_attn = getattr(layers[l], "is_attention_layer", False)
        layer_type = "attn" if is_attn else "conv"
        mean_cos = sum(cos_vals) / len(cos_vals)
        min_cos = min(cos_vals)
        mean_l2 = sum(l2_vals) / len(l2_vals)

        marker = ""
        if mean_cos < 0.95:
            marker = " <-- KEY DIVERGENCE"
        elif mean_cos < 0.99:
            marker = " <-- diverging"

        print(f"  {l:>6} {layer_type:>6}  {mean_cos:>10.6f}  {min_cos:>10.6f}  {mean_l2:>10.4f}{marker}")

    # Entropy comparison
    mp_ent = [p["results"]["mp"]["entropy"] for p in MATCHED_PAIRS]
    mt_ent = [p["results"]["mt"]["entropy"] for p in MATCHED_PAIRS]
    print("\n  Mean logit entropy:")
    print(f"    MP: {sum(mp_ent)/len(mp_ent):.3f}")
    print(f"    MT: {sum(mt_ent)/len(mt_ent):.3f}")
    print("    (Higher = less certain about next token)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for pair in MATCHED_PAIRS:
            entry = {"domain": pair["domain"]}
            for form in ["mp", "mt"]:
                r = pair["results"][form]
                entry[form] = {
                    "correct": r["correct"],
                    "response": r["response"][:500],
                    "norms": r["norms"],
                    "entropy": r["entropy"],
                    "top_tokens": [(t, p) for t, p in r["top_tokens"]],
                }
            serializable.append(entry)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
