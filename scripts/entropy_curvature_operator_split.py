#!/usr/bin/env python3
"""CR-EC-001: Entropy Operator Split — Logit vs Attention Entropy.

Collects BOTH entropy operators (logit entropy via Entropy-Lens, attention
weight entropy via manual QK^T extraction) alongside sublayer curvature
decomposition, then runs architecture-conditioned falsifier tests.

This resolves the open question from the SOTA audit (ACT-016): which entropy
operator drives the r=0.507 correlation with angular curvature?

Measurements per layer:
    1. H_logit: logit entropy (project h_l through unembedding, softmax, Shannon H)
    1b. H_logit_norm: normalized logit entropy (apply RMSNorm before unembedding, removing ||h||² confound)
    2. H_attn: attention weight entropy (Shannon H of softmax(QK^T/sqrt(d_k)))
    3. theta_core: angular change h_in -> h_post_core (core operator = attention or conv)
    4. theta_mlp: angular change h_post_core -> h_out
    5. theta_total: angular change h_in -> h_out
    6. G_mlp: MLP angular gain = theta_mlp / theta_core
    7. E_core, E_mlp, E_mix: update-energy decomposition
       ||delta_total||^2 = ||delta_core||^2 + ||delta_mlp||^2 + 2<delta_core, delta_mlp>

Analysis:
    - Per-model Spearman correlations for both operators vs all curvature measures
    - Per-model sign table (H_logit and H_attn columns)
    - Depth-controlled partial correlations
    - Falsifier table: F1, F3 (LFM2-qualified), F5

Usage:
    poetry run python scripts/entropy_curvature_operator_split.py
    poetry run python scripts/entropy_curvature_operator_split.py --smoke
    poetry run python scripts/entropy_curvature_operator_split.py --models LFM2-700M
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_BASE = os.environ.get("MC_MODELS_BASE", "/Volumes/CodeCypher/models")


# ---------------------------------------------------------------------------
# Detection floor utilities (Fisher-SE MDE + Bretherton autocorrelation)
# ---------------------------------------------------------------------------


def _residualize_ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y after removing linear effect of x."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def _effective_df(residuals: np.ndarray) -> tuple[float, float]:
    """Effective degrees of freedom under AR(1) autocorrelation.

    Bretherton et al. (1999) Eq. 31: n_eff = n * (1 - rho_1) / (1 + rho_1).
    """
    n = len(residuals)
    if n < 4:
        return float(n), 0.0
    r = np.asarray(residuals, dtype=float)
    r_centered = r - np.mean(r)
    denom = np.sum(r_centered ** 2)
    if abs(denom) < 1e-15:
        return float(n), 0.0
    rho_1 = float(np.sum(r_centered[:-1] * r_centered[1:]) / denom)
    rho_1_clamped = max(-0.99, min(0.99, rho_1))
    n_eff = n * (1 - rho_1_clamped) / (1 + rho_1_clamped)
    # Floor at 4 (minimum for partial correlation), ceiling at n (cannot
    # have more independent observations than physical layers).
    n_eff = max(4.0, min(float(n), n_eff))
    return n_eff, rho_1


def _minimum_detectable_effect(n_eff: float, n_controls: int = 1) -> float:
    """Fisher-SE MDE: smallest |r| with SNR >= 1 at given effective df."""
    df = n_eff - n_controls - 2
    if df <= 0:
        return 1.0
    return float(np.tanh(1.0 / np.sqrt(df)))


MODEL_REGISTRY = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "L": 16, "d": 1024,
        "architecture": "lfm2",
        "gqa_ratio": 2,
    },
    "LFM2-700M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-700M-bf16",
        "L": 16, "d": 1280,
        "architecture": "lfm2",
        "gqa_ratio": 3,
    },
    "Qwen3.5-0.8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-0.8B-bf16",
        "L": 24, "d": 1024,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,
    },
    "Qwen2.5-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "L": 36, "d": 2048,
        "architecture": "qwen2.5",
        "gqa_ratio": 8,
    },
    "Qwen3.5-4B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-bf16",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,
    },
    "Llama-3.2-3B": {
        "path": f"{MODELS_BASE}/mlx-community/Llama-3.2-3B-Instruct-bf16",
        "L": 28, "d": 3072,
        "architecture": "llama",
        "gqa_ratio": 3,
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        "L": 36, "d": 4096,
        "architecture": "qwen3",
        "gqa_ratio": 4,
    },
    "Qwen3.5-2B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-2B-bf16",
        "L": 24, "d": 2048,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,
    },
    "Qwen3.5-4B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-bf16",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,
    },
    "Qwen3.5-4B-4bit": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3.5-4B-4bit-g64",
        "L": 32, "d": 2560,
        "architecture": "qwen3.5",
        "gqa_ratio": 4,
    },
    "Mistral-7B": {
        "path": f"{MODELS_BASE}/mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "L": 32, "d": 4096,
        "architecture": "mistral",
        "gqa_ratio": 4,
    },
}

# Same probes as entropy_curvature_verification.py for cross-validation
PROBES = [
    # Retrieval
    "The capital of France is",
    "Who wrote Romeo and Juliet?",
    "The chemical symbol for water is",
    "The largest planet in our solar system is",
    "The first president of the United States was",
    "The boiling point of water at sea level is",
    # Arithmetic
    "What is 347 + 528?",
    "What is 15 * 23?",
    "What is 1024 / 16?",
    "What is 99 - 37?",
    "What is 8 * 7 + 13?",
    "What is 256 + 384 - 100?",
    # Reasoning
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "There are 48 people on a bus. At the first stop, 8 get off and 5 get on. How many now?",
    "A lily pad doubles in size every day. It takes 48 days to cover the lake. When is it half covered?",
    "If 5 machines make 5 widgets in 5 minutes, how long for 100 machines to make 100 widgets?",
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    # Creative
    "Write a haiku about the ocean.",
    "Describe a sunset over the mountains in one vivid sentence.",
    "Write a short poem about the passage of time.",
    "Describe the taste of your favorite food using only three words.",
    "Write a one-sentence story with a twist ending.",
    "Describe the sound of rain on a tin roof.",
    # Narrative
    "Once upon a time in a faraway kingdom, there lived a",
    "The old lighthouse keeper watched the storm approach from",
    "In the year 2150, humanity had finally achieved",
    "She opened the letter and read the first line:",
    "The forest was silent except for the sound of",
    "He had been walking for three days when he finally saw",
]


# =============================================================================
# Core measurements (reused from entropy_curvature_verification.py)
# =============================================================================


def angular_change(v1: np.ndarray, v2: np.ndarray) -> float:
    """Geodesic distance on unit sphere: arccos(cosine_similarity)."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_sim = np.dot(v1, v2) / (n1 * n2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(np.arccos(cos_sim))


def shannon_entropy(weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy H = -sum(p log(p)), numerically stable."""
    eps = 1e-10
    w = np.clip(weights, eps, 1.0)
    return -np.sum(w * np.log(w), axis=axis)


# =============================================================================
# Backbone resolution (from entropy_curvature_verification.py)
# =============================================================================


def _resolve_backbone(model):
    """Resolve model backbone to the level with embed_tokens and layers."""
    base = getattr(model, "model", None)
    if base is not None:
        if getattr(base, "layers", None) is not None and getattr(base, "embed_tokens", None) is not None:
            return base
        lm = getattr(base, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if inner is not None and getattr(inner, "layers", None) is not None:
                return inner
            if getattr(lm, "layers", None) is not None:
                return lm
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and getattr(inner, "layers", None) is not None:
            return inner
        if getattr(lm, "layers", None) is not None:
            return lm
    return model


def _get_head_config(model, attn_module):
    """Get n_heads, n_kv_heads, head_dim from module or model config."""
    n_heads = getattr(attn_module, "n_heads", None)
    n_kv_heads = getattr(attn_module, "n_kv_heads", None)

    if n_heads is not None and n_kv_heads is not None:
        q_out = attn_module.q_proj.weight.shape[0]
        head_dim = q_out // n_heads
        return n_heads, n_kv_heads, head_dim

    head_dim = None
    for norm_name in ("q_norm", "q_layernorm"):
        qn = getattr(attn_module, norm_name, None)
        if qn is not None and hasattr(qn, "weight"):
            head_dim = qn.weight.shape[0]
            break

    q_out = attn_module.q_proj.weight.shape[0]
    k_out = attn_module.k_proj.weight.shape[0]

    if head_dim is not None:
        n_heads = q_out // head_dim
        n_kv_heads = k_out // head_dim
        return n_heads, n_kv_heads, head_dim

    args = getattr(model, "args", getattr(model, "config", None))
    if args is not None:
        tc = getattr(args, "text_config", None)
        if isinstance(tc, dict):
            n_heads = tc.get("num_attention_heads")
            n_kv_heads = tc.get("num_key_value_heads", n_heads)
            head_dim = tc.get("head_dim")
        else:
            n_heads = getattr(args, "num_attention_heads", None)
            n_kv_heads = getattr(args, "num_key_value_heads", n_heads)
            head_dim = getattr(args, "head_dim", None)

    if n_heads is not None and head_dim is None:
        head_dim = q_out // n_heads

    if n_heads is None:
        return None, None, None

    return n_heads, n_kv_heads or n_heads, head_dim


def _get_pre_norm(layer):
    """Find the pre-attention normalization layer."""
    for name in ("input_layernorm", "operator_norm", "attention_norm"):
        norm = getattr(layer, name, None)
        if norm is not None:
            return norm
    return None


def _call_with_fallback(module, x, mask=None):
    """Call module with tolerant signature fallback."""
    try:
        if mask is not None:
            return module(x, mask=mask)
        return module(x)
    except Exception:
        pass
    try:
        if mask is not None:
            return module(x, mask=mask, cache=None)
        return module(x, cache=None)
    except Exception:
        pass
    try:
        if mask is not None:
            return module(x, mask)
        return module(x)
    except Exception:
        pass
    return module(x)


def _is_full_attention_layer(layer) -> bool:
    """Check if a layer has standard softmax attention."""
    if hasattr(layer, "is_attention_layer"):
        return layer.is_attention_layer
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return False
    return hasattr(attn, "q_proj") and hasattr(attn, "k_proj")


# =============================================================================
# Attention entropy (from entropy_curvature_verification.py)
# =============================================================================


def compute_attention_entropy(layer, h_in, seq_len: int, model=None):
    """Compute Shannon entropy of attention weights for one layer."""
    import mlx.core as mx

    if not _is_full_attention_layer(layer):
        return None

    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None

    pre_norm = _get_pre_norm(layer)
    if pre_norm is None:
        return None

    n_heads, n_kv_heads, head_dim = _get_head_config(model, attn)
    if n_heads is None:
        return None

    try:
        normed = pre_norm(h_in)
        q = attn.q_proj(normed)
        k = attn.k_proj(normed)

        B, L, _ = q.shape

        q = q.reshape(B, L, n_heads, head_dim)
        k_head_dim = k.shape[-1] // n_kv_heads
        k = k.reshape(B, L, n_kv_heads, k_head_dim)

        for qn_name in ("q_layernorm", "q_norm"):
            qn = getattr(attn, qn_name, None)
            if qn is not None:
                q = qn(q)
                break
        for kn_name in ("k_layernorm", "k_norm"):
            kn = getattr(attn, kn_name, None)
            if kn is not None:
                k = kn(k)
                break

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)

        rope_fn = getattr(attn, "rope", getattr(attn, "rotary_emb", None))
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)

        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)

        scale = getattr(attn, "scale", 1.0 / math.sqrt(k_head_dim))
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale

        causal = mx.triu(mx.full((L, L), -1e9, dtype=scores.dtype), k=1)
        scores = scores + causal

        weights = mx.softmax(scores, axis=-1)
        mx.eval(weights)

        weights_np = np.array(weights[0].tolist(), dtype=np.float32)
        per_head_per_pos = shannon_entropy(weights_np, axis=-1)
        per_head = np.mean(per_head_per_pos, axis=-1)
        layer_entropy = float(np.mean(per_head))

        return layer_entropy

    except Exception as exc:
        logger.debug("Attention entropy extraction failed: %s", exc)
        return None


# =============================================================================
# Logit entropy (via LayerEntropyProjector)
# =============================================================================


def collect_logit_entropy(model, tokenizer, prompts: list[str], num_layers: int, backend):
    """Collect per-layer logit entropy using Entropy-Lens (unembedding projection).

    Returns dict with:
        H_logit: list of per-layer mean unnormalized logit entropy (original operator)
        H_logit_norm: list of per-layer mean normalized logit entropy (RMSNorm applied first)

    Each list has length num_layers; entries are None if measurement failed.

    The normalized variant applies the model's final norm (RMSNorm) before projecting
    through the unembedding, matching what the model actually computes at inference.
    This removes the ||h||² confound where softmax sharpness scales with hidden state norm.
    """
    import mlx.core as mx
    from modelcypher.core.domain.entropy.layer_entropy_projector import LayerEntropyProjector

    # Resolve backbone once; used by both projector setup and manual fallback paths.
    base = _resolve_backbone(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone for logit entropy")

    # Resolve final norm for H_logit_norm (matches model's actual readout path).
    # Pattern from curvature_accumulation_analysis.py:418-419.
    final_norm = getattr(base, "embedding_norm", None) or getattr(base, "norm", None)
    if final_norm is None:
        logger.warning("No final norm found on backbone; H_logit_norm will equal H_logit")

    def _resolve_output_head():
        """Find a callable output projection module for direct logit projection."""
        candidates = []
        if hasattr(model, "lm_head"):
            candidates.append(getattr(model, "lm_head"))
        if hasattr(model, "model") and hasattr(model.model, "lm_head"):
            candidates.append(getattr(model.model, "lm_head"))
        if hasattr(base, "lm_head"):
            candidates.append(getattr(base, "lm_head"))
        if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
            candidates.append(getattr(base.language_model, "lm_head"))

        for head in candidates:
            if callable(head):
                return head
        return None

    def _entropy_from_logits(logits):
        """Stable Shannon entropy from logits tensor."""
        logits = logits.astype(mx.float32)
        logits = logits - mx.max(logits, axis=-1, keepdims=True)
        probs = mx.softmax(logits, axis=-1)
        probs = probs + 1e-12
        entropy = -mx.sum(probs * mx.log(probs), axis=-1)
        mx.eval(entropy)
        return float(np.array(entropy.tolist(), dtype=np.float32).reshape(-1)[0])

    projector = LayerEntropyProjector(backend=backend)
    output_head = _resolve_output_head()
    use_output_head_projection = False
    fallback_logged = False

    # Try standard set_unembedding_matrix first
    try:
        projector.set_unembedding_matrix(model)
    except ValueError:
        # Qwen3.5: model.model.language_model.embed_tokens
        # Resolve backbone and manually set unembedding from embed_tokens
        if embed is None:
            raise RuntimeError("Cannot find embed_tokens on resolved backbone")
        weight = embed.weight
        # Quantized embeddings store packed weights; dequantize to full [vocab, hidden].
        if hasattr(embed, "scales") and hasattr(embed, "bits"):
            weight = mx.dequantize(weight, embed.scales, embed.biases, embed.group_size, embed.bits)
        projector._unembedding_matrix = backend.astype(weight, "float32")
        projector._vocab_size = weight.shape[0]
        projector._hidden_dim = weight.shape[1]
        projector._unembedding_source = "embed_tokens_transposed"
        logger.info("Using resolved backbone embed_tokens: vocab=%d, hidden=%d",
                     projector._vocab_size, projector._hidden_dim)

    layer_entropies = [[] for _ in range(num_layers)]
    layer_entropies_norm = [[] for _ in range(num_layers)]

    for pi, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)

        try:
            mask = backend.create_causal_mask(input_ids.shape[1], hidden.dtype)
        except Exception:
            mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = mask

            try:
                hidden = layer(hidden, mask=layer_mask)
            except (TypeError, ValueError):
                try:
                    hidden = layer(hidden, layer_mask)
                except (TypeError, ValueError):
                    hidden = layer(hidden)

            if isinstance(hidden, tuple):
                hidden = hidden[0]

            # Compute logit entropy at this layer
            h_last = hidden[:, -1, :].astype(mx.float32)
            mx.eval(h_last)
            if (
                not use_output_head_projection
                and output_head is not None
                and projector._hidden_dim not in (0, int(h_last.shape[-1]))
            ):
                # Quantized heads may expose packed weights (e.g., 4-bit), so
                # projector hidden_dim can mismatch true hidden state width.
                use_output_head_projection = True
                if not fallback_logged:
                    logger.warning(
                        "Unembedding hidden_dim=%d mismatches hidden state dim=%d; "
                        "falling back to direct output-head projection for logit entropy.",
                        projector._hidden_dim,
                        int(h_last.shape[-1]),
                    )
                    fallback_logged = True

            if use_output_head_projection:
                logits = output_head(h_last)
                if isinstance(logits, tuple):
                    logits = logits[0]
                entropy = _entropy_from_logits(logits)
            else:
                try:
                    entropy, _ = projector.compute_layer_entropy(h_last)
                except ValueError as exc:
                    if output_head is None:
                        raise
                    use_output_head_projection = True
                    if not fallback_logged:
                        logger.warning(
                            "LayerEntropyProjector projection failed (%s); "
                            "falling back to direct output-head projection.",
                            exc,
                        )
                        fallback_logged = True
                    logits = output_head(h_last)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    entropy = _entropy_from_logits(logits)
            layer_entropies[i].append(entropy)

            # Compute normalized logit entropy (apply final RMSNorm before unembedding)
            if final_norm is not None:
                h_normed = final_norm(h_last)
                mx.eval(h_normed)
                if use_output_head_projection and output_head is not None:
                    logits_norm = output_head(h_normed)
                    if isinstance(logits_norm, tuple):
                        logits_norm = logits_norm[0]
                    entropy_norm = _entropy_from_logits(logits_norm)
                else:
                    # Direct unembedding projection on normalized hidden state
                    W = projector._unembedding_matrix
                    logits_norm = h_normed @ W.T
                    entropy_norm = _entropy_from_logits(logits_norm)
                layer_entropies_norm[i].append(entropy_norm)
            else:
                layer_entropies_norm[i].append(entropy)

        mx.eval(hidden)

    result_logit = []
    result_logit_norm = []
    for i in range(num_layers):
        if layer_entropies[i]:
            result_logit.append(float(np.mean(layer_entropies[i])))
        else:
            result_logit.append(None)
        if layer_entropies_norm[i]:
            result_logit_norm.append(float(np.mean(layer_entropies_norm[i])))
        else:
            result_logit_norm.append(None)

    return {"H_logit": result_logit, "H_logit_norm": result_logit_norm}


# =============================================================================
# Combined data collection
# =============================================================================


def collect_all_data(
    model, tokenizer, prompts: list[str], num_layers: int, backend,
) -> dict:
    """Collect sublayer activations, attention entropy, and logit entropy.

    Returns dict with:
        sublayer_data: list of per-layer dicts with h_in, h_post_attn, h_out, attn_entropy
        logit_entropy: list of per-layer mean logit entropy (unnormalized)
        logit_entropy_norm: list of per-layer mean logit entropy (RMSNorm applied)
    """
    import mlx.core as mx

    base = _resolve_backbone(model)
    embed = getattr(base, "embed_tokens", None)
    layers = getattr(base, "layers", None)
    if layers is None or embed is None:
        raise RuntimeError("Cannot resolve model backbone layers")

    # --- Sublayer activations + attention entropy ---
    layer_h_in = [[] for _ in range(num_layers)]
    layer_h_post_core = [[] for _ in range(num_layers)]
    layer_h_out = [[] for _ in range(num_layers)]
    layer_attn_entropy = [[] for _ in range(num_layers)]
    layer_core_operator = [[] for _ in range(num_layers)]

    for pi, prompt in enumerate(prompts):
        if pi % 10 == 0:
            logger.info("  Probe %d/%d (sublayer collection)", pi + 1, len(prompts))

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        hidden = embed(input_ids)
        seq_len = input_ids.shape[1]

        try:
            numeric_mask = backend.create_causal_mask(seq_len, hidden.dtype)
        except Exception:
            numeric_mask = None

        for i, layer in enumerate(layers):
            if i >= num_layers:
                break

            h_in = hidden
            if hasattr(layer, "is_attention_layer"):
                layer_mask = "causal" if layer.is_attention_layer else None
            else:
                layer_mask = numeric_mask

            h_post_core = None
            core_operator = None
            input_norm = _get_pre_norm(layer)
            self_attn = getattr(layer, "self_attn", None)
            conv = getattr(layer, "conv", None)
            post_attn_norm = getattr(layer, "post_attention_layernorm", None)
            if post_attn_norm is None:
                post_attn_norm = getattr(layer, "ffn_norm", None)
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                mlp = getattr(layer, "feed_forward", None)

            # 1) Attention-core decomposition
            if input_norm is not None and self_attn is not None and mlp is not None:
                try:
                    normed = input_norm(h_in)
                    attn_out = _call_with_fallback(self_attn, normed, mask=layer_mask)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    h_post_core = h_in + attn_out
                    core_operator = "attention"

                    normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
                    mlp_out = mlp(normed2)
                    h_out = h_post_core + mlp_out
                except Exception:
                    h_post_core = None
                    core_operator = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            # 2) Conv-core decomposition (hybrid/ShortConv blocks)
            elif input_norm is not None and conv is not None and mlp is not None:
                try:
                    normed = input_norm(h_in)
                    conv_out = _call_with_fallback(conv, normed, mask=layer_mask)
                    if isinstance(conv_out, tuple):
                        conv_out = conv_out[0]
                    h_post_core = h_in + conv_out
                    core_operator = "conv"

                    normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
                    mlp_out = mlp(normed2)
                    h_out = h_post_core + mlp_out
                except Exception:
                    h_post_core = None
                    core_operator = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            # 3) Linear-attention-core decomposition (GatedDeltaNet in Qwen3.5)
            elif input_norm is not None and getattr(layer, "linear_attn", None) is not None and mlp is not None:
                try:
                    normed = input_norm(h_in)
                    la_out = _call_with_fallback(layer.linear_attn, normed, mask=layer_mask)
                    if isinstance(la_out, tuple):
                        la_out = la_out[0]
                    h_post_core = h_in + la_out
                    core_operator = "linear_attn"

                    normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
                    mlp_out = mlp(normed2)
                    h_out = h_post_core + mlp_out
                except Exception:
                    h_post_core = None
                    core_operator = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            # 4) Identity-core decomposition (layers without explicit core operator)
            elif input_norm is not None and mlp is not None:
                try:
                    h_post_core = h_in
                    core_operator = "identity"
                    normed2 = post_attn_norm(h_post_core) if post_attn_norm else h_post_core
                    mlp_out = mlp(normed2)
                    if isinstance(mlp_out, tuple):
                        mlp_out = mlp_out[0]
                    h_out = h_post_core + mlp_out
                except Exception:
                    h_post_core = None
                    core_operator = None
                    try:
                        h_out = layer(h_in, mask=layer_mask)
                    except (TypeError, ValueError):
                        try:
                            h_out = layer(h_in, layer_mask)
                        except (TypeError, ValueError):
                            h_out = layer(h_in)
            else:
                try:
                    h_out = layer(h_in, mask=layer_mask)
                except (TypeError, ValueError):
                    try:
                        h_out = layer(h_in, layer_mask)
                    except (TypeError, ValueError):
                        h_out = layer(h_in)

            # Last-token representations
            h_in_last = h_in[:, -1, :].astype(mx.float32)
            h_out_last = h_out[:, -1, :].astype(mx.float32)
            mx.eval(h_in_last, h_out_last)

            layer_h_in[i].append(np.array(h_in_last[0].tolist(), dtype=np.float32))
            layer_h_out[i].append(np.array(h_out_last[0].tolist(), dtype=np.float32))

            if h_post_core is not None:
                h_pc_last = h_post_core[:, -1, :].astype(mx.float32)
                mx.eval(h_pc_last)
                layer_h_post_core[i].append(np.array(h_pc_last[0].tolist(), dtype=np.float32))
                layer_core_operator[i].append(core_operator)
            else:
                layer_h_post_core[i].append(None)
                layer_core_operator[i].append(None)

            # Attention entropy
            ent = compute_attention_entropy(layer, h_in, seq_len, model=model)
            layer_attn_entropy[i].append(ent)

            hidden = h_out

        mx.eval(hidden)

    # Build sublayer result
    sublayer_data = []
    for i in range(num_layers):
        valid_core_idx = [k for k, x in enumerate(layer_h_post_core[i]) if x is not None]
        has_decomp = len(valid_core_idx) > 0
        valid_ent = [e for e in layer_attn_entropy[i] if e is not None]
        has_attn_entropy = len(valid_ent) > 0
        valid_ops = [op for op in layer_core_operator[i] if op is not None]
        core_operator = valid_ops[0] if valid_ops and len(set(valid_ops)) == 1 else (
            "mixed" if valid_ops else None
        )
        core_operator_counts = {}
        for op in valid_ops:
            core_operator_counts[op] = core_operator_counts.get(op, 0) + 1

        sublayer_data.append({
            "h_in": np.stack(layer_h_in[i]),
            "h_out": np.stack(layer_h_out[i]),
            "h_in_core": (
                np.stack([layer_h_in[i][k] for k in valid_core_idx])
                if has_decomp else None
            ),
            "h_post_core": (
                np.stack([layer_h_post_core[i][k] for k in valid_core_idx])
                if has_decomp else None
            ),
            "h_out_core": (
                np.stack([layer_h_out[i][k] for k in valid_core_idx])
                if has_decomp else None
            ),
            "has_decomposition": has_decomp,
            "decomp_probe_fraction": (
                float(len(valid_core_idx) / len(layer_h_in[i]))
                if layer_h_in[i] else 0.0
            ),
            "attn_entropy": float(np.mean(valid_ent)) if has_attn_entropy else None,
            "is_attention_layer": has_attn_entropy,
            "core_operator": core_operator,
            "core_operator_counts": core_operator_counts,
        })

    # --- Logit entropy ---
    logger.info("  Collecting logit entropy (Entropy-Lens)...")
    logit_result = collect_logit_entropy(model, tokenizer, prompts, num_layers, backend)

    return {
        "sublayer_data": sublayer_data,
        "logit_entropy": logit_result["H_logit"],
        "logit_entropy_norm": logit_result["H_logit_norm"],
    }


# =============================================================================
# Per-layer measurements
# =============================================================================


def compute_measurements(data: dict, num_layers: int) -> list[dict]:
    """Compute per-layer curvature + both entropy operators."""
    sublayer_data = data["sublayer_data"]
    logit_entropy = data["logit_entropy"]
    logit_entropy_norm = data.get("logit_entropy_norm", [None] * num_layers)

    measurements = []
    for i in range(num_layers):
        sd = sublayer_data[i]
        h_in = sd["h_in"]
        h_out = sd["h_out"]
        n_probes = h_in.shape[0]

        total_angles = [angular_change(h_in[j], h_out[j]) for j in range(n_probes)]
        theta_total = float(np.mean(total_angles))
        h_in_norm_sq = float(np.mean(np.sum(h_in * h_in, axis=1)))
        h_out_norm_sq = float(np.mean(np.sum(h_out * h_out, axis=1)))
        delta_total = h_out - h_in
        E_total = float(np.mean(np.sum(delta_total * delta_total, axis=1)))

        layer_result = {
            "layer_idx": i,
            "depth_fraction": i / max(num_layers - 1, 1),
            "theta_total": theta_total,
            "theta_core": None,
            "theta_attn": None,
            "theta_conv": None,
            "theta_mlp": None,
            "theta_mlp_post_core": None,
            "core_operator": sd.get("core_operator"),
            "attn_fraction": None,
            "core_fraction": None,
            "G_mlp": None,
            "h_in_norm_sq": h_in_norm_sq,
            "h_out_norm_sq": h_out_norm_sq,
            "E_total": E_total,
            "E_total_decomp": None,
            "E_core": None,
            "E_mlp": None,
            "E_mix": None,
            "E_closure_error": None,
            "H_logit": logit_entropy[i],
            "H_logit_norm": logit_entropy_norm[i] if i < len(logit_entropy_norm) else None,
            "H_attn": sd["attn_entropy"],
            "is_attention_layer": sd["is_attention_layer"],
            "decomp_probe_fraction": sd.get("decomp_probe_fraction"),
        }

        if sd["has_decomposition"]:
            h_in_core = sd.get("h_in_core", h_in)
            h_post_core = sd["h_post_core"]
            h_out_core = sd.get("h_out_core", h_out)
            n_core = h_post_core.shape[0]
            core_angles = [angular_change(h_in_core[j], h_post_core[j]) for j in range(n_core)]
            mlp_angles = [angular_change(h_post_core[j], h_out_core[j]) for j in range(n_core)]
            delta_core = h_post_core - h_in_core
            delta_mlp = h_out_core - h_post_core
            delta_total_core = h_out_core - h_in_core
            E_core = float(np.mean(np.sum(delta_core * delta_core, axis=1)))
            E_mlp = float(np.mean(np.sum(delta_mlp * delta_mlp, axis=1)))
            E_mix = float(np.mean(2.0 * np.sum(delta_core * delta_mlp, axis=1)))
            E_total_decomp = float(np.mean(np.sum(delta_total_core * delta_total_core, axis=1)))
            E_closure_error = E_total_decomp - (E_core + E_mlp + E_mix)

            theta_core = float(np.mean(core_angles))
            theta_mlp = float(np.mean(mlp_angles))

            layer_result["theta_core"] = theta_core
            layer_result["theta_mlp"] = theta_mlp
            layer_result["theta_mlp_post_core"] = theta_mlp
            layer_result["core_fraction"] = (
                theta_core / (theta_core + theta_mlp)
                if (theta_core + theta_mlp) > 1e-10 else float("nan")
            )
            layer_result["G_mlp"] = (
                theta_mlp / theta_core if theta_core > 1e-10 else float("nan")
            )
            layer_result["E_total_decomp"] = E_total_decomp
            layer_result["E_core"] = E_core
            layer_result["E_mlp"] = E_mlp
            layer_result["E_mix"] = E_mix
            layer_result["E_closure_error"] = E_closure_error
            if sd.get("core_operator") == "attention":
                layer_result["theta_attn"] = theta_core
                layer_result["attn_fraction"] = layer_result["core_fraction"]
            elif sd.get("core_operator") == "conv":
                layer_result["theta_conv"] = theta_core

        measurements.append(layer_result)

    return measurements


# =============================================================================
# Correlation analysis
# =============================================================================


def safe_spearman(x, y):
    """Spearman correlation with NaN handling."""
    from scipy import stats
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    try:
        r, p = stats.spearmanr(x_arr[mask], y_arr[mask])
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def compute_operator_correlations(measurements: list[dict]) -> dict:
    """Compute correlations for both entropy operators vs all curvature measures.

    Returns sign table + correlation details for each operator.
    """
    # All layers with H_logit
    logit_layers = [m for m in measurements if m["H_logit"] is not None]
    # Attention layers only (have H_attn + decomposition)
    attn_layers = [
        m for m in measurements
        if m["H_attn"] is not None
        and m["theta_attn"] is not None
    ]

    result = {
        "n_logit_layers": len(logit_layers),
        "n_attn_layers": len(attn_layers),
    }

    # --- Logit entropy correlations (all layers) ---
    if len(logit_layers) >= 4:
        H_logit = [m["H_logit"] for m in logit_layers]
        theta_total_l = [m["theta_total"] for m in logit_layers]
        h_out_norm_sq_l = [m["h_out_norm_sq"] for m in logit_layers]
        E_total_l = [m["E_total"] for m in logit_layers]

        r, p = safe_spearman(H_logit, theta_total_l)
        result["H_logit_vs_theta_total"] = {"r": r, "p": p}
        r_hn, p_hn = safe_spearman(H_logit, h_out_norm_sq_l)
        result["H_logit_vs_h_out_norm_sq"] = {"r": r_hn, "p": p_hn}
        r_et, p_et = safe_spearman(H_logit, E_total_l)
        result["H_logit_vs_E_total"] = {"r": r_et, "p": p_et}

        # Core decomposition subset (attention or conv core)
        logit_decomp = [m for m in logit_layers if m["theta_core"] is not None]
        if len(logit_decomp) >= 4:
            H_ld = [m["H_logit"] for m in logit_decomp]
            theta_core_ld = [m["theta_core"] for m in logit_decomp]
            theta_mlp_ld = [m["theta_mlp"] for m in logit_decomp]
            G_mlp_ld = [m["G_mlp"] for m in logit_decomp]
            E_core_ld = [m["E_core"] for m in logit_decomp]
            E_mlp_ld = [m["E_mlp"] for m in logit_decomp]
            E_mix_ld = [m["E_mix"] for m in logit_decomp]
            E_total_decomp_ld = [m["E_total_decomp"] for m in logit_decomp]

            r_lc, p_lc = safe_spearman(H_ld, theta_core_ld)
            r_lm, p_lm = safe_spearman(H_ld, theta_mlp_ld)
            r_lg, p_lg = safe_spearman(H_ld, G_mlp_ld)
            r_ec, p_ec = safe_spearman(H_ld, E_core_ld)
            r_em, p_em = safe_spearman(H_ld, E_mlp_ld)
            r_ex, p_ex = safe_spearman(H_ld, E_mix_ld)
            r_ed, p_ed = safe_spearman(H_ld, E_total_decomp_ld)

            result["H_logit_vs_theta_core"] = {"r": r_lc, "p": p_lc}
            result["H_logit_vs_theta_mlp"] = {"r": r_lm, "p": p_lm}
            result["H_logit_vs_G_mlp"] = {"r": r_lg, "p": p_lg}
            result["H_logit_vs_E_core"] = {"r": r_ec, "p": p_ec}
            result["H_logit_vs_E_mlp"] = {"r": r_em, "p": p_em}
            result["H_logit_vs_E_mix"] = {"r": r_ex, "p": p_ex}
            result["H_logit_vs_E_total_decomp"] = {"r": r_ed, "p": p_ed}
            result["core_decomp_coverage"] = float(len(logit_decomp) / len(logit_layers))

            # Legacy attention-only correlation for continuity with prior artifacts.
            logit_attn_only = [m for m in logit_decomp if m["theta_attn"] is not None]
            if len(logit_attn_only) >= 4:
                H_la = [m["H_logit"] for m in logit_attn_only]
                theta_attn_la = [m["theta_attn"] for m in logit_attn_only]
                r_la, p_la = safe_spearman(H_la, theta_attn_la)
                result["H_logit_vs_theta_attn"] = {"r": r_la, "p": p_la}

    # --- Normalized logit entropy correlations (all layers with H_logit_norm) ---
    norm_layers = [m for m in measurements if m.get("H_logit_norm") is not None]
    result["n_norm_layers"] = len(norm_layers)
    if len(norm_layers) >= 4:
        H_ln = [m["H_logit_norm"] for m in norm_layers]
        theta_total_n = [m["theta_total"] for m in norm_layers]
        h_out_norm_sq_n = [m["h_out_norm_sq"] for m in norm_layers]
        E_total_n = [m["E_total"] for m in norm_layers]

        r_nt, p_nt = safe_spearman(H_ln, theta_total_n)
        result["H_logit_norm_vs_theta_total"] = {"r": r_nt, "p": p_nt}
        r_nn, p_nn = safe_spearman(H_ln, h_out_norm_sq_n)
        result["H_logit_norm_vs_h_out_norm_sq"] = {"r": r_nn, "p": p_nn}
        r_ne, p_ne = safe_spearman(H_ln, E_total_n)
        result["H_logit_norm_vs_E_total"] = {"r": r_ne, "p": p_ne}

        # Cross-check: H_logit_norm vs H_logit (prediction 3: should be < 0.9)
        H_logit_for_cross = [m["H_logit"] for m in norm_layers if m["H_logit"] is not None]
        H_norm_for_cross = [m["H_logit_norm"] for m in norm_layers if m["H_logit"] is not None]
        if len(H_logit_for_cross) >= 4:
            r_cross_norm, p_cross_norm = safe_spearman(H_logit_for_cross, H_norm_for_cross)
            result["H_logit_norm_vs_H_logit"] = {"r": r_cross_norm, "p": p_cross_norm}

        # Core decomposition subset
        norm_decomp = [m for m in norm_layers if m["theta_core"] is not None]
        if len(norm_decomp) >= 4:
            H_nd = [m["H_logit_norm"] for m in norm_decomp]
            r_nc, p_nc = safe_spearman(H_nd, [m["theta_core"] for m in norm_decomp])
            r_nm, p_nm = safe_spearman(H_nd, [m["theta_mlp"] for m in norm_decomp])
            r_ng, p_ng = safe_spearman(H_nd, [m["G_mlp"] for m in norm_decomp])
            r_nec, p_nec = safe_spearman(H_nd, [m["E_core"] for m in norm_decomp])
            r_nem, p_nem = safe_spearman(H_nd, [m["E_mlp"] for m in norm_decomp])
            r_nex, p_nex = safe_spearman(H_nd, [m["E_mix"] for m in norm_decomp])

            result["H_logit_norm_vs_theta_core"] = {"r": r_nc, "p": p_nc}
            result["H_logit_norm_vs_theta_mlp"] = {"r": r_nm, "p": p_nm}
            result["H_logit_norm_vs_G_mlp"] = {"r": r_ng, "p": p_ng}
            result["H_logit_norm_vs_E_core"] = {"r": r_nec, "p": p_nec}
            result["H_logit_norm_vs_E_mlp"] = {"r": r_nem, "p": p_nem}
            result["H_logit_norm_vs_E_mix"] = {"r": r_nex, "p": p_nex}

    # --- Attention entropy correlations (attention layers only) ---
    if len(attn_layers) >= 4:
        H_attn = [m["H_attn"] for m in attn_layers]
        theta_total_a = [m["theta_total"] for m in attn_layers]
        theta_attn_a = [m["theta_attn"] for m in attn_layers]
        theta_mlp_a = [m["theta_mlp"] for m in attn_layers]
        G_mlp_a = [m["G_mlp"] for m in attn_layers]

        r_at, p_at = safe_spearman(H_attn, theta_total_a)
        r_aa, p_aa = safe_spearman(H_attn, theta_attn_a)
        r_am, p_am = safe_spearman(H_attn, theta_mlp_a)
        r_ag, p_ag = safe_spearman(H_attn, G_mlp_a)

        result["H_attn_vs_theta_total"] = {"r": r_at, "p": p_at}
        result["H_attn_vs_theta_attn"] = {"r": r_aa, "p": p_aa}
        result["H_attn_vs_theta_mlp"] = {"r": r_am, "p": p_am}
        result["H_attn_vs_G_mlp"] = {"r": r_ag, "p": p_ag}

    # --- Cross-operator correlation ---
    cross_layers = [
        m for m in measurements
        if m["H_logit"] is not None and m["H_attn"] is not None
    ]
    if len(cross_layers) >= 4:
        H_logit_c = [m["H_logit"] for m in cross_layers]
        H_attn_c = [m["H_attn"] for m in cross_layers]
        r_cross, p_cross = safe_spearman(H_logit_c, H_attn_c)
        result["H_logit_vs_H_attn"] = {"r": r_cross, "p": p_cross}

    return result


def compute_depth_controlled_correlations(measurements: list[dict]) -> dict:
    """Partial correlations controlling for depth fraction.

    Uses rank-based partial correlation:
        r(X,Y|Z) = (r_XY - r_XZ * r_YZ) / sqrt((1 - r_XZ^2)(1 - r_YZ^2))
    """
    result = {}

    def depth_controlled_ols(x, y, depth):
        """OLS on depth-residualized variables."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        d_arr = np.asarray(depth, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(d_arr)
        if mask.sum() < 4:
            return {
                "slope": float("nan"),
                "intercept": float("nan"),
                "r_value": float("nan"),
                "p_value": float("nan"),
                "std_err": float("nan"),
                "r_squared": float("nan"),
                "n": int(mask.sum()),
            }
        x_res = _residualize_ols(x_arr[mask], d_arr[mask])
        y_res = _residualize_ols(y_arr[mask], d_arr[mask])
        if np.std(x_res) < 1e-15 or np.std(y_res) < 1e-15:
            return {
                "slope": float("nan"),
                "intercept": float("nan"),
                "r_value": float("nan"),
                "p_value": float("nan"),
                "std_err": float("nan"),
                "r_squared": float("nan"),
                "n": int(mask.sum()),
            }
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x_res, y_res)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "r_squared": float(r_value * r_value),
            "n": int(mask.sum()),
        }

    def partial_spearman(x, y, z):
        """Partial Spearman r(X,Y|Z)."""
        r_xy, _ = safe_spearman(x, y)
        r_xz, _ = safe_spearman(x, z)
        r_yz, _ = safe_spearman(y, z)
        if any(math.isnan(v) for v in [r_xy, r_xz, r_yz]):
            return float("nan")
        denom = math.sqrt(max(0, (1 - r_xz**2)) * max(0, (1 - r_yz**2)))
        if denom < 1e-10:
            return float("nan")
        return (r_xy - r_xz * r_yz) / denom

    # H_logit vs theta_total | depth
    logit_layers = [m for m in measurements if m["H_logit"] is not None]
    if len(logit_layers) >= 4:
        H_logit = [m["H_logit"] for m in logit_layers]
        theta_total = [m["theta_total"] for m in logit_layers]
        h_out_norm_sq = [m["h_out_norm_sq"] for m in logit_layers]
        E_total = [m["E_total"] for m in logit_layers]
        depth = [m["depth_fraction"] for m in logit_layers]

        result["partial_H_logit_vs_theta_total_given_depth"] = partial_spearman(
            H_logit, theta_total, depth
        )
        result["ols_H_logit_to_theta_total_given_depth"] = depth_controlled_ols(
            H_logit, theta_total, depth,
        )
        result["ols_H_logit_to_h_out_norm_sq_given_depth"] = depth_controlled_ols(
            H_logit, h_out_norm_sq, depth,
        )
        result["ols_H_logit_to_E_total_given_depth"] = depth_controlled_ols(
            H_logit, E_total, depth,
        )

        decomp_layers = [
            m for m in logit_layers
            if m.get("E_core") is not None
            and m.get("E_mlp") is not None
            and m.get("E_mix") is not None
        ]
        if len(decomp_layers) >= 4:
            H_decomp = [m["H_logit"] for m in decomp_layers]
            depth_decomp = [m["depth_fraction"] for m in decomp_layers]
            result["ols_H_logit_to_E_core_given_depth"] = depth_controlled_ols(
                H_decomp, [m["E_core"] for m in decomp_layers], depth_decomp,
            )
            result["ols_H_logit_to_E_mlp_given_depth"] = depth_controlled_ols(
                H_decomp, [m["E_mlp"] for m in decomp_layers], depth_decomp,
            )
            result["ols_H_logit_to_E_mix_given_depth"] = depth_controlled_ols(
                H_decomp, [m["E_mix"] for m in decomp_layers], depth_decomp,
            )
            result["ols_H_logit_to_E_total_decomp_given_depth"] = depth_controlled_ols(
                H_decomp, [m["E_total_decomp"] for m in decomp_layers], depth_decomp,
            )

            by_operator = {}
            operators = sorted({m.get("core_operator") for m in decomp_layers if m.get("core_operator")})
            for op in operators:
                op_rows = [m for m in decomp_layers if m.get("core_operator") == op]
                if len(op_rows) < 4:
                    continue
                H_op = [m["H_logit"] for m in op_rows]
                depth_op = [m["depth_fraction"] for m in op_rows]
                by_operator[op] = {
                    "ols_H_logit_to_h_out_norm_sq_given_depth": depth_controlled_ols(
                        H_op, [m["h_out_norm_sq"] for m in op_rows], depth_op,
                    ),
                    "ols_H_logit_to_E_total_decomp_given_depth": depth_controlled_ols(
                        H_op, [m["E_total_decomp"] for m in op_rows], depth_op,
                    ),
                    "ols_H_logit_to_E_mix_given_depth": depth_controlled_ols(
                        H_op, [m["E_mix"] for m in op_rows], depth_op,
                    ),
                }
            if by_operator:
                result["ols_H_logit_by_core_operator"] = by_operator

    # H_logit_norm vs theta_total | depth (norm-corrected)
    norm_layers = [m for m in measurements if m.get("H_logit_norm") is not None]
    if len(norm_layers) >= 4:
        H_logit_norm = [m["H_logit_norm"] for m in norm_layers]
        theta_total_n = [m["theta_total"] for m in norm_layers]
        h_out_norm_sq_n = [m["h_out_norm_sq"] for m in norm_layers]
        E_total_n = [m["E_total"] for m in norm_layers]
        depth_n = [m["depth_fraction"] for m in norm_layers]

        result["partial_H_logit_norm_vs_theta_total_given_depth"] = partial_spearman(
            H_logit_norm, theta_total_n, depth_n
        )
        result["ols_H_logit_norm_to_theta_total_given_depth"] = depth_controlled_ols(
            H_logit_norm, theta_total_n, depth_n,
        )
        result["ols_H_logit_norm_to_h_out_norm_sq_given_depth"] = depth_controlled_ols(
            H_logit_norm, h_out_norm_sq_n, depth_n,
        )
        result["ols_H_logit_norm_to_E_total_given_depth"] = depth_controlled_ols(
            H_logit_norm, E_total_n, depth_n,
        )

    # H_attn vs theta_total | depth (attention layers only)
    attn_layers = [m for m in measurements if m["H_attn"] is not None]
    if len(attn_layers) >= 4:
        H_attn = [m["H_attn"] for m in attn_layers]
        theta_total = [m["theta_total"] for m in attn_layers]
        depth = [m["depth_fraction"] for m in attn_layers]

        result["partial_H_attn_vs_theta_total_given_depth"] = partial_spearman(
            H_attn, theta_total, depth
        )

    return result


# =============================================================================
# Falsifier tests (from entropy-curvature-derivation.md)
# =============================================================================


def compute_falsifier_table(all_model_results: list[dict]) -> dict:
    """Run falsifier tests F1, F2, F3 (LFM2-qualified), F5.

    F1: Sign of H_logit -> theta_attn^2 coefficient (depth-controlled), per family
    F2: Geometry-conditioned effect corr(theta_total, E_mix | H_logit, depth)
    F3: |corr(H_logit, theta_attn)| > |corr(H_logit, theta_mlp)| — LFM2-qualified
    F5: Same sign of H_logit coefficient across families
    """
    falsifiers = {}

    # --- F1: H_logit -> theta_attn relationship (per model) ---
    f1_results = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        corrs = mr["correlations"]
        r_la = corrs.get("H_logit_vs_theta_attn", {}).get("r", float("nan"))
        f1_results[model_name] = {
            "r_H_logit_theta_attn": r_la,
            "sign": "positive" if r_la > 0 else ("negative" if r_la < 0 else "zero"),
            "pass": not math.isnan(r_la) and r_la >= 0,  # F1 predicts non-negative slope
        }
    f1_pass_count = sum(1 for v in f1_results.values() if v["pass"])
    falsifiers["F1_logit_theta_attn_sign"] = {
        "per_model": f1_results,
        "pass_count": f1_pass_count,
        "total": len(f1_results),
        "status": "PASS" if f1_pass_count == len(f1_results) else "FAIL",
    }

    # --- F2: Geometry-conditioned effect via E_mix at fixed H_logit + depth ---
    # Prediction: curvature varies with geometry proxy E_mix after entropy matching.
    f2_results = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        rows = []
        for m_row in mr.get("measurements", []):
            h_v = m_row.get("H_logit")
            e_mix_v = m_row.get("E_mix")
            th_v = m_row.get("theta_total")
            d_v = m_row.get("depth_fraction")
            if (
                h_v is not None and e_mix_v is not None and th_v is not None and d_v is not None
                and math.isfinite(h_v) and math.isfinite(e_mix_v)
                and math.isfinite(th_v) and math.isfinite(d_v)
            ):
                rows.append((float(h_v), float(e_mix_v), float(th_v), float(d_v)))

        if len(rows) < 6:
            f2_results[model_name] = {
                "status": "INSUFFICIENT_DATA",
                "n": len(rows),
            }
            continue

        h = np.array([r[0] for r in rows], dtype=float)
        e_mix = np.array([r[1] for r in rows], dtype=float)
        th = np.array([r[2] for r in rows], dtype=float)
        depth = np.array([r[3] for r in rows], dtype=float)

        X = np.column_stack([np.ones(len(rows)), h, depth])
        th_res = th - X @ np.linalg.lstsq(X, th, rcond=None)[0]
        e_mix_res = e_mix - X @ np.linalg.lstsq(X, e_mix, rcond=None)[0]

        if np.std(th_res) < 1e-15 or np.std(e_mix_res) < 1e-15:
            f2_results[model_name] = {
                "status": "DEGENERATE_RESIDUALS",
                "n": len(rows),
            }
            continue

        pearson_r = float(np.corrcoef(e_mix_res, th_res)[0, 1])
        rho_s, p_s = sp_stats.spearmanr(e_mix_res, th_res)
        rho_s = float(rho_s)
        p_s = float(p_s)

        n_eff_em, rho1_em = _effective_df(e_mix_res)
        n_eff_th, rho1_th = _effective_df(th_res)
        if abs(rho1_em) >= abs(rho1_th):
            n_eff_used, rho1_used, rho_src = n_eff_em, rho1_em, "E_mix_residuals"
        else:
            n_eff_used, rho1_used, rho_src = n_eff_th, rho1_th, "theta_total_residuals"
        mde = _minimum_detectable_effect(n_eff_used, n_controls=2)
        obs_abs_r = abs(pearson_r) if math.isfinite(pearson_r) else 0.0
        resolvable = obs_abs_r > mde

        if math.isfinite(pearson_r) and len(rows) > 4:
            df = len(rows) - 4  # two controls (H_logit, depth) + intercept
            denom_sq = max(1e-15, 1 - pearson_r ** 2)
            t_stat = pearson_r * math.sqrt(df) / math.sqrt(denom_sq)
            p_val = float(2 * sp_stats.t.sf(abs(t_stat), df))
        else:
            p_val = float("nan")

        f2_results[model_name] = {
            "status": "OK",
            "n": len(rows),
            "pearson_partial_r": pearson_r,
            "pearson_partial_p_approx": p_val,
            "spearman_partial_r": rho_s,
            "spearman_partial_p": p_s,
            "detection_floor": {
                "n_raw": len(rows),
                "rho_1_used": rho1_used,
                "rho_1_source": rho_src,
                "n_eff": n_eff_used,
                "mde": mde,
                "observed_abs_r": obs_abs_r,
                "resolvable": resolvable,
            },
        }

    f2_testable = [k for k, v in f2_results.items() if v.get("status") == "OK"]
    f2_detected = [
        k for k in f2_testable
        if f2_results[k].get("detection_floor", {}).get("resolvable", False)
    ]
    if not f2_testable:
        f2_status = "INCONCLUSIVE"
    elif f2_detected:
        f2_status = "PASS"
    else:
        f2_status = "BELOW_MEASUREMENT_RESOLUTION"

    falsifiers["F2_geometry_conditioned_E_mix"] = {
        "test": "corr(theta_total, E_mix | H_logit, depth) with Fisher-SE MDE floor",
        "per_model": f2_results,
        "testable_models": f2_testable,
        "geometry_effect_detected_models": f2_detected,
        "status": f2_status,
    }

    # --- F3: |corr(H_logit, theta_attn)| > |corr(H_logit, theta_mlp)| (LFM2-qualified) ---
    f3_results = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        corrs = mr["correlations"]
        r_la = corrs.get("H_logit_vs_theta_attn", {}).get("r", float("nan"))
        r_lm = corrs.get("H_logit_vs_theta_mlp", {}).get("r", float("nan"))
        if math.isnan(r_la) or math.isnan(r_lm):
            f3_results[model_name] = {
                "abs_r_logit_attn": float("nan"),
                "abs_r_logit_mlp": float("nan"),
                "attn_dominates": None,
                "qualified": mr["architecture"] == "lfm2",
            }
        else:
            f3_results[model_name] = {
                "abs_r_logit_attn": abs(r_la),
                "abs_r_logit_mlp": abs(r_lm),
                "attn_dominates": abs(r_la) > abs(r_lm),
                "qualified": mr["architecture"] == "lfm2",
            }

    # F3 status: LFM2 must pass; others are informational
    lfm2_f3 = [v for v in f3_results.values() if v["qualified"]]
    f3_lfm2_pass = all(v.get("attn_dominates", False) for v in lfm2_f3) if lfm2_f3 else None
    falsifiers["F3_attn_dominates_lfm2_qualified"] = {
        "per_model": f3_results,
        "lfm2_pass": f3_lfm2_pass,
        "status": "PASS" if f3_lfm2_pass else ("FAIL" if f3_lfm2_pass is False else "INCONCLUSIVE"),
    }

    # --- F5: Depth-controlled sign consistency across families ---
    # The raw Spearman sign of ρ(H_logit, θ_total) is confounded by depth:
    # both H_logit and θ_total trend with depth, creating spurious positive
    # correlation.  The depth-controlled partial correlation removes this
    # confound and reveals the true coupling sign.
    #
    # Detection floor: Fisher-SE MDE with Bretherton (1999) autocorrelation
    # correction. The MDE is the measurement resolution of the partial
    # correlation estimator — not an imposed threshold.
    f5_depth = {}
    f5_raw = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        corrs = mr["correlations"]
        dc = mr["depth_controlled"]

        # Raw Spearman (for confound documentation, NOT the test)
        r_raw = corrs.get("H_logit_vs_theta_total", {}).get("r", float("nan"))
        p_raw = corrs.get("H_logit_vs_theta_total", {}).get("p", float("nan"))
        f5_raw[model_name] = {
            "r": r_raw,
            "p": p_raw,
            "sign": (
                ("positive" if r_raw > 0 else "negative")
                if not math.isnan(r_raw) else "unknown"
            ),
        }

        # Depth-controlled partial Spearman
        partial_r = dc.get("partial_H_logit_vs_theta_total_given_depth", float("nan"))
        n = mr["num_layers"]

        # Compute detection floor from per-layer measurements.
        measurements = mr.get("measurements", [])
        h_vals, th_vals, d_vals = [], [], []
        for m_row in measurements:
            h_v = m_row.get("H_logit")
            th_v = m_row.get("theta_total")
            d_v = m_row.get("depth_fraction")
            if h_v is not None and th_v is not None and d_v is not None:
                if math.isfinite(h_v) and math.isfinite(th_v) and math.isfinite(d_v):
                    h_vals.append(h_v)
                    th_vals.append(th_v)
                    d_vals.append(d_v)

        det_floor = {}
        if len(h_vals) >= 6:
            h_arr = np.array(h_vals)
            th_arr = np.array(th_vals)
            d_arr = np.array(d_vals)
            h_resid = _residualize_ols(h_arr, d_arr)
            th_resid = _residualize_ols(th_arr, d_arr)

            n_eff_h, rho1_h = _effective_df(h_resid)
            n_eff_th, rho1_th = _effective_df(th_resid)
            if abs(rho1_h) >= abs(rho1_th):
                n_eff_used, rho1_used = n_eff_h, rho1_h
            else:
                n_eff_used, rho1_used = n_eff_th, rho1_th

            mde = _minimum_detectable_effect(n_eff_used)
            obs_abs_r = abs(partial_r) if not math.isnan(partial_r) else 0.0
            resolvable = obs_abs_r > mde

            det_floor = {
                "n_raw": len(h_vals),
                "rho_1_used": rho1_used,
                "n_eff": n_eff_used,
                "mde": mde,
                "observed_abs_r": obs_abs_r,
                "resolvable": resolvable,
            }

        if not math.isnan(partial_r) and n >= 6:
            # Approximate p-value: t = r*sqrt(n-3) / sqrt(1-r^2), df = n-3
            df = n - 3
            denom_sq = max(1e-15, 1 - partial_r ** 2)
            t_stat = partial_r * math.sqrt(df) / math.sqrt(denom_sq)
            p_val = float(2 * sp_stats.t.sf(abs(t_stat), df))
            f5_depth[model_name] = {
                "partial_r": partial_r,
                "approx_p": p_val,
                "n": n,
                "sign": "positive" if partial_r > 0 else "negative",
                "detection_floor": det_floor,
            }
        else:
            f5_depth[model_name] = {
                "partial_r": float("nan") if math.isnan(partial_r) else partial_r,
                "approx_p": float("nan"),
                "n": n,
                "sign": "unknown",
                "detection_floor": det_floor,
            }

    # --- Detection-floor-based sign consistency ---
    resolvable_models = [
        k for k, v in f5_depth.items()
        if v.get("detection_floor", {}).get("resolvable", False)
    ]
    below_floor_models = [
        k for k, v in f5_depth.items()
        if not v.get("detection_floor", {}).get("resolvable", False)
    ]
    resolvable_signs = [
        f5_depth[k]["sign"] for k in resolvable_models
        if f5_depth[k]["sign"] != "unknown"
    ]
    all_signs = [v["sign"] for v in f5_depth.values() if v["sign"] != "unknown"]

    if len(resolvable_signs) == 0:
        f5_status = "BELOW_MEASUREMENT_RESOLUTION"
    elif len(set(resolvable_signs)) == 1:
        f5_status = "CONSISTENT_SIGN"
    else:
        f5_status = "SIGN_DISAGREEMENT"

    falsifiers["F5_same_sign_across_families"] = {
        "test": "depth-controlled partial Spearman with Fisher-SE MDE detection floor",
        "per_model_depth_controlled": f5_depth,
        "per_model_raw_spearman": f5_raw,
        "all_signs": all_signs,
        "resolvable_models": resolvable_models,
        "below_floor_models": below_floor_models,
        "resolvable_signs": resolvable_signs,
        "threshold_status": "DERIVED",
        "threshold_derivation": (
            "Fisher-SE MDE with Bretherton (1999) autocorrelation correction. "
            "MDE = tanh(1/sqrt(n_eff - 3)), n_eff = n*(1-rho1)/(1+rho1)."
        ),
        "status": f5_status,
    }

    # --- F_norm: Norm confound validation ---
    # Prediction 1: r(H_logit_norm, ||h||²) ≈ 0 (|r| < 0.3)
    # Prediction 3: r(H_logit_norm, H_logit) < 0.9 (normalization is non-trivial)
    f_norm_results = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        corrs = mr["correlations"]

        r_norm_vs_normsq = corrs.get("H_logit_norm_vs_h_out_norm_sq", {}).get("r", float("nan"))
        r_norm_vs_logit = corrs.get("H_logit_norm_vs_H_logit", {}).get("r", float("nan"))
        r_logit_vs_normsq = corrs.get("H_logit_vs_h_out_norm_sq", {}).get("r", float("nan"))

        pred1_pass = not math.isnan(r_norm_vs_normsq) and abs(r_norm_vs_normsq) < 0.3
        pred3_pass = not math.isnan(r_norm_vs_logit) and r_norm_vs_logit < 0.9

        f_norm_results[model_name] = {
            "r_H_logit_norm_vs_h_out_norm_sq": r_norm_vs_normsq,
            "r_H_logit_norm_vs_H_logit": r_norm_vs_logit,
            "r_H_logit_vs_h_out_norm_sq": r_logit_vs_normsq,
            "prediction_1_pass": pred1_pass,
            "prediction_3_pass": pred3_pass,
        }

    p1_pass_count = sum(1 for v in f_norm_results.values() if v["prediction_1_pass"])
    p3_pass_count = sum(1 for v in f_norm_results.values() if v["prediction_3_pass"])
    falsifiers["F_norm_confound_validation"] = {
        "test": "r(H_logit_norm, ||h||²) ≈ 0 (|r| < 0.3) AND r(H_logit_norm, H_logit) < 0.9",
        "per_model": f_norm_results,
        "prediction_1_pass_count": p1_pass_count,
        "prediction_3_pass_count": p3_pass_count,
        "total": len(f_norm_results),
        "status": (
            "PASS" if p1_pass_count == len(f_norm_results) and p3_pass_count == len(f_norm_results)
            else "PARTIAL" if p1_pass_count > 0 or p3_pass_count > 0
            else "FAIL"
        ),
    }

    # --- F5_norm: Depth-controlled sign consistency for H_logit_norm ---
    f5_norm_depth = {}
    for mr in all_model_results:
        model_name = mr["model_name"]
        dc = mr["depth_controlled"]
        partial_r = dc.get("partial_H_logit_norm_vs_theta_total_given_depth", float("nan"))
        n = mr["num_layers"]

        if not math.isnan(partial_r) and n >= 6:
            df = n - 3
            denom_sq = max(1e-15, 1 - partial_r ** 2)
            t_stat = partial_r * math.sqrt(df) / math.sqrt(denom_sq)
            p_val = float(2 * sp_stats.t.sf(abs(t_stat), df))
            f5_norm_depth[model_name] = {
                "partial_r": partial_r,
                "approx_p": p_val,
                "n": n,
                "sign": "positive" if partial_r > 0 else "negative",
            }
        else:
            f5_norm_depth[model_name] = {
                "partial_r": float("nan") if math.isnan(partial_r) else partial_r,
                "approx_p": float("nan"),
                "n": n,
                "sign": "unknown",
            }

    norm_signs = [v["sign"] for v in f5_norm_depth.values() if v["sign"] != "unknown"]
    if len(norm_signs) == 0:
        f5_norm_status = "INCONCLUSIVE"
    elif len(set(norm_signs)) == 1:
        f5_norm_status = "CONSISTENT_SIGN"
    else:
        f5_norm_status = "SIGN_DISAGREEMENT"

    falsifiers["F5_norm_sign_across_families"] = {
        "test": "depth-controlled partial Spearman for H_logit_norm vs theta_total",
        "per_model": f5_norm_depth,
        "signs": norm_signs,
        "status": f5_norm_status,
    }

    return falsifiers


# =============================================================================
# Model runner
# =============================================================================


def run_single_model(
    model_name: str, model_info: dict, probes: list[str], backend,
) -> dict:
    """Run operator-split analysis for one model."""
    logger.info("Loading model: %s from %s", model_name, model_info["path"])
    model, tokenizer = backend.load_model(model_info["path"])

    base = _resolve_backbone(model)
    layers = getattr(base, "layers", None)
    num_layers = len(layers) if layers else 0

    logger.info("Model loaded: %d layers, d=%d", num_layers, model_info["d"])

    t0 = time.time()
    data = collect_all_data(model, tokenizer, probes, num_layers, backend)
    logger.info("Data collection: %.1fs", time.time() - t0)

    measurements = compute_measurements(data, num_layers)
    correlations = compute_operator_correlations(measurements)
    depth_controlled = compute_depth_controlled_correlations(measurements)

    # Log key results
    for key, val in correlations.items():
        if isinstance(val, dict) and "r" in val:
            logger.info("  %s: r=%.3f, p=%.3f", key, val["r"], val["p"])

    for key, val in depth_controlled.items():
        if isinstance(val, (int, float)):
            logger.info("  %s: %.3f", key, val)
        elif isinstance(val, dict):
            if "r_value" in val and "r_squared" in val:
                logger.info(
                    "  %s: r=%.3f, r2=%.3f, p=%.3f, n=%d",
                    key,
                    val.get("r_value", float("nan")),
                    val.get("r_squared", float("nan")),
                    val.get("p_value", float("nan")),
                    val.get("n", 0),
                )
            else:
                logger.info("  %s: %s", key, json.dumps(val, default=str))
        else:
            logger.info("  %s: %s", key, val)

    del model, tokenizer, data
    gc.collect()

    return {
        "model_name": model_name,
        "architecture": model_info["architecture"],
        "gqa_ratio": model_info.get("gqa_ratio", 1),
        "num_layers": num_layers,
        "d_model": model_info["d"],
        "n_probes": len(probes),
        "measurements": measurements,
        "correlations": correlations,
        "depth_controlled": depth_controlled,
    }


# =============================================================================
# Cross-model summary
# =============================================================================


def build_cross_model_summary(all_results: list[dict]) -> dict:
    """Build the sign table and cross-model summary."""
    sign_table = {}
    for r in all_results:
        model = r["model_name"]
        corrs = r["correlations"]
        sign_table[model] = {
            "r_H_logit_theta_core": corrs.get("H_logit_vs_theta_core", {}).get("r"),
            "r_H_logit_theta_attn": corrs.get("H_logit_vs_theta_attn", {}).get("r"),
            "r_H_logit_theta_mlp": corrs.get("H_logit_vs_theta_mlp", {}).get("r"),
            "r_H_attn_theta_attn": corrs.get("H_attn_vs_theta_attn", {}).get("r"),
            "r_H_attn_theta_mlp": corrs.get("H_attn_vs_theta_mlp", {}).get("r"),
            "r_H_logit_theta_total": corrs.get("H_logit_vs_theta_total", {}).get("r"),
            "r_H_attn_theta_total": corrs.get("H_attn_vs_theta_total", {}).get("r"),
            "r_H_logit_vs_H_attn": corrs.get("H_logit_vs_H_attn", {}).get("r"),
            "r_H_logit_h_out_norm_sq": corrs.get("H_logit_vs_h_out_norm_sq", {}).get("r"),
            "r_H_logit_E_core": corrs.get("H_logit_vs_E_core", {}).get("r"),
            "r_H_logit_E_mlp": corrs.get("H_logit_vs_E_mlp", {}).get("r"),
            "r_H_logit_E_mix": corrs.get("H_logit_vs_E_mix", {}).get("r"),
            "r_H_logit_norm_theta_total": corrs.get("H_logit_norm_vs_theta_total", {}).get("r"),
            "r_H_logit_norm_h_out_norm_sq": corrs.get("H_logit_norm_vs_h_out_norm_sq", {}).get("r"),
            "r_H_logit_norm_E_total": corrs.get("H_logit_norm_vs_E_total", {}).get("r"),
            "r_H_logit_norm_vs_H_logit": corrs.get("H_logit_norm_vs_H_logit", {}).get("r"),
            "core_decomp_coverage": corrs.get("core_decomp_coverage"),
        }

    depth_summary = {}
    for r in all_results:
        model = r["model_name"]
        dc = r["depth_controlled"]
        depth_summary[model] = dc

    # Which operator has stronger depth-controlled correlation?
    operator_comparison = {}
    for r in all_results:
        model = r["model_name"]
        dc = r["depth_controlled"]
        p_logit = dc.get("partial_H_logit_vs_theta_total_given_depth", float("nan"))
        p_attn = dc.get("partial_H_attn_vs_theta_total_given_depth", float("nan"))
        p_norm = dc.get("partial_H_logit_norm_vs_theta_total_given_depth", float("nan"))
        entry = {}
        if not math.isnan(p_logit) and not math.isnan(p_attn):
            winner = "H_logit" if abs(p_logit) > abs(p_attn) else "H_attn"
            entry["partial_r_logit"] = p_logit
            entry["partial_r_attn"] = p_attn
            entry["stronger_operator"] = winner
        if not math.isnan(p_norm):
            entry["partial_r_logit_norm"] = p_norm
        if entry:
            operator_comparison[model] = entry

    return {
        "sign_table": sign_table,
        "depth_controlled": depth_summary,
        "operator_comparison": operator_comparison,
    }


# =============================================================================
# Main
# =============================================================================


def run_experiment(args: argparse.Namespace) -> None:
    """Run entropy operator split experiment."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    if args.smoke:
        model_names = ["LFM2-700M"]
        probes = PROBES[:6]
    elif args.models:
        model_names = args.models
        probes = PROBES
    else:
        model_names = list(MODEL_REGISTRY.keys())
        probes = PROBES

    logger.info("CR-EC-001 Operator Split: %d models, %d probes", len(model_names), len(probes))

    all_results = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning("Unknown model: %s, skipping", model_name)
            continue
        result = run_single_model(model_name, MODEL_REGISTRY[model_name], probes, backend)
        all_results.append(result)
        gc.collect()

    # Cross-model analysis
    cross_summary = build_cross_model_summary(all_results)
    falsifier_table = compute_falsifier_table(all_results)

    # Print summary
    logger.info("\n%s", "=" * 70)
    logger.info("CR-EC-001 OPERATOR SPLIT SUMMARY")
    logger.info("%s", "=" * 70)

    logger.info("\nSign table:")
    for model, signs in cross_summary["sign_table"].items():
        logger.info("  %s:", model)
        for k, v in signs.items():
            if v is not None:
                logger.info("    %s: %.3f", k, v)

    logger.info("\nDepth-controlled partial correlations:")
    for model, dc in cross_summary["depth_controlled"].items():
        logger.info("  %s:", model)
        for k, v in dc.items():
            if isinstance(v, (int, float)):
                logger.info("    %s: %.3f", k, v)
            elif isinstance(v, dict):
                if "r_value" in v and "r_squared" in v:
                    logger.info(
                        "    %s: r=%.3f, r2=%.3f, p=%.3f, n=%d",
                        k,
                        v.get("r_value", float("nan")),
                        v.get("r_squared", float("nan")),
                        v.get("p_value", float("nan")),
                        v.get("n", 0),
                    )
                else:
                    logger.info("    %s: %s", k, json.dumps(v, default=str))
            else:
                logger.info("    %s: %s", k, v)

    logger.info("\nOperator comparison:")
    for model, comp in cross_summary.get("operator_comparison", {}).items():
        logger.info("  %s: stronger = %s (partial_r: logit=%.3f, attn=%.3f)",
                     model, comp["stronger_operator"],
                     comp["partial_r_logit"], comp["partial_r_attn"])

    logger.info("\nFalsifier table:")
    for fname, fdata in falsifier_table.items():
        logger.info("  %s: %s", fname, fdata.get("status", "?"))

    # Save per-model results
    output_base = Path(args.output)
    for r in all_results:
        model_dir = output_base / r["model_name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "operator_split.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        logger.info("Saved: %s", model_dir / "operator_split.json")

    # Save cross-model summary
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "cross_model_summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2, default=str)

    with open(output_base / "falsifier_outcomes.json", "w") as f:
        json.dump(falsifier_table, f, indent=2, default=str)

    logger.info("\nAll results saved to %s", output_base)


def main():
    parser = argparse.ArgumentParser(
        description="CR-EC-001: Entropy Operator Split (Logit vs Attention)"
    )
    parser.add_argument(
        "--output", default="results/entropy_curvature_operator_split/",
        help="Output directory",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific models to test",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test (1 model, 6 probes)",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
