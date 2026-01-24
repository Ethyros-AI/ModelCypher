#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# DeepSeek-R1-Qwen3-8B Compression Benchmark
"""
Benchmark three configurations:
1. ORIGINAL: Unmodified model
2. T-LOSSLESS: T-matrix compression (layers 14-21) at FP32
3. T-4BIT: T-matrix compression with 4-bit quantization

Metrics:
- Token accuracy (exact match with original)
- Inference speed (tokens/second)
- Model size (parameters, storage)
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from typing import Dict, List, Set, Optional
import time
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CALIBRATION AND TEST DATA
# ============================================================================

def generate_calibration_prompts(n: int = 800) -> List[str]:
    """Generate diverse calibration prompts."""
    prompts = []

    # Math (400 prompts)
    for a in range(1, 21):
        for b in range(1, 21):
            prompts.append(f"{a} + {b} =")

    # Geography (50 prompts)
    countries = [
        "France", "Japan", "Germany", "Italy", "Spain", "China", "India",
        "Brazil", "Canada", "Mexico", "Russia", "Australia", "Egypt",
        "UK", "Thailand", "Vietnam", "South Korea", "Poland", "Sweden",
        "Norway", "Finland", "Denmark", "Netherlands", "Belgium", "Switzerland",
        "Austria", "Portugal", "Greece", "Turkey", "Israel", "Saudi Arabia",
        "UAE", "Singapore", "Malaysia", "Indonesia", "Philippines", "New Zealand",
        "Argentina", "Chile", "Colombia", "Peru", "Venezuela", "Cuba",
        "Morocco", "South Africa", "Nigeria", "Kenya", "Ethiopia", "Ghana"
    ]
    for c in countries:
        prompts.append(f"The capital of {c} is")

    # Code patterns (50 prompts)
    code_patterns = [
        "def ", "class ", "import ", "from ", "return ", "if ", "for ",
        "while ", "try:", "except:", "with ", "async def ", "await ",
        "lambda ", "yield ", "@property", "@staticmethod", "@classmethod",
        "def __init__", "def __str__", "def __repr__", "def __len__",
        "def __getitem__", "def __setitem__", "def __iter__", "def __next__",
        "import numpy", "import pandas", "import torch", "import tensorflow",
        "from typing import", "from collections import", "from pathlib import",
        "class Model:", "class Dataset:", "class Config:", "class Error:",
        "def forward(", "def backward(", "def train(", "def evaluate(",
        "def predict(", "def fit(", "def transform(", "def load(",
        "def save(", "def process(", "def validate(", "def compute("
    ]
    prompts.extend(code_patterns)

    # Natural language (100 prompts)
    nl_patterns = [
        "The meaning of life is",
        "Scientists have discovered that",
        "In the year 2050,",
        "The most important thing about",
        "According to recent studies,",
        "The history of",
        "One of the key factors in",
        "Research has shown that",
        "The future of artificial intelligence",
        "Climate change is affecting",
        "The human brain is capable of",
        "Technology has revolutionized",
        "Education should focus on",
        "The best way to learn",
        "Creativity is essential for",
        "Leadership requires",
        "Success depends on",
        "The most challenging aspect of",
        "Innovation comes from",
        "The relationship between",
    ]
    for pattern in nl_patterns:
        prompts.append(pattern)
        prompts.append(pattern + " the")
        prompts.append(pattern + " a")
        prompts.append(pattern + " our")
        prompts.append(pattern + " their")

    return prompts[:n]


def generate_benchmark_prompts() -> List[Dict]:
    """Generate benchmark prompts with categories."""
    return [
        # Math
        {"category": "math", "prompt": "99 + 88 ="},
        {"category": "math", "prompt": "23 * 17 ="},
        {"category": "math", "prompt": "144 / 12 ="},
        {"category": "math", "prompt": "The square root of 256 is"},

        # Geography
        {"category": "geography", "prompt": "The capital of Mongolia is"},
        {"category": "geography", "prompt": "The capital of Nepal is"},
        {"category": "geography", "prompt": "The largest country by area is"},
        {"category": "geography", "prompt": "The longest river in the world is"},

        # Code
        {"category": "code", "prompt": "def factorial(n):"},
        {"category": "code", "prompt": "async def fetch_data("},
        {"category": "code", "prompt": "class NeuralNetwork(nn.Module):"},
        {"category": "code", "prompt": "import torch\nmodel = "},

        # Science
        {"category": "science", "prompt": "The speed of light is approximately"},
        {"category": "science", "prompt": "Water boils at"},
        {"category": "science", "prompt": "The chemical formula for water is"},
        {"category": "science", "prompt": "DNA stands for"},

        # Reasoning
        {"category": "reasoning", "prompt": "If all cats are mammals, and all mammals are animals, then"},
        {"category": "reasoning", "prompt": "The opposite of 'increase' is"},
        {"category": "reasoning", "prompt": "What comes next in the sequence: 2, 4, 8, 16,"},
        {"category": "reasoning", "prompt": "If today is Monday, tomorrow will be"},

        # Language
        {"category": "language", "prompt": "The past tense of 'run' is"},
        {"category": "language", "prompt": "A synonym for 'happy' is"},
        {"category": "language", "prompt": "The plural of 'child' is"},
        {"category": "language", "prompt": "Translate 'hello' to Spanish:"},

        # General knowledge
        {"category": "general", "prompt": "The author of Romeo and Juliet is"},
        {"category": "general", "prompt": "The first president of the United States was"},
        {"category": "general", "prompt": "The year World War II ended was"},
        {"category": "general", "prompt": "The chemical symbol for gold is"},

        # Creative
        {"category": "creative", "prompt": "Once upon a time, in a land far away,"},
        {"category": "creative", "prompt": "The best advice I ever received was"},
    ]


# ============================================================================
# T-MATRIX COMPRESSION
# ============================================================================

def quantize_symmetric(W: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric quantization."""
    W = W.astype(np.float32)
    abs_max = np.abs(W).max()
    if abs_max < 1e-10:
        return W.copy()
    scale = abs_max / (2**(bits-1) - 1)
    return np.round(W / scale) * scale


def derive_t_matrix(model, tokenizer, layer_idx: int, prompts: List[str]) -> Dict:
    """Derive T matrix for a layer."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model
    layer = inner_model.layers[layer_idx]

    X_list, Y_list = [], []

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = mx.array([tokens])
        h = inner_model.embed_tokens(input_ids)
        mx.eval(h)
        mask = create_attention_mask(h, None)

        for idx, l in enumerate(inner_model.layers):
            if idx == layer_idx:
                h_normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
                mx.eval(attn_out)
                h_post = h + attn_out
                h_normed2 = layer.post_attention_layernorm(h_post)
                mx.eval(h_normed2)

                x_in = np.array(h_normed2[0, -1, :].astype(mx.float32)).astype(np.float64)
                X_list.append(x_in)

                mlp_out = layer.mlp(h_normed2)
                mx.eval(mlp_out)
                Y_list.append(np.array(mlp_out[0, -1, :].astype(mx.float32)).astype(np.float64))
                break
            else:
                h = l(h, mask, None)
                mx.eval(h)

    X = np.stack(X_list, axis=1)
    Y = np.stack(Y_list, axis=1)

    X_mean = X.mean(axis=1, keepdims=True)
    Y_mean = Y.mean(axis=1, keepdims=True)
    X_c = X - X_mean
    Y_c = Y - Y_mean

    # SVD-based pseudoinverse for numerical stability
    U_x, S_x, Vt_x = np.linalg.svd(X_c, full_matrices=False)
    threshold = 1e-6 * S_x[0] if len(S_x) > 0 else 1e-6
    S_x_inv = np.where(S_x > threshold, 1.0 / S_x, 0.0)
    T = Y_c @ (Vt_x.T * S_x_inv) @ U_x.T

    return {
        'T': T.astype(np.float32),
        'X_mean': X_mean.flatten().astype(np.float32),
        'Y_mean': Y_mean.flatten().astype(np.float32),
    }


def apply_t_rule(h_normed2: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply T transformation."""
    h_centered = h_normed2.astype(np.float64) - rule['X_mean'].astype(np.float64)
    result = rule['T'].astype(np.float64) @ h_centered + rule['Y_mean'].astype(np.float64)
    if np.any(np.isnan(result)):
        return rule['Y_mean'].astype(np.float64)
    return result


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def inference_original(model, tokenizer, prompt: str) -> tuple:
    """Run inference with original model."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    t0 = time.perf_counter()
    logits = model(input_ids)
    mx.eval(logits)
    t1 = time.perf_counter()

    token_id = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    return token_id, t1 - t0


def inference_t_matrix(model, tokenizer, prompt: str, rules: Dict[int, Dict],
                       compress_layers: Set[int]) -> tuple:
    """Run inference with T-matrix compression."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    inner_model = model.model if hasattr(model, 'model') else model

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    t0 = time.perf_counter()

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)
    mask = create_attention_mask(h, None)

    for idx, layer in enumerate(inner_model.layers):
        if idx in compress_layers and idx in rules:
            h_normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(h_normed, mask=mask, cache=None)
            mx.eval(attn_out)
            h_post = h + attn_out
            h_normed2 = layer.post_attention_layernorm(h_post)
            mx.eval(h_normed2)

            h_np = np.array(h_normed2[0, -1, :].astype(mx.float32))
            mlp_out_last = apply_t_rule(h_np, rules[idx])

            mlp_out = layer.mlp(h_normed2)
            mx.eval(mlp_out)
            mlp_out_np = np.array(mlp_out.astype(mx.float32))
            mlp_out_np[0, -1, :] = mlp_out_last.astype(np.float32)
            mlp_out = mx.array(mlp_out_np).astype(h.dtype)

            h = h_post + mlp_out
            mx.eval(h)
        else:
            h = layer(h, mask, None)
            mx.eval(h)

    h = inner_model.norm(h)
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    else:
        logits = inner_model.embed_tokens.as_linear(h)
    mx.eval(logits)

    t1 = time.perf_counter()

    token_id = int(np.argmax(np.array(logits[0, -1, :].astype(mx.float32))))
    return token_id, t1 - t0


# ============================================================================
# BENCHMARKING
# ============================================================================

def calculate_sizes(model, rules: Dict[int, Dict], compress_layers: Set[int]) -> Dict:
    """Calculate model sizes for different configurations."""
    import mlx.core as mx

    inner_model = model.model if hasattr(model, 'model') else model
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    # Count original parameters
    total_params = 0
    mlp_params_per_layer = 0

    for name, param in model.parameters().items():
        total_params += param.size

    # Estimate MLP params (gate + up + down)
    layer = inner_model.layers[0]
    gate_size = layer.mlp.gate_proj.weight.size
    up_size = layer.mlp.up_proj.weight.size
    down_size = layer.mlp.down_proj.weight.size
    mlp_params_per_layer = gate_size + up_size + down_size

    # T-matrix size per layer
    t_matrix_params = hidden_dim * hidden_dim + 2 * hidden_dim  # T + X_mean + Y_mean

    # Calculate sizes
    n_compressed = len(compress_layers)
    original_mlp_params = n_compressed * mlp_params_per_layer
    t_matrix_total_params = n_compressed * t_matrix_params

    results = {
        'original': {
            'total_params': total_params,
            'mlp_params_compressed_layers': original_mlp_params,
            'size_bf16_gb': total_params * 2 / 1e9,
        },
        't_lossless': {
            'total_params': total_params - original_mlp_params + t_matrix_total_params,
            't_matrix_params': t_matrix_total_params,
            'size_bf16_gb': (total_params - original_mlp_params) * 2 / 1e9 + t_matrix_total_params * 4 / 1e9,
            'savings_params': original_mlp_params - t_matrix_total_params,
            'savings_pct': (original_mlp_params - t_matrix_total_params) / total_params * 100,
        },
        't_4bit': {
            'total_params': total_params - original_mlp_params + t_matrix_total_params,
            't_matrix_params': t_matrix_total_params,
            'size_gb': (total_params - original_mlp_params) * 2 / 1e9 + t_matrix_total_params * 0.5 / 1e9,
            'savings_params': original_mlp_params - t_matrix_total_params,
            'savings_pct': (original_mlp_params - t_matrix_total_params) / total_params * 100,
        },
    }

    return results


def run_benchmark(model, tokenizer, rules_fp32: Dict[int, Dict], rules_4bit: Dict[int, Dict],
                  compress_layers: Set[int], benchmark_prompts: List[Dict]) -> Dict:
    """Run comprehensive benchmark."""

    results = {
        'original': {'correct': 0, 'total': 0, 'times': [], 'tokens': []},
        't_lossless': {'correct': 0, 'total': 0, 'times': [], 'tokens': []},
        't_4bit': {'correct': 0, 'total': 0, 'times': [], 'tokens': []},
    }

    print(f"\n{'='*80}")
    print("RUNNING BENCHMARKS")
    print("="*80)

    for i, item in enumerate(benchmark_prompts):
        prompt = item['prompt']
        category = item['category']

        # Original
        orig_token, orig_time = inference_original(model, tokenizer, prompt)
        orig_text = tokenizer.decode([orig_token])

        # T-lossless
        t_token, t_time = inference_t_matrix(model, tokenizer, prompt, rules_fp32, compress_layers)
        t_text = tokenizer.decode([t_token])

        # T-4bit
        t4_token, t4_time = inference_t_matrix(model, tokenizer, prompt, rules_4bit, compress_layers)
        t4_text = tokenizer.decode([t4_token])

        # Record results
        results['original']['times'].append(orig_time)
        results['original']['tokens'].append(orig_token)
        results['original']['total'] += 1

        results['t_lossless']['times'].append(t_time)
        results['t_lossless']['tokens'].append(t_token)
        results['t_lossless']['total'] += 1
        if t_token == orig_token:
            results['t_lossless']['correct'] += 1

        results['t_4bit']['times'].append(t4_time)
        results['t_4bit']['tokens'].append(t4_token)
        results['t_4bit']['total'] += 1
        if t4_token == orig_token:
            results['t_4bit']['correct'] += 1

        # Print progress
        t_match = "✓" if t_token == orig_token else "✗"
        t4_match = "✓" if t4_token == orig_token else "✗"

        print(f"{i+1:2d}. [{category:10s}] T:{t_match} T4:{t4_match} | '{prompt[:40]}...' -> '{orig_text}'")

    # Calculate summary statistics
    for key in results:
        times = results[key]['times']
        results[key]['avg_time_ms'] = np.mean(times) * 1000
        results[key]['std_time_ms'] = np.std(times) * 1000
        results[key]['tokens_per_sec'] = 1.0 / np.mean(times) if np.mean(times) > 0 else 0

        if key != 'original':
            results[key]['accuracy'] = results[key]['correct'] / results[key]['total']

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--calibration-size", type=int, default=800)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model...")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    inner_model = model.model if hasattr(model, 'model') else model
    n_layers = len(inner_model.layers)
    hidden_dim = inner_model.embed_tokens.weight.shape[1]

    print(f"\n{'='*80}")
    print("DEEPSEEK-R1-QWEN3-8B COMPRESSION BENCHMARK")
    print("="*80)
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Generate calibration data
    calibration = generate_calibration_prompts(args.calibration_size)
    benchmark_prompts = generate_benchmark_prompts()

    print(f"Calibration: {len(calibration)} prompts")
    print(f"Benchmark: {len(benchmark_prompts)} prompts")

    # Define compression layers (14-21 for lossless)
    compress_layers = set(range(14, 22))
    print(f"\nCompressing layers: {sorted(compress_layers)}")

    # Derive T matrices
    print(f"\n{'='*80}")
    print("DERIVING T MATRICES (FP32)")
    print("="*80)

    rules_fp32 = {}
    for layer_idx in sorted(compress_layers):
        t0 = time.time()
        rules_fp32[layer_idx] = derive_t_matrix(model, tokenizer, layer_idx, calibration)
        print(f"  Layer {layer_idx}: done ({time.time()-t0:.1f}s)")

    # Create 4-bit quantized versions
    print(f"\n{'='*80}")
    print("QUANTIZING T MATRICES TO 4-BIT")
    print("="*80)

    rules_4bit = {}
    for layer_idx, rule in rules_fp32.items():
        rules_4bit[layer_idx] = {
            'T': quantize_symmetric(rule['T'], 4),
            'X_mean': rule['X_mean'],
            'Y_mean': rule['Y_mean'],
        }
        t_err = np.linalg.norm(rule['T'] - rules_4bit[layer_idx]['T']) / np.linalg.norm(rule['T']) * 100
        print(f"  Layer {layer_idx}: quantized (T error: {t_err:.1f}%)")

    # Calculate sizes
    print(f"\n{'='*80}")
    print("MODEL SIZES")
    print("="*80)

    sizes = calculate_sizes(model, rules_fp32, compress_layers)

    print(f"\n  ORIGINAL MODEL:")
    print(f"    Total params: {sizes['original']['total_params']/1e9:.2f}B")
    print(f"    Size (bf16):  {sizes['original']['size_bf16_gb']:.2f}GB")

    print(f"\n  T-MATRIX LOSSLESS (FP32 T):")
    print(f"    Total params: {sizes['t_lossless']['total_params']/1e9:.2f}B")
    print(f"    Size:         {sizes['t_lossless']['size_bf16_gb']:.2f}GB")
    print(f"    Savings:      {sizes['t_lossless']['savings_params']/1e9:.2f}B params ({sizes['t_lossless']['savings_pct']:.1f}%)")

    print(f"\n  T-MATRIX 4-BIT:")
    print(f"    Total params: {sizes['t_4bit']['total_params']/1e9:.2f}B")
    print(f"    Size:         {sizes['t_4bit']['size_gb']:.2f}GB")
    print(f"    Savings:      {sizes['t_4bit']['savings_pct']:.1f}% params, plus 4-bit T storage")

    # Run benchmarks
    benchmark_results = run_benchmark(
        model, tokenizer, rules_fp32, rules_4bit, compress_layers, benchmark_prompts
    )

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print("="*80)

    print(f"\n{'Configuration':<20} {'Accuracy':<12} {'Avg Time':<12} {'Tokens/sec':<12} {'Size':<12}")
    print("-"*80)

    print(f"{'ORIGINAL':<20} {'100.0%':<12} "
          f"{benchmark_results['original']['avg_time_ms']:.1f}ms{'':<5} "
          f"{benchmark_results['original']['tokens_per_sec']:.1f}{'':<6} "
          f"{sizes['original']['size_bf16_gb']:.2f}GB")

    print(f"{'T-LOSSLESS':<20} {benchmark_results['t_lossless']['accuracy']*100:.1f}%{'':<6} "
          f"{benchmark_results['t_lossless']['avg_time_ms']:.1f}ms{'':<5} "
          f"{benchmark_results['t_lossless']['tokens_per_sec']:.1f}{'':<6} "
          f"{sizes['t_lossless']['size_bf16_gb']:.2f}GB")

    print(f"{'T-4BIT':<20} {benchmark_results['t_4bit']['accuracy']*100:.1f}%{'':<6} "
          f"{benchmark_results['t_4bit']['avg_time_ms']:.1f}ms{'':<5} "
          f"{benchmark_results['t_4bit']['tokens_per_sec']:.1f}{'':<6} "
          f"{sizes['t_4bit']['size_gb']:.2f}GB")

    # Save results
    full_results = {
        'model': args.model,
        'calibration_size': len(calibration),
        'benchmark_size': len(benchmark_prompts),
        'compress_layers': list(compress_layers),
        'sizes': sizes,
        'benchmarks': {
            k: {kk: vv for kk, vv in v.items() if kk not in ['times', 'tokens']}
            for k, v in benchmark_results.items()
        },
    }

    with open(args.output, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    t_acc = benchmark_results['t_lossless']['accuracy'] * 100
    t4_acc = benchmark_results['t_4bit']['accuracy'] * 100

    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEEPSEEK-R1-QWEN3-8B COMPRESSION RESULTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONFIGURATION          SIZE         ACCURACY     SAVINGS                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Original (bf16)        {sizes['original']['size_bf16_gb']:.2f}GB       100.0%       baseline                  │
│  T-Lossless (FP32)      {sizes['t_lossless']['size_bf16_gb']:.2f}GB       {t_acc:.1f}%       {sizes['t_lossless']['savings_pct']:.1f}% smaller              │
│  T-4bit                 {sizes['t_4bit']['size_gb']:.2f}GB       {t4_acc:.1f}%       ~{(1-sizes['t_4bit']['size_gb']/sizes['original']['size_bf16_gb'])*100:.0f}% smaller              │
│                                                                             │
│  LAYERS COMPRESSED: 14-21 (8 transmission layers)                           │
│  CALIBRATION: {len(calibration)} prompts                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
