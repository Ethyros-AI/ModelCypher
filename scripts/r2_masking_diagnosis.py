#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""R2 Masking Diagnosis: Disentangle prompt distribution, masking, and generation mode.

The R2 inference-collapse blocker shows train CKA ~0.95 but inference CKA 0.01-0.58.
However, ``collect_hidden_activations()`` passes ``mask=None`` (bidirectional) for ALL
layers, while real inference uses causal masking via KV cache. This confounds three
variables:

  1. Prompt distribution (training eval vs benchmark prompts)
  2. Masking mode (bidirectional vs causal)
  3. Generation mode (prefill vs autoregressive decode)

Five conditions isolate each variable:

  C1: Canonical train probes + mask_mode="none"   -> reproduces shipped "healthy" CKA
  C2: Benchmark prompts     + mask_mode="none"    -> tests prompt distribution alone
  C3: Benchmark prompts     + mask_mode="causal"  -> tests masking mode
  C4: Benchmark prompts     + KV-cache prefill    -> tests cache path (no generation)
  C5: Benchmark prompts     + KV-cache decode     -> tests autoregressive feedback

C1 replicates the exact canonical probe derivation: ALL eval samples, char-truncated
by ``seq_length * median_chars_per_token``, one text at a time, mean-pooled over seq.
C2-C3 use the backend's ``collect_hidden_activations(mask_mode=...)`` — no forked code.
C4-C5 need manual layer-by-layer forwarding (KV cache path is architecturally distinct).

Usage:
    poetry run python scripts/r2_masking_diagnosis.py
    poetry run python scripts/r2_masking_diagnosis.py --adapter-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
ADAPTER_PATH = "/Volumes/CodeCypher/models/adapters/350m-geometric-lora-r1"
EVAL_DATA = "data/training/r1_quick_aligned_val.jsonl"
OUTPUT_DIR = Path("results/r2_masking_diagnosis")

# Derived from the adapter's training_plan.json
CANONICAL_SEQ_LENGTH = 768

N_BENCH_PER_TASK = 10  # Benchmark prompts per task for C2-C5
N_DECODE_STEPS = 5     # Autoregressive steps for C5
N_LAYERS = 16

# LFM2-350M attention layer indices (the rest are ShortConv)
ATTN_LAYER_INDICES = {2, 5, 8, 10, 12, 14}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R2 masking diagnosis: 5-condition CKA matrix.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=ADAPTER_PATH,
        help=f"Path to adapter (default: {ADAPTER_PATH}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--n-bench-per-task", type=int, default=N_BENCH_PER_TASK,
        help=f"Benchmark probes per task for C2-C5 (default: {N_BENCH_PER_TASK}).",
    )
    parser.add_argument(
        "--n-decode-steps", type=int, default=N_DECODE_STEPS,
        help=f"Autoregressive decode steps for C5 (default: {N_DECODE_STEPS}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# C1: Canonical probe derivation (matches shipped verifier exactly)
# ---------------------------------------------------------------------------

def derive_canonical_probes(
    tokenizer: Any,
    backend: Any,
    log: logging.Logger,
) -> list[str]:
    """Replicate _derive_probe_texts + _collect_probe_activations_from_texts.

    Uses ALL eval samples with char truncation derived from
    seq_length * median_chars_per_token — same as
    _dataset_training_service_helpers_mixin.py:238-272.
    """
    eval_samples: list[dict[str, Any]] = []
    with open(EVAL_DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_samples.append(json.loads(line))

    # Measure chars/token on the actual eval data
    ratios: list[float] = []
    for s in eval_samples:
        text = s.get("text")
        if not isinstance(text, str) or not text:
            continue
        n_tokens = len(backend.encode_tokens(tokenizer, text))
        if n_tokens > 0:
            ratios.append(len(text) / float(n_tokens))

    if not ratios:
        probes = [s["text"] for s in eval_samples
                  if isinstance(s.get("text"), str) and s["text"]]
        log.info("  C1 probes: %d (no truncation, fallback)", len(probes))
        return probes

    ratios.sort()
    mid = len(ratios) // 2
    median_cpt = (ratios[mid] if len(ratios) % 2 == 1
                  else (ratios[mid - 1] + ratios[mid]) / 2.0)

    char_budget = max(1, int(math.ceil(CANONICAL_SEQ_LENGTH * median_cpt)))

    probes: list[str] = []
    for s in eval_samples:
        text = s.get("text")
        if isinstance(text, str) and text:
            probes.append(text[:char_budget])

    log.info("  C1 probes: %d texts, char_budget=%d (median_cpt=%.2f, seq_len=%d)",
             len(probes), char_budget, median_cpt, CANONICAL_SEQ_LENGTH)
    return probes


def collect_canonical_activations(
    model: Any,
    tokenizer: Any,
    probe_texts: list[str],
    backend: Any,
    log: logging.Logger,
) -> dict[int, Any]:
    """Collect activations matching the canonical verifier path exactly.

    One text at a time, mask_mode="none" (bidirectional), mean-pooled over seq.
    This is what the shipped pipeline uses for training CKA.
    """
    activations: dict[int, list] = {}
    for i, text in enumerate(probe_texts):
        acts = backend.collect_hidden_activations(
            model, tokenizer, [text], mask_mode="none",
        )
        for layer_idx, act in acts.items():
            # act: [1, seq, hidden] -> mean over seq -> [hidden]
            pooled = backend.mean(act, axis=1)
            pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)
        if (i + 1) % 100 == 0:
            log.info("    canonical: %d/%d", i + 1, len(probe_texts))

    stacked: dict[int, Any] = {}
    for layer_idx, layer_acts in activations.items():
        if layer_acts:
            stacked[layer_idx] = backend.stack(layer_acts)
            backend.eval(stacked[layer_idx])
    return stacked


# ---------------------------------------------------------------------------
# C2-C3: Backend-based collection with mask_mode parameter
# ---------------------------------------------------------------------------

def collect_backend_activations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    backend: Any,
    mask_mode: str,
    pool: str,
    log: logging.Logger,
    label: str = "",
) -> dict[int, Any]:
    """Collect activations using the backend API with specified mask_mode.

    Collects one prompt at a time (variable-length safe). Pools each to [hidden].

    Args:
        pool: "mean" for mean-over-seq (matches canonical), "last" for last-token.
    """
    activations: dict[int, list] = {}
    for i, prompt in enumerate(prompts):
        acts = backend.collect_hidden_activations(
            model, tokenizer, [prompt], mask_mode=mask_mode,
        )
        for layer_idx, act in acts.items():
            if pool == "last":
                # act: [1, seq, hidden] -> last token -> [hidden]
                pooled = act[0, -1, :]
            else:
                # act: [1, seq, hidden] -> mean over seq -> [hidden]
                pooled = backend.mean(act, axis=1)
                pooled = backend.reshape(pooled, (-1,))
            backend.eval(pooled)
            activations.setdefault(layer_idx, []).append(pooled)
        if (i + 1) % 10 == 0:
            log.info("    %s: %d/%d", label, i + 1, len(prompts))

    stacked: dict[int, Any] = {}
    for layer_idx, layer_acts in activations.items():
        if layer_acts:
            stacked[layer_idx] = backend.stack(layer_acts)
            backend.eval(stacked[layer_idx])
    return stacked


# ---------------------------------------------------------------------------
# C4: KV-cache prefill (causal by construction, no generation)
# ---------------------------------------------------------------------------

def collect_activations_kv_prefill(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    mx: Any,
    log: logging.Logger,
) -> dict[int, Any]:
    """Collect last-token activations using KV-cache prefill path.

    The KV cache provides causal masking by construction during prefill.
    This does NOT generate any new tokens — it only processes the prompt.
    """
    from modelcypher.core.domain.geometry.model_utils import resolve_model_base

    base = resolve_model_base(model)
    n_layers = len(base.layers)

    per_layer: dict[int, list[Any]] = {i: [] for i in range(n_layers)}

    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        except (ImportError, TypeError):
            cache = [None] * n_layers

        h = base.embed_tokens(input_ids)
        mx.eval(h)

        for layer_idx, layer in enumerate(base.layers):
            layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
            result = layer(h, mask=None, cache=layer_cache)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result
            mx.eval(h)
            # Last token activation
            per_layer[layer_idx].append(h[0, -1, :])
            mx.eval(per_layer[layer_idx][-1])

        if (i + 1) % 10 == 0:
            log.info("    kv_prefill: %d/%d", i + 1, len(prompts))

    result = {}
    for layer_idx, acts in per_layer.items():
        if acts:
            result[layer_idx] = mx.stack(acts, axis=0)
            mx.eval(result[layer_idx])
    return result


# ---------------------------------------------------------------------------
# C5: KV-cache decode (autoregressive — tests generation feedback)
# ---------------------------------------------------------------------------

def collect_activations_kv_decode(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    n_decode_steps: int,
    mx: Any,
    log: logging.Logger,
) -> dict[int, Any]:
    """Collect activations after real autoregressive decode steps.

    1. Prefill: process the full prompt through KV cache.
    2. Decode: generate n_decode_steps tokens (greedy argmax).
    3. Collect: per-layer hidden state at the LAST decoded token.

    This tests whether generation feedback (seeing own output) causes collapse.
    """
    from modelcypher.core.domain.geometry.model_utils import resolve_model_base

    base = resolve_model_base(model)
    n_layers = len(base.layers)

    per_layer: dict[int, list[Any]] = {i: [] for i in range(n_layers)}

    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        except (ImportError, TypeError):
            cache = [None] * n_layers

        # Phase 1: Prefill (process entire prompt, populate KV cache)
        h = base.embed_tokens(input_ids)
        mx.eval(h)

        for layer_idx, layer in enumerate(base.layers):
            layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
            result = layer(h, mask=None, cache=layer_cache)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

        # Get logits from prefill to find first token
        if hasattr(base, "embedding_norm"):
            h_norm = base.embedding_norm(h)
        elif hasattr(base, "norm"):
            h_norm = base.norm(h)
        else:
            h_norm = h
        mx.eval(h_norm)

        if hasattr(base, "embed_tokens") and hasattr(base.embed_tokens, "as_linear"):
            logits = base.embed_tokens.as_linear(h_norm)
        elif hasattr(model, "lm_head"):
            logits = model.lm_head(h_norm)
        else:
            logits = h_norm
        mx.eval(logits)

        # Phase 2: Decode n steps
        for step in range(n_decode_steps):
            # Greedy: argmax of last-position logits
            next_token = mx.argmax(logits[0, -1, :])
            mx.eval(next_token)
            next_input = mx.reshape(next_token, (1, 1))

            # Forward single token through all layers with cache
            h = base.embed_tokens(next_input)
            mx.eval(h)

            for layer_idx, layer in enumerate(base.layers):
                layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
                result = layer(h, mask=None, cache=layer_cache)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)

            # Compute logits for next step
            if hasattr(base, "embedding_norm"):
                h_norm = base.embedding_norm(h)
            elif hasattr(base, "norm"):
                h_norm = base.norm(h)
            else:
                h_norm = h
            mx.eval(h_norm)

            if hasattr(base, "embed_tokens") and hasattr(base.embed_tokens, "as_linear"):
                logits = base.embed_tokens.as_linear(h_norm)
            elif hasattr(model, "lm_head"):
                logits = model.lm_head(h_norm)
            else:
                logits = h_norm
            mx.eval(logits)

        # Collect activations from the LAST decode step
        # h is [1, 1, hidden] from the last decode token
        for layer_idx in range(n_layers):
            # We need to re-forward to get per-layer activations at the
            # last decode position. But we already have h from the last
            # step — we stored it during the last decode iteration.
            # However, we need PER-LAYER activations, not just the final h.
            pass

        # Alternative approach: re-run the last decode step to capture
        # per-layer activations.
        next_token_replay = mx.argmax(logits[0, -1, :])
        mx.eval(next_token_replay)
        replay_input = mx.reshape(next_token_replay, (1, 1))

        h_replay = base.embed_tokens(replay_input)
        mx.eval(h_replay)

        for layer_idx, layer in enumerate(base.layers):
            layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
            result = layer(h_replay, mask=None, cache=layer_cache)
            h_replay = result[0] if isinstance(result, tuple) else result
            mx.eval(h_replay)
            # h_replay is [1, 1, hidden] — squeeze to [hidden]
            per_layer[layer_idx].append(h_replay[0, 0, :])
            mx.eval(per_layer[layer_idx][-1])

        if (i + 1) % 5 == 0:
            log.info("    kv_decode: %d/%d", i + 1, len(prompts))

    result = {}
    for layer_idx, acts in per_layer.items():
        if acts:
            result[layer_idx] = mx.stack(acts, axis=0)
            mx.eval(result[layer_idx])
    return result


# ---------------------------------------------------------------------------
# CKA computation
# ---------------------------------------------------------------------------

def compute_per_layer_cka(
    base_acts: dict[int, Any],
    adapted_acts: dict[int, Any],
) -> dict[int, float]:
    """Compute linear CKA per layer between base and adapted activations."""
    from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations

    cka_per_layer: dict[int, float] = {}
    for layer_idx in sorted(base_acts.keys()):
        if layer_idx not in adapted_acts:
            continue
        cka = compute_linear_cka_from_activations(
            base_acts[layer_idx], adapted_acts[layer_idx],
        )
        cka_per_layer[layer_idx] = cka
    return cka_per_layer


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", mode="w"),
        ],
    )
    log = logging.getLogger("r2_masking_diagnosis")

    # Pre-flight checks
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.adapter_path).exists():
        print(f"ERROR: Adapter not found: {args.adapter_path}", file=sys.stderr)
        sys.exit(2)

    log.info("=" * 70)
    log.info("R2 Masking Diagnosis: 5-Condition CKA Matrix")
    log.info("  Model:   %s", MODEL_PATH)
    log.info("  Adapter: %s", args.adapter_path)
    log.info("=" * 70)

    t_start = time.time()

    import mlx.core as mx
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.backends import initialize_default_backend

    initialize_default_backend()
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()
    loader = ModelLoader(backend)

    # --- Load prompts ---
    log.info("Loading prompts...")

    # C1: canonical probe derivation (ALL eval samples, char-truncated)
    model_tmp, tokenizer = loader.load_model(MODEL_PATH)
    c1_probes = derive_canonical_probes(tokenizer, backend, log)
    del model_tmp
    mx.eval()

    # C2-C5: benchmark prompts
    from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader
    bench_loader = BenchmarkLoader()
    bench_prompts: dict[str, list[str]] = {}
    for task_name in ["gsm8k", "arc_easy", "boolq"]:
        try:
            bench = bench_loader.load(task_name, split="test", limit=args.n_bench_per_task)
            bench_prompts[task_name] = [s.prompt for s in bench.samples]
            log.info("  Loaded %d %s prompts", len(bench_prompts[task_name]), task_name)
        except Exception as e:
            log.warning("  Failed to load %s: %s", task_name, e)

    all_bench: list[str] = []
    for task_prompts in bench_prompts.values():
        all_bench.extend(task_prompts)
    log.info("  Total benchmark prompts: %d", len(all_bench))

    if not all_bench:
        log.error("No benchmark prompts loaded. Cannot proceed.")
        sys.exit(1)

    # --- Load both models ---
    log.info("Loading base model...")
    model_base, tokenizer = loader.load_model(MODEL_PATH)
    mem_base = mx.metal.get_active_memory() / (1024**3)
    log.info("  Base model loaded. Active memory: %.2f GB", mem_base)

    log.info("Loading adapted model...")
    model_adapted, _ = loader.load_model(MODEL_PATH, adapter_path=args.adapter_path)
    mem_adapted = mx.metal.get_active_memory() / (1024**3)
    log.info("  Adapted model loaded. Active memory: %.2f GB", mem_adapted)

    # --- Identity sanity check ---
    log.info("--- Identity check: base vs base (must be 1.0) ---")
    id_acts = collect_backend_activations(
        model_base, tokenizer, c1_probes[:5], backend,
        mask_mode="none", pool="mean", log=log, label="identity",
    )
    identity_cka = compute_per_layer_cka(id_acts, id_acts)
    del id_acts
    mx.eval()
    identity_ok = all(v > 0.999 for v in identity_cka.values())
    log.info("  Identity check: %s (min=%.6f)",
             "PASS" if identity_ok else "FAIL",
             min(identity_cka.values()) if identity_cka else 0.0)

    all_conditions: dict[str, dict[int, float]] = {}

    # --- C1: Canonical train probes + bidirectional ---
    log.info("=== C1: Canonical train probes + mask_mode=none ===")
    c1_base = collect_canonical_activations(model_base, tokenizer, c1_probes, backend, log)
    c1_adapted = collect_canonical_activations(model_adapted, tokenizer, c1_probes, backend, log)
    all_conditions["C1_train_bidir"] = compute_per_layer_cka(c1_base, c1_adapted)
    del c1_base, c1_adapted
    mx.eval()
    _log_cka_summary("C1", all_conditions["C1_train_bidir"], log)

    # --- C2: Benchmark + bidirectional ---
    log.info("=== C2: Benchmark prompts + mask_mode=none ===")
    c2_base = collect_backend_activations(
        model_base, tokenizer, all_bench, backend,
        mask_mode="none", pool="last", log=log, label="C2_base",
    )
    c2_adapted = collect_backend_activations(
        model_adapted, tokenizer, all_bench, backend,
        mask_mode="none", pool="last", log=log, label="C2_adapted",
    )
    all_conditions["C2_bench_bidir"] = compute_per_layer_cka(c2_base, c2_adapted)
    del c2_base, c2_adapted
    mx.eval()
    _log_cka_summary("C2", all_conditions["C2_bench_bidir"], log)

    # --- C3: Benchmark + causal ---
    log.info("=== C3: Benchmark prompts + mask_mode=causal ===")
    c3_base = collect_backend_activations(
        model_base, tokenizer, all_bench, backend,
        mask_mode="causal", pool="last", log=log, label="C3_base",
    )
    c3_adapted = collect_backend_activations(
        model_adapted, tokenizer, all_bench, backend,
        mask_mode="causal", pool="last", log=log, label="C3_adapted",
    )
    all_conditions["C3_bench_causal"] = compute_per_layer_cka(c3_base, c3_adapted)
    del c3_base, c3_adapted
    mx.eval()
    _log_cka_summary("C3", all_conditions["C3_bench_causal"], log)

    # --- C4: Benchmark + KV-cache prefill ---
    log.info("=== C4: Benchmark prompts + KV-cache prefill ===")
    c4_base = collect_activations_kv_prefill(model_base, tokenizer, all_bench, mx, log)
    c4_adapted = collect_activations_kv_prefill(model_adapted, tokenizer, all_bench, mx, log)
    all_conditions["C4_bench_prefill"] = compute_per_layer_cka(c4_base, c4_adapted)
    del c4_base, c4_adapted
    mx.eval()
    _log_cka_summary("C4", all_conditions["C4_bench_prefill"], log)

    # --- C5: Benchmark + KV-cache decode ---
    log.info("=== C5: Benchmark prompts + KV-cache decode (%d steps) ===", args.n_decode_steps)
    c5_base = collect_activations_kv_decode(
        model_base, tokenizer, all_bench, args.n_decode_steps, mx, log,
    )
    c5_adapted = collect_activations_kv_decode(
        model_adapted, tokenizer, all_bench, args.n_decode_steps, mx, log,
    )
    all_conditions["C5_bench_decode"] = compute_per_layer_cka(c5_base, c5_adapted)
    del c5_base, c5_adapted
    mx.eval()
    _log_cka_summary("C5", all_conditions["C5_bench_decode"], log)

    # --- Unload models ---
    del model_base, model_adapted
    mx.eval()

    # --- Assemble results ---
    elapsed = time.time() - t_start

    results = {
        "model_path": MODEL_PATH,
        "adapter_path": args.adapter_path,
        "canonical_seq_length": CANONICAL_SEQ_LENGTH,
        "n_canonical_probes": len(c1_probes),
        "n_bench_probes": len(all_bench),
        "n_decode_steps": args.n_decode_steps,
        "bench_tasks": {k: len(v) for k, v in bench_prompts.items()},
        "elapsed_seconds": elapsed,
        "identity_check": {str(k): v for k, v in identity_cka.items()},
        "identity_ok": identity_ok,
        "conditions": {
            name: {str(k): v for k, v in cka.items()}
            for name, cka in all_conditions.items()
        },
        "notes": {
            "C1": "Canonical probe derivation: ALL eval samples, char-truncated by "
                  "seq_length*median_cpt, mean-pooled, mask=None (bidirectional)",
            "C2": "Benchmark prompts, last-token pooled, mask=None (bidirectional)",
            "C3": "Benchmark prompts, last-token pooled, mask=causal per layer type",
            "C4": "Benchmark prompts, last-token, KV-cache prefill (no generation)",
            "C5": f"Benchmark prompts, last decode token after {args.n_decode_steps} "
                  "greedy steps, KV-cache autoregressive",
        },
    }

    # --- Write JSON ---
    json_path = output_dir / "cka_5conditions.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", json_path)

    # --- Build and write analysis ---
    analysis = build_analysis(results)
    analysis_path = output_dir / "ANALYSIS.md"
    analysis_path.write_text(analysis, encoding="utf-8")
    log.info("Wrote %s", analysis_path)

    # Print to stdout
    print()
    print(analysis)
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {json_path}")
    print(f"Analysis: {analysis_path}")


def _log_cka_summary(label: str, cka: dict[int, float], log: logging.Logger) -> None:
    if cka:
        vals = list(cka.values())
        log.info("  %s: min=%.4f, mean=%.4f", label, min(vals), sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------

def _cka_summary(cka: dict[str, float]) -> tuple[float, float]:
    """Return (min, mean) from a layer->cka dict."""
    vals = list(cka.values())
    if not vals:
        return 0.0, 0.0
    return min(vals), sum(vals) / len(vals)


def build_analysis(results: dict) -> str:
    """Build markdown analysis from the 5-condition results."""
    conditions = results["conditions"]
    lines = [
        "# R2 Masking Diagnosis: 5-Condition CKA Matrix",
        "",
        f"**Model:** {results['model_path']}",
        f"**Adapter:** {results['adapter_path']}",
        f"**Canonical probes:** {results['n_canonical_probes']} (seq_length={results['canonical_seq_length']})",
        f"**Benchmark probes:** {results['n_bench_probes']} ({results['bench_tasks']})",
        f"**Decode steps (C5):** {results['n_decode_steps']}",
        "",
    ]

    # Identity check
    id_min, id_mean = _cka_summary(results["identity_check"])
    lines.append(f"**Identity check (base vs base):** min={id_min:.6f} "
                 f"{'PASS' if results['identity_ok'] else 'FAIL'}")
    lines.append("")

    # Condition descriptions
    for cond, note in results.get("notes", {}).items():
        lines.append(f"- **{cond}:** {note}")
    lines.append("")

    # Summary table
    condition_order = ["C1_train_bidir", "C2_bench_bidir", "C3_bench_causal",
                       "C4_bench_prefill", "C5_bench_decode"]
    lines.extend([
        "## Condition Summary",
        "",
        "| Condition | Min CKA | Mean CKA |",
        "|-----------|--------:|---------:|",
    ])

    summaries = {}
    for cond_name in condition_order:
        cka = conditions.get(cond_name, {})
        min_cka, mean_cka = _cka_summary(cka)
        summaries[cond_name] = (min_cka, mean_cka)
        lines.append(f"| {cond_name} | {min_cka:.4f} | {mean_cka:.4f} |")
    lines.append("")

    # Per-layer comparison table
    all_layers = sorted(set(int(k) for cond in conditions.values() for k in cond.keys()))
    header = "| Layer | Type |"
    sep = "|------:|------|"
    for cn in condition_order:
        short = cn.split("_", 1)[0]
        header += f" {short} |"
        sep += "------:|"
    lines.extend(["## Per-Layer CKA Comparison", "", header, sep])

    for layer in all_layers:
        layer_str = str(layer)
        layer_type = "attn" if layer in ATTN_LAYER_INDICES else "conv"
        row = f"| {layer:2d} | {layer_type} |"
        for cn in condition_order:
            val = conditions.get(cn, {}).get(layer_str, 0.0)
            row += f" {val:.4f} |"
        lines.append(row)
    lines.append("")

    # Delta table
    lines.extend([
        "## Condition-to-Condition Deltas (mean CKA)",
        "",
        "| Transition | Delta | Variable Isolated |",
        "|------------|------:|-------------------|",
    ])
    transitions = [
        ("C1->C2", "C1_train_bidir", "C2_bench_bidir", "Prompt distribution"),
        ("C2->C3", "C2_bench_bidir", "C3_bench_causal", "Causal masking"),
        ("C3->C4", "C3_bench_causal", "C4_bench_prefill", "KV-cache prefill path"),
        ("C4->C5", "C4_bench_prefill", "C5_bench_decode", "Autoregressive feedback"),
    ]
    deltas = {}
    for label, from_c, to_c, variable in transitions:
        from_mean = summaries.get(from_c, (0, 0))[1]
        to_mean = summaries.get(to_c, (0, 0))[1]
        delta = from_mean - to_mean
        deltas[label] = delta
        lines.append(f"| {label} | {delta:+.4f} | {variable} |")
    lines.append("")

    # Diagnostic interpretation
    lines.extend(["## Diagnostic Interpretation", ""])

    c1_min = summaries.get("C1_train_bidir", (0, 0))[0]
    c1_mean = summaries.get("C1_train_bidir", (0, 0))[1]

    if c1_min < 0.85:
        lines.append("**WARNING:** C1 did not reproduce healthy training CKA (min < 0.85).")
        lines.append("This may indicate probe derivation mismatch. Interpret cautiously.")
        lines.append("")

    for label, delta in deltas.items():
        variable = next(t[3] for t in transitions if t[0] == label)
        if abs(delta) > 0.1:
            direction = "degrades" if delta > 0 else "improves"
            lines.append(f"**{label}: {variable} {direction} CKA by {delta:+.4f}**")
        elif abs(delta) > 0.05:
            lines.append(f"{label}: {variable} has moderate effect ({delta:+.4f})")
        else:
            lines.append(f"{label}: {variable} has minimal effect ({delta:+.4f})")
    lines.append("")

    # Attribution
    total_drop = summaries.get("C1_train_bidir", (0, 0))[1] - summaries.get("C5_bench_decode", (0, 0))[1]
    if total_drop > 0.01:
        lines.extend(["## Attribution (% of total C1->C5 drop)", ""])
        for label, delta in deltas.items():
            pct = (delta / total_drop) * 100 if total_drop > 0 else 0
            variable = next(t[3] for t in transitions if t[0] == label)
            lines.append(f"- {variable}: {pct:.0f}%")
        lines.append("")

    # Worst layers
    lines.extend(["## Worst Layers by Condition", ""])
    for cn in condition_order:
        cka = conditions.get(cn, {})
        if cka:
            worst = min(cka.items(), key=lambda x: x[1])
            lines.append(f"- **{cn}:** layer {worst[0]} (CKA={worst[1]:.4f})")
    lines.append("")

    # Recommended next step
    lines.extend(["## Recommended Next Step", ""])
    if total_drop < 0.1:
        lines.append("Collapse is minimal across all conditions. Check whether the")
        lines.append("existing inference CKA used different probes or pooling strategy.")
    else:
        dominant = max(deltas.items(), key=lambda x: x[1])
        variable = next(t[3] for t in transitions if t[0] == dominant[0])
        lines.append(f"**Dominant factor: {variable}** ({dominant[0]}: {dominant[1]:+.4f})")
        lines.append("")
        if "Prompt" in variable:
            lines.append("Next: Phase C — layerwise benchmark CKA per task to identify")
            lines.append("which tasks and layers are most affected by prompt distribution.")
        elif "masking" in variable.lower():
            lines.append("The CKA measurement was wrong — bidirectional masking doesn't")
            lines.append("match the inference path. The mask_mode='causal' fix in the")
            lines.append("backend is the product fix. Retrain with corrected telemetry.")
        elif "prefill" in variable.lower():
            lines.append("KV-cache prefill changes representations beyond just masking.")
            lines.append("Investigate whether the cache-based attention computation")
            lines.append("differs from the non-cache path.")
        elif "feedback" in variable.lower():
            lines.append("Autoregressive generation degrades representations. The adapter")
            lines.append("breaks when the model sees its own output. Phase D (logit-")
            lines.append("trajectory analysis) should trace where generation diverges.")
    lines.append("")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
