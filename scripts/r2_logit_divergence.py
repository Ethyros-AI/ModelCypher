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

"""R2 Logit Divergence: Locate where base and adapted models diverge during generation.

Phase A showed that 88% of CKA collapse comes from autoregressive feedback.
This script answers: does the adapted model diverge at step 0 (readout-margin
instability) or after several matched steps (gradual cascade)?

Three measurements:
  D.1: Step-0 logit comparison on identical prompts
  D.2: First divergence index under greedy decode (max 20 steps)
  D.3: (conditional) Teacher-forced replay on shared prefix at divergence point

Usage:
    poetry run python scripts/r2_logit_divergence.py
    poetry run python scripts/r2_logit_divergence.py --adapter-path /path/to/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
ADAPTER_PATH = "/Volumes/CodeCypher/models/adapters/350m-geometric-lora-r1"
OUTPUT_DIR = Path("results/r2_logit_divergence")

N_BENCH_PER_TASK = 10
MAX_DECODE_STEPS = 20

# LFM2-350M attention layer indices
ATTN_LAYER_INDICES = {2, 5, 8, 10, 12, 14}
N_LAYERS = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Step0Comparison:
    """Step-0 logit comparison for a single prompt."""
    task: str
    prompt_idx: int
    base_top1_token: int
    adapted_top1_token: int
    top1_match: bool
    base_top1_text: str
    adapted_top1_text: str
    base_margin: float        # base top1 - top2
    adapted_margin: float     # adapted top1 - top2
    base_top1_logit: float
    adapted_logit_at_base_top1: float   # how much adapted shifts base's choice
    adapted_top1_logit: float
    base_logit_at_adapted_top1: float   # how much base supports adapted's choice
    top5_overlap: int         # how many tokens in common in top 5
    top10_overlap: int


@dataclass
class DivergenceResult:
    """First divergence index for a single prompt."""
    task: str
    prompt_idx: int
    divergence_step: int      # -1 if never diverges within max_steps
    max_steps: int
    base_tokens: list[int]
    adapted_tokens: list[int]
    base_texts: list[str]
    adapted_texts: list[str]
    margin_at_divergence_base: float    # base margin at divergence step
    margin_at_divergence_adapted: float


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R2 logit divergence: step-0 comparison + first divergence index.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=ADAPTER_PATH,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--n-bench-per-task", type=int, default=N_BENCH_PER_TASK,
    )
    parser.add_argument(
        "--max-decode-steps", type=int, default=MAX_DECODE_STEPS,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logit utilities
# ---------------------------------------------------------------------------

def get_step0_logits(
    model: Any,
    tokenizer: Any,
    prompt: str,
    mx: Any,
) -> Any:
    """Get last-position logits from a single prompt forward pass.

    Uses model.__call__ which handles masking internally.
    Returns 1D logit array [vocab_size].
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)
    if logits.ndim == 3:
        return logits[0, -1, :]
    elif logits.ndim == 2:
        return logits[0, :]
    return logits


def topk_tokens(logits: Any, k: int, mx: Any) -> list[int]:
    """Return indices of top-k logits."""
    # MLX doesn't have topk, use argsort
    sorted_idx = mx.argsort(logits)
    mx.eval(sorted_idx)
    n = int(sorted_idx.shape[0])
    top = [int(sorted_idx[n - 1 - i]) for i in range(min(k, n))]
    return top


def logit_margin(logits: Any, mx: Any) -> float:
    """Top-1 minus top-2 logit value."""
    sorted_vals = mx.sort(logits)
    mx.eval(sorted_vals)
    n = int(sorted_vals.shape[0])
    top1 = float(sorted_vals[n - 1].item())
    top2 = float(sorted_vals[n - 2].item())
    return top1 - top2


# ---------------------------------------------------------------------------
# Greedy decode with token tracking
# ---------------------------------------------------------------------------

def greedy_decode_with_tracking(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_steps: int,
    mx: Any,
) -> tuple[list[int], list[float]]:
    """Run greedy decode, return (token_ids, margins) at each step.

    Uses model.__call__ with KV cache for autoregressive generation.
    """
    from modelcypher.core.domain.geometry.model_utils import resolve_model_base

    base = resolve_model_base(model)

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Initialize KV cache
    try:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
    except (ImportError, TypeError):
        cache = [None] * N_LAYERS

    # Prefill
    h = base.embed_tokens(input_ids)
    mx.eval(h)
    for layer_idx, layer in enumerate(base.layers):
        layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
        result = layer(h, mask=None, cache=layer_cache)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

    # Compute logits
    logits = _compute_logits(base, model, h, mx)

    generated_tokens: list[int] = []
    margins: list[float] = []

    for step in range(max_steps):
        margin = logit_margin(logits[0, -1, :] if logits.ndim == 3 else logits, mx)
        margins.append(margin)

        if logits.ndim == 3:
            next_token = mx.argmax(logits[0, -1, :])
        else:
            next_token = mx.argmax(logits)
        mx.eval(next_token)
        generated_tokens.append(int(next_token.item()))

        # Feed next token
        next_input = mx.reshape(next_token, (1, 1))
        h = base.embed_tokens(next_input)
        mx.eval(h)

        for layer_idx, layer in enumerate(base.layers):
            layer_cache = cache[layer_idx] if cache and layer_idx < len(cache) else None
            result = layer(h, mask=None, cache=layer_cache)
            h = result[0] if isinstance(result, tuple) else result
            mx.eval(h)

        logits = _compute_logits(base, model, h, mx)

    return generated_tokens, margins


def _compute_logits(base: Any, model: Any, h: Any, mx: Any) -> Any:
    """Compute logits from hidden state (LFM2 compatible)."""
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
    return logits


# ---------------------------------------------------------------------------
# D.1: Step-0 logit comparison
# ---------------------------------------------------------------------------

def run_step0_comparison(
    model_base: Any,
    model_adapted: Any,
    tokenizer: Any,
    prompts: list[tuple[str, str, int]],  # (task, prompt_text, idx)
    mx: Any,
    log: logging.Logger,
) -> list[Step0Comparison]:
    """Compare step-0 logits between base and adapted models."""
    results = []

    for task, prompt, idx in prompts:
        base_logits = get_step0_logits(model_base, tokenizer, prompt, mx)
        adapted_logits = get_step0_logits(model_adapted, tokenizer, prompt, mx)
        mx.eval(base_logits, adapted_logits)

        # Top-1 tokens
        base_top1 = int(mx.argmax(base_logits).item())
        adapted_top1 = int(mx.argmax(adapted_logits).item())

        # Margins
        base_margin = logit_margin(base_logits, mx)
        adapted_margin = logit_margin(adapted_logits, mx)

        # Cross-logit values
        base_top1_logit = float(base_logits[base_top1].item())
        adapted_at_base = float(adapted_logits[base_top1].item())
        adapted_top1_logit = float(adapted_logits[adapted_top1].item())
        base_at_adapted = float(base_logits[adapted_top1].item())

        # Top-k overlap
        base_top5 = set(topk_tokens(base_logits, 5, mx))
        adapted_top5 = set(topk_tokens(adapted_logits, 5, mx))
        base_top10 = set(topk_tokens(base_logits, 10, mx))
        adapted_top10 = set(topk_tokens(adapted_logits, 10, mx))

        # Decode tokens to text
        base_text = tokenizer.decode([base_top1])
        adapted_text = tokenizer.decode([adapted_top1])

        comp = Step0Comparison(
            task=task,
            prompt_idx=idx,
            base_top1_token=base_top1,
            adapted_top1_token=adapted_top1,
            top1_match=(base_top1 == adapted_top1),
            base_top1_text=base_text,
            adapted_top1_text=adapted_text,
            base_margin=base_margin,
            adapted_margin=adapted_margin,
            base_top1_logit=base_top1_logit,
            adapted_logit_at_base_top1=adapted_at_base,
            adapted_top1_logit=adapted_top1_logit,
            base_logit_at_adapted_top1=base_at_adapted,
            top5_overlap=len(base_top5 & adapted_top5),
            top10_overlap=len(base_top10 & adapted_top10),
        )
        results.append(comp)

        log.info(
            "  %s[%d] top1: %s | base=%r adapted=%r | margins: %.2f / %.2f | top5_overlap=%d",
            task, idx,
            "MATCH" if comp.top1_match else "DIVERGE",
            base_text, adapted_text,
            base_margin, adapted_margin,
            comp.top5_overlap,
        )

    return results


# ---------------------------------------------------------------------------
# D.2: First divergence index
# ---------------------------------------------------------------------------

def run_divergence_scan(
    model_base: Any,
    model_adapted: Any,
    tokenizer: Any,
    prompts: list[tuple[str, str, int]],
    max_steps: int,
    mx: Any,
    log: logging.Logger,
) -> list[DivergenceResult]:
    """Find first token mismatch under greedy decode."""
    results = []

    for task, prompt, idx in prompts:
        base_tokens, base_margins = greedy_decode_with_tracking(
            model_base, tokenizer, prompt, max_steps, mx,
        )
        adapted_tokens, adapted_margins = greedy_decode_with_tracking(
            model_adapted, tokenizer, prompt, max_steps, mx,
        )

        # Find first divergence
        div_step = -1
        for step in range(min(len(base_tokens), len(adapted_tokens))):
            if base_tokens[step] != adapted_tokens[step]:
                div_step = step
                break

        # Decode token sequences for readability
        base_texts = [tokenizer.decode([t]) for t in base_tokens[:min(10, len(base_tokens))]]
        adapted_texts = [tokenizer.decode([t]) for t in adapted_tokens[:min(10, len(adapted_tokens))]]

        margin_base = base_margins[div_step] if div_step >= 0 and div_step < len(base_margins) else -1.0
        margin_adapted = adapted_margins[div_step] if div_step >= 0 and div_step < len(adapted_margins) else -1.0

        result = DivergenceResult(
            task=task,
            prompt_idx=idx,
            divergence_step=div_step,
            max_steps=max_steps,
            base_tokens=base_tokens,
            adapted_tokens=adapted_tokens,
            base_texts=base_texts,
            adapted_texts=adapted_texts,
            margin_at_divergence_base=margin_base,
            margin_at_divergence_adapted=margin_adapted,
        )
        results.append(result)

        if div_step >= 0:
            log.info(
                "  %s[%d] diverges at step %d | base=%r adapted=%r | margins: %.2f / %.2f",
                task, idx, div_step,
                base_texts[div_step] if div_step < len(base_texts) else "?",
                adapted_texts[div_step] if div_step < len(adapted_texts) else "?",
                margin_base, margin_adapted,
            )
        else:
            log.info("  %s[%d] NO divergence in %d steps", task, idx, max_steps)

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_analysis(
    step0: list[Step0Comparison],
    divergence: list[DivergenceResult],
) -> str:
    lines = [
        "# R2 Logit Divergence Analysis",
        "",
        f"**Prompts:** {len(step0)}",
        "",
    ]

    # D.1 Summary
    n_match = sum(1 for s in step0 if s.top1_match)
    n_total = len(step0)
    lines.extend([
        "## D.1: Step-0 Logit Comparison",
        "",
        f"**Top-1 match rate:** {n_match}/{n_total} ({n_match/n_total*100:.0f}%)",
        "",
    ])

    # Per-task breakdown
    tasks = sorted(set(s.task for s in step0))
    lines.extend([
        "| Task | Match | Diverge | Avg Base Margin | Avg Adapted Margin | Avg Top5 Overlap |",
        "|------|------:|--------:|----------------:|-------------------:|-----------------:|",
    ])
    for task in tasks:
        task_items = [s for s in step0 if s.task == task]
        n_m = sum(1 for s in task_items if s.top1_match)
        n_d = len(task_items) - n_m
        avg_bm = sum(s.base_margin for s in task_items) / len(task_items)
        avg_am = sum(s.adapted_margin for s in task_items) / len(task_items)
        avg_top5 = sum(s.top5_overlap for s in task_items) / len(task_items)
        lines.append(f"| {task} | {n_m} | {n_d} | {avg_bm:.2f} | {avg_am:.2f} | {avg_top5:.1f}/5 |")
    lines.append("")

    # Per-prompt detail for divergent prompts
    diverged = [s for s in step0 if not s.top1_match]
    if diverged:
        lines.extend([
            "### Step-0 Divergent Prompts",
            "",
            "| Task | Idx | Base Token | Adapted Token | Base Margin | Adapted Margin | Top5 |",
            "|------|----:|-----------|--------------|------------:|---------------:|-----:|",
        ])
        for s in diverged:
            lines.append(
                f"| {s.task} | {s.prompt_idx} | {s.base_top1_text!r} | {s.adapted_top1_text!r} "
                f"| {s.base_margin:.2f} | {s.adapted_margin:.2f} | {s.top5_overlap}/5 |"
            )
        lines.append("")

    # D.2 Summary
    lines.extend(["## D.2: First Divergence Index", ""])

    div_steps = [d.divergence_step for d in divergence if d.divergence_step >= 0]
    never_div = sum(1 for d in divergence if d.divergence_step < 0)

    if div_steps:
        avg_div = sum(div_steps) / len(div_steps)
        step0_div = sum(1 for d in div_steps if d == 0)
        step01_div = sum(1 for d in div_steps if d <= 1)
        lines.append(f"**Average divergence step:** {avg_div:.1f}")
        lines.append(f"**Diverge at step 0:** {step0_div}/{len(div_steps)}")
        lines.append(f"**Diverge at step 0-1:** {step01_div}/{len(div_steps)}")
        lines.append(f"**Never diverge (within {divergence[0].max_steps} steps):** {never_div}")
        lines.append("")

        # Distribution
        lines.extend([
            "### Divergence Step Distribution",
            "",
            "| Step | Count |",
            "|-----:|------:|",
        ])
        from collections import Counter
        counts = Counter(div_steps)
        for step in sorted(counts.keys()):
            lines.append(f"| {step} | {counts[step]} |")
        lines.append("")
    else:
        lines.append("No divergence detected within max steps.")
        lines.append("")

    # Per-task divergence
    lines.extend([
        "### Per-Task Divergence",
        "",
        "| Task | Avg Step | Step-0 | Step-0-1 | Never |",
        "|------|--------:|-------:|---------:|------:|",
    ])
    for task in tasks:
        task_divs = [d for d in divergence if d.task == task]
        task_steps = [d.divergence_step for d in task_divs if d.divergence_step >= 0]
        task_never = sum(1 for d in task_divs if d.divergence_step < 0)
        if task_steps:
            avg = sum(task_steps) / len(task_steps)
            s0 = sum(1 for s in task_steps if s == 0)
            s01 = sum(1 for s in task_steps if s <= 1)
        else:
            avg = -1.0
            s0 = s01 = 0
        lines.append(f"| {task} | {avg:.1f} | {s0} | {s01} | {task_never} |")
    lines.append("")

    # Decision
    lines.extend(["## Diagnosis", ""])

    if div_steps:
        pct_step01 = step01_div / len(div_steps) * 100
        if pct_step01 >= 70:
            lines.append(f"**VERDICT: Readout-margin instability.** {pct_step01:.0f}% of prompts "
                         "diverge at step 0 or 1.")
            lines.append("")
            lines.append("The adapter preserves internal geometry but shifts the logit "
                         "distribution enough to flip the greedy argmax at the very first "
                         "generation step. Once the first token differs, the cascade "
                         "diverges exponentially.")
            lines.append("")
            lines.append("**This is NOT a representation-collapse problem.** The CKA collapse "
                         "observed in the R2 training pipeline's inference measurement is a "
                         "downstream consequence of token divergence, not its cause.")
            lines.append("")
            lines.append("**Recommended next steps:**")
            lines.append("1. Measure the logit displacement (adapter logit - base logit) "
                         "at the base model's top-1 token across all prompts")
            lines.append("2. Determine if this can be corrected by logit calibration "
                         "(scaling/shifting) or requires training changes")
            lines.append("3. Check if the adapter's top-1 token is still a 'reasonable' "
                         "answer even when it differs from the base")
        elif pct_step01 >= 30:
            lines.append(f"**VERDICT: Mixed.** {pct_step01:.0f}% diverge early, rest later.")
            lines.append("Some prompts show immediate margin instability, others show "
                         "gradual cascade. Investigate both patterns.")
        else:
            lines.append(f"**VERDICT: Gradual cascade.** Only {pct_step01:.0f}% diverge at "
                         "step 0-1. Most stay matched for several steps before breaking.")
            lines.append("Recommend Step D.3: teacher-forced replay on shared prefix.")
    else:
        lines.append("**VERDICT: No divergence detected.** Models generate identical tokens "
                     f"for all {len(divergence)} prompts within {divergence[0].max_steps} steps.")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
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
    log = logging.getLogger("r2_logit_divergence")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.adapter_path).exists():
        print(f"ERROR: Adapter not found: {args.adapter_path}", file=sys.stderr)
        sys.exit(2)

    log.info("=" * 70)
    log.info("R2 Logit Divergence: Step-0 + First Divergence Index")
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

    # Load benchmark prompts
    from modelcypher.core.use_cases.curriculum.benchmark_loader import BenchmarkLoader
    bench_loader = BenchmarkLoader()
    prompts: list[tuple[str, str, int]] = []  # (task, text, idx)
    for task_name in ["gsm8k", "arc_easy", "boolq"]:
        try:
            bench = bench_loader.load(task_name, split="test", limit=args.n_bench_per_task)
            for i, sample in enumerate(bench.samples):
                prompts.append((task_name, sample.prompt, i))
            log.info("  Loaded %d %s prompts", len(bench.samples), task_name)
        except Exception as e:
            log.warning("  Failed to load %s: %s", task_name, e)

    if not prompts:
        log.error("No prompts loaded.")
        sys.exit(1)

    # Load models
    log.info("Loading base model...")
    model_base, tokenizer = loader.load_model(MODEL_PATH)
    log.info("Loading adapted model...")
    model_adapted, _ = loader.load_model(MODEL_PATH, adapter_path=args.adapter_path)

    # D.1: Step-0 logit comparison
    log.info("=== D.1: Step-0 Logit Comparison ===")
    step0_results = run_step0_comparison(
        model_base, model_adapted, tokenizer, prompts, mx, log,
    )

    # D.2: First divergence index
    log.info("=== D.2: First Divergence Index (max %d steps) ===", args.max_decode_steps)
    divergence_results = run_divergence_scan(
        model_base, model_adapted, tokenizer, prompts,
        args.max_decode_steps, mx, log,
    )

    # Unload
    del model_base, model_adapted
    mx.eval()

    elapsed = time.time() - t_start

    # Write results
    step0_path = output_dir / "step0_logits.json"
    step0_path.write_text(
        json.dumps([asdict(s) for s in step0_results], indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", step0_path)

    div_path = output_dir / "divergence_index.json"
    div_path.write_text(
        json.dumps([asdict(d) for d in divergence_results], indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", div_path)

    # Analysis
    analysis = build_analysis(step0_results, divergence_results)
    analysis_path = output_dir / "ANALYSIS.md"
    analysis_path.write_text(analysis, encoding="utf-8")
    log.info("Wrote %s", analysis_path)

    print()
    print(analysis)
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
