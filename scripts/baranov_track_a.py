#!/usr/bin/env python3
"""Baranov Track A: Alignment-Tax Replication.

EXPERIMENTAL: Not validated for production use.

Tests whether post-training alignment reduces recoverable factual recall
under parameter-efficient updates as model scale increases.  Evaluates
both ``raw_completion`` and ``chat_template`` pathways at each scale.

Research question (from replication protocol §4.1):
    Does post-training alignment behavior reduce recoverable factual recall
    under parameter-efficient updates as model scale increases, when
    evaluated separately for raw_completion and chat_template pathways?

Pass criteria:
    - Directional trend of recall suppression with scale in at least one
      mode split, with CI excluding zero effect.
    - Suppression co-occurs with geometry signatures (CKA drift / preserved-
      fraction collapse).

Fail criteria:
    - No consistent trend across scales, or trend reverses under controls,
      or CIs include no-effect across all splits.

Usage:
    poetry run python scripts/baranov_track_a.py --output results/baranov/track_a/
    poetry run python scripts/baranov_track_a.py --smoke
    poetry run python scripts/baranov_track_a.py --models LFM2-350M LFM2-1.2B

References:
    docs/research/baranov_replication_protocol_2026_02.md §4
    docs/research/baranov_sleeping_llm_intake_2026_02.md (claims C4, C6)
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model registry
# =============================================================================

MODELS_BASE = "/Volumes/CodeCypher/models"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "LFM2-350M": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-350M-MLX-bf16",
        "quantization": "bf16",
        "architecture": "lfm2",
    },
    "LFM2-1.2B": {
        "path": f"{MODELS_BASE}/mlx-community/LFM2-1.2B-bf16",
        "quantization": "bf16",
        "architecture": "lfm2",
    },
    "Qwen3-8B": {
        "path": f"{MODELS_BASE}/mlx-community/Qwen3-8B-bf16",
        "quantization": "bf16",
        "architecture": "qwen3",
    },
}

# Default model order: smallest to largest (for scaling trend)
DEFAULT_MODEL_ORDER = ["LFM2-350M", "LFM2-1.2B", "Qwen3-8B"]


# =============================================================================
# Fact pool
# =============================================================================

# Synthetic fact pool for initial testing.  A production run would load
# from a versioned JSON file with a content hash.
FACT_POOL = [
    {"subject": "Paris", "relation": "capital_of", "object": "France", "fact_id": "f001"},
    {"subject": "Berlin", "relation": "capital_of", "object": "Germany", "fact_id": "f002"},
    {"subject": "Tokyo", "relation": "capital_of", "object": "Japan", "fact_id": "f003"},
    {"subject": "Ottawa", "relation": "capital_of", "object": "Canada", "fact_id": "f004"},
    {"subject": "Canberra", "relation": "capital_of", "object": "Australia", "fact_id": "f005"},
    {"subject": "Rome", "relation": "capital_of", "object": "Italy", "fact_id": "f006"},
    {"subject": "Madrid", "relation": "capital_of", "object": "Spain", "fact_id": "f007"},
    {"subject": "Lisbon", "relation": "capital_of", "object": "Portugal", "fact_id": "f008"},
    {"subject": "Athens", "relation": "capital_of", "object": "Greece", "fact_id": "f009"},
    {"subject": "Vienna", "relation": "capital_of", "object": "Austria", "fact_id": "f010"},
    {"subject": "Water", "relation": "chemical_formula", "object": "H2O", "fact_id": "f011"},
    {"subject": "Gold", "relation": "chemical_symbol", "object": "Au", "fact_id": "f012"},
    {"subject": "Iron", "relation": "chemical_symbol", "object": "Fe", "fact_id": "f013"},
    {"subject": "Shakespeare", "relation": "wrote", "object": "Hamlet", "fact_id": "f014"},
    {"subject": "Einstein", "relation": "developed", "object": "relativity", "fact_id": "f015"},
    {"subject": "Newton", "relation": "formulated", "object": "gravity", "fact_id": "f016"},
    {"subject": "Darwin", "relation": "proposed", "object": "evolution", "fact_id": "f017"},
    {"subject": "Turing", "relation": "invented", "object": "Turing machine", "fact_id": "f018"},
    {"subject": "Mars", "relation": "has_moons", "object": "Phobos", "fact_id": "f019"},
    {"subject": "Jupiter", "relation": "largest_moon", "object": "Ganymede", "fact_id": "f020"},
]

SMOKE_FACT_COUNT = 5


def _compute_fact_pool_hash(facts: list[dict[str, str]]) -> str:
    """SHA-256 of the sorted JSON representation of the fact pool."""
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _get_git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model_recall(
    model_name: str,
    model_info: dict[str, Any],
    facts: list[dict[str, str]],
    backend: Any,
) -> dict[str, Any]:
    """Evaluate recall on a single model in both modes.

    Returns a dict with per-mode recall results and geometry metrics.
    """
    from modelcypher.core.domain.chat_template import ChatTemplate
    from modelcypher.experimental.baranov.models import FactTriple
    from modelcypher.experimental.baranov.simple_recall_evaluator import (
        SimpleRecallEvaluator,
    )
    from modelcypher.experimental.baranov.recall_evaluator import RecallMode

    logger.info("Loading model: %s", model_name)
    model_path = model_info["path"]
    model, tokenizer = backend.load_model(model_path)

    fact_triples = [FactTriple.from_dict(f) for f in facts]
    evaluator = SimpleRecallEvaluator(max_tokens=64)

    def generate_fn(m: Any, t: Any, prompt: str, max_tokens: int, verbose: bool = False) -> str:
        return backend.generate(m, t, prompt, max_tokens=max_tokens)

    # --- Raw completion mode ---
    logger.info("  Evaluating raw_completion recall (%d facts)...", len(fact_triples))
    t0 = time.monotonic()
    raw_result = evaluator.evaluate_recall(
        facts=fact_triples,
        generate_fn=generate_fn,
        model=model,
        tokenizer=tokenizer,
        mode=RecallMode.raw_completion,
    )
    raw_elapsed = time.monotonic() - t0
    logger.info(
        "  raw_completion: %d/%d recalled (%.1f%%) in %.1fs",
        raw_result.aggregate.recalled_count,
        raw_result.aggregate.total,
        raw_result.aggregate.recall_rate * 100,
        raw_elapsed,
    )

    # --- Chat template mode ---
    arch = model_info.get("architecture", "")
    template = ChatTemplate.detect(arch)
    logger.info("  Evaluating chat_template recall (template=%s)...", template.value)
    t0 = time.monotonic()
    chat_result = evaluator.evaluate_recall(
        facts=fact_triples,
        generate_fn=generate_fn,
        model=model,
        tokenizer=tokenizer,
        mode=RecallMode.chat_template,
        chat_template=template.value,
    )
    chat_elapsed = time.monotonic() - t0
    logger.info(
        "  chat_template: %d/%d recalled (%.1f%%) in %.1fs",
        chat_result.aggregate.recalled_count,
        chat_result.aggregate.total,
        chat_result.aggregate.recall_rate * 100,
        chat_elapsed,
    )

    # --- CKA drift (baseline vs self — placeholder for pre/post comparison) ---
    # In the full experiment, CKA drift is measured before vs after LoRA
    # injection.  Here we record the baseline geometry fingerprint.
    cka_drift = 0.0  # Placeholder: no edit applied yet
    preserved_fraction = 1.0  # Placeholder: no edit applied yet

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "quantization": model_info.get("quantization", "unknown"),
        "architecture": arch,
        "chat_template": template.value,
        "n_facts": len(fact_triples),
        "raw_completion": {
            "recall_rate": raw_result.aggregate.recall_rate,
            "recalled_count": raw_result.aggregate.recalled_count,
            "total": raw_result.aggregate.total,
            "confidence_interval": (
                list(raw_result.aggregate.confidence_interval)
                if raw_result.aggregate.confidence_interval
                else None
            ),
            "elapsed_s": raw_elapsed,
            "per_fact": [o.as_dict() for o in raw_result.per_fact_outcomes],
        },
        "chat_template_result": {
            "recall_rate": chat_result.aggregate.recall_rate,
            "recalled_count": chat_result.aggregate.recalled_count,
            "total": chat_result.aggregate.total,
            "confidence_interval": (
                list(chat_result.aggregate.confidence_interval)
                if chat_result.aggregate.confidence_interval
                else None
            ),
            "elapsed_s": chat_elapsed,
            "per_fact": [o.as_dict() for o in chat_result.per_fact_outcomes],
        },
        "geometry": {
            "cka_drift": cka_drift,
            "preserved_fraction": preserved_fraction,
        },
    }

    # Cleanup
    del model, tokenizer
    gc.collect()

    return result


# =============================================================================
# Manifest and artifact writing
# =============================================================================


def build_manifest(
    results: list[dict[str, Any]],
    facts: list[dict[str, str]],
    run_id: str,
) -> dict[str, Any]:
    """Build a Track A manifest dict from results."""
    from modelcypher.experimental.baranov.manifest import (
        CodeInfo,
        ControlFlags,
        DataHashes,
        ModelInfo,
        PreRegisteredDecision,
        ReplicationManifest,
    )

    # Use first model as representative for the manifest model field
    first = results[0]

    # Aggregate metrics across models (use worst-case for manifest)
    raw_rates = [r["raw_completion"]["recall_rate"] for r in results]
    chat_rates = [r["chat_template_result"]["recall_rate"] for r in results]

    commit = _get_git_commit()

    manifest = ReplicationManifest.from_metrics_dict(
        run_id=run_id,
        track="A",
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model=ModelInfo(
            id=first["model_name"],
            quantization=first["quantization"],
            backend="mlx",
        ),
        code=CodeInfo(
            modelcypher_commit=commit,
            experiment_module_commit=commit,
        ),
        data=DataHashes(
            fact_pool_hash=_compute_fact_pool_hash(facts),
            split_manifest_hash="none",
            reference_corpus_hash="none",
        ),
        controls=ControlFlags(
            base_control=True,
            lora_only_control=False,
            edit_only_control=False,
        ),
        metrics_dict={
            "cka_drift": max(r["geometry"]["cka_drift"] for r in results),
            "preserved_fraction": min(
                r["geometry"]["preserved_fraction"] for r in results
            ),
            "perplexity_drift_identity": 0.0,
            "perplexity_drift_general": 0.0,
            "recall_raw_completion": min(raw_rates),
            "recall_chat_template": min(chat_rates),
            "null_rank": 0.0,
            "condition_number": 0.0,
            "spectral_gap": 0.0,
        },
        pre_registered_decision=PreRegisteredDecision(
            criteria_version="v1",
            outcome="inconclusive",
            reason="Baseline recall measurement only — no LoRA intervention applied yet.",
        ),
    )
    return manifest.as_dict()


def write_artifacts(
    results: list[dict[str, Any]],
    manifest_dict: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write all Track A artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest_path = output_dir / "track_a_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote manifest: %s", manifest_path)

    # Metrics CSV rows
    csv_rows = []
    for r in results:
        csv_rows.append({
            "model": r["model_name"],
            "mode": "raw_completion",
            "recall_rate": r["raw_completion"]["recall_rate"],
            "recalled_count": r["raw_completion"]["recalled_count"],
            "total": r["raw_completion"]["total"],
            "cka_drift": r["geometry"]["cka_drift"],
            "preserved_fraction": r["geometry"]["preserved_fraction"],
        })
        csv_rows.append({
            "model": r["model_name"],
            "mode": "chat_template",
            "recall_rate": r["chat_template_result"]["recall_rate"],
            "recalled_count": r["chat_template_result"]["recalled_count"],
            "total": r["chat_template_result"]["total"],
            "cka_drift": r["geometry"]["cka_drift"],
            "preserved_fraction": r["geometry"]["preserved_fraction"],
        })

    from modelcypher.experimental.baranov.artifact_writer import write_metrics_csv

    metrics_path = output_dir / "track_a_metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    write_metrics_csv(csv_rows, metrics_path)
    logger.info("Wrote metrics: %s", metrics_path)

    # Full recall curves (per-fact detail)
    recall_curves = {
        r["model_name"]: {
            "raw_completion": r["raw_completion"]["per_fact"],
            "chat_template": r["chat_template_result"]["per_fact"],
        }
        for r in results
    }
    curves_path = output_dir / "track_a_recall_curves.json"
    curves_path.write_text(
        json.dumps(recall_curves, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote recall curves: %s", curves_path)

    # Geometry summary
    geometry = {
        r["model_name"]: r["geometry"]
        for r in results
    }
    geometry_path = output_dir / "track_a_geometry.json"
    geometry_path.write_text(
        json.dumps(geometry, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote geometry: %s", geometry_path)


# =============================================================================
# Main
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baranov Track A: Alignment-Tax Replication",
    )
    parser.add_argument(
        "--output",
        default="results/baranov/track_a/",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names to evaluate (default: {DEFAULT_MODEL_ORDER})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Smoke test: use only {SMOKE_FACT_COUNT} facts",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID (default: auto-generated from timestamp)",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Execute the Track A experiment."""
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    model_names = args.models or DEFAULT_MODEL_ORDER
    for name in model_names:
        if name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {name}. Available: {sorted(MODEL_REGISTRY)}",
            )

    facts = FACT_POOL
    if args.smoke:
        facts = facts[:SMOKE_FACT_COUNT]
        logger.info("SMOKE TEST: using %d facts", len(facts))

    run_id = args.run_id or f"track-a-{int(time.time())}"
    logger.info("Run ID: %s", run_id)
    logger.info("Models: %s", model_names)
    logger.info("Facts: %d", len(facts))

    results: list[dict[str, Any]] = []
    for model_name in model_names:
        model_info = MODEL_REGISTRY[model_name]
        result = evaluate_model_recall(model_name, model_info, facts, backend)
        results.append(result)

    # Build manifest and write artifacts
    manifest_dict = build_manifest(results, facts, run_id)
    output_dir = Path(args.output)
    write_artifacts(results, manifest_dict, output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("Track A Summary")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            "  %s: raw=%.1f%% chat=%.1f%%",
            r["model_name"],
            r["raw_completion"]["recall_rate"] * 100,
            r["chat_template_result"]["recall_rate"] * 100,
        )
    logger.info("Artifacts written to: %s", output_dir)


def main() -> None:
    args = _parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
