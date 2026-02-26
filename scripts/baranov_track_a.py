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

Intervention modes:
    - baseline:  Pre-measurement only, no training or post-measurement.
    - no_op:     Full pre/post measurement cycle, no intervention applied.
                 Measures the pipeline noise floor — all deltas should be ~0.
    - lora_only: Full pre/post cycle with LoRA training on the fact pool.
                 The real alignment-tax measurement.

Usage:
    poetry run python scripts/baranov_track_a.py --output results/baranov/track_a/
    poetry run python scripts/baranov_track_a.py --smoke
    poetry run python scripts/baranov_track_a.py --intervention lora_only --smoke
    poetry run python scripts/baranov_track_a.py --intervention lora_only --seeds 42 123 456
    poetry run python scripts/baranov_track_a.py --intervention no_op --models LFM2-350M
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
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Intervention mode
# =============================================================================


class InterventionMode(str, Enum):
    """Intervention type for Track A experiment."""

    baseline = "baseline"
    no_op = "no_op"
    lora_only = "lora_only"


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
# Recall measurement
# =============================================================================


def _make_generate_fn(backend: Any) -> Any:
    """Create a generate_fn callback for the evaluator."""

    def generate_fn(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int,
        verbose: bool = False,
    ) -> str:
        return backend.generate(model, tokenizer, prompt, max_tokens=max_tokens)

    return generate_fn


def measure_recall(
    model: Any,
    tokenizer: Any,
    facts: list[Any],
    backend: Any,
    architecture: str,
) -> dict[str, Any]:
    """Measure recall on a model in both raw_completion and chat_template modes.

    Returns a dict with per-mode recall results (rate, count, per-fact outcomes).
    """
    from modelcypher.core.domain.chat_template import ChatTemplate
    from modelcypher.experimental.baranov.recall_evaluator import RecallMode
    from modelcypher.experimental.baranov.simple_recall_evaluator import (
        SimpleRecallEvaluator,
    )

    evaluator = SimpleRecallEvaluator(max_tokens=64)
    generate_fn = _make_generate_fn(backend)

    # --- Raw completion mode ---
    logger.info("  Evaluating raw_completion recall (%d facts)...", len(facts))
    t0 = time.monotonic()
    raw_result = evaluator.evaluate_recall(
        facts=facts,
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
    template = ChatTemplate.detect(architecture)
    logger.info("  Evaluating chat_template recall (template=%s)...", template.value)
    t0 = time.monotonic()
    chat_result = evaluator.evaluate_recall(
        facts=facts,
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

    return {
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
    }


# =============================================================================
# Intervention: LoRA training
# =============================================================================


def run_lora_intervention(
    model_path: str,
    facts: list[Any],
    output_dir: Path,
    model_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a LoRA adapter on the fact pool.

    Returns a dict with training metadata and adapter_path.
    """
    from modelcypher.cli.composition import get_dataset_training_service
    from modelcypher.experimental.baranov.fact_dataset import (
        write_fact_training_jsonl,
    )

    # Write facts to JSONL for training
    jsonl_path = output_dir / "training_data" / f"{model_name}_facts.jsonl"
    write_fact_training_jsonl(facts, jsonl_path, overwrite=True)
    logger.info("Wrote training JSONL: %s (%d facts)", jsonl_path, len(facts))

    # Train via DatasetTrainingService
    adapter_dir = output_dir / "adapters" / f"{model_name}_seed{seed}"
    service = get_dataset_training_service()

    logger.info("Starting LoRA training on %s (seed=%d)...", model_name, seed)
    t0 = time.monotonic()
    train_result = service.train_from_dataset_research(
        model_path=model_path,
        dataset_path=str(jsonl_path),
        output_path=str(adapter_dir),
        seed=seed,
    )
    train_elapsed = time.monotonic() - t0

    logger.info(
        "LoRA training complete: %d iters, loss %.4f → %.4f, stop=%s (%.1fs)",
        train_result.train_iters,
        train_result.initial_loss,
        train_result.final_loss,
        train_result.stop_reason,
        train_elapsed,
    )

    return {
        "adapter_path": train_result.adapter_path,
        "seed": seed,
        "train_iters": train_result.train_iters,
        "initial_loss": train_result.initial_loss,
        "final_loss": train_result.final_loss,
        "stop_reason": train_result.stop_reason,
        "n_lora_layers": train_result.n_lora_layers,
        "n_trainable_params": train_result.n_trainable_params,
        "training_time_s": train_elapsed,
        "training_min_cka": train_result.min_cka,
        "training_mean_cka": train_result.mean_cka,
    }


# =============================================================================
# Full model evaluation pipeline
# =============================================================================


def evaluate_model(
    model_name: str,
    model_info: dict[str, Any],
    facts: list[dict[str, str]],
    fact_triples: list[Any],
    backend: Any,
    intervention: InterventionMode,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full evaluation pipeline for a single model and seed.

    Phases:
        1. Pre-measurement: load base model, measure recall + geometry
        2. Intervention: train LoRA (if lora_only), or skip
        3. Post-measurement: load model (with adapter if applicable),
           measure recall + geometry, compute CKA drift
        4. Compute deltas
    """
    from modelcypher.experimental.baranov.geometry_measurement import (
        CKADriftResult,
        collect_probe_activations,
        compute_cka_drift,
    )
    from modelcypher.experimental.baranov.simple_recall_evaluator import (
        _build_raw_prompt,
    )

    model_path = model_info["path"]
    arch = model_info.get("architecture", "")

    # Probe texts for geometry: use the raw prompts (subject + relation)
    probe_texts = [_build_raw_prompt(f) for f in fact_triples]

    # ----- Phase 1: Pre-measurement -----
    logger.info("=== %s: Pre-measurement ===", model_name)
    model, tokenizer = backend.load_model(model_path)

    pre_recall = measure_recall(model, tokenizer, fact_triples, backend, arch)

    pre_geometry = None
    if intervention != InterventionMode.baseline:
        logger.info("  Collecting pre-intervention geometry...")
        pre_geometry = collect_probe_activations(
            model, tokenizer, probe_texts, backend,
        )

    del model, tokenizer
    gc.collect()

    # ----- Phase 2: Intervention -----
    training_meta: dict[str, Any] | None = None
    adapter_path: str | None = None

    if intervention == InterventionMode.lora_only:
        logger.info("=== %s: LoRA intervention (seed=%d) ===", model_name, seed)
        training_meta = run_lora_intervention(
            model_path, fact_triples, output_dir, model_name, seed=seed,
        )
        adapter_path = training_meta["adapter_path"]
    elif intervention == InterventionMode.no_op:
        logger.info("=== %s: No-op control (no intervention) ===", model_name)

    # ----- Phase 3: Post-measurement -----
    post_recall: dict[str, Any] | None = None
    drift_result: CKADriftResult | None = None

    if intervention != InterventionMode.baseline:
        logger.info("=== %s: Post-measurement ===", model_name)
        if adapter_path:
            model, tokenizer = backend.load_model(model_path, adapter_path=adapter_path)
        else:
            model, tokenizer = backend.load_model(model_path)

        post_recall = measure_recall(model, tokenizer, fact_triples, backend, arch)

        logger.info("  Collecting post-intervention geometry...")
        post_geometry = collect_probe_activations(
            model, tokenizer, probe_texts, backend,
        )

        del model, tokenizer
        gc.collect()

        # CKA drift
        assert pre_geometry is not None
        drift_result = compute_cka_drift(pre_geometry, post_geometry, backend)

    # ----- Phase 4: Assemble result -----
    # Geometry: use drift result if available, otherwise placeholders
    if drift_result is not None:
        geometry = drift_result.as_dict()
    else:
        geometry = {
            "per_layer_cka": {},
            "min_cka": 1.0,
            "mean_cka": 1.0,
            "cka_drift": 0.0,
            "preserved_fraction": 1.0,
        }

    # Deltas (only if we have post-measurement)
    deltas: dict[str, float] | None = None
    if post_recall is not None:
        deltas = {
            "delta_raw_recall": (
                post_recall["raw_completion"]["recall_rate"]
                - pre_recall["raw_completion"]["recall_rate"]
            ),
            "delta_chat_recall": (
                post_recall["chat_template_result"]["recall_rate"]
                - pre_recall["chat_template_result"]["recall_rate"]
            ),
            "delta_cka_drift": geometry["cka_drift"],
            "delta_preserved_fraction": geometry["preserved_fraction"] - 1.0,
        }

    result: dict[str, Any] = {
        "model_name": model_name,
        "model_path": model_path,
        "quantization": model_info.get("quantization", "unknown"),
        "architecture": arch,
        "intervention": intervention.value,
        "seed": seed,
        "n_facts": len(fact_triples),
        "pre": pre_recall,
        "post": post_recall,
        "deltas": deltas,
        "geometry": geometry,
        "training_meta": training_meta,
    }

    # Log summary
    _log_model_summary(model_name, result)

    return result


def _log_model_summary(model_name: str, result: dict[str, Any]) -> None:
    """Log a compact summary of results for a single model."""
    pre = result["pre"]
    logger.info(
        "  %s PRE:  raw=%.1f%% chat=%.1f%%",
        model_name,
        pre["raw_completion"]["recall_rate"] * 100,
        pre["chat_template_result"]["recall_rate"] * 100,
    )
    if result["post"] is not None:
        post = result["post"]
        logger.info(
            "  %s POST: raw=%.1f%% chat=%.1f%%",
            model_name,
            post["raw_completion"]["recall_rate"] * 100,
            post["chat_template_result"]["recall_rate"] * 100,
        )
    if result["deltas"] is not None:
        d = result["deltas"]
        logger.info(
            "  %s DELTA: raw=%+.1f%% chat=%+.1f%% cka_drift=%.4f",
            model_name,
            d["delta_raw_recall"] * 100,
            d["delta_chat_recall"] * 100,
            d["delta_cka_drift"],
        )


# =============================================================================
# Manifest and artifact writing
# =============================================================================


def build_manifest(
    results: list[dict[str, Any]],
    facts: list[dict[str, str]],
    run_id: str,
    intervention: InterventionMode,
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

    first = results[0]
    commit = _get_git_commit()

    # Aggregate pre-measurement metrics across models (worst-case)
    pre_raw_rates = [r["pre"]["raw_completion"]["recall_rate"] for r in results]
    pre_chat_rates = [r["pre"]["chat_template_result"]["recall_rate"] for r in results]

    # Use real geometry if available
    cka_drift = max(r["geometry"]["cka_drift"] for r in results)
    preserved_fraction = min(r["geometry"]["preserved_fraction"] for r in results)

    # Decision logic
    if intervention == InterventionMode.baseline:
        outcome = "inconclusive"
        reason = "Baseline recall measurement only — no intervention applied."
    elif intervention == InterventionMode.no_op:
        outcome = "inconclusive"
        reason = "No-op control run — validates measurement pipeline noise floor."
    else:
        # lora_only: check if we have any delta signal
        has_deltas = any(r["deltas"] is not None for r in results)
        if not has_deltas:
            outcome = "inconclusive"
            reason = "LoRA intervention completed but no deltas computed."
        else:
            outcome = "inconclusive"
            reason = (
                "LoRA intervention completed. "
                "Requires multi-seed replication before pass/fail determination."
            )

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
            base_control=(intervention == InterventionMode.baseline),
            lora_only_control=(intervention == InterventionMode.lora_only),
            edit_only_control=False,
        ),
        metrics_dict={
            "cka_drift": cka_drift,
            "preserved_fraction": preserved_fraction,
            "perplexity_drift_identity": 0.0,
            "perplexity_drift_general": 0.0,
            "recall_raw_completion": min(pre_raw_rates),
            "recall_chat_template": min(pre_chat_rates),
            "null_rank": 0.0,
            "condition_number": 0.0,
            "spectral_gap": 0.0,
        },
        pre_registered_decision=PreRegisteredDecision(
            criteria_version="v1",
            outcome=outcome,
            reason=reason,
        ),
    )
    return manifest.as_dict()


def build_summary(
    results: list[dict[str, Any]],
    intervention: InterventionMode,
    run_id: str,
) -> dict[str, Any]:
    """Build a decision-ready summary from all model results."""
    models_summary = {}
    for r in results:
        name = r["model_name"]
        entry: dict[str, Any] = {
            "intervention": r["intervention"],
            "n_facts": r["n_facts"],
            "pre_raw_recall": r["pre"]["raw_completion"]["recall_rate"],
            "pre_chat_recall": r["pre"]["chat_template_result"]["recall_rate"],
        }
        if r["post"] is not None:
            entry["post_raw_recall"] = r["post"]["raw_completion"]["recall_rate"]
            entry["post_chat_recall"] = r["post"]["chat_template_result"]["recall_rate"]
        if r["deltas"] is not None:
            entry["deltas"] = r["deltas"]
        if r["geometry"]:
            entry["cka_drift"] = r["geometry"]["cka_drift"]
            entry["preserved_fraction"] = r["geometry"]["preserved_fraction"]
        if r["training_meta"] is not None:
            entry["training"] = {
                k: v
                for k, v in r["training_meta"].items()
                if k != "adapter_path"
            }
        models_summary[name] = entry

    return {
        "run_id": run_id,
        "intervention": intervention.value,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": models_summary,
    }


def write_artifacts(
    results: list[dict[str, Any]],
    manifest_dict: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
    intervention: InterventionMode,
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
    csv_rows = _build_csv_rows(results, intervention)

    from modelcypher.experimental.baranov.artifact_writer import write_metrics_csv

    metrics_path = output_dir / "track_a_metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    write_metrics_csv(csv_rows, metrics_path)
    logger.info("Wrote metrics: %s", metrics_path)

    # Full recall curves (per-fact detail)
    recall_curves = _build_recall_curves(results)
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

    # Decision-ready summary
    summary_path = output_dir / "track_a_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote summary: %s", summary_path)


def _build_csv_rows(
    results: list[dict[str, Any]],
    intervention: InterventionMode,
) -> list[dict[str, Any]]:
    """Build CSV rows from results, including pre/post/delta phases."""
    rows: list[dict[str, Any]] = []

    for r in results:
        # Pre-measurement rows
        rows.append({
            "model": r["model_name"],
            "mode": "raw_completion",
            "phase": "pre",
            "recall_rate": r["pre"]["raw_completion"]["recall_rate"],
            "recalled_count": r["pre"]["raw_completion"]["recalled_count"],
            "total": r["pre"]["raw_completion"]["total"],
            "cka_drift": r["geometry"]["cka_drift"],
            "preserved_fraction": r["geometry"]["preserved_fraction"],
        })
        rows.append({
            "model": r["model_name"],
            "mode": "chat_template",
            "phase": "pre",
            "recall_rate": r["pre"]["chat_template_result"]["recall_rate"],
            "recalled_count": r["pre"]["chat_template_result"]["recalled_count"],
            "total": r["pre"]["chat_template_result"]["total"],
            "cka_drift": r["geometry"]["cka_drift"],
            "preserved_fraction": r["geometry"]["preserved_fraction"],
        })

        # Post-measurement rows (if available)
        if r["post"] is not None:
            rows.append({
                "model": r["model_name"],
                "mode": "raw_completion",
                "phase": "post",
                "recall_rate": r["post"]["raw_completion"]["recall_rate"],
                "recalled_count": r["post"]["raw_completion"]["recalled_count"],
                "total": r["post"]["raw_completion"]["total"],
                "cka_drift": r["geometry"]["cka_drift"],
                "preserved_fraction": r["geometry"]["preserved_fraction"],
            })
            rows.append({
                "model": r["model_name"],
                "mode": "chat_template",
                "phase": "post",
                "recall_rate": r["post"]["chat_template_result"]["recall_rate"],
                "recalled_count": r["post"]["chat_template_result"]["recalled_count"],
                "total": r["post"]["chat_template_result"]["total"],
                "cka_drift": r["geometry"]["cka_drift"],
                "preserved_fraction": r["geometry"]["preserved_fraction"],
            })

        # Delta rows (if available)
        if r["deltas"] is not None:
            rows.append({
                "model": r["model_name"],
                "mode": "raw_completion",
                "phase": "delta",
                "recall_rate": r["deltas"]["delta_raw_recall"],
                "recalled_count": 0,
                "total": r["pre"]["raw_completion"]["total"],
                "cka_drift": r["deltas"]["delta_cka_drift"],
                "preserved_fraction": r["geometry"]["preserved_fraction"],
            })
            rows.append({
                "model": r["model_name"],
                "mode": "chat_template",
                "phase": "delta",
                "recall_rate": r["deltas"]["delta_chat_recall"],
                "recalled_count": 0,
                "total": r["pre"]["chat_template_result"]["total"],
                "cka_drift": r["deltas"]["delta_cka_drift"],
                "preserved_fraction": r["geometry"]["preserved_fraction"],
            })

    return rows


def _build_recall_curves(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build recall curves JSON from results."""
    curves: dict[str, Any] = {}
    for r in results:
        name = r["model_name"]
        entry: dict[str, Any] = {
            "pre": {
                "raw_completion": r["pre"]["raw_completion"]["per_fact"],
                "chat_template": r["pre"]["chat_template_result"]["per_fact"],
            },
        }
        if r["post"] is not None:
            entry["post"] = {
                "raw_completion": r["post"]["raw_completion"]["per_fact"],
                "chat_template": r["post"]["chat_template_result"]["per_fact"],
            }
        curves[name] = entry
    return curves


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
    parser.add_argument(
        "--intervention",
        type=str,
        choices=[m.value for m in InterventionMode],
        default=InterventionMode.baseline.value,
        help="Intervention mode (default: baseline)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Training seeds for multi-seed runs (default: [42])",
    )
    parser.add_argument(
        "--noop-dir",
        default=None,
        help="Path to no_op results dir for noise floor (optional)",
    )
    return parser.parse_args()


def _load_noop_noise_floor(noop_dir: str | None) -> Any:
    """Load noise floor from a prior no_op run, if available."""
    from modelcypher.experimental.baranov.decision import NoiseFloor

    if noop_dir is None:
        return NoiseFloor()

    summary_path = Path(noop_dir) / "track_a_summary.json"
    if not summary_path.exists():
        logger.warning("No no_op summary at %s, using zero noise floor", summary_path)
        return NoiseFloor()

    summary = json.loads(summary_path.read_text())
    max_raw = 0.0
    max_chat = 0.0
    for model_data in summary.get("models", {}).values():
        deltas = model_data.get("deltas", {})
        max_raw = max(max_raw, abs(deltas.get("delta_raw_recall", 0.0)))
        max_chat = max(max_chat, abs(deltas.get("delta_chat_recall", 0.0)))

    floor = NoiseFloor(raw=max_raw, chat=max_chat)
    logger.info("Loaded noise floor from %s: raw=%.4f chat=%.4f", noop_dir, floor.raw, floor.chat)
    return floor


def run_experiment(args: argparse.Namespace) -> None:
    """Execute the Track A experiment."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.experimental.baranov.decision import (
        compute_model_verdict,
        compute_track_a_decision,
    )
    from modelcypher.experimental.baranov.models import FactTriple

    backend = initialize_default_backend()
    intervention = InterventionMode(args.intervention)
    seeds = args.seeds

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

    fact_triples = [FactTriple.from_dict(f) for f in facts]
    n_facts = len(facts)

    run_id = args.run_id or f"track-a-{int(time.time())}"
    output_dir = Path(args.output)

    logger.info("Run ID: %s", run_id)
    logger.info("Models: %s", model_names)
    logger.info("Facts: %d", n_facts)
    logger.info("Intervention: %s", intervention.value)
    logger.info("Seeds: %s", seeds)
    logger.info("Output: %s", output_dir)

    # Collect per-model, per-seed results
    # model_seed_results[model_name] = [result_per_seed, ...]
    model_seed_results: dict[str, list[dict[str, Any]]] = {}

    for model_name in model_names:
        model_info = MODEL_REGISTRY[model_name]

        if intervention == InterventionMode.baseline:
            # Baseline: single run, no seeds
            result = evaluate_model(
                model_name=model_name,
                model_info=model_info,
                facts=facts,
                fact_triples=fact_triples,
                backend=backend,
                intervention=intervention,
                output_dir=output_dir,
            )
            model_seed_results[model_name] = [result]
        else:
            # Multi-seed: run each seed
            seed_results = []
            for seed in seeds:
                logger.info("=== %s seed=%d ===", model_name, seed)
                result = evaluate_model(
                    model_name=model_name,
                    model_info=model_info,
                    facts=facts,
                    fact_triples=fact_triples,
                    backend=backend,
                    intervention=intervention,
                    output_dir=output_dir,
                    seed=seed,
                )
                seed_results.append(result)
            model_seed_results[model_name] = seed_results

    # Flatten to first-seed results for existing artifact format
    flat_results = [results[0] for results in model_seed_results.values()]

    # Build artifacts (backward-compatible)
    manifest_dict = build_manifest(flat_results, facts, run_id, intervention)
    summary = build_summary(flat_results, intervention, run_id)
    write_artifacts(flat_results, manifest_dict, summary, output_dir, intervention)

    # Compute decision (if intervention has post-measurement)
    decision = None
    if intervention != InterventionMode.baseline:
        noise_floor = _load_noop_noise_floor(args.noop_dir)

        model_verdicts = {}
        for model_name in model_names:
            seed_results = model_seed_results[model_name]
            # Pre is constant across seeds; use first seed
            pre_raw = seed_results[0]["pre"]["raw_completion"]["recall_rate"]
            pre_chat = seed_results[0]["pre"]["chat_template_result"]["recall_rate"]

            # Build per-seed data for decision module
            per_seed_data = []
            for r in seed_results:
                if r["post"] is not None:
                    per_seed_data.append({
                        "post_raw_rate": r["post"]["raw_completion"]["recall_rate"],
                        "post_chat_rate": r["post"]["chat_template_result"]["recall_rate"],
                        "cka_drift": r["geometry"]["cka_drift"],
                        "preserved_fraction": r["geometry"]["preserved_fraction"],
                    })

            if per_seed_data:
                model_verdicts[model_name] = compute_model_verdict(
                    model=model_name,
                    pre_raw_rate=pre_raw,
                    pre_chat_rate=pre_chat,
                    seed_results=per_seed_data,
                    n_facts=n_facts,
                    noise_floor=noise_floor,
                )

        if model_verdicts:
            decision = compute_track_a_decision(model_verdicts, model_names)

            # Write decision artifact
            decision_path = output_dir / "track_a_decision.json"
            decision_path.write_text(
                json.dumps(decision.as_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote decision: %s", decision_path)

            # Update manifest with decision verdict
            manifest_dict["pre_registered_decision"]["outcome"] = decision.overall_verdict
            manifest_dict["pre_registered_decision"]["reason"] = decision.overall_reason
            manifest_path = output_dir / "track_a_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_dict, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    # Print final summary
    _print_final_summary(model_seed_results, intervention, output_dir, decision)


def _print_final_summary(
    model_seed_results: dict[str, list[dict[str, Any]]],
    intervention: InterventionMode,
    output_dir: Path,
    decision: Any,
) -> None:
    """Print a compact final summary to the log."""
    logger.info("=" * 60)
    logger.info("Track A Summary (%s)", intervention.value)
    logger.info("=" * 60)

    for model_name, seed_results in model_seed_results.items():
        r = seed_results[0]
        pre = r["pre"]
        line = f"  {model_name}: pre_raw={pre['raw_completion']['recall_rate']*100:.1f}%"
        line += f" pre_chat={pre['chat_template_result']['recall_rate']*100:.1f}%"

        if r["post"] is not None:
            # Show mean post across seeds
            post_raw_rates = [
                s["post"]["raw_completion"]["recall_rate"]
                for s in seed_results if s["post"] is not None
            ]
            post_chat_rates = [
                s["post"]["chat_template_result"]["recall_rate"]
                for s in seed_results if s["post"] is not None
            ]
            if post_raw_rates:
                mean_post_raw = sum(post_raw_rates) / len(post_raw_rates)
                mean_post_chat = sum(post_chat_rates) / len(post_chat_rates)
                line += f" | post_raw={mean_post_raw*100:.1f}%"
                line += f" post_chat={mean_post_chat*100:.1f}%"
                line += f" ({len(seed_results)} seeds)"

        if r["deltas"] is not None:
            d = r["deltas"]
            line += f" | cka_drift={d['delta_cka_drift']:.4f}"

        logger.info(line)

    if decision is not None:
        logger.info("-" * 60)
        logger.info("Decision: %s", decision.overall_verdict.upper())
        logger.info("Reason: %s", decision.overall_reason)

    logger.info("Artifacts written to: %s", output_dir)


def main() -> None:
    args = _parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
