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

"""Unified atlas CLI commands."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_model_path
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
from modelcypher.core.domain.geometry.concept_dimensionality import (
    ConceptDimensionalityAnalyzer,
    ConceptDimensionalityReport,
    ConceptDimensionalityStudy,
)
from modelcypher.core.domain.geometry.riemannian_utils import frechet_mean
from modelcypher.core.support.array_utils import array_to_list

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


class BackboneActivationProvider:
    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        frechet_k_neighbors: int | None = None,
        frechet_max_k_neighbors: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend
        self._frechet_k_neighbors = frechet_k_neighbors
        self._frechet_max_k_neighbors = frechet_max_k_neighbors

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        activations = []
        pending = []

        for text in texts:
            if not text:
                continue
            try:
                tokens = self._tokenizer.encode(text)
                if not tokens:
                    continue
                input_ids = self._backend.array([tokens])
                hidden = forward_through_backbone(
                    input_ids,
                    self._embed_tokens,
                    self._layers,
                    self._norm,
                    target_layer=layer,
                    backend=self._backend,
                )
                mean = frechet_mean(
                    hidden[0],
                    backend=self._backend,
                    k_neighbors=self._frechet_k_neighbors,
                    max_k_neighbors=self._frechet_max_k_neighbors,
                )
                self._backend.async_eval(mean)
                pending.append(mean)
                activations.append(mean)
            except Exception as exc:
                logger.debug("Activation failed for text '%s': %s", text, exc)
                continue

        if pending:
            self._backend.eval(*pending)

        return [array_to_list(self._backend, vec) for vec in activations]


def _apply_layer(layer, hidden, mask):
    """Apply a transformer layer with graceful mask fallback.

    Some architectures (e.g., LFM2) accept a mask parameter but create their own
    batch-aware masks internally. Passing our 2D causal mask causes ValueError
    on broadcast. We catch this and retry without mask.
    """
    try:
        return layer(hidden, mask=mask)
    except (TypeError, ValueError):
        # TypeError: layer doesn't accept mask parameter
        # ValueError: mask shape incompatible with batched hidden (e.g., LFM2)
        try:
            return layer(hidden, mask)
        except (TypeError, ValueError):
            return layer(hidden)


def _normalize_probe_text(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) < 2:
        return None
    return cleaned


class AtlasProgress:
    def __init__(self, logger: logging.Logger, min_interval_s: float = 1.0) -> None:
        self._logger = logger
        self._min_interval_s = min_interval_s
        self._start = time.monotonic()
        self._last_emit = 0.0

    def emit(
        self,
        phase: str,
        processed: int,
        total: int,
        extra: dict | None = None,
    ) -> None:
        now = time.monotonic()
        if processed < total and now - self._last_emit < self._min_interval_s:
            return

        elapsed = now - self._start
        rate = processed / elapsed if elapsed > 0 else None
        eta = ((total - processed) / rate) if rate and total >= processed else None
        payload = {
            "phase": phase,
            "processed": processed,
            "total": total,
            "elapsedS": round(elapsed, 2),
            "rate": round(rate, 2) if rate is not None else None,
            "etaS": round(eta, 2) if eta is not None else None,
        }
        if extra:
            payload.update(extra)
        self._logger.info("ATLAS_PROGRESS %s", json.dumps(payload, sort_keys=True))
        self._last_emit = now

    def callback(self, phase: str, total: int, extra: dict | None = None) -> Callable:
        def _cb(
            processed: int,
            total_override: int | None = None,
            analyzed: int | None = None,
            skipped: int | None = None,
        ) -> None:
            payload = dict(extra or {})
            if analyzed is not None:
                payload["analyzed"] = analyzed
            if skipped is not None:
                payload["skipped"] = skipped
            self.emit(phase, processed, total_override or total, payload)

        return _cb


class AtlasActivationCache:
    def __init__(
        self,
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        pooling: str = "frechet",
        batch_size: int = 8,
        frechet_k_neighbors: int | None = None,
        frechet_max_k_neighbors: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._embed_tokens = embed_tokens
        self._layers = layers
        self._norm = norm
        self._backend = backend
        self._pooling = pooling
        self._batch_size = max(1, batch_size)
        self._frechet_k_neighbors = frechet_k_neighbors
        self._frechet_max_k_neighbors = frechet_max_k_neighbors
        self._progress_callback = progress_callback
        self._cache: dict[int, dict[str, list[float]]] = {}
        self._token_cache: dict[str, list[int]] = {}

    def get_activations(self, texts: list[str], layer: int) -> list[list[float]]:
        normalized = [_normalize_probe_text(text) for text in texts]
        missing = [
            text
            for text in normalized
            if text is not None and text not in self._cache.get(layer, {})
        ]
        if missing:
            self.preload_layers(missing, [layer])

        activations = []
        layer_cache = self._cache.get(layer, {})
        for text in normalized:
            if text is None:
                continue
            vec = layer_cache.get(text)
            if vec is not None:
                activations.append(vec)
        return activations

    def clear_layers(self, layers: list[int]) -> None:
        for layer in layers:
            self._cache.pop(layer, None)

    def preload_layers(
        self,
        texts: list[str],
        layers: list[int],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if not layers:
            return

        normalized = []
        seen: set[str] = set()
        for text in texts:
            clean = _normalize_probe_text(text)
            if clean is None or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)

        missing = [
            text
            for text in normalized
            if any(text not in self._cache.get(layer, {}) for layer in layers)
        ]
        if not missing:
            return

        callback = progress_callback or self._progress_callback
        total = len(missing)
        processed = 0
        for batch_start in range(0, total, self._batch_size):
            batch_texts = missing[batch_start: batch_start + self._batch_size]
            pooled_by_layer = self._collect_layer_pools(batch_texts, layers)
            for layer, pooled in pooled_by_layer.items():
                layer_cache = self._cache.setdefault(layer, {})
                for text, vec in zip(batch_texts, pooled):
                    if vec is None:
                        continue
                    layer_cache[text] = array_to_list(self._backend, vec)

            processed += len(batch_texts)
            if callback:
                callback(processed, total)

    def _tokenize(self, text: str) -> list[int]:
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached
        tokens = self._tokenizer.encode(text)
        if isinstance(tokens, list):
            token_ids = tokens
        elif hasattr(tokens, "ids"):
            token_ids = list(tokens.ids)
        else:
            token_ids = list(tokens)
        self._token_cache[text] = token_ids
        return token_ids

    def _pool_hidden(self, hidden, lengths: list[int]) -> list:
        pooled = []
        pending = []
        for i, seq_len in enumerate(lengths):
            if seq_len <= 0:
                pooled.append(None)
                continue
            slice_arr = hidden[i, :seq_len, :]
            if self._pooling == "mean":
                vec = self._backend.mean(slice_arr, axis=0)
            else:
                vec = frechet_mean(
                    slice_arr,
                    backend=self._backend,
                    k_neighbors=self._frechet_k_neighbors,
                    max_k_neighbors=self._frechet_max_k_neighbors,
                )
            self._backend.async_eval(vec)
            pending.append(vec)
            pooled.append(vec)
        if pending:
            self._backend.eval(*pending)
        return pooled

    def _collect_layer_pools(self, texts: list[str], layers: list[int]) -> dict[int, list]:
        if not texts:
            return {layer: [] for layer in layers}
        target_layers = sorted(set(layers))
        max_target = max(target_layers)
        target_set = set(target_layers)

        token_ids = [self._tokenize(text) for text in texts]
        lengths = [len(ids) for ids in token_ids]
        max_len = max(lengths)
        if max_len <= 0:
            return {layer: [None for _ in texts] for layer in target_layers}

        pad_id = getattr(self._tokenizer, "pad_token_id", 0) or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_ids]
        input_ids = self._backend.array(padded)

        hidden = self._embed_tokens(input_ids)
        mask = self._backend.create_causal_mask(max_len, hidden.dtype)

        results: dict[int, list] = {layer: [] for layer in target_layers}
        for layer_idx, layer in enumerate(self._layers):
            hidden = _apply_layer(layer, hidden, mask)
            if layer_idx == len(self._layers) - 1 and self._norm is not None:
                hidden = self._norm(hidden)
            if layer_idx in target_set:
                results[layer_idx] = self._pool_hidden(hidden, lengths)
            if layer_idx >= max_target:
                break

        return results


def _report_payload(report: ConceptDimensionalityReport) -> dict:
    return {
        "layer": report.layer,
        "totalProbes": report.total_probes,
        "analyzedCount": report.analyzed_count,
        "skippedCount": report.skipped_count,
        "meanDimension": report.mean_dimension,
        "weightedMeanDimension": report.weighted_mean_dimension,
        "dimensionHistogram": report.dimension_histogram,
        "domainSummaries": [
            {
                "domain": summary.domain,
                "probeCount": summary.probe_count,
                "meanDimension": summary.mean_dimension,
                "dimensionHistogram": summary.dimension_histogram,
            }
            for summary in report.domain_summaries
        ],
        "results": [
            {
                "probeID": result.probe_id,
                "name": result.name,
                "source": result.source,
                "domain": result.domain,
                "category": result.category,
                "layer": result.layer,
                "supportTextCount": result.support_text_count,
                "sampleCount": result.sample_count,
                "usableCount": result.usable_count,
                "intrinsicDimension": result.intrinsic_dimension,
                "calibrationWeight": result.calibration_weight,
                "confidenceLower": result.ci_lower,
                "confidenceUpper": result.ci_upper,
            }
            for result in report.results
        ],
        "skipped": [
            {
                "probeID": item.probe_id,
                "name": item.name,
                "reason": item.reason,
                "supportTextCount": item.support_text_count,
                "calibrationWeight": item.calibration_weight,
                "activationCount": item.activation_count,
                "invalidCounts": item.invalid_counts,
            }
            for item in report.skipped
        ],
    }


def _collect_probe_texts(probes) -> list[str]:
    texts: list[str] = []
    for probe in probes:
        texts.extend(ConceptDimensionalityAnalyzer._build_support_texts(probe))
    return texts


@app.command("dimensionality")
def atlas_dimensionality(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    batch_size: int = typer.Option(
        8, "--batch-size", help="Batch size for probe activation collection"
    ),
    pooling: str = typer.Option(
        "frechet", "--pooling", help="Token pooling: frechet or mean"
    ),
) -> None:
    """Measure intrinsic dimension for UnifiedAtlas probes at a model layer."""
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)
    target_layer = num_layers - 1

    probes = UnifiedAtlasInventory.all_probes()
    probe_texts = _collect_probe_texts(probes)
    calibration_weights = {}

    backend = get_default_backend()
    pool_mode = pooling.strip().lower()
    if pool_mode not in {"frechet", "mean"}:
        raise typer.BadParameter("Pooling must be 'frechet' or 'mean'.")
    if batch_size < 1:
        raise typer.BadParameter("Batch size must be >= 1.")

    progress = AtlasProgress(logger)
    unique_texts = {
        text for text in (_normalize_probe_text(t) for t in probe_texts) if text is not None
    }
    logger.info(
        "ATLAS: %d probes, %d texts (%d unique)",
        len(probes),
        len(probe_texts),
        len(unique_texts),
    )

    provider = AtlasActivationCache(
        tokenizer,
        embed_tokens,
        layers,
        norm,
        backend,
        pooling=pool_mode,
        batch_size=batch_size,
        frechet_k_neighbors=None,
        frechet_max_k_neighbors=None,
        progress_callback=progress.callback(
            "activations",
            total=len(unique_texts),
            extra={"layers": 1, "batchSize": batch_size},
        ),
    )
    analyzer = ConceptDimensionalityAnalyzer(backend=backend)
    provider.preload_layers(probe_texts, [target_layer])
    report = analyzer.analyze(
        probes=probes,
        activation_provider=provider,
        layer=target_layer,
        calibration_weights=calibration_weights,
        progress_callback=progress.callback(
            "probes",
            total=len(probes),
            extra={"layer": target_layer},
        ),
    )

    payload = {
        "_schema": "mc.geometry.atlas.dimensionality.v1",
        "modelPath": model_path,
        **_report_payload(report),
    }

    if context.output_format == "text":
        histogram = report.dimension_histogram
        lines = [
            "CONCEPT DIMENSIONALITY (UNIFIED ATLAS)",
            f"Model: {model_path}",
            f"Layer: {target_layer}",
            f"Analyzed: {report.analyzed_count}/{report.total_probes} "
            f"(skipped {report.skipped_count})",
        ]
        if report.mean_dimension is not None:
            lines.append(f"Mean Dimension: {report.mean_dimension:.2f}")
        if report.weighted_mean_dimension is not None:
            lines.append(f"Weighted Mean: {report.weighted_mean_dimension:.2f}")
        lines.extend(
            [
                "Histogram: "
                f"1D {histogram.get('1D', 0)} | "
                f"2D {histogram.get('2D', 0)} | "
                f"3D {histogram.get('3D', 0)} | "
                f"4D+ {histogram.get('4D+', 0)}",
                "",
                "Domain Summaries:",
            ]
        )
        for summary in report.domain_summaries:
            mean_dim = summary.mean_dimension
            mean_text = f"{mean_dim:.2f}" if mean_dim is not None else "n/a"
            lines.append(
                f"  {summary.domain}: mean {mean_text}, "
                f"1D {summary.dimension_histogram.get('1D', 0)}, "
                f"2D {summary.dimension_histogram.get('2D', 0)}, "
                f"3D {summary.dimension_histogram.get('3D', 0)}, "
                f"4D+ {summary.dimension_histogram.get('4D+', 0)}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dimensionality-study")
def atlas_dimensionality_study(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    include_results: bool = typer.Option(
        False,
        "--include-results/--summary-only",
        help="Include per-probe results for each layer",
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", help="Batch size for probe activation collection"
    ),
    pooling: str = typer.Option(
        "frechet", "--pooling", help="Token pooling: frechet or mean"
    ),
    layer_chunk_size: int = typer.Option(
        4,
        "--layer-chunk-size",
        help="Layers per activation pass (higher = faster, more memory)",
    ),
) -> None:
    """Run atlas dimensionality across all layers and summarize structure."""
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend

    model, tokenizer = load_model_for_training(model_path)
    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)
    if not resolved:
        raise typer.BadParameter("Could not resolve model architecture.")

    embed_tokens, layers_module, norm = resolved
    num_layers = len(layers_module)
    resolved_layers = list(range(num_layers))

    probes = UnifiedAtlasInventory.all_probes()
    probe_texts = _collect_probe_texts(probes)
    calibration_weights = {}

    backend = get_default_backend()
    pool_mode = pooling.strip().lower()
    if pool_mode not in {"frechet", "mean"}:
        raise typer.BadParameter("Pooling must be 'frechet' or 'mean'.")
    if batch_size < 1:
        raise typer.BadParameter("Batch size must be >= 1.")

    chunk_size = layer_chunk_size
    if chunk_size <= 0:
        chunk_size = len(resolved_layers)
    chunk_size = min(chunk_size, len(resolved_layers))

    progress = AtlasProgress(logger)
    unique_texts = {
        text for text in (_normalize_probe_text(t) for t in probe_texts) if text is not None
    }
    logger.info(
        "ATLAS: %d probes, %d texts (%d unique), %d layers, chunk=%d",
        len(probes),
        len(probe_texts),
        len(unique_texts),
        len(resolved_layers),
        chunk_size,
    )

    provider = AtlasActivationCache(
        tokenizer,
        embed_tokens,
        layers_module,
        norm,
        backend,
        pooling=pool_mode,
        batch_size=batch_size,
        frechet_k_neighbors=None,
        frechet_max_k_neighbors=None,
        progress_callback=progress.callback(
            "activations",
            total=len(unique_texts),
            extra={"layers": chunk_size, "batchSize": batch_size},
        ),
    )
    analyzer = ConceptDimensionalityAnalyzer(backend=backend)

    reports: list[ConceptDimensionalityReport] = []
    chunks = [
        resolved_layers[i : i + chunk_size]
        for i in range(0, len(resolved_layers), chunk_size)
    ]
    for chunk_idx, chunk in enumerate(chunks, start=1):
        logger.info(
            "ATLAS: Preloading chunk %d/%d (layers %s)",
            chunk_idx,
            len(chunks),
            ", ".join(str(layer) for layer in chunk),
        )
        provider.preload_layers(
            probe_texts,
            chunk,
            progress_callback=progress.callback(
                "activations",
                total=len(unique_texts),
                extra={"layers": len(chunk), "batchSize": batch_size},
            ),
        )
        for layer_idx in chunk:
            logger.info(
                "ATLAS: Analyzing layer %d/%d",
                layer_idx + 1,
                len(resolved_layers),
            )
            report = analyzer.analyze(
                probes=probes,
                activation_provider=provider,
                layer=layer_idx,
                calibration_weights=calibration_weights,
                progress_callback=progress.callback(
                    "probes",
                    total=len(probes),
                    extra={"layer": layer_idx},
                ),
            )
            reports.append(report)
        provider.clear_layers(chunk)

    study = ConceptDimensionalityStudy.summarize(reports)

    payload = {
        "_schema": "mc.geometry.atlas.dimensionality_study.v1",
        "modelPath": model_path,
        "layers": study.layers,
        "bottleneckLayer": study.bottleneck_layer,
        "bottleneckMeanDimension": study.bottleneck_mean_dimension,
        "endpointMeanDimension": study.endpoint_mean_dimension,
        "collapseRatio": study.collapse_ratio,
        "meanDomainRankCorrelation": study.mean_domain_rank_correlation,
        "domainRankCorrelations": [
            {
                "layerA": item.layer_a,
                "layerB": item.layer_b,
                "domainCount": item.domain_count,
                "spearman": item.spearman,
            }
            for item in study.domain_rank_correlations
        ],
        "layerSummaries": [
            {
                "layer": summary.layer,
                "meanDimension": summary.mean_dimension,
                "dimensionHistogram": summary.dimension_histogram,
                "domainMeanDimensions": summary.domain_mean_dimensions,
                "domainRank": summary.domain_rank,
            }
            for summary in study.layer_summaries
        ],
        "layerReports": [_report_payload(report) for report in reports]
        if include_results
        else None,
    }

    if context.output_format == "text":
        lines = [
            "ATLAS DIMENSIONALITY STUDY",
            f"Model: {model_path}",
            f"Layers: {', '.join(str(layer) for layer in study.layers)}",
        ]
        if study.bottleneck_layer is not None:
            lines.append(f"Bottleneck Layer: {study.bottleneck_layer}")
        if study.bottleneck_mean_dimension is not None:
            lines.append(f"Bottleneck Mean Dimension: {study.bottleneck_mean_dimension:.3f}")
        if study.collapse_ratio is not None:
            lines.append(f"Collapse Ratio: {study.collapse_ratio:.3f}")
        if study.mean_domain_rank_correlation is not None:
            lines.append(
                f"Mean Domain Rank Correlation: {study.mean_domain_rank_correlation:.3f}"
            )
        lines.append("")
        lines.append("Layer Summaries:")
        for summary in study.layer_summaries:
            mean_dim = summary.mean_dimension
            mean_text = f"{mean_dim:.3f}" if mean_dim is not None else "n/a"
            hist = summary.dimension_histogram
            lines.append(
                f"  L{summary.layer}: mean {mean_text} | "
                f"1D {hist.get('1D', 0)}, 2D {hist.get('2D', 0)}, "
                f"3D {hist.get('3D', 0)}, 4D+ {hist.get('4D+', 0)}"
            )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
