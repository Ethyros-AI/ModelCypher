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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.sparse_region_domains import (
    DomainDefinition,
    ProbeCorpus,
    SparseRegionDomains,
)
from modelcypher.core.domain.geometry.sparse_region_locator import (
    LayerActivationStats,
    SparseRegionLocator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - prompts_per_domain: use all available prompts from corpus
# - max_tokens_per_prompt: derived from prompt length (2x prompt tokens)
# =============================================================================


@dataclass(frozen=True)
class ProbeProgress:
    current_prompt: int
    total_prompts: int
    domain_name: str
    status: str

    @property
    def percentage(self) -> float:
        return float(self.current_prompt) / float(max(1, self.total_prompts))


@dataclass(frozen=True)
class DomainProbeResult:
    domain: DomainDefinition
    layer_stats: list[LayerActivationStats]
    prompts_processed: int
    tokens_generated: int
    duration: float
    prompt_activations: list[dict[int, float]]

    def generate_report(self) -> str:
        report_lines = [
            f"# Domain Probe Report: {self.domain.name}",
            "",
            "## Overview",
            f"- Description: {self.domain.description}",
            f"- Category: {self.domain.category.value}",
            f"- Prompts Processed: {self.prompts_processed}",
            f"- Tokens Generated: {self.tokens_generated}",
            f"- Duration: {self.duration:.2f}s",
            "",
            "## Layer Activations",
        ]

        for stat in sorted(self.layer_stats, key=lambda item: item.layer_index):
            report_lines.append(
                f"Layer {stat.layer_index}: mean={stat.mean_activation:.4f}, max={stat.max_activation:.4f}, var={stat.activation_variance:.4f}"
            )

        means = [stat.mean_activation for stat in self.layer_stats]
        if means:
            avg_mean = sum(means) / float(len(means))
            max_mean = max(means)
            min_mean = min(means)
            report_lines.extend(
                [
                    "",
                    "## Summary",
                    f"- Average Layer Mean: {avg_mean:.4f}",
                    f"- Max Layer Mean: {max_mean:.4f}",
                    f"- Min Layer Mean: {min_mean:.4f}",
                    f"- Layer Range: {(max_mean - min_mean):.4f}",
                ]
            )

        return "\n".join(report_lines)


class SparseRegionProber:
    """Probes sparse regions in neural network layers.

    All parameters are derived from data - no configuration needed.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _derive_max_tokens(prompt: str) -> int:
        """Derive max tokens from prompt length.

        Response length is proportional to prompt complexity.
        Uses word count as a proxy for semantic content - response should
        be proportional to input complexity.
        """
        # Word count as proxy for semantic complexity
        word_count = len(prompt.split())
        # Response proportional to input (minimum 1 token to capture activations)
        return max(1, word_count)

    def probe(
        self,
        domain: DomainDefinition,
        total_layers: int,
        generate_tokens: Callable[[str, int, Callable[[dict[int, float]], None]], int],
        progress: Callable[[ProbeProgress], None] | None = None,
    ) -> DomainProbeResult:
        start_time = time.time()
        # Use all available prompts from corpus (no artificial limit)
        corpus = ProbeCorpus(domain=domain, max_prompts=None, shuffle=True)

        all_prompt_activations: list[dict[int, float]] = []
        tokens_generated = 0
        prompts_processed = 0

        for index, prompt in enumerate(corpus.prompts):
            if progress:
                progress(
                    ProbeProgress(
                        current_prompt=index + 1,
                        total_prompts=corpus.count,
                        domain_name=domain.name,
                        status="Probing...",
                    )
                )

            prompt_layer_activations: dict[int, list[float]] = {}

            def _capture(layer_activations: dict[int, float]) -> None:
                for layer, activation in layer_activations.items():
                    prompt_layer_activations.setdefault(layer, []).append(float(activation))

            try:
                # Derive max tokens from prompt length
                max_tokens = self._derive_max_tokens(prompt)
                tokens = generate_tokens(prompt, max_tokens, _capture)
                prompt_means = {
                    layer: sum(values) / float(len(values))
                    for layer, values in prompt_layer_activations.items()
                    if values
                }
                all_prompt_activations.append(prompt_means)
                tokens_generated += tokens
                prompts_processed += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Probe failed for prompt %s in domain %s: %s", index, domain.name, exc
                )

        layer_stats = self._aggregate_to_stats(all_prompt_activations, total_layers)
        duration = time.time() - start_time

        logger.info(
            "Probed domain %s: %s prompts, %s tokens, %.2fs",
            domain.name,
            prompts_processed,
            tokens_generated,
            duration,
        )

        return DomainProbeResult(
            domain=domain,
            layer_stats=layer_stats,
            prompts_processed=prompts_processed,
            tokens_generated=tokens_generated,
            duration=duration,
            prompt_activations=all_prompt_activations,
        )

    def probe_baseline(
        self,
        total_layers: int,
        generate_tokens: Callable[[str, int, Callable[[dict[int, float]], None]], int],
        progress: Callable[[ProbeProgress], None] | None = None,
    ) -> DomainProbeResult:
        return self.probe(
            domain=SparseRegionDomains.baseline,
            total_layers=total_layers,
            generate_tokens=generate_tokens,
            progress=progress,
        )

    def analyze_sparsity(
        self,
        domain: DomainDefinition,
        total_layers: int,
        generate_tokens: Callable[[str, int, Callable[[dict[int, float]], None]], int],
        dare_analysis=None,
        progress: Callable[[ProbeProgress], None] | None = None,
    ):
        """Analyze sparsity for a domain.

        All parameters are derived from data - no configuration needed.
        """
        baseline = self.probe_baseline(
            total_layers=total_layers,
            generate_tokens=generate_tokens,
            progress=(
                lambda p: progress(
                    ProbeProgress(
                        current_prompt=p.current_prompt,
                        total_prompts=p.total_prompts * 2,
                        domain_name="baseline",
                        status="Probing baseline...",
                    )
                )
                if progress
                else None
            ),
        )

        domain_result = self.probe(
            domain=domain,
            total_layers=total_layers,
            generate_tokens=generate_tokens,
            progress=(
                lambda p: progress(
                    ProbeProgress(
                        current_prompt=p.current_prompt + p.total_prompts,
                        total_prompts=p.total_prompts * 2,
                        domain_name=domain.name,
                        status="Probing domain...",
                    )
                )
                if progress
                else None
            ),
        )

        # All locator parameters derived from data - no config needed
        locator = SparseRegionLocator()
        return locator.analyze(
            domain_stats=domain_result.layer_stats,
            baseline_stats=baseline.layer_stats,
            dare_analysis=dare_analysis,
            domain=domain.name,
        )

    def probe_domains(
        self,
        domains: list[DomainDefinition],
        total_layers: int,
        generate_tokens: Callable[[str, int, Callable[[dict[int, float]], None]], int],
        progress: Callable[[ProbeProgress], None] | None = None,
    ) -> list[DomainProbeResult]:
        """Probe multiple domains.

        Uses all available prompts from each domain corpus.
        """
        results: list[DomainProbeResult] = []
        total_domains = len(domains)
        prompts_completed = 0

        for domain_index, domain in enumerate(domains):
            # Capture current offset for progress reporting
            current_offset = prompts_completed

            def make_progress_fn(offset: int, domain_name: str):
                def fn(p: ProbeProgress) -> None:
                    if progress:
                        progress(
                            ProbeProgress(
                                current_prompt=offset + p.current_prompt,
                                total_prompts=p.total_prompts * total_domains,
                                domain_name=domain_name,
                                status=f"Probing {domain_name}...",
                            )
                        )
                return fn

            result = self.probe(
                domain=domain,
                total_layers=total_layers,
                generate_tokens=generate_tokens,
                progress=make_progress_fn(current_offset, domain.name) if progress else None,
            )
            prompts_completed += result.prompts_processed
            results.append(result)
        return results

    @staticmethod
    def activations_from_hidden_states(states: dict[int, object]) -> dict[int, float]:
        return {
            layer: SparseRegionProber.compute_activation(state) for layer, state in states.items()
        }

    @staticmethod
    def compute_activation(hidden_state: object) -> float:
        from modelcypher.core.domain._backend import get_default_backend

        b = get_default_backend()
        arr = hidden_state if hasattr(hidden_state, "shape") else b.array(hidden_state)
        norms = geodesic_norms(b.reshape(arr, (1, -1)), b)
        b.eval(norms)
        return float(b.to_scalar(norms[0]))

    def _aggregate_to_stats(
        self, prompt_activations: list[dict[int, float]], total_layers: int
    ) -> list[LayerActivationStats]:
        if not prompt_activations:
            return []

        layer_values: dict[int, list[float]] = {layer: [] for layer in range(total_layers)}
        for prompt in prompt_activations:
            for layer, value in prompt.items():
                layer_values.setdefault(layer, []).append(float(value))

        stats: list[LayerActivationStats] = []
        backend = get_default_backend()
        for layer, values in layer_values.items():
            if not values:
                continue
            # Use backend for mean, max, and variance computation
            arr = backend.array(values)
            mean_arr = backend.mean(arr)
            max_arr = backend.max(arr)
            backend.eval(mean_arr, max_arr)
            mean = float(backend.to_scalar(mean_arr))
            max_val = float(backend.to_scalar(max_arr))

            # Variance using backend ops
            if len(values) > 1:
                diff = arr - mean
                sum_sq = backend.sum(diff * diff)
                backend.eval(sum_sq)
                variance = float(backend.to_scalar(sum_sq)) / float(len(values) - 1)
            else:
                variance = 0.0
            stats.append(
                LayerActivationStats(
                    layer_index=layer,
                    mean_activation=mean,
                    max_activation=max_val,
                    activation_variance=variance,
                    prompt_count=len(values),
                )
            )
        stats.sort(key=lambda item: item.layer_index)
        return stats

    @staticmethod
    def _sum_squares(value: object) -> float:
        if isinstance(value, (list, tuple)):
            return sum(SparseRegionProber._sum_squares(item) for item in value)
        if hasattr(value, "shape") or hasattr(value, "tolist"):
            from modelcypher.core.domain._backend import get_default_backend

            backend = get_default_backend()
            try:
                sum_sq = backend.sum(value * value)
                backend.eval(sum_sq)
                return float(backend.to_scalar(sum_sq))
            except Exception:
                try:
                    return SparseRegionProber._sum_squares(backend.tolist(value))
                except Exception:
                    return SparseRegionProber._sum_squares(value.tolist())
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return 0.0
        return scalar * scalar
