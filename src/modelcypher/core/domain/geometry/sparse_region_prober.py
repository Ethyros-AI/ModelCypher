from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Callable, Optional

from modelcypher.core.domain.geometry.sparse_region_domains import ProbeCorpus, SparseRegionDomains, DomainDefinition
from modelcypher.core.domain.geometry.sparse_region_locator import LayerActivationStats, SparseRegionLocator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Configuration:
    prompts_per_domain: int = 8
    max_tokens_per_prompt: int = 50
    temperature: float = 0.1
    capture_all_layers: bool = True
    warmup_prompts: int = 1


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
    def __init__(self, configuration: Configuration | None = None) -> None:
        self.config = configuration or Configuration()

    def probe(
        self,
        domain: DomainDefinition,
        total_layers: int,
        generate_tokens: Callable[[str, int, Callable[[dict[int, float]], None]], int],
        progress: Callable[[ProbeProgress], None] | None = None,
    ) -> DomainProbeResult:
        start_time = time.time()
        corpus = ProbeCorpus(domain=domain, max_prompts=self.config.prompts_per_domain, shuffle=True)

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
                tokens = generate_tokens(prompt, self.config.max_tokens_per_prompt, _capture)
                prompt_means = {
                    layer: sum(values) / float(len(values))
                    for layer, values in prompt_layer_activations.items()
                    if values
                }
                all_prompt_activations.append(prompt_means)
                tokens_generated += tokens
                prompts_processed += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Probe failed for prompt %s in domain %s: %s", index, domain.name, exc)

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
        results: list[DomainProbeResult] = []
        total_domains = len(domains)
        for domain_index, domain in enumerate(domains):
            result = self.probe(
                domain=domain,
                total_layers=total_layers,
                generate_tokens=generate_tokens,
                progress=(
                    lambda p: progress(
                        ProbeProgress(
                            current_prompt=domain_index * self.config.prompts_per_domain + p.current_prompt,
                            total_prompts=total_domains * self.config.prompts_per_domain,
                            domain_name=domain.name,
                            status=f"Probing {domain.name}...",
                        )
                    )
                    if progress
                    else None
                ),
            )
            results.append(result)
        return results

    @staticmethod
    def activations_from_hidden_states(states: dict[int, object]) -> dict[int, float]:
        return {layer: SparseRegionProber.compute_activation(state) for layer, state in states.items()}

    @staticmethod
    def compute_activation(hidden_state: object) -> float:
        from modelcypher.core.domain._backend import get_default_backend

        if hasattr(hidden_state, "shape"):
            b = get_default_backend()
            norm = b.norm(hidden_state)
            b.eval(norm)
            return float(b.to_numpy(norm).item())

        total = SparseRegionProber._sum_squares(hidden_state)
        return math.sqrt(total)

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
        for layer, values in layer_values.items():
            if not values:
                continue
            mean = sum(values) / float(len(values))
            max_val = max(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / float(max(1, len(values) - 1))
                if len(values) > 1
                else 0.0
            )
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
        if hasattr(value, "tolist"):
            return SparseRegionProber._sum_squares(value.tolist())
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return 0.0
        return scalar * scalar
