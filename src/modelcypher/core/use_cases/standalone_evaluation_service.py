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

"""Standalone evaluation service: evaluate adapters independently of training.

Three modes:
1. Inference comparison (--prompts): Generate with base vs adapted, compare per-prompt.
2. Loss evaluation (--data): Compute loss/perplexity on a dataset.
3. Benchmark (--benchmark): Run lm-eval suite.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.agent_protocol import (
    AgentDiagnostics,
    AgentEnvelope,
    AgentRecommendation,
    make_metadata,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class InferenceComparison:
    """Per-prompt comparison between base and adapted model."""

    prompt: str
    base_response: str
    adapted_response: str
    reference: str | None
    verdict: str  # "improved" | "degraded" | "unchanged" | "degenerated" | "unmeasured"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt": self.prompt,
            "base_response": self.base_response,
            "adapted_response": self.adapted_response,
            "verdict": self.verdict,
        }
        if self.reference is not None:
            d["reference"] = self.reference
        return d


@dataclass
class StandaloneEvalResult:
    """Result of standalone evaluation."""

    model_path: str
    adapter_path: str | None
    mode: str  # "inference" | "loss" | "benchmark"
    n_prompts: int = 0
    n_improved: int = 0
    n_degraded: int = 0
    n_unchanged: int = 0
    n_degenerated: int = 0
    n_unmeasured: int = 0
    base_perplexity: float | None = None
    adapted_perplexity: float | None = None
    base_loss: float | None = None
    adapted_loss: float | None = None
    per_prompt: list[InferenceComparison] = field(default_factory=list)
    benchmark_results: dict[str, Any] | None = None
    overall_verdict: str = "neutral"  # "improved" | "degraded" | "neutral" | "degenerated"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "mode": self.mode,
            "overall_verdict": self.overall_verdict,
        }
        if self.mode == "inference":
            d["n_prompts"] = self.n_prompts
            d["n_improved"] = self.n_improved
            d["n_degraded"] = self.n_degraded
            d["n_unchanged"] = self.n_unchanged
            d["n_degenerated"] = self.n_degenerated
            d["n_unmeasured"] = self.n_unmeasured
            d["per_prompt"] = [p.to_dict() for p in self.per_prompt]
        if self.base_perplexity is not None:
            d["base_perplexity"] = self.base_perplexity
        if self.adapted_perplexity is not None:
            d["adapted_perplexity"] = self.adapted_perplexity
        if self.base_loss is not None:
            d["base_loss"] = self.base_loss
        if self.adapted_loss is not None:
            d["adapted_loss"] = self.adapted_loss
        if self.benchmark_results is not None:
            d["benchmark_results"] = self.benchmark_results
        return d


class StandaloneEvaluationService:
    """Evaluate trained adapters independently of the training pipeline."""

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def evaluate(
        self,
        model_path: Path,
        adapter_path: Path | None = None,
        prompts_path: Path | None = None,
        data_path: Path | None = None,
        benchmark_suite: str | None = None,
        max_tokens: int = 256,
    ) -> StandaloneEvalResult:
        """Run evaluation in one of three modes.

        Args:
            model_path: Path to base model.
            adapter_path: Optional adapter to evaluate.
            prompts_path: JSONL with {"prompt": "...", "reference": "..."} for inference mode.
            data_path: JSONL dataset for loss/perplexity mode.
            benchmark_suite: lm-eval suite name for benchmark mode.
            max_tokens: Max tokens for inference generation.
        """
        if prompts_path is not None:
            return self._evaluate_inference(
                model_path, adapter_path, prompts_path, max_tokens,
            )
        elif data_path is not None:
            return self._evaluate_loss(model_path, adapter_path, data_path)
        elif benchmark_suite is not None:
            return self._evaluate_benchmark(
                model_path, adapter_path, benchmark_suite,
            )
        else:
            raise ValueError(
                "Specify one of --prompts, --data, or --benchmark for evaluation"
            )

    def make_envelope(
        self,
        result: StandaloneEvalResult,
        model_id_value: str | None = None,
        eval_data_path: str | None = None,
        benchmark_suite: str | None = None,
    ) -> AgentEnvelope:
        """Wrap an eval result in an AgentEnvelope."""
        observations: list[str] = []
        recommendations: list[AgentRecommendation] = []

        if result.mode == "inference":
            obs_parts = [
                f"{result.n_improved} improved",
                f"{result.n_degraded} degraded",
                f"{result.n_unchanged} unchanged",
                f"{result.n_degenerated} degenerated",
            ]
            if result.n_unmeasured > 0:
                obs_parts.append(f"{result.n_unmeasured} unmeasured")
            observations.append(
                f"Inference comparison: {', '.join(obs_parts)} "
                f"out of {result.n_prompts} prompts"
            )
            if result.n_degenerated > 0:
                recommendations.append(
                    AgentRecommendation(
                        action="investigate_degeneration",
                        reason=f"{result.n_degenerated} prompts show degeneration "
                        "(repetitive or incoherent output)",
                    )
                )
        elif result.mode == "loss":
            if result.base_loss is not None and result.adapted_loss is not None:
                delta = result.adapted_loss - result.base_loss
                observations.append(
                    f"Loss: base={result.base_loss:.4f}, "
                    f"adapted={result.adapted_loss:.4f} (Δ={delta:+.4f})"
                )
            if result.base_perplexity is not None and result.adapted_perplexity is not None:
                observations.append(
                    f"Perplexity: base={result.base_perplexity:.2f}, "
                    f"adapted={result.adapted_perplexity:.2f}"
                )

        observations.append(f"Overall verdict: {result.overall_verdict}")

        if result.overall_verdict == "degraded":
            recommendations.append(
                AgentRecommendation(
                    action="try_different_data",
                    reason="Model performance degraded. Consider different training data.",
                )
            )
        elif result.overall_verdict == "improved" and result.adapter_path:
            if result.mode in ("loss", "benchmark"):
                recommendations.append(
                    AgentRecommendation(
                        action="deploy",
                        reason="Adapter improves model performance on measured evaluation.",
                    )
                )
            elif result.mode == "inference":
                recommendations.append(
                    AgentRecommendation(
                        action="evaluate_benchmark",
                        reason="Adapter shows improvement on prompt references. "
                        "Run benchmark evaluation for deployment decision.",
                        command=f"mc train evaluate -m {result.model_path} "
                        f"-a {result.adapter_path} --benchmark quick",
                    )
                )

        summary = self._build_summary(result)

        return AgentEnvelope(
            command="mc train evaluate",
            status="success",
            result=result.to_dict(),
            diagnostics=AgentDiagnostics(
                summary=summary,
                observations=observations,
                recommendations=recommendations,
            ),
            metadata=make_metadata(
                model=str(result.model_path),
                adapter_path=str(result.adapter_path) if result.adapter_path else None,
                model_id_value=model_id_value,
                eval_data_path=eval_data_path,
                benchmark_suite=benchmark_suite,
            ),
        )

    # ------------------------------------------------------------------
    # Internal evaluation modes
    # ------------------------------------------------------------------

    def _evaluate_inference(
        self,
        model_path: Path,
        adapter_path: Path | None,
        prompts_path: Path,
        max_tokens: int,
    ) -> StandaloneEvalResult:
        """Mode 1: Inference comparison (base vs adapted)."""
        from modelcypher.core.domain.training.degeneration import ngram_repetition_rate

        prompts = self._load_prompts(prompts_path)
        if not prompts:
            return StandaloneEvalResult(
                model_path=str(model_path),
                adapter_path=str(adapter_path) if adapter_path else None,
                mode="inference",
                overall_verdict="neutral",
            )

        # Load base model
        model_base, tokenizer = self._backend.load_model(str(model_path))

        # Generate base responses
        base_responses: list[str] = []
        for p in prompts:
            try:
                resp = self._backend.generate(
                    model_base, tokenizer, p["prompt"], max_tokens=max_tokens,
                )
                base_responses.append(resp)
            except Exception as e:
                logger.warning("Base generation failed for prompt: %s", e)
                base_responses.append("")

        # Generate adapted responses (if adapter provided)
        adapted_responses: list[str] = []
        if adapter_path:
            model_adapted, _ = self._backend.load_model(
                str(model_path), adapter_path=str(adapter_path),
            )
            for p in prompts:
                try:
                    resp = self._backend.generate(
                        model_adapted, tokenizer, p["prompt"], max_tokens=max_tokens,
                    )
                    adapted_responses.append(resp)
                except Exception as e:
                    logger.warning("Adapted generation failed for prompt: %s", e)
                    adapted_responses.append("")
        else:
            adapted_responses = [""] * len(prompts)

        # Compare
        comparisons: list[InferenceComparison] = []
        n_improved = n_degraded = n_unchanged = n_degenerated = n_unmeasured = 0

        sqrt_eps = math.sqrt(float(self._backend.finfo().eps))

        for prompt_data, base_resp, adapted_resp in zip(
            prompts, base_responses, adapted_responses,
        ):
            reference = prompt_data.get("reference")

            # Compute ngram repetition rate for BOTH base and adapted
            base_degen_rate = 0.0
            adapted_degen_rate = 0.0
            if base_resp:
                try:
                    base_degen_rate = ngram_repetition_rate(base_resp, n=3)
                except Exception:
                    pass
            if adapted_resp:
                try:
                    adapted_degen_rate = ngram_repetition_rate(adapted_resp, n=3)
                except Exception:
                    pass

            # Degeneration: adapted is measurably more repetitive than base
            if adapted_degen_rate > base_degen_rate + sqrt_eps:
                verdict = "degenerated"
                n_degenerated += 1
            elif reference:
                # Reference substring matching (binary, not heuristic)
                base_match = reference.strip().lower() in base_resp.strip().lower()
                adapted_match = reference.strip().lower() in adapted_resp.strip().lower()
                if adapted_match and not base_match:
                    verdict = "improved"
                    n_improved += 1
                elif base_match and not adapted_match:
                    verdict = "degraded"
                    n_degraded += 1
                else:
                    verdict = "unchanged"
                    n_unchanged += 1
            else:
                # No reference — cannot measure improvement
                verdict = "unmeasured"
                n_unmeasured += 1

            comparisons.append(
                InferenceComparison(
                    prompt=prompt_data["prompt"],
                    base_response=base_resp,
                    adapted_response=adapted_resp,
                    reference=reference,
                    verdict=verdict,
                )
            )

        overall = self._determine_verdict(
            n_improved, n_degraded, n_unchanged, n_degenerated, n_unmeasured,
        )

        return StandaloneEvalResult(
            model_path=str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
            mode="inference",
            n_prompts=len(prompts),
            n_improved=n_improved,
            n_degraded=n_degraded,
            n_unchanged=n_unchanged,
            n_degenerated=n_degenerated,
            n_unmeasured=n_unmeasured,
            per_prompt=comparisons,
            overall_verdict=overall,
        )

    def _evaluate_loss(
        self,
        model_path: Path,
        adapter_path: Path | None,
        data_path: Path,
    ) -> StandaloneEvalResult:
        """Mode 2: Loss/perplexity comparison."""
        from modelcypher.core.use_cases.evaluation_service import EvaluationService

        # Use existing EvaluationService for loss computation
        # Load without adapter for baseline
        eval_service = EvaluationService(
            store=_NullEvalStore(), model_loader=_NullModelLoader(),
        )

        base_result = eval_service.run(
            model=str(model_path), dataset=str(data_path),
        )

        adapted_result = None
        if adapter_path:
            adapted_result = eval_service.run(
                model=str(model_path),
                dataset=str(data_path),
                adapter=str(adapter_path),
            )

        base_loss = base_result.average_loss
        base_ppl = base_result.perplexity
        adapted_loss = adapted_result.average_loss if adapted_result else None
        adapted_ppl = adapted_result.perplexity if adapted_result else None

        if adapted_loss is not None and base_loss > 0:
            if adapted_loss < base_loss:
                verdict = "improved"
            elif adapted_loss > base_loss:
                verdict = "degraded"
            else:
                verdict = "neutral"
        else:
            verdict = "neutral"

        return StandaloneEvalResult(
            model_path=str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
            mode="loss",
            base_loss=base_loss,
            base_perplexity=base_ppl,
            adapted_loss=adapted_loss,
            adapted_perplexity=adapted_ppl,
            overall_verdict=verdict,
        )

    def _evaluate_benchmark(
        self,
        model_path: Path,
        adapter_path: Path | None,
        suite: str,
    ) -> StandaloneEvalResult:
        """Mode 3: lm-eval benchmark."""
        from modelcypher.core.use_cases.benchmark_service import BenchmarkService

        service = BenchmarkService(backend=self._backend)

        base_scores = service.run_suite(
            model_path=model_path, suite_name=suite,
        )

        adapted_scores = None
        if adapter_path:
            adapted_scores = service.run_suite(
                model_path=model_path,
                suite_name=suite,
                adapter_path=adapter_path,
            )

        results: dict[str, Any] = {"base": base_scores.to_dict()}
        if adapted_scores:
            results["adapted"] = adapted_scores.to_dict()

        verdict = "neutral"
        if adapted_scores and hasattr(adapted_scores, "overall") and hasattr(base_scores, "overall"):
            if adapted_scores.overall > base_scores.overall:
                verdict = "improved"
            elif adapted_scores.overall < base_scores.overall:
                verdict = "degraded"

        return StandaloneEvalResult(
            model_path=str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
            mode="benchmark",
            benchmark_results=results,
            overall_verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_prompts(path: Path) -> list[dict[str, str]]:
        """Load prompts from JSONL file."""
        prompts: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "prompt" in data:
                    prompts.append(data)
            except json.JSONDecodeError:
                continue
        return prompts

    @staticmethod
    def _determine_verdict(
        improved: int,
        degraded: int,
        unchanged: int,
        degenerated: int,
        unmeasured: int = 0,
    ) -> str:
        """Determine overall verdict from per-prompt counts.

        Unmeasured prompts are excluded from tallies.
        Degeneration dominates if it outweighs improvement (a comparison,
        not a threshold).
        """
        measured = improved + degraded + unchanged + degenerated
        if measured == 0:
            return "neutral"
        if degenerated > 0 and degenerated >= improved:
            return "degenerated"
        if improved > degraded:
            return "improved"
        if degraded > improved:
            return "degraded"
        return "neutral"

    @staticmethod
    def _build_summary(result: StandaloneEvalResult) -> str:
        """Build a one-sentence summary."""
        if result.mode == "inference":
            return (
                f"Inference evaluation on {result.n_prompts} prompts: "
                f"{result.n_improved} improved, {result.n_degraded} degraded, "
                f"{result.n_degenerated} degenerated. "
                f"Overall: {result.overall_verdict}."
            )
        elif result.mode == "loss":
            parts = []
            if result.base_loss is not None:
                parts.append(f"base_loss={result.base_loss:.4f}")
            if result.adapted_loss is not None:
                parts.append(f"adapted_loss={result.adapted_loss:.4f}")
            return f"Loss evaluation: {', '.join(parts)}. Overall: {result.overall_verdict}."
        else:
            return f"Benchmark evaluation completed. Overall: {result.overall_verdict}."


# ---------------------------------------------------------------------------
# Null implementations for EvaluationService dependency
# ---------------------------------------------------------------------------


class _NullEvalStore:
    """Minimal EvaluationStore that doesn't persist."""

    def save_evaluation(self, result: Any) -> None:
        pass

    def list_evaluations(self, limit: int = 50) -> list:
        return []

    def get_evaluation(self, eval_id: str) -> None:
        return None


class _NullModelLoader:
    """Minimal ModelLoaderPort stub."""

    pass
