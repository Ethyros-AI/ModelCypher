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

"""Training comparison service: side-by-side adapter and run comparison."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.agent_protocol import (
    AgentDiagnostics,
    AgentEnvelope,
    NextAction,
    AgentRecommendation,
    make_metadata,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass
class MetricDelta:
    """Side-by-side comparison of a single metric."""

    metric: str
    value_a: float | None
    value_b: float | None
    delta: float | None  # b - a
    better: str | None  # "a", "b", "tie"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "delta": self.delta,
            "better": self.better,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two training runs or adapters."""

    label_a: str
    label_b: str
    metrics: list[MetricDelta] = field(default_factory=list)
    winner: str | None = None  # "a", "b", None
    winner_reason: str = ""
    commensurable: bool = True
    # Raw data dicts for envelope construction (not serialized)
    _raw_a: dict[str, Any] = field(default_factory=dict, repr=False)
    _raw_b: dict[str, Any] = field(default_factory=dict, repr=False)
    _meta_a: dict[str, Any] = field(default_factory=dict, repr=False)
    _meta_b: dict[str, Any] = field(default_factory=dict, repr=False)
    _commensurability_notes: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "metrics": [m.to_dict() for m in self.metrics],
            "winner": self.winner,
            "winner_reason": self.winner_reason,
            "commensurable": self.commensurable,
        }


class TrainingComparisonService:
    """Compare training runs or adapters side-by-side."""

    def compare_results(
        self,
        result_a_path: Path,
        result_b_path: Path,
    ) -> ComparisonResult:
        """Compare two saved training result JSON files."""
        raw_a = json.loads(result_a_path.read_text(encoding="utf-8"))
        raw_b = json.loads(result_b_path.read_text(encoding="utf-8"))

        # Extract metadata before unwrapping envelope
        meta_a = raw_a.get("metadata", {}) if isinstance(raw_a, dict) else {}
        meta_b = raw_b.get("metadata", {}) if isinstance(raw_b, dict) else {}

        # Handle AgentEnvelope wrapping
        data_a = raw_a
        data_b = raw_b
        if "result" in data_a and "command" in data_a:
            data_a = data_a["result"]
        if "result" in data_b and "command" in data_b:
            data_b = data_b["result"]

        label_a = data_a.get("adapter_path") or str(result_a_path.stem)
        label_b = data_b.get("adapter_path") or str(result_b_path.stem)

        metrics = self._compare_dicts(data_a, data_b)
        winner, reason = self._determine_winner(metrics, data_a, data_b)

        # Check commensurability from metadata identity fields
        commensurable, mismatch_observations = self._check_commensurability(
            meta_a, meta_b,
        )

        return ComparisonResult(
            label_a=label_a,
            label_b=label_b,
            metrics=metrics,
            winner=winner,
            winner_reason=reason,
            commensurable=commensurable,
            _raw_a=data_a,
            _raw_b=data_b,
            _meta_a=meta_a,
            _meta_b=meta_b,
            _commensurability_notes=mismatch_observations,
        )

    def compare_adapters(
        self,
        model_path: Path,
        adapter_a_path: Path | None,
        adapter_b_path: Path | None,
        data_path: Path | None = None,
        backend: "Backend | None" = None,
    ) -> ComparisonResult:
        """Evaluate two adapters on the same data and compare."""
        if backend is None:
            raise ValueError("Backend required for adapter comparison")

        from modelcypher.core.use_cases.standalone_evaluation_service import (
            StandaloneEvaluationService,
        )

        eval_service = StandaloneEvaluationService(backend=backend)

        if data_path:
            result_a = eval_service.evaluate(
                model_path=model_path,
                adapter_path=adapter_a_path,
                data_path=data_path,
            )
            result_b = eval_service.evaluate(
                model_path=model_path,
                adapter_path=adapter_b_path,
                data_path=data_path,
            )
        else:
            raise ValueError("--data required for adapter comparison")

        dict_a = result_a.to_dict()
        dict_b = result_b.to_dict()

        metrics = self._compare_dicts(dict_a, dict_b)
        winner, reason = self._determine_winner(metrics, dict_a, dict_b)

        return ComparisonResult(
            label_a=str(adapter_a_path) if adapter_a_path is not None else "base_model",
            label_b=str(adapter_b_path) if adapter_b_path is not None else "base_model",
            metrics=metrics,
            winner=winner,
            winner_reason=reason,
            _raw_a=dict_a,
            _raw_b=dict_b,
            _meta_a={"model": str(model_path), "adapter_path": str(adapter_a_path) if adapter_a_path else None},
            _meta_b={"model": str(model_path), "adapter_path": str(adapter_b_path) if adapter_b_path else None},
        )

    def make_envelope(self, result: ComparisonResult) -> AgentEnvelope:
        """Wrap a ComparisonResult in an AgentEnvelope."""
        observations: list[str] = []
        next_actions: list[NextAction] = []
        for m in result.metrics:
            if m.delta is not None:
                observations.append(
                    f"{m.metric}: A={m.value_a}, B={m.value_b} "
                    f"(Δ={m.delta:+.4f}, better={m.better})"
                )

        # Report pipeline gate status (informational, not a winner blocker)
        for label, data in [("A", result._raw_a), ("B", result._raw_b)]:
            if not data:
                continue
            gate = data.get("pipeline_gate_passed")
            if gate is True:
                observations.append(f"Run {label}: pipeline gate passed")
            elif gate is False:
                observations.append(f"Run {label}: pipeline gate failed")
                # None / missing = not evaluated, no observation

        # Report commensurability mismatches
        observations.extend(result._commensurability_notes)

        recs: list[AgentRecommendation] = []
        if not result.commensurable:
            # Fail closed: non-commensurable runs get no winner recommendation
            summary = (
                "Comparison: results are not commensurable — "
                "different model, data, or evaluation surface. "
                "Metric deltas are reported but no winner is recommended."
            )
            status = "partial"
        elif result.winner:
            winner_label = result.label_a if result.winner == "a" else result.label_b
            winner_meta = result._meta_a if result.winner == "a" else result._meta_b
            winner_adapter = winner_meta.get("adapter_path")
            winner_model = winner_meta.get("model")
            recs.append(
                AgentRecommendation(
                    action=f"use_{result.winner}",
                    reason=f"{winner_label} is the better option. {result.winner_reason}",
                )
            )
            if winner_adapter and winner_model:
                export_output = (
                    Path(winner_adapter).parent
                    / f"{Path(winner_adapter).name}-deployment"
                )
                next_actions.append(
                    NextAction(
                        name="export",
                        reason="Export the winning adapter to a deployment-ready target.",
                        command=(
                            f"mc train export -m {winner_model} --adapter {winner_adapter} "
                            f"-o {export_output} --target deployment_quantized"
                        ),
                    )
                )
            summary = (
                f"Comparison: {result.winner.upper()} wins. {result.winner_reason}"
            )
            status = "success"
        else:
            summary = "Comparison: no clear winner between the two runs."
            status = "success"

        result_payload = result.to_dict()
        result_payload["next_actions"] = [action.to_dict() for action in next_actions]
        return AgentEnvelope(
            command="mc train compare",
            status=status,
            result=result_payload,
            diagnostics=AgentDiagnostics(
                summary=summary,
                observations=observations,
                recommendations=recs,
            ),
            metadata=make_metadata(),
            next_actions=next_actions,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # Metrics where lower is better
    _LOWER_IS_BETTER = {
        "post_loss", "final_loss", "post_perplexity",
        "adapted_loss", "adapted_perplexity",
        "base_loss", "base_perplexity",
    }

    # Metrics where higher is better
    _HIGHER_IS_BETTER = {
        "min_cka", "mean_cka", "pipeline_gate_passed",
        "n_improved",
    }

    # Metrics to compare
    _COMPARE_KEYS = [
        "post_loss", "post_perplexity", "final_loss",
        "min_cka", "mean_cka",
        "adapter_saturation_median_ratio",
        "training_time_seconds", "train_iters",
        "adapted_loss", "adapted_perplexity",
        "n_improved", "n_degraded", "n_degenerated",
        "pipeline_gate_passed",
    ]

    def _compare_dicts(
        self, a: dict[str, Any], b: dict[str, Any],
    ) -> list[MetricDelta]:
        """Compare two result dicts on standard metrics."""
        metrics: list[MetricDelta] = []
        for key in self._COMPARE_KEYS:
            va = a.get(key)
            vb = b.get(key)
            if va is None and vb is None:
                continue
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                continue

            delta = float(vb) - float(va)

            if key in self._LOWER_IS_BETTER:
                if delta < 0:
                    better = "b"
                elif delta > 0:
                    better = "a"
                else:
                    better = "tie"
            elif key in self._HIGHER_IS_BETTER:
                if delta > 0:
                    better = "b"
                elif delta < 0:
                    better = "a"
                else:
                    better = "tie"
            else:
                better = None

            metrics.append(
                MetricDelta(
                    metric=key,
                    value_a=float(va),
                    value_b=float(vb),
                    delta=delta,
                    better=better,
                )
            )
        return metrics

    def _determine_winner(
        self,
        metrics: list[MetricDelta],
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Determine overall winner from metric comparisons."""
        # Primary: post_loss (lower is better)
        for m in metrics:
            if m.metric == "post_loss" and m.delta is not None:
                if m.delta < -1e-4:  # TODO: derive tolerance from machine epsilon
                    return "b", f"Lower post-training loss ({m.value_b:.4f} vs {m.value_a:.4f})"
                elif m.delta > 1e-4:  # TODO: derive tolerance from machine epsilon
                    return "a", f"Lower post-training loss ({m.value_a:.4f} vs {m.value_b:.4f})"

        # Secondary: adapted_loss
        for m in metrics:
            if m.metric == "adapted_loss" and m.delta is not None:
                if m.delta < -1e-4:  # TODO: derive tolerance from machine epsilon
                    return "b", f"Lower adapted loss ({m.value_b:.4f} vs {m.value_a:.4f})"
                elif m.delta > 1e-4:  # TODO: derive tolerance from machine epsilon
                    return "a", f"Lower adapted loss ({m.value_a:.4f} vs {m.value_b:.4f})"

        # Tertiary: min_cka (higher is better)
        for m in metrics:
            if m.metric == "min_cka" and m.delta is not None:
                if m.delta > 0.01:  # TODO: derive tolerance from CKA precision
                    return "b", f"Better CKA preservation ({m.value_b:.3f} vs {m.value_a:.3f})"
                elif m.delta < -0.01:  # TODO: derive tolerance from CKA precision
                    return "a", f"Better CKA preservation ({m.value_a:.3f} vs {m.value_b:.3f})"

        # Count wins
        a_wins = sum(1 for m in metrics if m.better == "a")
        b_wins = sum(1 for m in metrics if m.better == "b")

        if a_wins > b_wins:
            return "a", f"Wins on {a_wins}/{len(metrics)} metrics"
        elif b_wins > a_wins:
            return "b", f"Wins on {b_wins}/{len(metrics)} metrics"

        return None, "No clear winner — results are comparable"

    @staticmethod
    def _check_commensurability(
        meta_a: dict[str, Any],
        meta_b: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Check if two results are commensurable based on identity metadata.

        Returns (commensurable, mismatch_observations).
        Results without metadata are treated as commensurable (backward compat).
        """
        observations: list[str] = []
        commensurable = True

        identity_fields = {
            "model_id": "model architecture",
            "data_hash": "training data",
            "eval_data_hash": "evaluation data",
            "benchmark_suite": "benchmark suite",
        }

        for field, label in identity_fields.items():
            val_a = meta_a.get(field)
            val_b = meta_b.get(field)
            if val_a is not None and val_b is not None and val_a != val_b:
                observations.append(
                    f"Results use different {label} — comparison may not be meaningful"
                )
                commensurable = False

        return commensurable, observations
