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

"""Experimental benchmark runner for routed adapter generation.

Produces side-by-side lenient and strict scoring to separate extraction artifacts
from true behavioral gains.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.star.problem_generator import (
    extract_final_answer,
    normalize_text,
    parse_last_decimal,
    parse_yes_no,
)
from modelcypher.core.use_cases.adapter_routing_service import AdapterRoutingService

if TYPE_CHECKING:
    from modelcypher.core.domain.inference.adapter_routing import AdapterPool

logger = logging.getLogger(__name__)

_NEGATION_TOKENS = {
    "no",
    "not",
    "never",
    "cannot",
    "cant",
    "won't",
    "wont",
    "neither",
}

_YES_TOKENS = {"yes", "true", "correct"}
_NO_TOKENS = {"no", "false", "incorrect"}


@dataclass(frozen=True)
class ScoreResult:
    """Score payload for one response."""

    is_correct: bool
    rule: str
    contradiction: bool
    expected_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "rule": self.rule,
            "contradiction": self.contradiction,
            "expected_type": self.expected_type,
        }


class AdapterRoutingBenchmarkService:
    """Run and score routed-vs-single adapter generation experiments."""

    def __init__(self, routing_service: AdapterRoutingService | None = None) -> None:
        self._routing_service = routing_service or AdapterRoutingService()
        self._backend = self._routing_service._backend

    def sweep_selectors(
        self,
        *,
        base_model_path: str,
        adapter_paths: list[str],
        eval_suite_path: str,
        max_tokens: int = 128,
        pair_label: str | None = None,
        selectors: list[str] | None = None,
        output_directory: str | None = None,
    ) -> dict[str, Any]:
        """Run a selector sweep and return comparative strict-scoring summary."""
        selector_list = selectors or ["min_kl", "max_kl", "max_cosine", "min_cosine"]
        reports: dict[str, dict[str, Any]] = {}
        selectors_run: list[str] = []

        output_root: Path | None = None
        if output_directory is not None:
            output_root = Path(output_directory).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)

        for selector in selector_list:
            try:
                report = self.run(
                    base_model_path=base_model_path,
                    adapter_paths=adapter_paths,
                    eval_suite_path=eval_suite_path,
                    selection_method=selector,
                    max_tokens=max_tokens,
                    pair_label=pair_label,
                )
            except Exception:
                logger.exception("Selector sweep failed for %s", selector)
                continue

            reports[selector] = report
            selectors_run.append(selector)
            if output_root is not None:
                (output_root / f"{selector}.json").write_text(
                    json.dumps(report, indent=2),
                    encoding="utf-8",
                )

        per_selector = {
            selector: report.get("summary", {})
            for selector, report in reports.items()
        }
        comparison: dict[str, dict[str, Any]] = {}
        for selector, report in reports.items():
            summary = report.get("summary", {})
            systems_summary = summary.get("systems", {})
            strict_correct = {
                system: int(values.get("strict", {}).get("correct", 0))
                for system, values in systems_summary.items()
                if isinstance(values, dict)
            }
            dominance_gate = summary.get("dominance_gate", {})
            comparison[selector] = {
                "strict_correct": strict_correct,
                "recommended_mode": dominance_gate.get("recommended_mode"),
                "recommended_system": dominance_gate.get("recommended_system"),
            }

        routed_scores: dict[str, int] = {}
        for selector, report in reports.items():
            systems = report.get("systems", [])
            summary = report.get("summary", {})
            systems_summary = summary.get("systems", {})
            routed_system = next((system for system in systems if str(system).startswith("routed_")), None)
            if routed_system is None:
                for system in systems_summary:
                    if str(system).startswith("routed_"):
                        routed_system = str(system)
                        break
            if routed_system is None:
                continue
            routed_scores[selector] = int(
                systems_summary.get(routed_system, {}).get("strict", {}).get("correct", 0),
            )

        best_selector: str | None = None
        if routed_scores:
            max_score = max(routed_scores.values())
            best_selector = sorted(
                selector for selector, score in routed_scores.items() if score == max_score
            )[0]

        sweep_summary = {
            "selectors_run": selectors_run,
            "per_selector": per_selector,
            "comparison": comparison,
            "best_selector": best_selector,
        }
        if output_root is not None:
            (output_root / "sweep_summary.json").write_text(
                json.dumps(sweep_summary, indent=2),
                encoding="utf-8",
            )
        return sweep_summary

    def run_and_save(
        self,
        *,
        base_model_path: str,
        adapter_paths: list[str],
        eval_suite_path: str,
        selection_method: str = "min_kl",
        max_tokens: int = 128,
        output_path: str | None = None,
        pair_label: str | None = None,
    ) -> str:
        """Run benchmark and persist a JSON report."""
        report = self.run(
            base_model_path=base_model_path,
            adapter_paths=adapter_paths,
            eval_suite_path=eval_suite_path,
            selection_method=selection_method,
            max_tokens=max_tokens,
            pair_label=pair_label,
        )

        output = Path(output_path).expanduser().resolve() if output_path else self._default_output_path(
            selection_method=selection_method,
            pair_label=pair_label,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return str(output)

    def run(
        self,
        *,
        base_model_path: str,
        adapter_paths: list[str],
        eval_suite_path: str,
        selection_method: str = "min_kl",
        max_tokens: int = 128,
        pair_label: str | None = None,
    ) -> dict[str, Any]:
        """Run routed generation benchmark for one adapter pair."""
        pool = self._routing_service.load_adapter_pool(base_model_path, adapter_paths)
        cases = self._load_eval_cases(Path(eval_suite_path))
        adapter_ids = [identity.id for identity in pool.adapter_identities]
        systems = ["base", *adapter_ids, f"routed_{selection_method}"]

        supported_layers = self._supported_layers_by_adapter(pool)
        supported_union = set().union(*supported_layers.values()) if supported_layers else set()

        results_root = Path("results/adapter_routing").expanduser().resolve()
        results_root.mkdir(parents=True, exist_ok=True)
        composites_dir = results_root / f"composites_{selection_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        composites_dir.mkdir(parents=True, exist_ok=True)

        lenient_correct = Counter()
        strict_correct = Counter()
        category_totals = Counter()
        category_lenient = {system: Counter() for system in systems}
        category_strict = {system: Counter() for system in systems}
        layer_winner_counts = defaultdict(Counter)

        case_records: list[dict[str, Any]] = []

        for index, case in enumerate(cases):
            prompt = case["prompt"]
            expected = case["expected"]
            category = case.get("category", "unknown")
            case_id = case.get("id", f"case_{index+1:03d}")

            trace = self._routing_service.collect_routing_measurements(
                pool=pool,
                prompt=prompt,
                selection_method=selection_method,
            )
            filtered_routing = {
                layer: adapter_id
                for layer, adapter_id in trace.selected_adapter_per_layer.items()
                if layer in supported_layers.get(adapter_id, set())
            }
            dropped_layers = sorted(set(trace.selected_adapter_per_layer) - set(filtered_routing))

            if not filtered_routing:
                fallback = adapter_ids[0]
                filtered_routing = {layer: fallback for layer in sorted(supported_union)}
                dropped_layers = sorted(set(trace.selected_adapter_per_layer))

            for layer, adapter_id in filtered_routing.items():
                layer_winner_counts[layer][adapter_id] += 1

            composite_case_dir = composites_dir / f"{index:03d}_{case_id}"
            composite_path = self._routing_service.build_composite_adapter(
                pool=pool,
                routing_decisions=filtered_routing,
                output_directory=str(composite_case_dir),
            )

            composite_model, composite_tokenizer = self._backend.load_model(
                pool.base_model_path,
                adapter_path=composite_path,
            )

            responses: dict[str, str] = {
                "base": self._generate_greedy(pool.base_model, pool.base_tokenizer, prompt, max_tokens),
            }
            for adapter_id, (model, tokenizer) in sorted(pool.adapter_models.items()):
                responses[adapter_id] = self._generate_greedy(model, tokenizer, prompt, max_tokens)

            routed_key = f"routed_{selection_method}"
            responses[routed_key] = self._generate_greedy(
                composite_model,
                composite_tokenizer,
                prompt,
                max_tokens,
            )

            del composite_model, composite_tokenizer
            self._backend.clear_cache()

            lenient_scores: dict[str, dict[str, Any]] = {}
            strict_scores: dict[str, dict[str, Any]] = {}
            expected_type = self._expected_type(expected)

            for system in systems:
                lenient = self.score_lenient(expected, responses[system])
                strict = self.score_strict(expected, responses[system])
                lenient_scores[system] = lenient.to_dict()
                strict_scores[system] = strict.to_dict()
                lenient_correct[system] += int(lenient.is_correct)
                strict_correct[system] += int(strict.is_correct)
                category_lenient[system][category] += int(lenient.is_correct)
                category_strict[system][category] += int(strict.is_correct)

            category_totals[category] += 1
            case_records.append(
                {
                    "id": case_id,
                    "category": category,
                    "prompt": prompt,
                    "expected": expected,
                    "expected_type": expected_type,
                    "routing": {
                        "selection_method": selection_method,
                        "raw_selected_adapter_per_layer": trace.selected_adapter_per_layer,
                        "filtered_selected_adapter_per_layer": filtered_routing,
                        "dropped_layers": dropped_layers,
                    },
                    "scores": {
                        "lenient": lenient_scores,
                        "strict": strict_scores,
                    },
                    "responses": responses,
                    "composite_adapter_path": composite_path,
                },
            )

        n_cases = len(cases)
        summary = {
            "systems": {
                system: {
                    "lenient": self._metric_payload(lenient_correct[system], n_cases),
                    "strict": self._metric_payload(strict_correct[system], n_cases),
                }
                for system in systems
            },
            "category": {
                category: {
                    "total": int(category_totals[category]),
                    **{
                        system: {
                            "lenient": self._metric_payload(
                                category_lenient[system][category],
                                category_totals[category],
                            ),
                            "strict": self._metric_payload(
                                category_strict[system][category],
                                category_totals[category],
                            ),
                        }
                        for system in systems
                    },
                }
                for category in sorted(category_totals)
            },
            "layer_winner_counts_after_filter": {
                str(layer): dict(counts)
                for layer, counts in sorted(layer_winner_counts.items())
            },
        }
        summary["dominance_gate"] = self._build_dominance_gate(
            systems=systems,
            systems_summary=summary["systems"],
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "pair_label": pair_label,
            "base_model_path": base_model_path,
            "eval_suite_path": str(Path(eval_suite_path).expanduser().resolve()),
            "selection_method": selection_method,
            "max_tokens": max_tokens,
            "n_cases": n_cases,
            "adapter_identities": [
                {
                    "id": identity.id,
                    "path": identity.path,
                    "rank": identity.rank,
                    "scale": identity.scale,
                    "supported_layers": sorted(supported_layers.get(identity.id, set())),
                    "module_keys": list(identity.module_keys),
                }
                for identity in pool.adapter_identities
            ],
            "systems": systems,
            "summary": summary,
            "cases": case_records,
        }

    def rescore_existing_report(self, report_path: str, output_path: str | None = None) -> str:
        """Re-score an existing report with strict+lenient scorers."""
        source = Path(report_path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "cases" not in payload:
            raise ValueError(f"Invalid report structure at {source}")

        systems = payload.get("systems")
        if not isinstance(systems, list):
            inferred = set()
            for case in payload.get("cases", []):
                if isinstance(case, dict):
                    responses = case.get("responses")
                    if isinstance(responses, dict):
                        inferred.update(responses.keys())
            systems = sorted(inferred)

        if not systems:
            raise ValueError(f"No systems found in report {source}")

        lenient_correct = Counter()
        strict_correct = Counter()
        category_totals = Counter()
        category_lenient = {system: Counter() for system in systems}
        category_strict = {system: Counter() for system in systems}

        updated_cases: list[dict[str, Any]] = []
        for case in payload.get("cases", []):
            if not isinstance(case, dict):
                continue
            expected = str(case.get("expected", ""))
            category = str(case.get("category", "unknown"))
            responses = case.get("responses")
            if not isinstance(responses, dict):
                continue

            lenient_scores: dict[str, dict[str, Any]] = {}
            strict_scores: dict[str, dict[str, Any]] = {}
            for system in systems:
                response = str(responses.get(system, ""))
                lenient = self.score_lenient(expected, response)
                strict = self.score_strict(expected, response)
                lenient_scores[system] = lenient.to_dict()
                strict_scores[system] = strict.to_dict()
                lenient_correct[system] += int(lenient.is_correct)
                strict_correct[system] += int(strict.is_correct)
                category_lenient[system][category] += int(lenient.is_correct)
                category_strict[system][category] += int(strict.is_correct)

            category_totals[category] += 1

            updated = dict(case)
            updated["expected_type"] = self._expected_type(expected)
            updated["scores"] = {
                "lenient": lenient_scores,
                "strict": strict_scores,
            }
            updated_cases.append(updated)

        n_cases = len(updated_cases)
        summary = {
            "systems": {
                system: {
                    "lenient": self._metric_payload(lenient_correct[system], n_cases),
                    "strict": self._metric_payload(strict_correct[system], n_cases),
                }
                for system in systems
            },
            "category": {
                category: {
                    "total": int(category_totals[category]),
                    **{
                        system: {
                            "lenient": self._metric_payload(
                                category_lenient[system][category],
                                category_totals[category],
                            ),
                            "strict": self._metric_payload(
                                category_strict[system][category],
                                category_totals[category],
                            ),
                        }
                        for system in systems
                    },
                }
                for category in sorted(category_totals)
            },
        }
        summary["dominance_gate"] = self._build_dominance_gate(
            systems=systems,
            systems_summary=summary["systems"],
        )

        rewritten = dict(payload)
        rewritten["rescored_at"] = datetime.now().isoformat()
        rewritten["scorer"] = "adapter_routing_benchmark_service_v1"
        rewritten["systems"] = systems
        rewritten["n_cases"] = n_cases
        rewritten["summary"] = summary
        rewritten["cases"] = updated_cases

        if output_path is None:
            output = source.with_name(f"{source.stem}_rescored.json")
        else:
            output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")
        return str(output)

    def score_lenient(self, expected: str, response: str) -> ScoreResult:
        """Lenient scorer with contradiction guards."""
        expected_type = self._expected_type(expected)
        extracted = extract_final_answer(response)

        if expected_type == "yes_no":
            contradiction = self._has_yes_no_contradiction(response)
            expected_bool = parse_yes_no(expected)
            observed = parse_yes_no(extracted)
            if observed is None:
                observed = parse_yes_no(response)
            is_correct = observed is not None and observed == expected_bool and not contradiction
            return ScoreResult(
                is_correct=is_correct,
                rule="yes_no_lenient",
                contradiction=contradiction,
                expected_type=expected_type,
            )

        if expected_type == "numeric":
            expected_num = parse_last_decimal(expected)
            observed = self._extract_numeric_candidate(response)
            return ScoreResult(
                is_correct=(observed is not None and expected_num is not None and observed == expected_num),
                rule="numeric_lenient",
                contradiction=False,
                expected_type=expected_type,
            )

        expected_norm = normalize_text(expected)
        response_norm = normalize_text(response)
        extracted_norm = normalize_text(extracted)

        contradiction = self._is_text_contradicted(expected_norm, response_norm)
        if expected_norm and (expected_norm in response_norm or expected_norm in extracted_norm) and not contradiction:
            return ScoreResult(
                is_correct=True,
                rule="text_substring_with_guard",
                contradiction=False,
                expected_type=expected_type,
            )

        required_terms = [
            token
            for token in expected_norm.split()
            if len(token) >= 4 and token not in {"therefore", "answer", "must", "with"}
        ]
        if not required_terms:
            return ScoreResult(
                is_correct=False,
                rule="text_no_required_terms",
                contradiction=contradiction,
                expected_type=expected_type,
            )

        is_correct = all(term in response_norm or term in extracted_norm for term in required_terms)
        if contradiction:
            is_correct = False
        return ScoreResult(
            is_correct=is_correct,
            rule="text_term_overlap_with_guard",
            contradiction=contradiction,
            expected_type=expected_type,
        )

    def score_strict(self, expected: str, response: str) -> ScoreResult:
        """Strict scorer emphasizing final answer consistency."""
        expected_type = self._expected_type(expected)
        extracted = extract_final_answer(response)

        if expected_type == "yes_no":
            contradiction = self._has_yes_no_contradiction(response)
            expected_bool = parse_yes_no(expected)
            observed = parse_yes_no(extracted)
            if observed is None:
                observed = parse_yes_no(response)
            return ScoreResult(
                is_correct=(observed is not None and observed == expected_bool and not contradiction),
                rule="yes_no_strict",
                contradiction=contradiction,
                expected_type=expected_type,
            )

        if expected_type == "numeric":
            expected_num = parse_last_decimal(expected)
            observed = self._extract_numeric_final(response)
            return ScoreResult(
                is_correct=(observed is not None and expected_num is not None and observed == expected_num),
                rule="numeric_strict_final",
                contradiction=False,
                expected_type=expected_type,
            )

        expected_norm = normalize_text(expected)
        extracted_norm = normalize_text(extracted)
        contradiction = self._is_text_contradicted(expected_norm, normalize_text(response))
        is_correct = bool(expected_norm and expected_norm in extracted_norm and not contradiction)
        return ScoreResult(
            is_correct=is_correct,
            rule="text_exact_in_final_with_guard",
            contradiction=contradiction,
            expected_type=expected_type,
        )

    def _expected_type(self, expected: str) -> str:
        if parse_yes_no(expected) is not None:
            return "yes_no"
        if parse_last_decimal(expected) is not None:
            return "numeric"
        return "text"

    def _supported_layers_by_adapter(self, pool: "AdapterPool") -> dict[str, set[int]]:
        return {
            identity.id: self._extract_supported_layers(identity.module_keys)
            for identity in pool.adapter_identities
        }

    def _extract_supported_layers(self, module_keys: tuple[str, ...]) -> set[int]:
        layers: set[int] = set()
        for key in module_keys:
            parts = key.split(".")
            if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                try:
                    layers.add(int(parts[2]))
                except ValueError:
                    continue
        return layers

    def _generate_greedy(self, model: Any, tokenizer: Any, prompt: str, max_tokens: int) -> str:
        try:
            return self._backend.generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=0.0,
                top_p=1.0,
            )
        except TypeError:
            return self._backend.generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
            )

    def _load_eval_cases(self, eval_suite_path: Path) -> list[dict[str, str]]:
        cases: list[dict[str, str]] = []
        for line in eval_suite_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            prompt = payload.get("prompt")
            expected = payload.get("expected")
            if not isinstance(prompt, str) or not isinstance(expected, str):
                continue
            cases.append(
                {
                    "id": str(payload.get("id", f"case_{len(cases)+1:03d}")),
                    "category": str(payload.get("category", "unknown")),
                    "prompt": prompt,
                    "expected": expected,
                },
            )
        if not cases:
            raise ValueError(f"No valid benchmark cases found in {eval_suite_path}")
        return cases

    def _default_output_path(self, selection_method: str, pair_label: str | None) -> Path:
        root = Path("results/adapter_routing").expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{pair_label}" if pair_label else ""
        return root / f"routed_generation_benchmark_{selection_method}{suffix}_{stamp}.json"

    def _metric_payload(self, correct: int, total: int) -> dict[str, Any]:
        return {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(correct / total) if total else 0.0,
        }

    def _build_dominance_gate(
        self,
        *,
        systems: list[str],
        systems_summary: dict[str, dict[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        strict_correct = {
            system: int(systems_summary[system]["strict"]["correct"])
            for system in systems
            if system in systems_summary and "strict" in systems_summary[system]
        }
        if not strict_correct:
            return {
                "criterion": "max_strict_correct",
                "recommended_mode": "unknown",
                "recommended_system": None,
                "tied_systems": [],
                "dominant_single_adapter": False,
                "dominant_routed": False,
                "strict_margins": {},
            }

        routed_systems = [s for s in systems if s.startswith("routed_") and s in strict_correct]
        single_systems = [s for s in systems if s != "base" and not s.startswith("routed_") and s in strict_correct]

        max_correct = max(strict_correct.values())
        tied_systems = sorted([system for system, correct in strict_correct.items() if correct == max_correct])

        recommended_system = tied_systems[0] if len(tied_systems) == 1 else None
        if len(tied_systems) > 1:
            recommended_mode = "tie"
        elif recommended_system in routed_systems:
            recommended_mode = "routed"
        elif recommended_system in single_systems:
            recommended_mode = "single_adapter"
        else:
            recommended_mode = "base"

        best_single = max(single_systems, key=lambda system: strict_correct.get(system, -1), default=None)
        best_routed = max(routed_systems, key=lambda system: strict_correct.get(system, -1), default=None)

        strict_margins: dict[str, int] = {}
        if best_routed is not None and best_single is not None:
            strict_margins["routed_minus_best_single_correct"] = (
                strict_correct[best_routed] - strict_correct[best_single]
            )
        if best_routed is not None and "base" in strict_correct:
            strict_margins["routed_minus_base_correct"] = strict_correct[best_routed] - strict_correct["base"]
        if best_single is not None and "base" in strict_correct:
            strict_margins["best_single_minus_base_correct"] = strict_correct[best_single] - strict_correct["base"]

        return {
            "criterion": "max_strict_correct",
            "recommended_mode": recommended_mode,
            "recommended_system": recommended_system,
            "tied_systems": tied_systems if len(tied_systems) > 1 else [],
            "best_single_system": best_single,
            "best_routed_system": best_routed,
            "dominant_single_adapter": (
                recommended_mode == "single_adapter"
                and best_single is not None
                and best_routed is not None
                and strict_correct[best_single] > strict_correct[best_routed]
            ),
            "dominant_routed": (
                recommended_mode == "routed"
                and best_single is not None
                and best_routed is not None
                and strict_correct[best_routed] > strict_correct[best_single]
            ),
            "strict_margins": strict_margins,
        }

    def _has_yes_no_contradiction(self, response: str) -> bool:
        tokens = [token.strip(".,;:!?") for token in normalize_text(response).split()]
        seen_yes = any(token in _YES_TOKENS for token in tokens)
        seen_no = any(token in _NO_TOKENS for token in tokens)
        return seen_yes and seen_no

    def _is_text_contradicted(self, expected_norm: str, response_norm: str) -> bool:
        if not expected_norm:
            return False
        response_tokens = response_norm.split()
        expected_tokens = expected_norm.split()
        if not expected_tokens:
            return False

        positions: list[int] = []
        for idx in range(0, len(response_tokens) - len(expected_tokens) + 1):
            if response_tokens[idx : idx + len(expected_tokens)] == expected_tokens:
                positions.append(idx)

        if not positions:
            return False

        for start in positions:
            window_start = max(0, start - 4)
            context = response_tokens[window_start:start]
            if not any(token in _NEGATION_TOKENS for token in context):
                return False
        return True

    def _extract_numeric_candidate(self, response: str) -> Decimal | None:
        final = self._extract_numeric_final(response)
        if final is not None:
            return final
        return self._parse_last_decimal(response)

    def _extract_numeric_final(self, response: str) -> Decimal | None:
        hash_matches = re.findall(r"####\\s*([-+]?\\d+(?:\\.\\d+)?)", response)
        if hash_matches:
            try:
                return Decimal(hash_matches[-1])
            except InvalidOperation:
                pass
        extracted = extract_final_answer(response)
        candidate = self._parse_last_decimal(extracted)
        if candidate is not None:
            return candidate
        return None

    def _parse_last_decimal(self, text: str) -> Decimal | None:
        raw = parse_last_decimal(text)
        if raw is not None:
            return raw
        matches = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", text)
        if not matches:
            return None
        try:
            return Decimal(matches[-1])
        except InvalidOperation:
            return None


__all__ = ["AdapterRoutingBenchmarkService", "ScoreResult"]
