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

"""STaR (Self-Taught Reasoner) round orchestration service.

Core loop:
1. Generate on novel, programmatically verifiable problems
2. Filter by deterministic correctness
3. Rationalize failed items with known correct answers
4. Retrain with CE (existing DatasetTrainingService)
5. Evaluate and measure geometric diagnostics
6. Log reproducible artifacts (hashes, seeds, per-problem traces)
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import fmean, median
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.dataset_loading import load_jsonl_dataset
from modelcypher.core.domain.geometry.cka import compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.star.problem_generator import (
    StarProblem,
    StarProblemGenerator,
    extract_final_answer,
    normalize_text,
    parse_yes_no,
)
from modelcypher.core.domain.star.prompting import (
    build_forward_prompt,
    build_rationalization_prompt,
)
from modelcypher.core.domain.training.exceptions import TrainingDerivationError
from modelcypher.core.domain.training.geometric_lora import analyze_weight_geometries
from modelcypher.core.domain.training.spectral_budget import compute_budget_ratios

if TYPE_CHECKING:
    from modelcypher.core.use_cases.dataset_training_service import DatasetTrainingService
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)

STRATEGY_FRESH_BASE = "fresh_base"
STRATEGY_CUMULATIVE_ADAPTER = "cumulative_adapter"


@dataclass(frozen=True)
class StarRunResult:
    """Serializable result payload for a full STaR run."""

    output_root: str
    config: dict[str, Any]
    baseline_eval: dict[str, Any]
    rounds: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "config": self.config,
            "baseline_eval": self.baseline_eval,
            "rounds": self.rounds,
        }


class StarTrainingService:
    """Run multi-round STaR over the existing CE + geometric training pipeline."""

    def __init__(
        self,
        backend: "Backend",
        dataset_training_service: "DatasetTrainingService",
        training_adapter: Any,
    ) -> None:
        self._backend = backend
        self._dataset_training_service = dataset_training_service
        self._training_adapter = training_adapter
        self._id_estimator = IntrinsicDimension(backend)

    def run(
        self,
        model_path: str | Path,
        base_dataset_path: str | Path,
        output_root: str | Path,
        *,
        eval_dataset_path: str | Path | None = None,
        eval_suite_path: str | Path | None = None,
        initial_adapter_path: str | Path | None = None,
        rounds: int = 3,
        problems_per_round: int = 500,
        seed: int = 42,
        few_shot_examples: int = 3,
        max_generation_tokens: int | None = None,
        training_strategy: str = STRATEGY_FRESH_BASE,
    ) -> StarRunResult:
        """Execute full STaR loop and persist per-round artifacts."""
        if rounds <= 0:
            raise ValueError(f"rounds must be positive, got {rounds}")
        if problems_per_round < 500:
            raise ValueError(
                "problems_per_round must be >= 500 to avoid memorization-prone "
                f"small pools, got {problems_per_round}",
            )
        if few_shot_examples < 2 or few_shot_examples > 3:
            raise ValueError(
                "few_shot_examples must be 2 or 3 for STaR prompting, got "
                f"{few_shot_examples}",
            )
        if training_strategy not in {STRATEGY_FRESH_BASE, STRATEGY_CUMULATIVE_ADAPTER}:
            raise ValueError(
                f"Unknown training_strategy: {training_strategy}. "
                f"Use {STRATEGY_FRESH_BASE} or {STRATEGY_CUMULATIVE_ADAPTER}.",
            )

        model_path_obj = Path(model_path).expanduser().resolve()
        base_dataset_path_obj = Path(base_dataset_path).expanduser().resolve()
        output_root_obj = Path(output_root).expanduser().resolve()
        output_root_obj.mkdir(parents=True, exist_ok=True)

        eval_dataset_path_obj = (
            Path(eval_dataset_path).expanduser().resolve() if eval_dataset_path else None
        )
        initial_adapter_obj = (
            Path(initial_adapter_path).expanduser().resolve() if initial_adapter_path else None
        )
        eval_suite_path_obj = (
            Path(eval_suite_path).expanduser().resolve()
            if eval_suite_path
            else self._default_eval_suite_path()
        )

        base_samples = load_jsonl_dataset(base_dataset_path_obj)
        eval_cases = self._load_eval_cases(eval_suite_path_obj)
        base_geometry = self._collect_model_geometry(model_path_obj, adapter_path=None)
        model_hash = self._hash_model_artifacts(model_path_obj)

        baseline_eval = self._evaluate_suite(
            model_path=model_path_obj,
            adapter_path=initial_adapter_obj,
            eval_cases=eval_cases,
            max_generation_tokens=max_generation_tokens,
        )

        run_config: dict[str, Any] = {
            "model_path": str(model_path_obj),
            "base_dataset_path": str(base_dataset_path_obj),
            "eval_dataset_path": str(eval_dataset_path_obj) if eval_dataset_path_obj else None,
            "eval_suite_path": str(eval_suite_path_obj),
            "initial_adapter_path": str(initial_adapter_obj) if initial_adapter_obj else None,
            "rounds": rounds,
            "problems_per_round": problems_per_round,
            "seed": seed,
            "few_shot_examples": few_shot_examples,
            "max_generation_tokens": max_generation_tokens,
            "training_strategy": training_strategy,
            "model_hash": model_hash,
        }
        self._write_json(output_root_obj / "config.json", run_config)

        round_records: list[dict[str, Any]] = []
        current_adapter = initial_adapter_obj
        cumulative_star_samples: list[dict[str, Any]] = []
        seen_problem_prompts: set[str] = set()

        for round_idx in range(1, rounds + 1):
            round_dir = output_root_obj / f"round_{round_idx:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            round_seed = seed + round_idx
            generator = StarProblemGenerator(round_seed)
            problems = self._generate_unique_problems(
                generator=generator,
                count=problems_per_round,
                seen_prompts=seen_problem_prompts,
            )
            self._write_jsonl(
                round_dir / "generated_problems.jsonl",
                [p.to_problem_record() for p in problems],
            )

            generation = self._run_generation_round(
                model_path=model_path_obj,
                adapter_path=current_adapter,
                problems=problems,
                round_idx=round_idx,
                few_shot_examples=few_shot_examples,
                max_generation_tokens=max_generation_tokens,
            )

            self._write_jsonl(round_dir / "generation_log.jsonl", generation["logs"])
            self._write_jsonl(round_dir / "accepted_samples.jsonl", generation["accepted_samples"])

            accepted_count = len(generation["accepted_samples"])
            if accepted_count == 0:
                round_record = {
                    "round_index": round_idx,
                    "seed": round_seed,
                    "status": "stopped_no_verified_samples",
                    "generation": generation["summary"],
                }
                self._write_json(round_dir / "round_summary.json", round_record)
                round_records.append(round_record)
                break

            train_samples: list[dict[str, Any]]
            init_adapter_for_training: Path | None = None

            if training_strategy == STRATEGY_FRESH_BASE:
                cumulative_star_samples.extend(generation["accepted_samples"])
                train_samples = self._dedup_samples(base_samples + cumulative_star_samples)
            else:
                train_samples = self._dedup_samples(generation["accepted_samples"])
                init_adapter_for_training = current_adapter

            train_dataset_path = round_dir / "train_dataset.jsonl"
            self._write_jsonl(train_dataset_path, train_samples)
            dataset_hash = self._hash_file(train_dataset_path)

            training_output_dir = round_dir / "adapter_delta"
            train_result = self._dataset_training_service.train_from_dataset(
                model_path=model_path_obj,
                dataset_path=train_dataset_path,
                output_path=training_output_dir,
                eval_dataset_path=eval_dataset_path_obj,
                seed=round_seed,
                init_adapter_path=init_adapter_for_training,
            )
            train_result_dict = train_result.to_dict()
            self._write_json(round_dir / "train_result.json", train_result_dict)

            produced_adapter = Path(train_result.adapter_path).expanduser().resolve()
            if training_strategy == STRATEGY_CUMULATIVE_ADAPTER and current_adapter is not None:
                composed_adapter = round_dir / "adapter"
                self._compose_adapters(
                    prior_adapter=current_adapter,
                    delta_adapter=produced_adapter,
                    output_adapter=composed_adapter,
                )
                current_adapter = composed_adapter
            else:
                current_adapter = produced_adapter

            eval_result = self._evaluate_suite(
                model_path=model_path_obj,
                adapter_path=current_adapter,
                eval_cases=eval_cases,
                max_generation_tokens=max_generation_tokens,
            )
            self._write_json(round_dir / "eval_result.json", eval_result)

            reasoning_prompts = [case["prompt"] for case in eval_cases]
            diagnostics = self._collect_round_diagnostics(
                model_path=model_path_obj,
                adapter_path=current_adapter,
                base_geometry=base_geometry,
                reasoning_prompts=reasoning_prompts,
                train_result=train_result_dict,
            )
            self._write_json(round_dir / "diagnostics.json", diagnostics)

            adapter_hash = self._hash_directory(current_adapter)
            round_record = {
                "round_index": round_idx,
                "seed": round_seed,
                "adapter_path": str(current_adapter),
                "adapter_hash": adapter_hash,
                "dataset_path": str(train_dataset_path),
                "dataset_hash": dataset_hash,
                "generation": generation["summary"],
                "training": train_result_dict,
                "evaluation": eval_result,
                "diagnostics": diagnostics,
            }
            self._write_json(round_dir / "round_summary.json", round_record)
            round_records.append(round_record)

        result = StarRunResult(
            output_root=str(output_root_obj),
            config=run_config,
            baseline_eval=baseline_eval,
            rounds=round_records,
        )
        self._write_json(output_root_obj / "summary.json", result.to_dict())
        return result

    def _run_generation_round(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        problems: list[StarProblem],
        round_idx: int,
        few_shot_examples: int,
        max_generation_tokens: int | None,
    ) -> dict[str, Any]:
        model, tokenizer = self._backend.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )

        logs: list[dict[str, Any]] = []
        accepted_samples: list[dict[str, Any]] = []
        direct_successes = 0
        rationalized_successes = 0
        failed_after_rationalization = 0
        direct_chain_lengths: list[int] = []
        rational_chain_lengths: list[int] = []
        success_difficulties: list[int] = []
        success_difficulties_by_type: dict[str, list[int]] = {}

        for problem in problems:
            forward_prompt = build_forward_prompt(problem, demonstrations=few_shot_examples)
            forward_raw = self._generate(
                model=model,
                tokenizer=tokenizer,
                prompt=forward_prompt,
                max_generation_tokens=max_generation_tokens,
            )
            forward_completion = self._strip_prompt_prefix(forward_raw, forward_prompt)
            forward_answer = extract_final_answer(forward_completion)
            forward_correct = problem.verify_response(forward_completion)
            forward_tokens = len(self._backend.encode_tokens(tokenizer, forward_completion))

            rational_prompt = ""
            rational_completion = ""
            rational_answer = ""
            rational_correct = False
            rational_tokens = 0
            accepted_mode = ""

            if forward_correct:
                direct_successes += 1
                direct_chain_lengths.append(forward_tokens)
                accepted_mode = "direct"
                accepted_samples.append(
                    self._build_training_sample(problem, forward_completion, round_idx, accepted_mode)
                )
            else:
                rational_prompt = build_rationalization_prompt(
                    problem,
                    demonstrations=max(2, few_shot_examples - 1),
                )
                rational_raw = self._generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rational_prompt,
                    max_generation_tokens=max_generation_tokens,
                )
                rational_completion = self._strip_prompt_prefix(rational_raw, rational_prompt)
                rational_answer = extract_final_answer(rational_completion)
                rational_correct = problem.verify_response(rational_completion)
                rational_tokens = len(self._backend.encode_tokens(tokenizer, rational_completion))

                if rational_correct:
                    rationalized_successes += 1
                    rational_chain_lengths.append(rational_tokens)
                    accepted_mode = "rationalized"
                    accepted_samples.append(
                        self._build_training_sample(
                            problem,
                            rational_completion,
                            round_idx,
                            accepted_mode,
                        )
                    )
                else:
                    failed_after_rationalization += 1

            if accepted_mode:
                success_difficulties.append(problem.difficulty)
                success_difficulties_by_type.setdefault(problem.problem_type, []).append(
                    problem.difficulty,
                )

            logs.append({
                "round": round_idx,
                "problem_id": problem.problem_id,
                "problem_type": problem.problem_type,
                "difficulty": problem.difficulty,
                "verification_fn": problem.verification_fn,
                "correct_answer": problem.correct_answer,
                "forward_answer": forward_answer,
                "forward_correct": forward_correct,
                "forward_chain_length_tokens": forward_tokens,
                "rationalized_answer": rational_answer,
                "rationalized_correct": rational_correct,
                "rationalized_chain_length_tokens": rational_tokens,
                "accepted_mode": accepted_mode or "discarded",
            })

        del model, tokenizer
        self._backend.clear_cache()
        gc.collect()

        total = len(problems)
        failed_forward = total - direct_successes
        mean_chain_lengths: list[int] = direct_chain_lengths + rational_chain_lengths

        summary = {
            "generated_total": total,
            "direct_successes": direct_successes,
            "failed_forward": failed_forward,
            "rationalized_successes": rationalized_successes,
            "failed_after_rationalization": failed_after_rationalization,
            "accepted_total": len(accepted_samples),
            "generation_success_rate": direct_successes / max(1, total),
            "rationalization_success_rate": rationalized_successes / max(1, failed_forward),
            "mean_chain_length_tokens": (
                fmean(mean_chain_lengths) if mean_chain_lengths else 0.0
            ),
            "problem_difficulty_distribution_of_successes": (
                self._summarize_difficulty(success_difficulties_by_type)
            ),
        }

        return {
            "logs": logs,
            "accepted_samples": accepted_samples,
            "summary": summary,
        }

    def _evaluate_suite(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        eval_cases: list[dict[str, str]],
        max_generation_tokens: int | None,
    ) -> dict[str, Any]:
        model, tokenizer = self._backend.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )

        case_results: list[dict[str, Any]] = []
        correct = 0

        for case in eval_cases:
            prompt = case["prompt"]
            raw = self._generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_generation_tokens=max_generation_tokens,
            )
            completion = self._strip_prompt_prefix(raw, prompt)
            extracted = extract_final_answer(completion)
            is_correct = self._verify_eval_case(case, completion, extracted)
            if is_correct:
                correct += 1

            case_results.append({
                "id": case["id"],
                "category": case["category"],
                "prompt": prompt,
                "expected": case["expected"],
                "response": completion,
                "extracted_answer": extracted,
                "correct": is_correct,
            })

        del model, tokenizer
        self._backend.clear_cache()
        gc.collect()

        total = len(eval_cases)
        return {
            "score": correct,
            "total": total,
            "accuracy": correct / max(1, total),
            "per_problem": case_results,
        }

    def _collect_round_diagnostics(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        base_geometry: dict[str, Any],
        reasoning_prompts: list[str],
        train_result: dict[str, Any],
    ) -> dict[str, Any]:
        adapted_geometry = self._collect_model_geometry(model_path, adapter_path=adapter_path)
        cka_stats = self._compute_cka_alignment(model_path, adapter_path, reasoning_prompts)
        id_stats = self._compute_intrinsic_dimensions(model_path, adapter_path, reasoning_prompts)
        weyl_stats = self._compute_weyl_ratios_from_adapter(adapter_path, base_geometry)

        effective_rank_by_layer = {
            layer: geom.effective_rank for layer, geom in adapted_geometry.items()
        }
        tail_dims_by_layer = {
            layer: geom.tail_dims for layer, geom in adapted_geometry.items()
        }
        tail_consumption = {}
        for layer, geom in adapted_geometry.items():
            base_geom = base_geometry.get(layer)
            if base_geom is None:
                continue
            tail_consumption[layer] = base_geom.tail_dims - geom.tail_dims

        return {
            "effective_rank_by_layer": effective_rank_by_layer,
            "tail_dims_by_layer": tail_dims_by_layer,
            "tail_dims_consumption_by_layer": tail_consumption,
            "intrinsic_dimension_by_layer": id_stats["by_layer"],
            "intrinsic_dimension_mean": id_stats["mean"],
            "cka_by_layer": cka_stats["by_layer"],
            "cka_min": cka_stats["min"],
            "cka_mean": cka_stats["mean"],
            "cka_status": cka_stats.get("status", "ok"),
            "cka_reason": cka_stats.get("reason"),
            "weyl_budget_ratio_by_layer": weyl_stats["by_layer"],
            "weyl_budget_ratio_median": weyl_stats["median"],
            "weyl_budget_ratio_max": weyl_stats["max"],
            "weyl_budget_status": weyl_stats.get("status", "ok"),
            "weyl_budget_reason": weyl_stats.get("reason"),
            "training_epoch_metrics": train_result.get("epoch_metrics"),
        }

    def _collect_model_geometry(
        self,
        model_path: Path,
        adapter_path: Path | None,
    ) -> dict[str, Any]:
        model, _tokenizer = self._backend.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        if hasattr(self._training_adapter, "analyze_model_geometry_streaming"):
            geometry = self._training_adapter.analyze_model_geometry_streaming(
                model, use_randomized=True,
            )
        else:
            weights = self._training_adapter.extract_weight_matrices(model)
            geometry = analyze_weight_geometries(weights, self._backend)
        del model, _tokenizer
        self._backend.clear_cache()
        gc.collect()
        return geometry

    def _compute_intrinsic_dimensions(
        self,
        model_path: Path,
        adapter_path: Path | None,
        reasoning_prompts: list[str],
    ) -> dict[str, Any]:
        model, tokenizer = self._backend.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )
        # Collect one prompt at a time and mean-pool (prompts have varying lengths)
        activations: dict[int, list] = {}
        for prompt in reasoning_prompts:
            acts = self._backend.collect_hidden_activations(model, tokenizer, [prompt])
            for layer_idx, act in acts.items():
                pooled = self._backend.mean(act, axis=1)  # [1, seq, h] → [1, h]
                pooled = self._backend.reshape(pooled, (-1,))  # [h]
                self._backend.eval(pooled)
                activations.setdefault(layer_idx, []).append(pooled)

        id_by_layer: dict[str, float] = {}
        for layer_idx, pooled_list in activations.items():
            stacked = self._backend.stack(pooled_list)  # [n, h]
            self._backend.eval(stacked)
            n_samples = int(stacked.shape[0])
            if n_samples < 3:
                continue
            estimate = self._id_estimator.compute(stacked, with_ci=False)
            id_by_layer[str(layer_idx)] = estimate.intrinsic_dimension

        del model, tokenizer
        self._backend.clear_cache()
        gc.collect()
        return {
            "by_layer": id_by_layer,
            "mean": fmean(id_by_layer.values()) if id_by_layer else 0.0,
        }

    def _compute_cka_alignment(
        self,
        model_path: Path,
        adapter_path: Path | None,
        prompts: list[str],
    ) -> dict[str, Any]:
        base_model, base_tokenizer = self._backend.load_model(str(model_path))
        adapted_model, adapted_tokenizer = self._backend.load_model(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path else None,
        )

        # Collect one prompt at a time and mean-pool (prompts have varying lengths)
        base_activations: dict[int, list] = {}
        adapted_activations: dict[int, list] = {}
        for prompt in prompts:
            b_acts = self._backend.collect_hidden_activations(base_model, base_tokenizer, [prompt])
            a_acts = self._backend.collect_hidden_activations(
                adapted_model, adapted_tokenizer, [prompt],
            )
            for layer_idx, act in b_acts.items():
                pooled = self._backend.mean(act, axis=1)
                pooled = self._backend.reshape(pooled, (-1,))
                self._backend.eval(pooled)
                base_activations.setdefault(layer_idx, []).append(pooled)
            for layer_idx, act in a_acts.items():
                pooled = self._backend.mean(act, axis=1)
                pooled = self._backend.reshape(pooled, (-1,))
                self._backend.eval(pooled)
                adapted_activations.setdefault(layer_idx, []).append(pooled)

        cka_by_layer: dict[str, float] = {}
        for layer_idx, base_list in base_activations.items():
            if layer_idx not in adapted_activations:
                continue
            base_stacked = self._backend.stack(base_list)
            adapted_stacked = self._backend.stack(adapted_activations[layer_idx])
            self._backend.eval(base_stacked, adapted_stacked)
            score = compute_linear_cka_from_activations(
                base_stacked, adapted_stacked, self._backend,
            )
            cka_by_layer[str(layer_idx)] = score

        del base_model, base_tokenizer, adapted_model, adapted_tokenizer
        self._backend.clear_cache()
        gc.collect()
        if not cka_by_layer:
            return {
                "by_layer": {},
                "min": None,
                "mean": None,
                "status": "unavailable_measurement",
                "reason": "no_overlapping_layers",
            }
        return {
            "by_layer": cka_by_layer,
            "min": min(cka_by_layer.values()),
            "mean": fmean(cka_by_layer.values()),
            "status": "ok",
            "reason": None,
        }

    def _compute_weyl_ratios_from_adapter(
        self,
        adapter_path: Path | None,
        base_geometry: dict[str, Any],
    ) -> dict[str, Any]:
        if adapter_path is None:
            return {
                "by_layer": {},
                "median": None,
                "max": None,
                "status": "unavailable_measurement",
                "reason": "no_adapter",
            }

        weights_path = self._adapter_weights_path(adapter_path)
        adapter_weights = self._backend.load_safetensors(str(weights_path))
        layer_ratios: dict[str, float] = {}
        ratio_values: list[float] = []

        for key in sorted(adapter_weights.keys()):
            if not key.endswith(".lora_a"):
                continue
            base_key = key[:-7]
            key_b = f"{base_key}.lora_b"
            if key_b not in adapter_weights:
                continue
            layer_key = f"{base_key}.weight"
            base_geom = base_geometry.get(layer_key)
            if base_geom is None or base_geom.sigma_k <= 0:
                continue

            lora_a = adapter_weights[key]
            lora_b = adapter_weights[key_b]
            ratios = compute_budget_ratios(
                [(1.0, lora_a, lora_b, base_geom.sigma_k)],
                self._backend,
            )
            if not ratios:
                continue
            layer_ratio = ratios[0]
            layer_ratios[layer_key] = layer_ratio
            ratio_values.append(layer_ratio)

        if not ratio_values:
            return {
                "by_layer": {},
                "median": None,
                "max": None,
                "status": "unavailable_measurement",
                "reason": "no_valid_budget_ratios",
            }
        return {
            "by_layer": layer_ratios,
            "median": median(ratio_values),
            "max": max(ratio_values),
            "status": "ok",
            "reason": None,
        }

    def _verify_eval_case(
        self,
        case: dict[str, str],
        response: str,
        extracted_answer: str,
    ) -> bool:
        expected = case["expected"].strip()
        expected_core = expected.split("(")[0].strip()
        expected_norm = normalize_text(expected_core or expected)
        response_norm = normalize_text(response)
        extracted_norm = normalize_text(extracted_answer)

        expected_yes_no = parse_yes_no(expected_core)
        if expected_yes_no is not None:
            observed = parse_yes_no(extracted_answer)
            if observed is None:
                observed = parse_yes_no(response)
            return observed is not None and observed == expected_yes_no

        target_number = self._extract_target_number(expected)
        if target_number is not None:
            observed = self._extract_target_number(extracted_answer)
            if observed is None:
                observed = self._extract_target_number(response)
            return observed is not None and observed == target_number

        if expected_norm and expected_norm in response_norm:
            return True
        if expected_norm and expected_norm in extracted_norm:
            return True

        required_terms = [
            token for token in expected_norm.split()
            if len(token) >= 4 and token not in {"therefore", "answer", "must", "with"}
        ]
        if not required_terms:
            return False
        return all(term in response_norm or term in extracted_norm for term in required_terms)

    def _build_training_sample(
        self,
        problem: StarProblem,
        raw_reasoning: str,
        round_idx: int,
        mode: str,
    ) -> dict[str, Any]:
        reasoning = self._sanitize_reasoning(raw_reasoning)
        final_answer_line = f"Final answer: {problem.correct_answer}"
        text = (
            f"{problem.prompt}\n"
            f"Reasoning:\n{reasoning}\n"
            f"{final_answer_line}"
        )
        return {
            "text": text,
            "answer_start": "Reasoning:",
            "logic_id": problem.problem_type,
            "template_id": f"star_{problem.problem_id}",
            "pair_type": f"star_round_{round_idx}_{mode}",
            "problem_type": problem.problem_type,
            "difficulty": problem.difficulty,
            "verification_fn": problem.verification_fn,
        }

    def _sanitize_reasoning(self, reasoning_text: str) -> str:
        cleaned_lines: list[str] = []
        for line in reasoning_text.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("final answer"):
                continue
            cleaned_lines.append(stripped)
        if not cleaned_lines:
            return "The conclusion follows from the stated premises."
        return "\n".join(cleaned_lines)

    def _generate(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_generation_tokens: int | None,
    ) -> str:
        if max_generation_tokens is None:
            return self._backend.generate(model, tokenizer, prompt=prompt)
        return self._backend.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_generation_tokens,
        )

    def _compose_adapters(
        self,
        *,
        prior_adapter: Path,
        delta_adapter: Path,
        output_adapter: Path,
    ) -> None:
        prior_manifest = self._load_geometry_manifest(prior_adapter)
        delta_manifest = self._load_geometry_manifest(delta_adapter)
        prior_hash = str(prior_manifest.get("base_model_hash", ""))
        delta_hash = str(delta_manifest.get("base_model_hash", ""))
        if not prior_hash or not delta_hash or prior_hash != delta_hash:
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail=(
                    "Cannot compose adapters with missing/incompatible geometry manifests: "
                    "base_model_hash mismatch."
                ),
                diagnostics={
                    "prior_adapter": str(prior_adapter),
                    "delta_adapter": str(delta_adapter),
                    "prior_base_model_hash": prior_hash,
                    "delta_base_model_hash": delta_hash,
                },
            )

        prior_weights = self._backend.load_safetensors(str(self._adapter_weights_path(prior_adapter)))
        delta_weights = self._backend.load_safetensors(str(self._adapter_weights_path(delta_adapter)))

        composed: dict[str, Any] = {}
        all_lora_a_keys = {
            key for key in prior_weights if key.endswith(".lora_a")
        } | {
            key for key in delta_weights if key.endswith(".lora_a")
        }

        for lora_a_key in sorted(all_lora_a_keys):
            base_key = lora_a_key[:-7]
            lora_b_key = f"{base_key}.lora_b"

            prior_a = prior_weights.get(lora_a_key)
            prior_b = prior_weights.get(lora_b_key)
            delta_a = delta_weights.get(lora_a_key)
            delta_b = delta_weights.get(lora_b_key)

            if prior_a is None or prior_b is None:
                if delta_a is None or delta_b is None:
                    continue
                composed[lora_a_key] = delta_a
                composed[lora_b_key] = delta_b
                continue

            if delta_a is None or delta_b is None:
                composed[lora_a_key] = prior_a
                composed[lora_b_key] = prior_b
                continue

            composed[lora_a_key] = self._backend.concatenate([prior_a, delta_a], axis=1)
            composed[lora_b_key] = self._backend.concatenate([prior_b, delta_b], axis=0)

        output_adapter.mkdir(parents=True, exist_ok=True)
        self._backend.save_safetensors(
            str(output_adapter / "adapters.safetensors"),
            composed,
            metadata={"composed_from": f"{prior_adapter.name}+{delta_adapter.name}"},
        )

        prior_config = self._load_adapter_config(prior_adapter)
        delta_config = self._load_adapter_config(delta_adapter)

        rank = 0
        if composed:
            sample_key = next(k for k in composed if k.endswith(".lora_a"))
            rank = int(composed[sample_key].shape[1])

        target_modules = sorted(set(
            prior_config.get("target_modules", [])
        ) | set(
            delta_config.get("target_modules", [])
        ))

        sigma_prior = prior_manifest.get("sigma_k_by_module", {})
        sigma_delta = delta_manifest.get("sigma_k_by_module", {})
        if not isinstance(sigma_prior, dict) or not isinstance(sigma_delta, dict):
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="Geometry manifest is missing sigma_k_by_module mapping.",
                diagnostics={
                    "prior_manifest_path": str(prior_adapter / "geometry_manifest.json"),
                    "delta_manifest_path": str(delta_adapter / "geometry_manifest.json"),
                },
            )

        per_module_scale: dict[str, dict[str, float]] = {}
        scale_candidates: list[float] = []
        for lora_a_key in sorted(k for k in composed if k.endswith(".lora_a")):
            base_key = lora_a_key[:-7]
            lora_b_key = f"{base_key}.lora_b"
            layer_key = f"{base_key}.weight"
            lora_b = composed.get(lora_b_key)
            if lora_b is None:
                continue

            sigma_candidates = []
            if layer_key in sigma_prior:
                sigma_candidates.append(float(sigma_prior[layer_key]))
            if layer_key in sigma_delta:
                sigma_candidates.append(float(sigma_delta[layer_key]))
            sigma_candidates = [v for v in sigma_candidates if math.isfinite(v) and v > 0.0]
            if not sigma_candidates:
                raise TrainingDerivationError(
                    failure_class="insufficient_adapter_geometry",
                    detail="Missing sigma_k for composed adapter layer.",
                    diagnostics={
                        "layer": layer_key,
                        "required_files": [
                            str(prior_adapter / "geometry_manifest.json"),
                            str(delta_adapter / "geometry_manifest.json"),
                            str(prior_adapter / "adapter_config.json"),
                            str(delta_adapter / "adapter_config.json"),
                        ],
                    },
                )
            sigma_k = min(sigma_candidates)
            ratios = compute_budget_ratios(
                [(1.0, composed[lora_a_key], lora_b, sigma_k)],
                self._backend,
            )
            if not ratios or not math.isfinite(ratios[0]) or ratios[0] <= 0.0:
                raise TrainingDerivationError(
                    failure_class="insufficient_adapter_geometry",
                    detail="Could not derive composed adapter scale from spectral ratio.",
                    diagnostics={
                        "layer": layer_key,
                        "sigma_k": sigma_k,
                    },
                )

            ratio = ratios[0]
            spectral_norm = ratio * sigma_k
            module_scale = sigma_k / spectral_norm
            per_module_scale[layer_key] = {
                "sigma_k": sigma_k,
                "spectral_norm": spectral_norm,
                "max_scale": module_scale,
            }
            scale_candidates.append(module_scale)

        if not scale_candidates:
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="Composed adapter has no valid modules for geometric scale derivation.",
                diagnostics={
                    "prior_adapter": str(prior_adapter),
                    "delta_adapter": str(delta_adapter),
                },
            )
        derived_scale = min(scale_candidates)

        config = {
            "fine_tune_type": "lora",
            "num_layers": prior_config.get("num_layers", delta_config.get("num_layers")),
            "lora_parameters": {
                "rank": rank,
                "scale": derived_scale,
                "dropout": 0.0,
                "keys": target_modules,
            },
            "target_modules": target_modules,
            "rank": rank,
            "method": "nb_lora_cayley_composed",
            "scale_derivation": {
                "method": "min_module_sigma_k_over_spectral_norm",
                "global_scale": derived_scale,
                "base_model_hash": prior_hash,
                "per_module": per_module_scale,
            },
            "metadata": {
                "prior_adapter": str(prior_adapter),
                "delta_adapter": str(delta_adapter),
            },
        }
        self._write_json(output_adapter / "adapter_config.json", config)
        geometry_manifest = {
            "base_model_hash": prior_hash,
            "target_modules": target_modules,
            "sigma_k_by_module": {
                layer: values["sigma_k"] for layer, values in per_module_scale.items()
            },
            "module_geometry": per_module_scale,
            "derivation": {
                "source": "compose_adapters",
                "method": "min_module_sigma_k_over_spectral_norm",
                "inputs": {
                    "prior_manifest": str(prior_adapter / "geometry_manifest.json"),
                    "delta_manifest": str(delta_adapter / "geometry_manifest.json"),
                },
            },
        }
        self._write_json(output_adapter / "geometry_manifest.json", geometry_manifest)

    def _load_eval_cases(self, eval_suite_path: Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with eval_suite_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    continue
                if {"id", "prompt", "expected", "category"}.issubset(payload):
                    rows.append({
                        "id": str(payload["id"]),
                        "prompt": str(payload["prompt"]),
                        "expected": str(payload["expected"]),
                        "category": str(payload["category"]),
                    })
        if not rows:
            raise ValueError(f"No valid eval cases found in {eval_suite_path}")
        return rows

    def _generate_unique_problems(
        self,
        *,
        generator: StarProblemGenerator,
        count: int,
        seen_prompts: set[str],
    ) -> list[StarProblem]:
        unique: list[StarProblem] = []
        while len(unique) < count:
            before = len(seen_prompts)
            candidates = generator.generate(count - len(unique))
            for problem in candidates:
                if problem.prompt in seen_prompts:
                    continue
                seen_prompts.add(problem.prompt)
                unique.append(problem)
            if len(seen_prompts) == before:
                raise RuntimeError("Unable to generate structurally novel STaR prompts")
        return unique

    def _summarize_difficulty(
        self,
        difficulties_by_type: dict[str, list[int]],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for problem_type, values in sorted(difficulties_by_type.items()):
            if not values:
                continue
            summary[problem_type] = {
                "count": len(values),
                "mean": fmean(values),
                "min": min(values),
                "max": max(values),
            }
        return summary

    def _dedup_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for sample in samples:
            key = (str(sample.get("text", "")), str(sample.get("answer_start", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(sample)
        return deduped

    def _extract_target_number(self, text: str) -> Any | None:
        matches = [
            token for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        ]
        if not matches:
            return None
        try:
            return Decimal(matches[-1])
        except Exception:
            return None

    def _strip_prompt_prefix(self, text: str, prompt: str) -> str:
        stripped = text.strip()
        prompt_stripped = prompt.strip()
        if stripped.startswith(prompt_stripped):
            return stripped[len(prompt_stripped):].strip()
        return stripped

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _hash_directory(self, directory: Path) -> str:
        digest = hashlib.sha256()
        for file_path in sorted(p for p in directory.rglob("*") if p.is_file()):
            digest.update(str(file_path.relative_to(directory)).encode("utf-8"))
            with file_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()

    def _hash_model_artifacts(self, model_path: Path) -> str:
        """Compute SHA256 over model config + full safetensor bytes."""
        digest = hashlib.sha256()
        files: list[Path] = []
        config = model_path / "config.json"
        if config.exists():
            files.append(config)
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            files.append(index_file)
        files.extend(sorted(model_path.glob("*.safetensors")))

        if not files:
            return self._hash_directory(model_path)

        for file_path in files:
            digest.update(str(file_path.relative_to(model_path)).encode("utf-8"))
            with file_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()

    def _adapter_weights_path(self, adapter_path: Path) -> Path:
        safetensor = adapter_path / "adapters.safetensors"
        if safetensor.exists():
            return safetensor
        fallback = adapter_path / "adapter.safetensors"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No adapter weights found under {adapter_path}")

    def _load_adapter_config(self, adapter_path: Path) -> dict[str, Any]:
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="Adapter composition requires adapter_config.json.",
                diagnostics={
                    "adapter_path": str(adapter_path),
                    "missing_path": str(config_path),
                    "required_files": [
                        str(adapter_path / "adapter_config.json"),
                        str(adapter_path / "geometry_manifest.json"),
                        str(adapter_path / "adapters.safetensors"),
                    ],
                },
            )
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        raise TrainingDerivationError(
            failure_class="insufficient_adapter_geometry",
            detail="adapter_config.json must contain a JSON object.",
            diagnostics={
                "adapter_path": str(adapter_path),
                "config_path": str(config_path),
                "payload_type": type(payload).__name__,
            },
        )

    def _load_geometry_manifest(self, adapter_path: Path) -> dict[str, Any]:
        manifest_path = adapter_path / "geometry_manifest.json"
        if not manifest_path.exists():
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="Adapter composition requires geometry_manifest.json.",
                diagnostics={
                    "adapter_path": str(adapter_path),
                    "missing_path": str(manifest_path),
                    "required_files": [
                        str(adapter_path / "geometry_manifest.json"),
                        str(adapter_path / "adapter_config.json"),
                        str(adapter_path / "adapters.safetensors"),
                    ],
                },
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TrainingDerivationError(
                failure_class="insufficient_adapter_geometry",
                detail="geometry_manifest.json must contain a JSON object.",
                diagnostics={
                    "adapter_path": str(adapter_path),
                    "manifest_path": str(manifest_path),
                    "payload_type": type(payload).__name__,
                },
            )
        return payload

    def _default_eval_suite_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "data" / "eval_prompts" / "nblora_inference_tests.jsonl"


__all__ = [
    "STRATEGY_CUMULATIVE_ADAPTER",
    "STRATEGY_FRESH_BASE",
    "StarRunResult",
    "StarTrainingService",
]
