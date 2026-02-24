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

"""Validation harness for derived dataset training.

Runs repeated training trials with derived settings and captures
counterexamples where post-training metrics fail to improve over baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


@dataclass(frozen=True)
class DerivedTrainingTrial:
    """Single repeated training trial result."""

    trial_index: int
    seed: int
    baseline_loss: float
    post_loss: float
    loss_delta: float
    baseline_perplexity: float
    post_perplexity: float
    perplexity_delta: float
    spectral_bounds_ok: bool
    max_spectral_ratio: float
    stop_reason: str
    train_iters: int
    training_time_seconds: float
    min_cka: float | None
    mean_cka: float | None
    passed: bool
    failure_modes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "seed": self.seed,
            "baseline_loss": self.baseline_loss,
            "post_loss": self.post_loss,
            "loss_delta": self.loss_delta,
            "baseline_perplexity": self.baseline_perplexity,
            "post_perplexity": self.post_perplexity,
            "perplexity_delta": self.perplexity_delta,
            "spectral_bounds_ok": self.spectral_bounds_ok,
            "max_spectral_ratio": self.max_spectral_ratio,
            "stop_reason": self.stop_reason,
            "train_iters": self.train_iters,
            "training_time_seconds": self.training_time_seconds,
            "min_cka": self.min_cka,
            "mean_cka": self.mean_cka,
            "passed": self.passed,
            "failure_modes": list(self.failure_modes),
        }


@dataclass(frozen=True)
class DerivedTrainingValidationResult:
    """Aggregate validation result across repeated trials."""

    model_path: str
    dataset_path: str
    eval_dataset_path: str | None
    trials_requested: int
    trial_results: tuple[DerivedTrainingTrial, ...]
    counterexamples: tuple[DerivedTrainingTrial, ...]
    pass_count: int
    fail_count: int
    all_passed: bool
    seed_source: str
    seeds: tuple[int, ...]
    min_loss_delta: float
    mean_loss_delta: float
    min_perplexity_delta: float
    mean_perplexity_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "eval_dataset_path": self.eval_dataset_path,
            "trials_requested": self.trials_requested,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "all_passed": self.all_passed,
            "seed_source": self.seed_source,
            "seeds": list(self.seeds),
            "min_loss_delta": self.min_loss_delta,
            "mean_loss_delta": self.mean_loss_delta,
            "min_perplexity_delta": self.min_perplexity_delta,
            "mean_perplexity_delta": self.mean_perplexity_delta,
            "trial_results": [trial.to_dict() for trial in self.trial_results],
            "counterexamples": [trial.to_dict() for trial in self.counterexamples],
        }


class DerivedTrainingValidationService:
    """Run repeated trials to validate derived training behavior."""

    def __init__(self, dataset_training_service: Any, backend: Any) -> None:
        self._dataset_training_service = dataset_training_service
        self._backend = backend

    def validate(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        eval_dataset_path: str | Path | None,
        trials: int,
        base_seed: int | None = None,
        seq_length: int | None = None,
    ) -> DerivedTrainingValidationResult:
        if trials <= 0:
            raise ValueError("trials must be positive")

        resolved_model_path = Path(model_path).expanduser().resolve()
        resolved_dataset_path = Path(dataset_path).expanduser().resolve()
        resolved_eval_path = (
            Path(eval_dataset_path).expanduser().resolve()
            if eval_dataset_path is not None
            else None
        )

        if base_seed is None:
            seed_root = self._derive_seed_root(
                model_path=resolved_model_path,
                dataset_path=resolved_dataset_path,
            )
            seed_source = "derived_from_model_dataset_hash"
        else:
            seed_root = int(base_seed)
            seed_source = "user_supplied_base_seed"

        seeds = tuple(seed_root + idx for idx in range(trials))

        trials_out: list[DerivedTrainingTrial] = []
        loss_deltas: list[float] = []
        ppl_deltas: list[float] = []

        for index, seed in enumerate(seeds):
            result = self._dataset_training_service.train_from_dataset_research(
                model_path=resolved_model_path,
                dataset_path=resolved_dataset_path,
                eval_dataset_path=resolved_eval_path,
                seed=seed,
                seq_length=seq_length,
                auto_regime=True,
                no_save=True,
            )
            loss_delta = float(result.baseline_loss - result.post_loss)
            perplexity_delta = float(result.baseline_perplexity - result.post_perplexity)
            loss_deltas.append(loss_delta)
            ppl_deltas.append(perplexity_delta)

            trial = self._build_trial(
                index=index,
                seed=seed,
                train_result=result,
                loss_delta=loss_delta,
                perplexity_delta=perplexity_delta,
            )
            trials_out.append(trial)

        pass_count = sum(1 for trial in trials_out if trial.passed)
        fail_count = len(trials_out) - pass_count
        counterexamples = tuple(trial for trial in trials_out if not trial.passed)

        return DerivedTrainingValidationResult(
            model_path=str(resolved_model_path),
            dataset_path=str(resolved_dataset_path),
            eval_dataset_path=str(resolved_eval_path) if resolved_eval_path else None,
            trials_requested=trials,
            trial_results=tuple(trials_out),
            counterexamples=counterexamples,
            pass_count=pass_count,
            fail_count=fail_count,
            all_passed=fail_count == 0,
            seed_source=seed_source,
            seeds=seeds,
            min_loss_delta=min(loss_deltas),
            mean_loss_delta=sum(loss_deltas) / len(loss_deltas),
            min_perplexity_delta=min(ppl_deltas),
            mean_perplexity_delta=sum(ppl_deltas) / len(ppl_deltas),
        )

    def _build_trial(
        self,
        index: int,
        seed: int,
        train_result: Any,
        loss_delta: float,
        perplexity_delta: float,
    ) -> DerivedTrainingTrial:
        eps = machine_epsilon(
            self._backend,
            self._backend.array(
                [
                    train_result.baseline_loss,
                    train_result.post_loss,
                    train_result.baseline_perplexity,
                    train_result.post_perplexity,
                ]
            ),
        )
        loss_improved = loss_delta > eps
        perplexity_improved = perplexity_delta > eps
        bounds_ok = bool(train_result.spectral_bounds_ok)

        failure_modes: list[str] = []
        if not loss_improved:
            failure_modes.append("loss_not_improved")
        if not perplexity_improved:
            failure_modes.append("perplexity_not_improved")
        if not bounds_ok:
            failure_modes.append("spectral_bounds_violation")

        return DerivedTrainingTrial(
            trial_index=index,
            seed=seed,
            baseline_loss=float(train_result.baseline_loss),
            post_loss=float(train_result.post_loss),
            loss_delta=loss_delta,
            baseline_perplexity=float(train_result.baseline_perplexity),
            post_perplexity=float(train_result.post_perplexity),
            perplexity_delta=perplexity_delta,
            spectral_bounds_ok=bounds_ok,
            max_spectral_ratio=float(train_result.max_spectral_ratio),
            stop_reason=str(train_result.stop_reason),
            train_iters=int(train_result.train_iters),
            training_time_seconds=float(train_result.training_time_seconds),
            min_cka=(
                float(train_result.min_cka)
                if train_result.min_cka is not None
                else None
            ),
            mean_cka=(
                float(train_result.mean_cka)
                if train_result.mean_cka is not None
                else None
            ),
            passed=(len(failure_modes) == 0),
            failure_modes=tuple(failure_modes),
        )

    @staticmethod
    def _derive_seed_root(model_path: Path, dataset_path: Path) -> int:
        payload = f"{model_path}:{dataset_path}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big")
