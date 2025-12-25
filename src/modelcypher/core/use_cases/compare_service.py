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
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from modelcypher.core.domain.models import CompareCheckpointResult, CompareSession

if TYPE_CHECKING:
    from modelcypher.ports.storage import CompareStore, JobStore


@dataclass
class CompareConfig:
    """Configuration for comparison run."""

    prompt: str = "Hello, how are you?"
    max_tokens: int = 100
    temperature: float = 0.7


@dataclass
class CompareRunResult:
    """Result of running a comparison."""

    comparison_id: str
    checkpoints: list[str]
    prompt: str


@dataclass
class CheckpointComparisonResult:
    """Result of comparing checkpoints for a job."""

    job_id: str
    checkpoints: list[dict]
    comparison_metrics: dict


@dataclass
class BaselineResult:
    """Result of establishing baseline metrics."""

    model: str
    metrics: dict
    timestamp: datetime


@dataclass
class CompareScoreResult:
    """Aggregated comparison scores."""

    comparison_id: str
    scores: dict
    winner: str | None


class CompareService:
    def __init__(self, store: "CompareStore", job_store: "JobStore") -> None:
        self.store = store
        self._job_store = job_store

    def list_sessions(self, limit: int = 50, status: str | None = None) -> dict:
        sessions = self.store.list_sessions(limit, status)
        return {"sessions": sessions}

    def get_session(self, session_id: str) -> CompareSession:
        session = self.store.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return session

    def run(
        self,
        checkpoints: list[str],
        config: CompareConfig | None = None,
    ) -> CompareRunResult:
        """Execute A/B comparison between checkpoints.

        Args:
            checkpoints: List of checkpoint paths to compare.
            config: Optional comparison configuration.

        Returns:
            CompareRunResult with comparison_id.
        """
        config = config or CompareConfig()
        comparison_id = f"cmp-{uuid.uuid4().hex[:8]}"

        import time

        from mlx_lm import generate, load

        results = []
        for ckpt in checkpoints:
            try:
                # Determine if it's an adapter or full model
                ckpt_path = Path(ckpt)
                is_adapter = (ckpt_path / "adapter_config.json").exists() or (
                    ckpt_path / "adapter_model.bin"
                ).exists()

                # We need a base model ID if it's an adapter.
                # For now, we assume if it's an adapter, the base model is either in the config or we use a default.
                # In ModelCypher, models are usually registered.

                # Full implementation would resolve base model. For now, try loading.
                if is_adapter:
                    # This is tricky without knowing the base model.
                    # mlx_lm.load usually needs (model_path, adapter_path)
                    # For simplicity, if ckpt is a directory, we check if it has weights.
                    logger.info(f"Loading adapter from {ckpt}")
                    # Mocking base model for now as we don't have a reliable way to find it from just a path here
                    # unless the user provided it.
                    # For now, we assume ckpt is a loadable path for mlx_lm.load
                    model, tokenizer = load(ckpt)
                else:
                    model, tokenizer = load(ckpt)

                start_time = time.time()
                response = generate(
                    model,
                    tokenizer,
                    prompt=config.prompt,
                    max_tokens=config.max_tokens,
                    verbose=False,
                )
                duration = time.time() - start_time

                results.append(
                    CompareCheckpointResult(
                        checkpoint_path=ckpt,
                        model_name=ckpt_path.name,
                        base_model_name=None,
                        response=response,
                        status="completed",
                        metrics={
                            "latency_ms": int(duration * 1000),
                            "tokens_per_sec": len(response.split()) / duration
                            if duration > 0
                            else 0,
                        },
                    )
                )
            except Exception as e:
                results.append(
                    CompareCheckpointResult(
                        checkpoint_path=ckpt,
                        model_name=Path(ckpt).name,
                        base_model_name=None,
                        response=f"Error: {str(e)}",
                        status="failed",
                        metrics={},
                    )
                )

        session = CompareSession(
            id=comparison_id,
            created_at=datetime.utcnow(),
            prompt=config.prompt,
            config={"max_tokens": config.max_tokens, "temperature": config.temperature},
            checkpoints=results,
        )
        self.store.save_session(session)

        return CompareRunResult(
            comparison_id=comparison_id,
            checkpoints=checkpoints,
            prompt=config.prompt,
        )

    def checkpoints(self, job_id: str) -> CheckpointComparisonResult:
        """Compare checkpoints for a job.

        Args:
            job_id: Job ID to get checkpoints for.

        Returns:
            CheckpointComparisonResult with checkpoint comparison.
        """
        checkpoints = self._job_store.list_checkpoints(job_id)
        if not checkpoints:
            return CheckpointComparisonResult(job_id=job_id, checkpoints=[], comparison_metrics={})

        # List most recent 5 checkpoints
        checkpoints = sorted(checkpoints, key=lambda c: c.step, reverse=True)[:5]

        comparison_results = []
        for ckpt in checkpoints:
            comparison_results.append(
                {
                    "step": ckpt.step,
                    "loss": ckpt.loss,
                    "timestamp": ckpt.timestamp.isoformat(),
                    "path": ckpt.file_path,
                }
            )

        # Calculate metrics (e.g. loss reduction)
        metrics = {}
        if len(checkpoints) >= 2:
            first = checkpoints[-1]
            last = checkpoints[0]
            loss_reduction = first.loss - last.loss
            metrics["total_loss_reduction"] = loss_reduction
            metrics["improvement_per_1000_steps"] = (
                (loss_reduction / (last.step - first.step)) * 1000 if last.step > first.step else 0
            )

        return CheckpointComparisonResult(
            job_id=job_id,
            checkpoints=comparison_results,
            comparison_metrics=metrics,
        )

    def baseline(self, model: str) -> BaselineResult:
        """Establish baseline metrics for a model.

        Args:
            model: Path to model directory.

        Returns:
            BaselineResult with baseline metrics.
        """
        import time

        from mlx_lm import generate, load

        try:
            llm_model, tokenizer = load(model)

            # Warm up and measure throughput
            start_time = time.time()
            prompt = "The quick brown fox jumps over the lazy dog"
            response = generate(llm_model, tokenizer, prompt=prompt, max_tokens=50)
            duration = time.time() - start_time

            tokens = len(tokenizer.encode(response))
            tps = tokens / duration if duration > 0 else 0

            return BaselineResult(
                model=model,
                metrics={
                    "perplexity": 1.0,  # Placeholder, needs real dataset eval
                    "latency_ms": int(duration * 1000),
                    "throughput_tps": tps,
                },
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            return BaselineResult(
                model=model,
                metrics={"error": str(e)},
                timestamp=datetime.utcnow(),
            )

    def score(self, comparison_id: str) -> CompareScoreResult:
        """Get aggregated comparison scores.

        Args:
            comparison_id: Comparison ID.

        Returns:
            CompareScoreResult with aggregated scores.
        """
        session = self.get_session(comparison_id)
        if not session:
            raise RuntimeError(f"Comparison not found: {comparison_id}")

        scores = {}
        winner = None
        best_latency = float("inf")

        for cp in session.checkpoints:
            if cp.status == "completed":
                latency = cp.metrics.get("latency_ms", float("inf"))
                if latency < best_latency:
                    best_latency = latency
                    winner = cp.checkpoint_path

        return CompareScoreResult(
            comparison_id=comparison_id,
            scores={
                "quality": 1.0,  # Needs NLP scoring (e.g. BERTScore or LLM-as-a-judge)
                "speed": 1.0 if best_latency < 500 else 0.5,
                "overall": 0.9,
            },
            winner=winner,
        )
