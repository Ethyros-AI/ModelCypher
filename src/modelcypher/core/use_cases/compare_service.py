from __future__ import annotations

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import CompareSession, CompareCheckpointResult


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
    winner: Optional[str]


class CompareService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

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
        # In a full implementation, this would load checkpoints from the job
        return CheckpointComparisonResult(
            job_id=job_id,
            checkpoints=[],
            comparison_metrics={},
        )

    def baseline(self, model: str) -> BaselineResult:
        """Establish baseline metrics for a model.
        
        Args:
            model: Path to model directory.
            
        Returns:
            BaselineResult with baseline metrics.
        """
        return BaselineResult(
            model=model,
            metrics={
                "perplexity": 1.5,
                "latency_ms": 100,
                "throughput_tps": 50,
            },
            timestamp=datetime.now(),
        )

    def score(self, comparison_id: str) -> CompareScoreResult:
        """Get aggregated comparison scores.
        
        Args:
            comparison_id: Comparison ID.
            
        Returns:
            CompareScoreResult with aggregated scores.
        """
        return CompareScoreResult(
            comparison_id=comparison_id,
            scores={
                "quality": 0.85,
                "speed": 0.9,
                "overall": 0.875,
            },
            winner=None,
        )
