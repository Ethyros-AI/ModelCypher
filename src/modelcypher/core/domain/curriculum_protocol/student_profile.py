"""Student profile data structures for curriculum generation protocol.

The StudentProfile is a JSON-serializable report card that tells a frontier
model what the student model knows and what its geometric capacity looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeometricProfile:
    """Summary of a model's geometric state."""

    architecture: str
    model_family: str
    parameter_count: int
    hidden_dim: int
    num_layers: int
    vocab_size: int
    context_length: int
    mean_effective_rank: float | None = None
    mean_intrinsic_dimension: float | None = None
    spectral_budget_remaining: float | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "architecture": self.architecture,
            "model_family": self.model_family,
            "parameter_count": self.parameter_count,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
        }
        if self.mean_effective_rank is not None:
            d["mean_effective_rank"] = self.mean_effective_rank
        if self.mean_intrinsic_dimension is not None:
            d["mean_intrinsic_dimension"] = self.mean_intrinsic_dimension
        if self.spectral_budget_remaining is not None:
            d["spectral_budget_remaining"] = self.spectral_budget_remaining
        return d

    @classmethod
    def from_dict(cls, d: dict) -> GeometricProfile:
        return cls(
            architecture=d["architecture"],
            model_family=d["model_family"],
            parameter_count=d["parameter_count"],
            hidden_dim=d["hidden_dim"],
            num_layers=d["num_layers"],
            vocab_size=d["vocab_size"],
            context_length=d["context_length"],
            mean_effective_rank=d.get("mean_effective_rank"),
            mean_intrinsic_dimension=d.get("mean_intrinsic_dimension"),
            spectral_budget_remaining=d.get("spectral_budget_remaining"),
        )


@dataclass(frozen=True)
class SkillAssessment:
    """Per-skill mastery measurement."""

    skill_name: str
    accuracy: float
    ci_lower: float
    ci_upper: float
    n_total: int
    n_correct: int
    regime: str
    is_mastered: bool
    answer_mode: str

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "accuracy": self.accuracy,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "regime": self.regime,
            "is_mastered": self.is_mastered,
            "answer_mode": self.answer_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillAssessment:
        return cls(**d)


@dataclass(frozen=True)
class StallDiagnostic:
    """Diagnostic data for a skill that has stalled during training."""

    skill_name: str
    rounds_attempted: int
    accuracy_history: tuple[float, ...]
    training_loss_history: tuple[float, ...]
    samples_trained_on: int
    cka_pre_post: float | None = None
    spectral_budget_consumed: float | None = None
    common_failure_patterns: tuple[str, ...] = ()
    n_distinct_wrong_answers: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "skill_name": self.skill_name,
            "rounds_attempted": self.rounds_attempted,
            "accuracy_history": list(self.accuracy_history),
            "training_loss_history": list(self.training_loss_history),
            "samples_trained_on": self.samples_trained_on,
            "common_failure_patterns": list(self.common_failure_patterns),
            "n_distinct_wrong_answers": self.n_distinct_wrong_answers,
        }
        if self.cka_pre_post is not None:
            d["cka_pre_post"] = self.cka_pre_post
        if self.spectral_budget_consumed is not None:
            d["spectral_budget_consumed"] = self.spectral_budget_consumed
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StallDiagnostic:
        return cls(
            skill_name=d["skill_name"],
            rounds_attempted=d["rounds_attempted"],
            accuracy_history=tuple(d["accuracy_history"]),
            training_loss_history=tuple(d["training_loss_history"]),
            samples_trained_on=d["samples_trained_on"],
            cka_pre_post=d.get("cka_pre_post"),
            spectral_budget_consumed=d.get("spectral_budget_consumed"),
            common_failure_patterns=tuple(d.get("common_failure_patterns", ())),
            n_distinct_wrong_answers=d.get("n_distinct_wrong_answers", 0),
        )


@dataclass(frozen=True)
class StudentProfile:
    """Complete student model profile for curriculum generation."""

    model_path: str
    model_id: str
    geometric_profile: GeometricProfile
    schema_version: str = "mc.student_profile.v1"
    skill_assessments: tuple[SkillAssessment, ...] = ()
    mastered_skills: tuple[str, ...] = ()
    frontier_skills: tuple[str, ...] = ()
    blocked_skills: tuple[str, ...] = ()
    benchmark_baselines: dict[str, float] = field(default_factory=dict)
    training_rounds_completed: int = 0
    total_training_samples_seen: int = 0
    profiled_at: str = ""
    stall_diagnostics: tuple[StallDiagnostic, ...] | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "schema_version": self.schema_version,
            "model_path": self.model_path,
            "model_id": self.model_id,
            "geometric_profile": self.geometric_profile.to_dict(),
            "skill_assessments": [a.to_dict() for a in self.skill_assessments],
            "mastered_skills": list(self.mastered_skills),
            "frontier_skills": list(self.frontier_skills),
            "blocked_skills": list(self.blocked_skills),
            "benchmark_baselines": dict(self.benchmark_baselines),
            "training_rounds_completed": self.training_rounds_completed,
            "total_training_samples_seen": self.total_training_samples_seen,
            "profiled_at": self.profiled_at,
        }
        if self.stall_diagnostics is not None:
            d["stall_diagnostics"] = [s.to_dict() for s in self.stall_diagnostics]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StudentProfile:
        stall = d.get("stall_diagnostics")
        return cls(
            schema_version=d.get("schema_version", "mc.student_profile.v1"),
            model_path=d["model_path"],
            model_id=d["model_id"],
            geometric_profile=GeometricProfile.from_dict(d["geometric_profile"]),
            skill_assessments=tuple(
                SkillAssessment.from_dict(a) for a in d.get("skill_assessments", ())
            ),
            mastered_skills=tuple(d.get("mastered_skills", ())),
            frontier_skills=tuple(d.get("frontier_skills", ())),
            blocked_skills=tuple(d.get("blocked_skills", ())),
            benchmark_baselines=dict(d.get("benchmark_baselines", {})),
            training_rounds_completed=d.get("training_rounds_completed", 0),
            total_training_samples_seen=d.get("total_training_samples_seen", 0),
            profiled_at=d.get("profiled_at", ""),
            stall_diagnostics=(
                tuple(StallDiagnostic.from_dict(s) for s in stall)
                if stall is not None
                else None
            ),
        )
