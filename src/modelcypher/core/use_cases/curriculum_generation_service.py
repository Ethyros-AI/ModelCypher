"""Orchestration service for the curriculum generation protocol.

Coordinates: building student profiles, constructing prompts for frontier
models, and ingesting frontier-generated curricula into the training pipeline.

No ML imports. No adapter imports. Uses only domain data structures and
existing curriculum infrastructure (SkillDAG, PhaseScheduler).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from modelcypher.core.domain.curriculum_protocol.curriculum_spec import (
    CurriculumSpec,
)
from modelcypher.core.domain.curriculum_protocol.prompt_template import (
    build_prompt,
)
from modelcypher.core.domain.curriculum_protocol.student_profile import (
    GeometricProfile,
    SkillAssessment,
    StallDiagnostic,
    StudentProfile,
)
from modelcypher.core.domain.curriculum_protocol.validation import (
    ValidationResult,
    validate_curriculum,
)
from modelcypher.core.use_cases.curriculum.phase_scheduler import (
    PhaseScheduler,
)
from modelcypher.core.use_cases.curriculum.skill_dag import SkillDAG

logger = logging.getLogger(__name__)


class CurriculumGenerationService:
    """Orchestrates the curriculum generation protocol.

    The three main operations:
    1. build_student_profile() — produce a report card for the frontier model
    2. build_prompt() — fill the prompt template with profile + goal
    3. ingest_curriculum() — validate and import frontier-generated curriculum
    """

    def build_student_profile(
        self,
        model_path: str,
        model_id: str = "",
        dag: SkillDAG | None = None,
        skill_assessments: list[SkillAssessment] | None = None,
        benchmark_baselines: dict[str, float] | None = None,
        geometric_profile: GeometricProfile | None = None,
        stall_diagnostics: list[StallDiagnostic] | None = None,
        training_rounds_completed: int = 0,
        total_training_samples_seen: int = 0,
    ) -> StudentProfile:
        """Build a StudentProfile from pre-computed data.

        This method does NOT load the model or run inference — that is
        adapter-layer work. It accepts pre-computed assessments, geometric
        profiles, and benchmarks, then derives mastered/frontier/blocked
        from the PhaseScheduler state.

        Args:
            model_path: Path to the model directory.
            model_id: Content hash or identifier for the model.
            dag: The current skill DAG (for computing frontier/blocked).
            skill_assessments: Pre-computed mastery evaluations.
            benchmark_baselines: Pre-computed benchmark scores.
            geometric_profile: Pre-computed geometric state summary.
            stall_diagnostics: Diagnostics for stalled skills.
            training_rounds_completed: How many rounds have been trained.
            total_training_samples_seen: Total samples across all rounds.
        """
        assessments = tuple(skill_assessments or ())

        # Derive mastered/frontier/blocked from assessments + DAG
        mastered: set[str] = set()
        for a in assessments:
            if a.is_mastered:
                mastered.add(a.skill_name)

        frontier_list: list[str] = []
        blocked_list: list[str] = []

        if dag is not None:
            ready = dag.ready_to_teach(mastered)
            frontier_list = [n.name for n in ready]
            all_names = {n.name for n in dag.nodes}
            blocked_list = sorted(
                all_names - mastered - set(frontier_list)
            )

        # Default geometric profile for a fresh/unknown model
        if geometric_profile is None:
            geometric_profile = GeometricProfile(
                architecture="unknown",
                model_family="unknown",
                parameter_count=0,
                hidden_dim=0,
                num_layers=0,
                vocab_size=0,
                context_length=0,
            )

        return StudentProfile(
            model_path=model_path,
            model_id=model_id,
            geometric_profile=geometric_profile,
            skill_assessments=assessments,
            mastered_skills=tuple(sorted(mastered)),
            frontier_skills=tuple(frontier_list),
            blocked_skills=tuple(blocked_list),
            benchmark_baselines=benchmark_baselines or {},
            training_rounds_completed=training_rounds_completed,
            total_training_samples_seen=total_training_samples_seen,
            profiled_at=datetime.now(timezone.utc).isoformat(),
            stall_diagnostics=(
                tuple(stall_diagnostics) if stall_diagnostics else None
            ),
        )

    def build_prompt(
        self,
        profile: StudentProfile,
        goal: str,
        target_domain: str = "",
        target_benchmark: str = "",
    ) -> str:
        """Build the prompt document for a frontier model."""
        return build_prompt(
            profile=profile,
            goal=goal,
            target_domain=target_domain,
            target_benchmark=target_benchmark,
        )

    def ingest_curriculum(
        self,
        curriculum_json: dict | str | Path,
        output_dir: Path,
        mastered_skills: set[str] | None = None,
    ) -> tuple[SkillDAG, PhaseScheduler, ValidationResult]:
        """Validate and ingest a frontier-generated curriculum.

        1. Parse JSON into CurriculumSpec
        2. Validate (hard errors reject, warnings logged)
        3. Write training data samples to JSONL files in output_dir
        4. Convert to SkillDAG + PhaseScheduler
        5. Save curriculum.json for reproducibility

        Args:
            curriculum_json: The curriculum as a dict, JSON string, or file path.
            output_dir: Directory to write JSONL files and curriculum.json.
            mastered_skills: Skills already mastered (valid prerequisite targets).

        Returns:
            Tuple of (SkillDAG, PhaseScheduler, ValidationResult).
            If validation fails, SkillDAG and PhaseScheduler will be empty.

        Raises:
            ValueError: If curriculum_json cannot be parsed.
        """
        # Parse input
        if isinstance(curriculum_json, Path):
            raw = json.loads(curriculum_json.read_text())
        elif isinstance(curriculum_json, str):
            raw = json.loads(curriculum_json)
        else:
            raw = curriculum_json

        spec = CurriculumSpec.from_dict(raw)

        # Validate
        result = validate_curriculum(spec, mastered_skills=mastered_skills)

        if not result.is_valid:
            logger.error(
                "Curriculum validation failed with %d errors", len(result.errors)
            )
            for err in result.errors:
                logger.error("  ERROR: %s", err)
            # Return empty DAG and scheduler
            empty_dag = SkillDAG([])
            empty_scheduler = PhaseScheduler(empty_dag)
            return empty_dag, empty_scheduler, result

        for warning in result.warnings:
            logger.warning("  WARNING: %s", warning)

        # Write JSONL files
        output_dir.mkdir(parents=True, exist_ok=True)
        for td in spec.training_data:
            filepath = output_dir / td.filename
            with open(filepath, "w") as f:
                for sample in td.samples:
                    f.write(json.dumps(sample.to_jsonl_dict()) + "\n")
            logger.info(
                "Wrote %d samples to %s", len(td.samples), filepath
            )

        # Build SkillDAG
        dag = spec.to_skill_dag(output_dir)

        # Build PhaseScheduler
        state_path = output_dir / "curriculum_state.json"
        scheduler = PhaseScheduler(dag, state_path=state_path)

        # Pre-populate mastery for already-mastered skills
        if mastered_skills:
            from modelcypher.core.use_cases.curriculum.phase_scheduler import (
                MasteryRecord,
            )

            for skill_name in mastered_skills:
                if skill_name not in {n.name for n in dag.nodes}:
                    continue
                record = MasteryRecord(
                    skill_name=skill_name,
                    regime="reinforce",
                    accuracy=1.0,
                    ci_lower=1.0,
                    ci_upper=1.0,
                    n_correct=100,
                    n_total=100,
                )
                scheduler.update_mastery(record)

        # Save curriculum spec for reproducibility
        spec_path = output_dir / "curriculum.json"
        with open(spec_path, "w") as f:
            json.dump(spec.to_dict(), f, indent=2)
        logger.info("Saved curriculum spec to %s", spec_path)

        return dag, scheduler, result

    def save_profile(self, profile: StudentProfile, path: Path) -> None:
        """Serialize StudentProfile to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load_profile(self, path: Path) -> StudentProfile:
        """Deserialize StudentProfile from JSON file."""
        with open(path) as f:
            return StudentProfile.from_dict(json.load(f))
