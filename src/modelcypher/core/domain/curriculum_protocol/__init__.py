"""Curriculum generation protocol: data structures, validation, and prompt building.

This package implements the structured protocol for frontier-model-generated
training curricula. No ML imports — pure data structures and logic.

See docs/curriculum/curriculum_generation_protocol.md for the full specification.
"""

from modelcypher.core.domain.curriculum_protocol.curriculum_spec import (
    CurriculumSpec,
    GeneratedSkillNode,
    TrainingDataSpec,
    TrainingSample,
    VerificationSpec,
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

__all__ = [
    "CurriculumSpec",
    "GeneratedSkillNode",
    "GeometricProfile",
    "SkillAssessment",
    "StallDiagnostic",
    "StudentProfile",
    "TrainingDataSpec",
    "TrainingSample",
    "ValidationResult",
    "VerificationSpec",
    "build_prompt",
    "validate_curriculum",
]
