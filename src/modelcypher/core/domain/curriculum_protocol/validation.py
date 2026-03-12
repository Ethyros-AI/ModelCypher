"""Validation for frontier-generated curricula.

Validates structural integrity (acyclicity, references, required fields) and
data quality (sample counts, diversity, held-out integrity) before ingestion.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass, field

from modelcypher.core.domain.curriculum_protocol.curriculum_spec import (
    CurriculumSpec,
)

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_ANSWER_MODES = {"exact", "numeric", "procedural"}
_MIN_EVAL_SAMPLES = 50


def _is_absolute(path: str) -> bool:
    """Check if a path string is absolute (cross-platform)."""
    return os.path.isabs(path)


@dataclass
class ValidationResult:
    """Result of validating a CurriculumSpec."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skill_count: int = 0
    total_train_samples: int = 0
    total_eval_samples: int = 0
    max_dag_depth: int = 0
    branch_names: list[str] = field(default_factory=list)


def validate_curriculum(
    spec: CurriculumSpec,
    mastered_skills: set[str] | None = None,
) -> ValidationResult:
    """Validate a CurriculumSpec for structural and data quality issues.

    Hard errors (is_valid=False):
        Schema version mismatch, missing required fields, non-unique names,
        non-snake_case names, dangling prerequisite references, cycles,
        missing train/eval data, invalid answer_mode, eval < 50 samples,
        train/eval text overlap.

    Warnings (is_valid=True):
        Train < 50 samples, low template diversity, negative ratio out of
        range, vague formal_statement, duplicate training texts.

    Args:
        spec: The curriculum to validate.
        mastered_skills: Skills already mastered (valid prerequisite targets
            even if not in this curriculum).

    Returns:
        ValidationResult with errors, warnings, and summary statistics.
    """
    result = ValidationResult()
    mastered = mastered_skills or set()

    # -- Schema version --
    if spec.schema_version != "mc.curriculum.v1":
        result.errors.append(
            f"Schema version mismatch: expected 'mc.curriculum.v1', "
            f"got '{spec.schema_version}'"
        )

    # -- Skill-level validation --
    skill_names: set[str] = set()
    skill_name_list: list[str] = []

    for skill in spec.skills:
        # Required fields
        if not skill.name:
            result.errors.append("Skill has empty name")
            continue
        if not skill.formal_statement:
            result.errors.append(f"Skill '{skill.name}': missing formal_statement")
        if not skill.branch:
            result.errors.append(f"Skill '{skill.name}': missing branch")

        # Unique names
        if skill.name in skill_names:
            result.errors.append(f"Duplicate skill name: '{skill.name}'")
        skill_names.add(skill.name)
        skill_name_list.append(skill.name)

        # Snake case
        if not _SNAKE_CASE_RE.match(skill.name):
            result.errors.append(
                f"Skill name '{skill.name}' is not snake_case "
                f"(must match ^[a-z][a-z0-9_]*$)"
            )

        # Valid answer_mode
        if skill.answer_mode not in _VALID_ANSWER_MODES:
            result.errors.append(
                f"Skill '{skill.name}': invalid answer_mode '{skill.answer_mode}' "
                f"(must be one of {_VALID_ANSWER_MODES})"
            )

    # -- Prerequisite validation --
    valid_targets = skill_names | mastered
    for skill in spec.skills:
        for prereq in skill.prerequisites:
            if prereq not in valid_targets:
                result.errors.append(
                    f"Skill '{skill.name}': prerequisite '{prereq}' not found "
                    f"in curriculum or mastered skills"
                )

    # -- Cycle detection (Kahn's algorithm) --
    if skill_names:
        in_degree: dict[str, int] = {name: 0 for name in skill_names}
        for skill in spec.skills:
            for prereq in skill.prerequisites:
                if prereq in skill_names:
                    in_degree[skill.name] = in_degree.get(skill.name, 0) + 1

        # Recompute: count only in-curriculum prerequisites
        in_degree = {}
        for skill in spec.skills:
            in_degree[skill.name] = sum(
                1 for p in skill.prerequisites if p in skill_names
            )

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        visited = 0
        while queue:
            name = queue.popleft()
            visited += 1
            for skill in spec.skills:
                if name in skill.prerequisites and skill.name in skill_names:
                    in_degree[skill.name] -= 1
                    if in_degree[skill.name] == 0:
                        queue.append(skill.name)

        if visited < len(skill_names):
            cycle_nodes = {
                name for name, deg in in_degree.items() if deg > 0
            }
            result.errors.append(
                f"Cycle detected in skill DAG involving: {cycle_nodes}"
            )

    # -- Training data validation --

    # Filename safety: reject path traversal and duplicates
    seen_filenames: set[str] = set()
    for td in spec.training_data:
        fn = td.filename
        # Path traversal: no slashes, no .., no absolute paths
        if "/" in fn or "\\" in fn or ".." in fn or _is_absolute(fn):
            result.errors.append(
                f"Unsafe filename '{fn}' for skill '{td.skill_name}': "
                f"filenames must be bare names (no path separators or '..')"
            )
        # Duplicate filenames
        if fn in seen_filenames:
            result.errors.append(
                f"Duplicate filename '{fn}': multiple training data entries "
                f"share the same filename (would overwrite each other)"
            )
        seen_filenames.add(fn)

    # Build skill -> data mapping
    train_data: dict[str, list] = {s.name: [] for s in spec.skills}
    eval_data: dict[str, list] = {s.name: [] for s in spec.skills}

    for td in spec.training_data:
        if td.skill_name not in skill_names:
            result.warnings.append(
                f"Training data references unknown skill '{td.skill_name}'"
            )
            continue
        if td.file_type == "train":
            train_data[td.skill_name].extend(td.samples)
        elif td.file_type == "eval":
            eval_data[td.skill_name].extend(td.samples)

    total_train = 0
    total_eval = 0

    for skill in spec.skills:
        train_samples = train_data.get(skill.name, [])
        eval_samples = eval_data.get(skill.name, [])
        total_train += len(train_samples)
        total_eval += len(eval_samples)

        # Hard: missing data
        if not train_samples:
            result.errors.append(
                f"Skill '{skill.name}': no training data"
            )
        if not eval_samples:
            result.errors.append(
                f"Skill '{skill.name}': no eval data"
            )

        # Hard: eval too small
        if eval_samples and len(eval_samples) < _MIN_EVAL_SAMPLES:
            result.errors.append(
                f"Skill '{skill.name}': eval has {len(eval_samples)} samples "
                f"(minimum {_MIN_EVAL_SAMPLES} for Clopper-Pearson CI)"
            )

        # Hard: train/eval text overlap
        if train_samples and eval_samples:
            train_texts = {s.text for s in train_samples}
            eval_texts = {s.text for s in eval_samples}
            overlap = train_texts & eval_texts
            if overlap:
                result.errors.append(
                    f"Skill '{skill.name}': {len(overlap)} train/eval text "
                    f"overlaps (held-out integrity violation)"
                )

        # Warning: train too small
        if train_samples and len(train_samples) < 50:
            result.warnings.append(
                f"Skill '{skill.name}': only {len(train_samples)} training "
                f"samples (recommend >= 50)"
            )

        # Warning: template diversity
        if train_samples:
            template_ids = {
                s.template_id for s in train_samples if s.template_id
            }
            if len(template_ids) < 3:
                result.warnings.append(
                    f"Skill '{skill.name}': only {len(template_ids)} distinct "
                    f"templates (recommend >= 3)"
                )

        # Warning: negative example ratio
        if train_samples:
            n_negative = sum(1 for s in train_samples if s.is_negative)
            ratio = n_negative / len(train_samples) if train_samples else 0
            if train_samples and (ratio < 0.10 or ratio > 0.20):
                if n_negative > 0:
                    result.warnings.append(
                        f"Skill '{skill.name}': negative example ratio "
                        f"{ratio:.1%} (recommend 10-20%)"
                    )

        # Warning: vague formal_statement
        if len(skill.formal_statement) < 10:
            result.warnings.append(
                f"Skill '{skill.name}': formal_statement is very short "
                f"({len(skill.formal_statement)} chars)"
            )

        # Warning: duplicate training texts
        if train_samples:
            texts = [s.text for s in train_samples]
            if len(texts) != len(set(texts)):
                n_dupes = len(texts) - len(set(texts))
                result.warnings.append(
                    f"Skill '{skill.name}': {n_dupes} duplicate training texts"
                )

    # -- Compute summary stats --
    result.skill_count = len(spec.skills)
    result.total_train_samples = total_train
    result.total_eval_samples = total_eval
    result.branch_names = sorted({s.branch for s in spec.skills})

    # Max DAG depth (only if no cycles)
    if not any("Cycle" in e for e in result.errors) and skill_names:
        depths: dict[str, int] = {}

        def _depth(name: str) -> int:
            if name in depths:
                return depths[name]
            skill = next((s for s in spec.skills if s.name == name), None)
            if skill is None or not skill.prerequisites:
                depths[name] = 0
                return 0
            in_curriculum = [p for p in skill.prerequisites if p in skill_names]
            if not in_curriculum:
                depths[name] = 0
                return 0
            d = 1 + max(_depth(p) for p in in_curriculum)
            depths[name] = d
            return d

        for name in skill_names:
            _depth(name)
        result.max_dag_depth = max(depths.values()) if depths else 0

    # -- Final verdict --
    result.is_valid = len(result.errors) == 0
    return result
