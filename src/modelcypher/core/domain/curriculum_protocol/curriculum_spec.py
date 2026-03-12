"""Curriculum specification data structures for frontier-generated curricula.

These structures represent the JSON curriculum a frontier model returns.
GeneratedSkillNode converts down to the existing SkillNode for compatibility
with PhaseScheduler and the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationSpec:
    """How to verify correctness for a skill's training/eval samples."""

    type: str  # "code_execution" | "substring_match" | "rubric"
    code: str | None = None
    rubric: tuple[str, ...] | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type}
        if self.code is not None:
            d["code"] = self.code
        if self.rubric is not None:
            d["rubric"] = list(self.rubric)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> VerificationSpec:
        rubric = d.get("rubric")
        return cls(
            type=d["type"],
            code=d.get("code"),
            rubric=tuple(rubric) if rubric is not None else None,
        )


@dataclass(frozen=True)
class GeneratedSkillNode:
    """Extended skill node from frontier model curriculum generation.

    Converts down to SkillNode via to_skill_node() for PhaseScheduler
    compatibility. Extended fields (verification, procedure_tokens, etc.)
    are metadata the protocol tracks but the scheduler doesn't need.
    """

    name: str
    formal_statement: str
    prerequisites: tuple[str, ...]
    branch: str
    proof_sketch: str = ""
    answer_mode: str = "exact"
    procedure_tokens: tuple[str, ...] = ()
    verification: VerificationSpec | None = None
    difficulty_tier: int = 0
    estimated_samples_needed: int = 100
    notes: str = ""

    def to_skill_node(
        self,
        train_files: tuple[str, ...],
        eval_files: tuple[str, ...],
        in_curriculum_prereqs: tuple[str, ...] | None = None,
    ):
        """Convert to SkillNode for PhaseScheduler compatibility.

        Args:
            train_files: Resolved paths to training JSONL files.
            eval_files: Resolved paths to eval JSONL files.
            in_curriculum_prereqs: If provided, use these instead of
                self.prerequisites. This allows stripping external mastered
                prerequisites that aren't in the current DAG.

        Imports SkillNode here to avoid circular imports at module level.
        """
        from modelcypher.core.use_cases.curriculum.skill_dag import SkillNode

        prereqs = in_curriculum_prereqs if in_curriculum_prereqs is not None else self.prerequisites

        return SkillNode(
            name=self.name,
            formal_statement=self.formal_statement,
            prerequisites=prereqs,
            train_files=train_files,
            eval_files=eval_files,
            branch=self.branch,
            answer_mode=self.answer_mode,
            notes=self.proof_sketch if not self.notes else self.notes,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "formal_statement": self.formal_statement,
            "prerequisites": list(self.prerequisites),
            "branch": self.branch,
            "proof_sketch": self.proof_sketch,
            "answer_mode": self.answer_mode,
            "procedure_tokens": list(self.procedure_tokens),
            "difficulty_tier": self.difficulty_tier,
            "estimated_samples_needed": self.estimated_samples_needed,
        }
        if self.verification is not None:
            d["verification"] = self.verification.to_dict()
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> GeneratedSkillNode:
        verification = d.get("verification")
        return cls(
            name=d["name"],
            formal_statement=d["formal_statement"],
            prerequisites=tuple(d.get("prerequisites", ())),
            branch=d["branch"],
            proof_sketch=d.get("proof_sketch", ""),
            answer_mode=d.get("answer_mode", "exact"),
            procedure_tokens=tuple(d.get("procedure_tokens", ())),
            verification=(
                VerificationSpec.from_dict(verification)
                if verification is not None
                else None
            ),
            difficulty_tier=d.get("difficulty_tier", 0),
            estimated_samples_needed=d.get("estimated_samples_needed", 100),
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True)
class TrainingSample:
    """A single training or eval sample."""

    text: str
    answer_start: int | None = None
    logic_id: str | None = None
    template_id: str | None = None
    is_negative: bool = False
    difficulty: int = 1
    composition_k: int = 1

    def to_jsonl_dict(self) -> dict:
        """Convert to JSONL-compatible dict for load_jsonl_dataset().

        Keeps only fields the existing data loader understands:
        text (required), answer_start, logic_id, template_id.
        Extended fields (is_negative, difficulty, composition_k) are stripped.
        """
        d: dict[str, Any] = {"text": self.text}
        if self.answer_start is not None:
            d["answer_start"] = self.answer_start
        if self.logic_id is not None:
            d["logic_id"] = self.logic_id
        if self.template_id is not None:
            d["template_id"] = self.template_id
        return d

    def to_dict(self) -> dict:
        """Full serialization including extended fields."""
        d = self.to_jsonl_dict()
        d["is_negative"] = self.is_negative
        d["difficulty"] = self.difficulty
        d["composition_k"] = self.composition_k
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainingSample:
        return cls(
            text=d["text"],
            answer_start=d.get("answer_start"),
            logic_id=d.get("logic_id"),
            template_id=d.get("template_id"),
            is_negative=d.get("is_negative", False),
            difficulty=d.get("difficulty", 1),
            composition_k=d.get("composition_k", 1),
        )


@dataclass(frozen=True)
class TrainingDataSpec:
    """A batch of training or eval samples for one skill."""

    skill_name: str
    filename: str
    file_type: str  # "train" | "eval"
    samples: tuple[TrainingSample, ...]

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "filename": self.filename,
            "file_type": self.file_type,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingDataSpec:
        return cls(
            skill_name=d["skill_name"],
            filename=d["filename"],
            file_type=d["file_type"],
            samples=tuple(TrainingSample.from_dict(s) for s in d.get("samples", ())),
        )


@dataclass(frozen=True)
class CurriculumSpec:
    """Complete curriculum specification from a frontier model."""

    curriculum_id: str
    goal: str
    target_domain: str
    skills: tuple[GeneratedSkillNode, ...]
    training_data: tuple[TrainingDataSpec, ...]
    schema_version: str = "mc.curriculum.v1"
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "curriculum_id": self.curriculum_id,
            "goal": self.goal,
            "target_domain": self.target_domain,
            "description": self.description,
            "skills": [s.to_dict() for s in self.skills],
            "training_data": [td.to_dict() for td in self.training_data],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CurriculumSpec:
        return cls(
            schema_version=d.get("schema_version", "mc.curriculum.v1"),
            curriculum_id=d["curriculum_id"],
            goal=d["goal"],
            target_domain=d["target_domain"],
            description=d.get("description", ""),
            skills=tuple(GeneratedSkillNode.from_dict(s) for s in d.get("skills", ())),
            training_data=tuple(
                TrainingDataSpec.from_dict(td) for td in d.get("training_data", ())
            ),
            metadata=dict(d.get("metadata", {})),
        )

    def to_skill_dag(
        self, data_dir: Path, mastered_skills: set[str] | None = None
    ):
        """Convert to SkillDAG, resolving file paths relative to data_dir.

        External prerequisites (in mastered_skills but not in this curriculum)
        are stripped before constructing SkillNodes, since SkillDAG._validate()
        requires all prerequisite names to reference nodes in the DAG.

        Imports SkillDAG here to avoid circular imports at module level.
        """
        from modelcypher.core.use_cases.curriculum.skill_dag import SkillDAG

        curriculum_skill_names = {s.name for s in self.skills}

        # Build a mapping: skill_name -> {train_files, eval_files}
        file_map: dict[str, dict[str, list[str]]] = {}
        for td in self.training_data:
            if td.skill_name not in file_map:
                file_map[td.skill_name] = {"train": [], "eval": []}
            path = str(data_dir / td.filename)
            file_map[td.skill_name][td.file_type].append(path)

        nodes = []
        for skill in self.skills:
            files = file_map.get(skill.name, {"train": [], "eval": []})
            # Filter prerequisites to only in-curriculum skills
            in_curriculum_prereqs = tuple(
                p for p in skill.prerequisites if p in curriculum_skill_names
            )
            node = skill.to_skill_node(
                train_files=tuple(files["train"]),
                eval_files=tuple(files["eval"]),
                in_curriculum_prereqs=in_curriculum_prereqs,
            )
            nodes.append(node)

        return SkillDAG(nodes)
