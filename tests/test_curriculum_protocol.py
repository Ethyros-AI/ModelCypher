"""Tests for the curriculum generation protocol.

Covers: data structure round-trips, validation rules (hard errors + warnings),
prompt template construction, and orchestration service (ingestion, profile).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    validate_curriculum,
)
from modelcypher.core.use_cases.curriculum_generation_service import (
    CurriculumGenerationService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_geometric_profile(**overrides) -> GeometricProfile:
    defaults = dict(
        architecture="TestArch",
        model_family="test",
        parameter_count=1000,
        hidden_dim=64,
        num_layers=4,
        vocab_size=1000,
        context_length=512,
    )
    defaults.update(overrides)
    return GeometricProfile(**defaults)


def _make_student_profile(**overrides) -> StudentProfile:
    defaults = dict(
        model_path="/tmp/test_model",
        model_id="test_hash_abc123",
        geometric_profile=_make_geometric_profile(),
    )
    defaults.update(overrides)
    return StudentProfile(**defaults)


def _make_samples(n: int, prefix: str = "sample", skill: str = "test_skill") -> tuple[TrainingSample, ...]:
    return tuple(
        TrainingSample(
            text=f"{prefix}_{i}: Question? Answer is {i}.",
            answer_start=len(f"{prefix}_{i}: Question? "),
            logic_id=skill,
            template_id=f"tpl_{i % 3}",
        )
        for i in range(n)
    )


def _make_valid_curriculum(n_train: int = 60, n_eval: int = 55) -> CurriculumSpec:
    """Build a valid 3-skill linear DAG: skill_a -> skill_b -> skill_c."""
    skills = (
        GeneratedSkillNode(
            name="skill_a",
            formal_statement="Base skill: no prerequisites required.",
            prerequisites=(),
            branch="test",
            proof_sketch="Root node: primitive skill.",
            answer_mode="exact",
        ),
        GeneratedSkillNode(
            name="skill_b",
            formal_statement="Depends on skill_a as a premise.",
            prerequisites=("skill_a",),
            branch="test",
            proof_sketch="Proof of B uses A: step 1 applies A, step 2 extends.",
            answer_mode="numeric",
        ),
        GeneratedSkillNode(
            name="skill_c",
            formal_statement="Depends on skill_b which chains through skill_a.",
            prerequisites=("skill_b",),
            branch="test",
            proof_sketch="C requires B which requires A. Transitive dependency.",
            answer_mode="exact",
        ),
    )

    training_data = []
    for skill in skills:
        training_data.append(
            TrainingDataSpec(
                skill_name=skill.name,
                filename=f"{skill.name}_train.jsonl",
                file_type="train",
                samples=_make_samples(n_train, prefix=f"{skill.name}_train", skill=skill.name),
            )
        )
        training_data.append(
            TrainingDataSpec(
                skill_name=skill.name,
                filename=f"{skill.name}_eval.jsonl",
                file_type="eval",
                samples=_make_samples(n_eval, prefix=f"{skill.name}_eval", skill=skill.name),
            )
        )

    return CurriculumSpec(
        curriculum_id="test_curriculum_v1",
        goal="Test the protocol",
        target_domain="test",
        skills=skills,
        training_data=tuple(training_data),
        metadata={"generator_model": "test"},
    )


# ---------------------------------------------------------------------------
# StudentProfile tests
# ---------------------------------------------------------------------------


class TestStudentProfile:
    def test_round_trip(self):
        profile = _make_student_profile(
            mastered_skills=("skill_a",),
            frontier_skills=("skill_b",),
            blocked_skills=("skill_c",),
            benchmark_baselines={"gsm8k": 0.05},
            training_rounds_completed=2,
            total_training_samples_seen=500,
            profiled_at="2026-03-12T10:00:00Z",
        )
        d = profile.to_dict()
        restored = StudentProfile.from_dict(d)
        assert restored.model_path == profile.model_path
        assert restored.model_id == profile.model_id
        assert restored.mastered_skills == profile.mastered_skills
        assert restored.frontier_skills == profile.frontier_skills
        assert restored.blocked_skills == profile.blocked_skills
        assert restored.benchmark_baselines == profile.benchmark_baselines
        assert restored.schema_version == "mc.student_profile.v1"

    def test_round_trip_with_stall_diagnostics(self):
        stall = StallDiagnostic(
            skill_name="carry_rule",
            rounds_attempted=3,
            accuracy_history=(0.12, 0.15, 0.14),
            training_loss_history=(2.1, 1.8, 1.75),
            samples_trained_on=450,
            cka_pre_post=0.97,
            common_failure_patterns=("always outputs 0",),
            n_distinct_wrong_answers=1,
        )
        profile = _make_student_profile(stall_diagnostics=(stall,))
        d = profile.to_dict()
        restored = StudentProfile.from_dict(d)
        assert restored.stall_diagnostics is not None
        assert len(restored.stall_diagnostics) == 1
        assert restored.stall_diagnostics[0].skill_name == "carry_rule"
        assert restored.stall_diagnostics[0].accuracy_history == (0.12, 0.15, 0.14)

    def test_round_trip_with_skill_assessments(self):
        assessment = SkillAssessment(
            skill_name="modus_ponens",
            accuracy=0.45,
            ci_lower=0.31,
            ci_upper=0.60,
            n_total=100,
            n_correct=45,
            regime="ce",
            is_mastered=False,
            answer_mode="exact",
        )
        profile = _make_student_profile(skill_assessments=(assessment,))
        d = profile.to_dict()
        restored = StudentProfile.from_dict(d)
        assert len(restored.skill_assessments) == 1
        assert restored.skill_assessments[0].skill_name == "modus_ponens"
        assert restored.skill_assessments[0].accuracy == 0.45

    def test_json_serializable(self):
        profile = _make_student_profile()
        json_str = json.dumps(profile.to_dict())
        assert json.loads(json_str)["schema_version"] == "mc.student_profile.v1"

    def test_none_optionals_excluded(self):
        profile = _make_student_profile()
        d = profile.to_dict()
        assert "stall_diagnostics" not in d
        gp = d["geometric_profile"]
        assert "mean_effective_rank" not in gp


# ---------------------------------------------------------------------------
# CurriculumSpec tests
# ---------------------------------------------------------------------------


class TestCurriculumSpec:
    def test_round_trip(self):
        spec = _make_valid_curriculum()
        d = spec.to_dict()
        restored = CurriculumSpec.from_dict(d)
        assert restored.curriculum_id == spec.curriculum_id
        assert len(restored.skills) == 3
        assert len(restored.training_data) == 6
        assert restored.schema_version == "mc.curriculum.v1"

    def test_generated_skill_node_to_skill_node(self):
        node = GeneratedSkillNode(
            name="test_skill",
            formal_statement="(A, B) -> A + B",
            prerequisites=(),
            branch="math",
            proof_sketch="Primitive operation.",
            answer_mode="numeric",
        )
        skill_node = node.to_skill_node(
            train_files=("data/train.jsonl",),
            eval_files=("data/eval.jsonl",),
        )
        assert skill_node.name == "test_skill"
        assert skill_node.formal_statement == "(A, B) -> A + B"
        assert skill_node.train_files == ("data/train.jsonl",)
        assert skill_node.eval_files == ("data/eval.jsonl",)
        assert skill_node.answer_mode == "numeric"
        assert skill_node.branch == "math"

    def test_to_skill_dag(self, tmp_path):
        spec = _make_valid_curriculum()
        dag = spec.to_skill_dag(tmp_path)
        sorted_nodes = dag.topological_sort()
        names = [n.name for n in sorted_nodes]
        assert names.index("skill_a") < names.index("skill_b")
        assert names.index("skill_b") < names.index("skill_c")

    def test_training_sample_to_jsonl_dict(self):
        sample = TrainingSample(
            text="What is 2+3? 5",
            answer_start=14,
            logic_id="addition",
            template_id="qa_simple",
            is_negative=True,
            difficulty=2,
            composition_k=1,
        )
        d = sample.to_jsonl_dict()
        assert d["text"] == "What is 2+3? 5"
        assert d["answer_start"] == 14
        assert d["logic_id"] == "addition"
        assert d["template_id"] == "qa_simple"
        # Extended fields NOT in jsonl dict
        assert "is_negative" not in d
        assert "difficulty" not in d
        assert "composition_k" not in d

    def test_training_sample_full_dict(self):
        sample = TrainingSample(
            text="test",
            is_negative=True,
            difficulty=3,
        )
        d = sample.to_dict()
        assert d["is_negative"] is True
        assert d["difficulty"] == 3

    def test_verification_spec_round_trip(self):
        v = VerificationSpec(
            type="code_execution",
            code="def verify(e, g): return int(e) == int(g)",
            rubric=None,
        )
        d = v.to_dict()
        restored = VerificationSpec.from_dict(d)
        assert restored.type == "code_execution"
        assert restored.code == v.code
        assert restored.rubric is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_curriculum_passes(self):
        spec = _make_valid_curriculum()
        result = validate_curriculum(spec)
        assert result.is_valid, f"Errors: {result.errors}"
        assert result.skill_count == 3
        assert result.max_dag_depth == 2
        assert result.branch_names == ["test"]

    def test_schema_version_mismatch(self):
        spec = CurriculumSpec(
            curriculum_id="bad",
            goal="test",
            target_domain="test",
            skills=(),
            training_data=(),
            schema_version="mc.curriculum.v999",
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("Schema version" in e for e in result.errors)

    def test_duplicate_skill_names(self):
        skills = (
            GeneratedSkillNode(name="dup", formal_statement="First definition.", prerequisites=(), branch="test"),
            GeneratedSkillNode(name="dup", formal_statement="Second definition.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=tuple(
                TrainingDataSpec(skill_name="dup", filename=f"dup_{ft}.jsonl", file_type=ft,
                                 samples=_make_samples(60 if ft == "train" else 55, prefix=f"dup_{ft}", skill="dup"))
                for ft in ("train", "eval")
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)

    def test_non_snake_case_name(self):
        skills = (
            GeneratedSkillNode(name="CamelCase", formal_statement="Bad name.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="CamelCase", filename="cc_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="CamelCase")),
                TrainingDataSpec(skill_name="CamelCase", filename="cc_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="CamelCase")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("snake_case" in e for e in result.errors)

    def test_dangling_prerequisite(self):
        skills = (
            GeneratedSkillNode(
                name="orphan", formal_statement="Has missing prereq.",
                prerequisites=("nonexistent_skill",), branch="test",
            ),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="orphan", filename="o_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="orphan")),
                TrainingDataSpec(skill_name="orphan", filename="o_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="orphan")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("nonexistent_skill" in e for e in result.errors)

    def test_mastered_skill_as_prerequisite_is_valid(self):
        skills = (
            GeneratedSkillNode(
                name="child_skill", formal_statement="Depends on mastered parent.",
                prerequisites=("already_mastered",), branch="test",
            ),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="child_skill", filename="c_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="child_skill")),
                TrainingDataSpec(skill_name="child_skill", filename="c_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="child_skill")),
            ),
        )
        result = validate_curriculum(spec, mastered_skills={"already_mastered"})
        assert result.is_valid, f"Errors: {result.errors}"

    def test_cycle_detection(self):
        skills = (
            GeneratedSkillNode(name="cycle_a", formal_statement="A depends on B.", prerequisites=("cycle_b",), branch="test"),
            GeneratedSkillNode(name="cycle_b", formal_statement="B depends on A.", prerequisites=("cycle_a",), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=tuple(
                TrainingDataSpec(skill_name=s.name, filename=f"{s.name}_{ft}.jsonl", file_type=ft,
                                 samples=_make_samples(60 if ft == "train" else 55, prefix=f"{s.name}_{ft}", skill=s.name))
                for s in skills for ft in ("train", "eval")
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("Cycle" in e for e in result.errors)

    def test_missing_train_data(self):
        skills = (
            GeneratedSkillNode(name="no_train", formal_statement="Has no training data.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="no_train", filename="nt_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="no_train")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("no training data" in e for e in result.errors)

    def test_missing_eval_data(self):
        skills = (
            GeneratedSkillNode(name="no_eval", formal_statement="Has no eval data.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="no_eval", filename="ne_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="no_eval")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("no eval data" in e for e in result.errors)

    def test_invalid_answer_mode(self):
        skills = (
            GeneratedSkillNode(name="bad_mode", formal_statement="Invalid mode.", prerequisites=(), branch="test", answer_mode="freestyle"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="bad_mode", filename="bm_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="bad_mode")),
                TrainingDataSpec(skill_name="bad_mode", filename="bm_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="bad_mode")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("answer_mode" in e for e in result.errors)

    def test_eval_below_minimum(self):
        skills = (
            GeneratedSkillNode(name="small_eval", formal_statement="Eval too small.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="small_eval", filename="se_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="small_eval")),
                TrainingDataSpec(skill_name="small_eval", filename="se_eval.jsonl", file_type="eval",
                                 samples=_make_samples(10, skill="small_eval")),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("10 samples" in e for e in result.errors)

    def test_train_eval_overlap(self):
        shared = _make_samples(55, prefix="shared", skill="overlap_skill")
        skills = (
            GeneratedSkillNode(name="overlap_skill", formal_statement="Has overlap.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="overlap_skill", filename="ol_train.jsonl", file_type="train",
                                 samples=shared),
                TrainingDataSpec(skill_name="overlap_skill", filename="ol_eval.jsonl", file_type="eval",
                                 samples=shared),
            ),
        )
        result = validate_curriculum(spec)
        assert not result.is_valid
        assert any("overlap" in e.lower() for e in result.errors)

    def test_warning_low_train_count(self):
        skills = (
            GeneratedSkillNode(name="small_train", formal_statement="Train is small but OK.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="small_train", filename="st_train.jsonl", file_type="train",
                                 samples=_make_samples(20, prefix="st_train", skill="small_train")),
                TrainingDataSpec(skill_name="small_train", filename="st_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, prefix="st_eval", skill="small_train")),
            ),
        )
        result = validate_curriculum(spec)
        assert result.is_valid
        assert any("20 training" in w for w in result.warnings)

    def test_warning_low_template_diversity(self):
        # All samples with same template_id
        samples = tuple(
            TrainingSample(text=f"q{i}? a{i}.", template_id="same_tpl", logic_id="mono_tpl")
            for i in range(60)
        )
        skills = (
            GeneratedSkillNode(name="mono_tpl", formal_statement="One template only.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="mono_tpl", filename="mt_train.jsonl", file_type="train", samples=samples),
                TrainingDataSpec(skill_name="mono_tpl", filename="mt_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, prefix="mt_eval", skill="mono_tpl")),
            ),
        )
        result = validate_curriculum(spec)
        assert result.is_valid
        assert any("1 distinct templates" in w for w in result.warnings)

    def test_warning_short_formal_statement(self):
        skills = (
            GeneratedSkillNode(name="short_fs", formal_statement="Hi.", prerequisites=(), branch="test"),
        )
        spec = CurriculumSpec(
            curriculum_id="test", goal="test", target_domain="test",
            skills=skills,
            training_data=(
                TrainingDataSpec(skill_name="short_fs", filename="sf_train.jsonl", file_type="train",
                                 samples=_make_samples(60, skill="short_fs")),
                TrainingDataSpec(skill_name="short_fs", filename="sf_eval.jsonl", file_type="eval",
                                 samples=_make_samples(55, skill="short_fs")),
            ),
        )
        result = validate_curriculum(spec)
        assert result.is_valid
        assert any("very short" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_build_prompt_contains_sections(self):
        profile = _make_student_profile(
            mastered_skills=("skill_a",),
            frontier_skills=("skill_b",),
            blocked_skills=("skill_c",),
        )
        prompt = build_prompt(profile, goal="Beat GSM8K", target_domain="math")
        assert "## Your Role" in prompt
        assert "## Student Model Profile" in prompt
        assert "## Training Goal" in prompt
        assert "Beat GSM8K" in prompt
        assert "## Constraints" in prompt
        assert "## Response Schema" in prompt
        assert "## Research Findings" in prompt

    def test_build_prompt_embeds_profile(self):
        profile = _make_student_profile(model_id="unique_hash_xyz")
        prompt = build_prompt(profile, goal="test")
        assert "unique_hash_xyz" in prompt

    def test_build_prompt_formats_skill_lists(self):
        profile = _make_student_profile(
            mastered_skills=("mp", "ds"),
            frontier_skills=("mt",),
            blocked_skills=("chain",),
        )
        prompt = build_prompt(profile, goal="test")
        assert "mp, ds" in prompt
        assert "mt" in prompt
        assert "chain" in prompt

    def test_build_prompt_handles_empty_skills(self):
        profile = _make_student_profile()
        prompt = build_prompt(profile, goal="test")
        assert "(none)" in prompt

    def test_build_prompt_embeds_response_schema(self):
        profile = _make_student_profile()
        prompt = build_prompt(profile, goal="test")
        assert "mc.curriculum.v1" in prompt
        assert "formal_statement" in prompt


# ---------------------------------------------------------------------------
# Orchestration service tests
# ---------------------------------------------------------------------------


class TestCurriculumGenerationService:
    def test_build_student_profile_minimal(self):
        svc = CurriculumGenerationService()
        profile = svc.build_student_profile(model_path="/tmp/model", model_id="abc")
        assert profile.model_path == "/tmp/model"
        assert profile.model_id == "abc"
        assert profile.schema_version == "mc.student_profile.v1"
        assert profile.geometric_profile.architecture == "unknown"

    def test_build_student_profile_with_dag(self):
        from modelcypher.core.use_cases.curriculum.skill_dag import (
            SkillDAG,
            SkillNode,
        )

        nodes = [
            SkillNode(name="a", formal_statement="A", prerequisites=(),
                      train_files=("t.jsonl",), eval_files=("e.jsonl",), branch="test"),
            SkillNode(name="b", formal_statement="B", prerequisites=("a",),
                      train_files=("t.jsonl",), eval_files=("e.jsonl",), branch="test"),
        ]
        dag = SkillDAG(nodes)
        mastered_a = SkillAssessment(
            skill_name="a", accuracy=1.0, ci_lower=1.0, ci_upper=1.0,
            n_total=100, n_correct=100, regime="reinforce",
            is_mastered=True, answer_mode="exact",
        )

        svc = CurriculumGenerationService()
        profile = svc.build_student_profile(
            model_path="/tmp/m", model_id="x", dag=dag,
            skill_assessments=[mastered_a],
        )
        assert "a" in profile.mastered_skills
        assert "b" in profile.frontier_skills
        assert len(profile.blocked_skills) == 0

    def test_ingest_valid_curriculum(self, tmp_path):
        spec = _make_valid_curriculum()
        svc = CurriculumGenerationService()
        dag, scheduler, result = svc.ingest_curriculum(
            curriculum_json=spec.to_dict(),
            output_dir=tmp_path / "output",
        )
        assert result.is_valid, f"Errors: {result.errors}"
        assert len(dag.nodes) == 3
        # Files were written
        assert (tmp_path / "output" / "skill_a_train.jsonl").exists()
        assert (tmp_path / "output" / "skill_a_eval.jsonl").exists()
        # JSONL is parseable
        with open(tmp_path / "output" / "skill_a_train.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 60
        assert "text" in lines[0]
        # curriculum.json saved
        assert (tmp_path / "output" / "curriculum.json").exists()
        # DAG is topologically valid
        sorted_names = [n.name for n in dag.topological_sort()]
        assert sorted_names.index("skill_a") < sorted_names.index("skill_b")
        # PhaseScheduler works
        next_skill = scheduler.next_to_teach()
        assert next_skill is not None
        assert next_skill.name == "skill_a"

    def test_ingest_invalid_curriculum_returns_errors(self, tmp_path):
        bad_spec = {
            "schema_version": "mc.curriculum.v999",
            "curriculum_id": "bad",
            "goal": "fail",
            "target_domain": "test",
            "skills": [],
            "training_data": [],
        }
        svc = CurriculumGenerationService()
        dag, scheduler, result = svc.ingest_curriculum(
            curriculum_json=bad_spec,
            output_dir=tmp_path / "bad_output",
        )
        assert not result.is_valid
        assert len(dag.nodes) == 0

    def test_ingest_from_json_string(self, tmp_path):
        spec = _make_valid_curriculum()
        json_str = json.dumps(spec.to_dict())
        svc = CurriculumGenerationService()
        dag, scheduler, result = svc.ingest_curriculum(
            curriculum_json=json_str,
            output_dir=tmp_path / "str_output",
        )
        assert result.is_valid
        assert len(dag.nodes) == 3

    def test_ingest_from_file_path(self, tmp_path):
        spec = _make_valid_curriculum()
        json_path = tmp_path / "curriculum_input.json"
        json_path.write_text(json.dumps(spec.to_dict()))
        svc = CurriculumGenerationService()
        dag, scheduler, result = svc.ingest_curriculum(
            curriculum_json=json_path,
            output_dir=tmp_path / "file_output",
        )
        assert result.is_valid
        assert len(dag.nodes) == 3

    def test_save_and_load_profile(self, tmp_path):
        svc = CurriculumGenerationService()
        profile = svc.build_student_profile(model_path="/tmp/m", model_id="test")
        path = tmp_path / "profile.json"
        svc.save_profile(profile, path)
        loaded = svc.load_profile(path)
        assert loaded.model_path == profile.model_path
        assert loaded.model_id == profile.model_id

    def test_written_jsonl_compatible_with_load_jsonl_dataset(self, tmp_path):
        """Verify ingested JSONL files work with the existing data loader."""
        from modelcypher.core.domain.dataset_loading import load_jsonl_dataset

        spec = _make_valid_curriculum()
        svc = CurriculumGenerationService()
        svc.ingest_curriculum(
            curriculum_json=spec.to_dict(),
            output_dir=tmp_path / "compat",
        )
        samples = load_jsonl_dataset(tmp_path / "compat" / "skill_a_train.jsonl")
        assert len(samples) == 60
        assert "text" in samples[0]


# ---------------------------------------------------------------------------
# Integration: hand-crafted DAG reconstruction
# ---------------------------------------------------------------------------


class TestIntegrationDAGReconstruction:
    def test_reconstruct_logic_subdag(self, tmp_path):
        """Build a curriculum mirroring MP -> MT -> HS from the existing DAG."""
        skills = (
            GeneratedSkillNode(
                name="modus_ponens",
                formal_statement="(P->Q, P) |- Q",
                prerequisites=(),
                branch="logic",
                proof_sketch="Primitive axiom.",
                answer_mode="exact",
            ),
            GeneratedSkillNode(
                name="modus_tollens",
                formal_statement="(P->Q, ~Q) |- ~P",
                prerequisites=("modus_ponens",),
                branch="logic",
                proof_sketch="MT applies MP to the contrapositive.",
                answer_mode="exact",
            ),
            GeneratedSkillNode(
                name="hypothetical_syllogism",
                formal_statement="(P->Q, Q->R) |- P->R",
                prerequisites=("modus_ponens",),
                branch="logic",
                proof_sketch="HS applies MP twice in sequence.",
                answer_mode="exact",
            ),
        )
        training_data = []
        for skill in skills:
            for ft in ("train", "eval"):
                n = 60 if ft == "train" else 55
                training_data.append(
                    TrainingDataSpec(
                        skill_name=skill.name,
                        filename=f"{skill.name}_{ft}.jsonl",
                        file_type=ft,
                        samples=_make_samples(n, prefix=f"{skill.name}_{ft}", skill=skill.name),
                    )
                )
        spec = CurriculumSpec(
            curriculum_id="logic_reconstruction",
            goal="Reconstruct logic sub-DAG",
            target_domain="logic",
            skills=skills,
            training_data=tuple(training_data),
        )

        svc = CurriculumGenerationService()
        dag, scheduler, result = svc.ingest_curriculum(
            curriculum_json=spec.to_dict(),
            output_dir=tmp_path / "logic",
        )
        assert result.is_valid
        # DAG structure matches: MP is root, MT and HS depend on MP
        sorted_nodes = dag.topological_sort()
        names = [n.name for n in sorted_nodes]
        assert names[0] == "modus_ponens"
        assert set(names[1:]) == {"modus_tollens", "hypothetical_syllogism"}
        # PhaseScheduler teaches MP first
        assert scheduler.next_to_teach().name == "modus_ponens"
