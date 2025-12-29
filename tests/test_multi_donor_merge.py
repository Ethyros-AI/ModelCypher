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

"""
Tests for the Multi-Donor Transplant Pipeline.

Validates:
- Data structure serialization/deserialization
- Program YAML loading and validation
- MultiDonorMergeService orchestration logic
- Status tracking and checkpointing
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from modelcypher.core.use_cases.multi_donor_merge import (
    BaseModelResult,
    BaseModelSpec,
    DonorSpec,
    DonorStageResult,
    EvaluationConfig,
    MultiDonorMergeService,
    MultiDonorResult,
    ProgramStatus,
    TransplantProgram,
)


# =============================================================================
# Test DonorSpec
# =============================================================================


class TestDonorSpec:
    """Test DonorSpec dataclass."""

    def test_from_dict_minimal(self):
        """Test minimal DonorSpec creation."""
        data = {
            "id": "test-donor",
            "source": "test/TestModel",
        }
        spec = DonorSpec.from_dict(data)
        assert spec.id == "test-donor"
        assert spec.source == "test/TestModel"
        assert spec.domains == ()
        assert spec.layers is None
        assert spec.priority == 0
        assert spec.boundary_k is None
        assert spec.geodesic_k is None

    def test_from_dict_full(self):
        """Test full DonorSpec creation with all fields."""
        data = {
            "id": "math-donor",
            "source": "deepseek-ai/DeepSeek-Math-7B",
            "domains": ["mathematical", "logical"],
            "layers": [5, 10, 15],
            "priority": 2,
            "boundary_k": 128,
            "geodesic_k": 64,
        }
        spec = DonorSpec.from_dict(data)
        assert spec.id == "math-donor"
        assert spec.source == "deepseek-ai/DeepSeek-Math-7B"
        assert spec.domains == ("mathematical", "logical")
        assert spec.layers == (5, 10, 15)
        assert spec.priority == 2
        assert spec.boundary_k == 128
        assert spec.geodesic_k == 64

    def test_to_dict_minimal(self):
        """Test minimal DonorSpec serialization."""
        spec = DonorSpec(
            id="test-donor",
            source="test/TestModel",
            domains=("coding",),
        )
        data = spec.to_dict()
        assert data["id"] == "test-donor"
        assert data["source"] == "test/TestModel"
        assert data["domains"] == ["coding"]
        assert "layers" not in data
        assert "priority" not in data  # 0 is default, not serialized

    def test_to_dict_full(self):
        """Test full DonorSpec serialization."""
        spec = DonorSpec(
            id="math-donor",
            source="deepseek-ai/DeepSeek-Math-7B",
            domains=("mathematical",),
            layers=(5, 10, 15),
            priority=3,
            boundary_k=128,
            geodesic_k=64,
        )
        data = spec.to_dict()
        assert data["priority"] == 3
        assert data["layers"] == [5, 10, 15]
        assert data["boundary_k"] == 128
        assert data["geodesic_k"] == 64

    def test_immutable(self):
        """Test DonorSpec is frozen/immutable."""
        spec = DonorSpec(id="test", source="path", domains=())
        with pytest.raises(AttributeError):
            spec.id = "changed"  # type: ignore


# =============================================================================
# Test BaseModelSpec
# =============================================================================


class TestBaseModelSpec:
    """Test BaseModelSpec dataclass."""

    def test_from_dict_minimal(self):
        """Test minimal BaseModelSpec creation."""
        data = {"id": "qwen3-8b", "source": "Qwen/Qwen3-8B"}
        spec = BaseModelSpec.from_dict(data)
        assert spec.id == "qwen3-8b"
        assert spec.source == "Qwen/Qwen3-8B"
        assert spec.alias is None

    def test_from_dict_with_alias(self):
        """Test BaseModelSpec with alias."""
        data = {"id": "qwen3-8b", "source": "Qwen/Qwen3-8B", "alias": "qwen3"}
        spec = BaseModelSpec.from_dict(data)
        assert spec.alias == "qwen3"

    def test_effective_alias_without_alias(self):
        """Test effective_alias returns id when no alias set."""
        spec = BaseModelSpec(id="qwen3-8b", source="Qwen/Qwen3-8B")
        assert spec.effective_alias == "qwen3-8b"

    def test_effective_alias_with_alias(self):
        """Test effective_alias returns alias when set."""
        spec = BaseModelSpec(id="qwen3-8b", source="Qwen/Qwen3-8B", alias="qwen3")
        assert spec.effective_alias == "qwen3"

    def test_to_dict_minimal(self):
        """Test minimal serialization."""
        spec = BaseModelSpec(id="qwen3-8b", source="Qwen/Qwen3-8B")
        data = spec.to_dict()
        assert data == {"id": "qwen3-8b", "source": "Qwen/Qwen3-8B"}

    def test_to_dict_with_alias(self):
        """Test serialization with alias."""
        spec = BaseModelSpec(id="qwen3-8b", source="Qwen/Qwen3-8B", alias="qwen3")
        data = spec.to_dict()
        assert data["alias"] == "qwen3"


# =============================================================================
# Test EvaluationConfig
# =============================================================================


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = EvaluationConfig()
        assert config.after_each_donor is True
        assert config.after_program_complete is True
        assert config.benchmarks == ()
        assert len(config.smoke_test_prompts) == 3

    def test_from_dict_none(self):
        """Test from_dict with None returns defaults."""
        config = EvaluationConfig.from_dict(None)
        assert config.after_each_donor is True

    def test_from_dict_custom(self):
        """Test from_dict with custom values."""
        data = {
            "after_each_donor": False,
            "after_program_complete": True,
            "benchmarks": ["mmlu_pro", "gpqa_diamond"],
            "smoke_test_prompts": ["Test prompt 1"],
        }
        config = EvaluationConfig.from_dict(data)
        assert config.after_each_donor is False
        assert config.benchmarks == ("mmlu_pro", "gpqa_diamond")
        assert config.smoke_test_prompts == ("Test prompt 1",)

    def test_to_dict(self):
        """Test serialization."""
        config = EvaluationConfig(
            after_each_donor=True,
            benchmarks=("mmlu_pro",),
            smoke_test_prompts=("Test 1", "Test 2"),
        )
        data = config.to_dict()
        assert data["after_each_donor"] is True
        assert data["benchmarks"] == ["mmlu_pro"]
        assert data["smoke_test_prompts"] == ["Test 1", "Test 2"]


# =============================================================================
# Test TransplantProgram
# =============================================================================


class TestTransplantProgram:
    """Test TransplantProgram dataclass."""

    @pytest.fixture
    def minimal_program_data(self):
        """Return minimal valid program data."""
        return {
            "name": "Test Program",
            "bases": [{"id": "base-1", "source": "test/Base"}],
            "donors": [{"id": "donor-1", "source": "test/Donor", "domains": ["coding"]}],
        }

    @pytest.fixture
    def full_program_data(self):
        """Return full program data with all fields."""
        return {
            "_schema": "mc.program.transplant.v1",
            "name": "Full Test Program",
            "description": "A test program with all fields",
            "bases": [
                {"id": "base-1", "source": "test/Base1", "alias": "b1"},
                {"id": "base-2", "source": "test/Base2", "alias": "b2"},
            ],
            "donors": [
                {"id": "donor-1", "source": "test/Donor1", "domains": ["coding"], "priority": 2},
                {"id": "donor-2", "source": "test/Donor2", "domains": ["math"], "priority": 1},
                {"id": "donor-3", "source": "test/Donor3", "domains": ["reasoning"], "priority": 2},
            ],
            "evaluation": {
                "after_each_donor": True,
                "benchmarks": ["mmlu_pro"],
            },
            "output_dir": "~/.modelcypher/test",
        }

    def test_from_dict_minimal(self, minimal_program_data):
        """Test minimal program creation."""
        program = TransplantProgram.from_dict(minimal_program_data)
        assert program.name == "Test Program"
        assert len(program.bases) == 1
        assert len(program.donors) == 1
        assert program.description == ""

    def test_from_dict_full(self, full_program_data):
        """Test full program creation."""
        program = TransplantProgram.from_dict(full_program_data)
        assert program.name == "Full Test Program"
        assert len(program.bases) == 2
        assert len(program.donors) == 3
        assert program.description == "A test program with all fields"
        assert program.evaluation.benchmarks == ("mmlu_pro",)
        assert program.output_dir == "~/.modelcypher/test"

    def test_donors_sorted_by_priority(self, full_program_data):
        """Test donors are sorted by priority (higher first)."""
        program = TransplantProgram.from_dict(full_program_data)
        # donor-1 (priority 2) and donor-3 (priority 2) come before donor-2 (priority 1)
        assert program.donors[0].priority == 2
        assert program.donors[1].priority == 2
        assert program.donors[2].priority == 1

    def test_missing_name_raises(self):
        """Test missing name raises ValueError."""
        data = {"bases": [{"id": "b", "source": "s"}], "donors": [{"id": "d", "source": "s"}]}
        with pytest.raises(ValueError, match="must have 'name'"):
            TransplantProgram.from_dict(data)

    def test_missing_bases_raises(self):
        """Test missing bases raises ValueError."""
        data = {"name": "Test", "donors": [{"id": "d", "source": "s"}]}
        with pytest.raises(ValueError, match="at least one base"):
            TransplantProgram.from_dict(data)

    def test_missing_donors_raises(self):
        """Test missing donors raises ValueError."""
        data = {"name": "Test", "bases": [{"id": "b", "source": "s"}]}
        with pytest.raises(ValueError, match="at least one donor"):
            TransplantProgram.from_dict(data)

    def test_empty_bases_raises(self):
        """Test empty bases list raises ValueError."""
        data = {"name": "Test", "bases": [], "donors": [{"id": "d", "source": "s"}]}
        with pytest.raises(ValueError, match="at least one base"):
            TransplantProgram.from_dict(data)

    def test_to_dict(self, minimal_program_data):
        """Test program serialization."""
        program = TransplantProgram.from_dict(minimal_program_data)
        data = program.to_dict()
        assert data["_schema"] == "mc.program.transplant.v1"
        assert data["name"] == "Test Program"
        assert len(data["bases"]) == 1
        assert len(data["donors"]) == 1

    def test_from_yaml_file(self, minimal_program_data):
        """Test loading program from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(minimal_program_data, f)
            f.flush()
            program = TransplantProgram.from_yaml(f.name)
        assert program.name == "Test Program"
        Path(f.name).unlink()

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TransplantProgram.from_yaml("/nonexistent/path.yaml")

    def test_to_yaml_file(self, minimal_program_data):
        """Test saving program to YAML file."""
        program = TransplantProgram.from_dict(minimal_program_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "program.yaml"
            program.to_yaml(yaml_path)
            assert yaml_path.exists()
            loaded = TransplantProgram.from_yaml(yaml_path)
            assert loaded.name == program.name

    def test_roundtrip_yaml(self, full_program_data):
        """Test YAML roundtrip preserves data."""
        program = TransplantProgram.from_dict(full_program_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "program.yaml"
            program.to_yaml(yaml_path)
            loaded = TransplantProgram.from_yaml(yaml_path)
            assert loaded.name == program.name
            assert len(loaded.bases) == len(program.bases)
            assert len(loaded.donors) == len(program.donors)
            assert loaded.evaluation.benchmarks == program.evaluation.benchmarks


# =============================================================================
# Test Result Dataclasses
# =============================================================================


class TestDonorStageResult:
    """Test DonorStageResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = DonorStageResult(
            donor_id="test-donor",
            donor_source="test/Donor",
            domains=("coding", "logical"),
            donor_index=0,
            cka_before=0.5,
            cka_after=0.7,
            cka_improvement=0.2,
            alignment_samples=3,
            mean_alignment_improvement=0.12,
            mean_core_dist_to_source_before=1.5,
            mean_core_dist_to_source_after=1.1,
            boundary_preserved=0.95,
            mean_boundary_relative_diff=0.03,
            max_boundary_relative_diff=0.08,
            layers_transplanted=12,
            weights_transplanted=48,
            mean_projection_loss=0.001,
            mean_null_dim=256.0,
            checkpoint_path="/path/to/checkpoint",
            duration_seconds=120.5,
            smoke_test_passed=True,
            intrinsic_dim_delta=0.5,
        )
        data = result.to_dict()
        assert data["donor_id"] == "test-donor"
        assert data["cka_improvement"] == 0.2
        assert data["domains"] == ["coding", "logical"]
        assert data["smoke_test_passed"] is True


class TestBaseModelResult:
    """Test BaseModelResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        stage = DonorStageResult(
            donor_id="d1",
            donor_source="s",
            domains=(),
            donor_index=0,
            cka_before=0.5,
            cka_after=0.7,
            cka_improvement=0.2,
            alignment_samples=2,
            mean_alignment_improvement=0.08,
            mean_core_dist_to_source_before=1.2,
            mean_core_dist_to_source_after=1.0,
            boundary_preserved=0.95,
            mean_boundary_relative_diff=0.03,
            max_boundary_relative_diff=0.08,
            layers_transplanted=12,
            weights_transplanted=48,
            mean_projection_loss=0.001,
            mean_null_dim=256.0,
            checkpoint_path="/path",
            duration_seconds=60.0,
        )
        result = BaseModelResult(
            base_id="base-1",
            base_source="test/Base",
            base_alias="b1",
            output_path="/output",
            donor_stages=[stage],
            total_cka_improvement=0.2,
            mean_boundary_preserved=0.95,
            total_donors_applied=1,
            status="completed",
            duration_seconds=65.0,
        )
        data = result.to_dict()
        assert data["base_id"] == "base-1"
        assert len(data["donor_stages"]) == 1
        assert data["status"] == "completed"
        assert data["error"] is None


class TestMultiDonorResult:
    """Test MultiDonorResult dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        result = MultiDonorResult(
            program_id="abc123",
            program_name="Test Program",
            base_results=[],
            total_duration_seconds=600.0,
            completed_at=datetime(2025, 1, 1, 12, 0, 0),
            status="completed",
        )
        data = result.to_dict()
        assert data["_schema"] == "mc.result.multi_donor.v1"
        assert data["program_id"] == "abc123"
        assert data["completed_at"] == "2025-01-01T12:00:00"
        assert data["status"] == "completed"


class TestProgramStatus:
    """Test ProgramStatus dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        status = ProgramStatus(
            program_id="abc123",
            program_name="Test Program",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            updated_at=datetime(2025, 1, 1, 12, 30, 0),
            status="in_progress",
            base_progress=[
                {"base_index": 0, "base_id": "b1", "completed_donors": 1, "total_donors": 3}
            ],
        )
        data = status.to_dict()
        assert data["program_id"] == "abc123"
        assert data["status"] == "in_progress"
        assert len(data["base_progress"]) == 1

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "program_id": "abc123",
            "program_name": "Test Program",
            "started_at": "2025-01-01T12:00:00",
            "updated_at": "2025-01-01T12:30:00",
            "status": "completed",
            "base_progress": [],
        }
        status = ProgramStatus.from_dict(data)
        assert status.program_id == "abc123"
        assert status.status == "completed"
        assert status.started_at.year == 2025

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = ProgramStatus(
            program_id="abc123",
            program_name="Test Program",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            updated_at=datetime(2025, 1, 1, 12, 30, 0),
            status="in_progress",
            base_progress=[{"idx": 0}],
        )
        data = original.to_dict()
        restored = ProgramStatus.from_dict(data)
        assert restored.program_id == original.program_id
        assert restored.status == original.status


# =============================================================================
# Test MultiDonorMergeService
# =============================================================================


class TestMultiDonorMergeService:
    """Test MultiDonorMergeService."""

    @pytest.fixture
    def temp_programs_dir(self):
        """Create temporary programs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def service(self, temp_programs_dir):
        """Create service with temp directory."""
        return MultiDonorMergeService(programs_dir=temp_programs_dir)

    @pytest.fixture
    def simple_program(self):
        """Create simple program for testing."""
        return TransplantProgram(
            name="Test Program",
            bases=(BaseModelSpec(id="base-1", source="test/Base"),),
            donors=(
                DonorSpec(id="donor-1", source="test/Donor", domains=("coding",)),
            ),
            output_dir="~/.modelcypher/test",
        )

    def test_initialization(self, temp_programs_dir):
        """Test service initialization."""
        service = MultiDonorMergeService(programs_dir=temp_programs_dir)
        assert service._programs_dir == temp_programs_dir
        assert service._backend is not None

    def test_programs_dir_created(self):
        """Test programs directory is created if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            programs_dir = Path(tmpdir) / "subdir" / "programs"
            service = MultiDonorMergeService(programs_dir=programs_dir)
            assert programs_dir.exists()

    def test_list_programs_empty(self, service):
        """Test listing programs when none exist."""
        programs = service.list_programs()
        assert programs == []

    def test_get_program_status_not_found(self, service):
        """Test getting status of non-existent program raises."""
        with pytest.raises(FileNotFoundError):
            service.get_program_status("nonexistent")

    def test_dry_run(self, service, simple_program):
        """Test dry run returns valid result without execution."""
        result = service.execute_program(simple_program, dry_run=True)
        assert result.status == "completed"
        assert result.base_results == []
        assert result.total_duration_seconds >= 0

    def test_dry_run_creates_program_dir(self, service, simple_program, temp_programs_dir):
        """Test dry run creates program directory."""
        result = service.execute_program(simple_program, dry_run=True)
        program_dir = temp_programs_dir / result.program_id
        assert program_dir.exists()
        assert (program_dir / "program.yaml").exists()
        assert (program_dir / "status.json").exists()


# =============================================================================
# Test Predefined Program Configs
# =============================================================================


class TestPredefinedPrograms:
    """Test loading and validation of predefined program configs."""

    @pytest.fixture
    def programs_dir(self):
        """Return path to predefined programs."""
        return Path(__file__).parent.parent / "src" / "modelcypher" / "data" / "programs"

    def test_program_a_loads(self, programs_dir):
        """Test Program A config loads successfully."""
        program_path = programs_dir / "program_a_permissive.yaml"
        if program_path.exists():
            program = TransplantProgram.from_yaml(program_path)
            assert "Permissive" in program.name
            assert len(program.bases) >= 1
            assert len(program.donors) >= 1

    def test_program_b_loads(self, programs_dir):
        """Test Program B config loads successfully."""
        program_path = programs_dir / "program_b_llama.yaml"
        if program_path.exists():
            program = TransplantProgram.from_yaml(program_path)
            assert "Mistral" in program.name or "Llama" in program.name
            assert len(program.bases) >= 1
            assert len(program.donors) >= 1

    def test_program_c_loads(self, programs_dir):
        """Test Program C config loads successfully."""
        program_path = programs_dir / "program_c_agentic.yaml"
        if program_path.exists():
            program = TransplantProgram.from_yaml(program_path)
            assert "Qwen" in program.name or "Agentic" in program.name
            assert len(program.bases) >= 1
            assert len(program.donors) >= 1

    def test_all_programs_have_valid_domains(self, programs_dir):
        """Test all programs have valid domain names."""
        valid_domains = {
            "mathematical", "logical", "coding", "reasoning",
            "factual", "computational", "philosophical",
        }
        for yaml_file in programs_dir.glob("*.yaml"):
            program = TransplantProgram.from_yaml(yaml_file)
            for donor in program.donors:
                for domain in donor.domains:
                    assert domain in valid_domains, (
                        f"Invalid domain '{domain}' in {yaml_file.name}"
                    )

    def test_all_programs_have_evaluation_config(self, programs_dir):
        """Test all programs have evaluation configuration."""
        for yaml_file in programs_dir.glob("*.yaml"):
            program = TransplantProgram.from_yaml(yaml_file)
            assert program.evaluation is not None
            assert len(program.evaluation.smoke_test_prompts) > 0
