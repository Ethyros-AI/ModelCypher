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

"""Tests for filesystem storage adapter."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from modelcypher.adapters.filesystem_storage import FileSystemStore, StoragePaths
from modelcypher.core.domain.models import (
    CheckpointRecord,
    CompareCheckpointResult,
    CompareSession,
    EvaluationResult,
    ModelInfo,
    TrainingJob,
)
from modelcypher.core.domain.training import TrainingStatus


@pytest.fixture
def temp_storage_paths():
    """Create temporary storage paths for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = StoragePaths()
        paths.base = Path(tmpdir)
        paths.models = paths.base / "models.json"
        paths.jobs = paths.base / "jobs"
        paths.jobs.mkdir(parents=True, exist_ok=True)
        paths.checkpoints = paths.base / "checkpoints.json"
        paths.evaluations = paths.base / "evaluations.json"
        paths.comparisons = paths.base / "comparisons.json"
        paths.logs = paths.base / "logs"
        paths.logs.mkdir(parents=True, exist_ok=True)
        yield paths


@pytest.fixture
def store(temp_storage_paths):
    """Create a FileSystemStore with temporary paths."""
    return FileSystemStore(paths=temp_storage_paths)


@pytest.fixture
def sample_model():
    """Create a sample ModelInfo for testing."""
    return ModelInfo(
        id="test-model-1",
        alias="test-model",
        architecture="llama",
        format="safetensors",
        path="/path/to/model",
        size_bytes=1024,
        parameter_count=7_000_000_000,
        is_default_chat=False,
        created_at=datetime(2025, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_job():
    """Create a sample TrainingJob for testing."""
    return TrainingJob(
        job_id="job-123",
        status=TrainingStatus.pending,
        model_id="test-model-1",
        dataset_path="/path/to/dataset",
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        updated_at=datetime(2025, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_checkpoint():
    """Create a sample CheckpointRecord for testing."""
    return CheckpointRecord(
        job_id="job-123",
        step=100,
        loss=0.5,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        file_path="/path/to/checkpoint.safetensors",
    )


@pytest.fixture
def sample_evaluation():
    """Create a sample EvaluationResult for testing."""
    return EvaluationResult(
        id="eval-123",
        model_path="/path/to/model",
        model_name="test-model",
        dataset_path="/path/to/dataset",
        dataset_name="test-dataset",
        average_loss=0.5,
        perplexity=1.65,
        sample_count=100,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        config={"batch_size": 8},
        sample_results=[],
    )


class TestStoragePaths:
    """Test StoragePaths initialization."""

    def test_creates_required_directories(self):
        """StoragePaths should create jobs and logs directories."""
        paths = StoragePaths()
        assert paths.jobs.exists()
        assert paths.logs.exists()

    def test_defines_all_required_paths(self):
        """StoragePaths should define all required file paths."""
        paths = StoragePaths()
        assert paths.models is not None
        assert paths.checkpoints is not None
        assert paths.evaluations is not None
        assert paths.comparisons is not None


class TestFileSystemStoreModels:
    """Test FileSystemStore model operations."""

    def test_list_models_empty(self, store):
        """list_models should return empty list when no models registered."""
        result = store.list_models()
        assert result == []

    def test_register_model(self, store, sample_model):
        """register_model should save model to storage."""
        store.register_model(sample_model)
        models = store.list_models()

        assert len(models) == 1
        assert models[0].id == sample_model.id
        assert models[0].alias == sample_model.alias

    def test_register_model_updates_existing(self, store, sample_model):
        """register_model should update existing model with same id."""
        store.register_model(sample_model)

        updated = ModelInfo(
            id=sample_model.id,
            alias="updated-alias",
            architecture="qwen2",
            format="safetensors",
            path="/new/path",
            size_bytes=2048,
            parameter_count=14_000_000_000,
            is_default_chat=True,
            created_at=datetime.utcnow(),
        )
        store.register_model(updated)

        models = store.list_models()
        assert len(models) == 1
        assert models[0].alias == "updated-alias"

    def test_get_model_by_id(self, store, sample_model):
        """get_model should retrieve model by id."""
        store.register_model(sample_model)
        result = store.get_model(sample_model.id)

        assert result is not None
        assert result.id == sample_model.id

    def test_get_model_by_alias(self, store, sample_model):
        """get_model should retrieve model by alias."""
        store.register_model(sample_model)
        result = store.get_model(sample_model.alias)

        assert result is not None
        assert result.alias == sample_model.alias

    def test_get_model_not_found(self, store):
        """get_model should return None for non-existent model."""
        result = store.get_model("non-existent")
        assert result is None

    def test_delete_model_by_id(self, store, sample_model):
        """delete_model should remove model by id."""
        store.register_model(sample_model)
        store.delete_model(sample_model.id)

        assert store.get_model(sample_model.id) is None

    def test_delete_model_by_alias(self, store, sample_model):
        """delete_model should remove model by alias."""
        store.register_model(sample_model)
        store.delete_model(sample_model.alias)

        assert store.get_model(sample_model.alias) is None


class TestFileSystemStoreJobs:
    """Test FileSystemStore job operations."""

    def test_save_job(self, store, sample_job):
        """save_job should persist job to storage."""
        store.save_job(sample_job)
        retrieved = store.get_job(sample_job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id
        assert retrieved.status == sample_job.status

    def test_update_job(self, store, sample_job):
        """update_job should update existing job."""
        store.save_job(sample_job)

        updated = TrainingJob(
            job_id=sample_job.job_id,
            status=TrainingStatus.running,
            model_id=sample_job.model_id,
            dataset_path=sample_job.dataset_path,
            created_at=sample_job.created_at,
            updated_at=datetime.utcnow(),
            current_step=50,
        )
        store.update_job(updated)

        retrieved = store.get_job(sample_job.job_id)
        assert retrieved.status == TrainingStatus.running
        assert retrieved.current_step == 50

    def test_list_jobs_empty(self, store):
        """list_jobs should return empty list when no jobs exist."""
        result = store.list_jobs()
        assert result == []

    def test_list_jobs_filter_by_status(self, store):
        """list_jobs should filter by status."""
        job1 = TrainingJob(
            job_id="job-1",
            status=TrainingStatus.pending,
            model_id="model-1",
            dataset_path="/path",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        job2 = TrainingJob(
            job_id="job-2",
            status=TrainingStatus.running,
            model_id="model-2",
            dataset_path="/path",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        store.save_job(job1)
        store.save_job(job2)

        pending_jobs = store.list_jobs(status=TrainingStatus.pending)
        assert len(pending_jobs) == 1
        assert pending_jobs[0].job_id == "job-1"

    def test_list_jobs_active_only(self, store):
        """list_jobs with active_only should filter inactive jobs."""
        job1 = TrainingJob(
            job_id="job-1",
            status=TrainingStatus.running,
            model_id="model-1",
            dataset_path="/path",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        job2 = TrainingJob(
            job_id="job-2",
            status=TrainingStatus.completed,
            model_id="model-2",
            dataset_path="/path",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        store.save_job(job1)
        store.save_job(job2)

        active_jobs = store.list_jobs(active_only=True)
        assert len(active_jobs) == 1
        assert active_jobs[0].job_id == "job-1"

    def test_get_job_not_found(self, store):
        """get_job should return None for non-existent job."""
        result = store.get_job("non-existent")
        assert result is None

    def test_delete_job(self, store, sample_job):
        """delete_job should remove job from storage."""
        store.save_job(sample_job)
        store.delete_job(sample_job.job_id)

        assert store.get_job(sample_job.job_id) is None


class TestFileSystemStoreCheckpoints:
    """Test FileSystemStore checkpoint operations."""

    def test_list_checkpoints_empty(self, store):
        """list_checkpoints should return empty list when no checkpoints exist."""
        result = store.list_checkpoints()
        assert result == []

    def test_add_checkpoint(self, store, sample_checkpoint):
        """add_checkpoint should save checkpoint to storage."""
        store.add_checkpoint(sample_checkpoint)
        checkpoints = store.list_checkpoints()

        assert len(checkpoints) == 1
        assert checkpoints[0].job_id == sample_checkpoint.job_id
        assert checkpoints[0].step == sample_checkpoint.step

    def test_list_checkpoints_filter_by_job_id(self, store):
        """list_checkpoints should filter by job_id."""
        cp1 = CheckpointRecord(
            job_id="job-1",
            step=100,
            loss=0.5,
            timestamp=datetime.utcnow(),
            file_path="/path/cp1.safetensors",
        )
        cp2 = CheckpointRecord(
            job_id="job-2",
            step=100,
            loss=0.6,
            timestamp=datetime.utcnow(),
            file_path="/path/cp2.safetensors",
        )
        store.add_checkpoint(cp1)
        store.add_checkpoint(cp2)

        result = store.list_checkpoints(job_id="job-1")
        assert len(result) == 1
        assert result[0].job_id == "job-1"

    def test_delete_checkpoint(self, store, sample_checkpoint, temp_storage_paths):
        """delete_checkpoint should remove checkpoint record."""
        # Create actual file
        cp_path = temp_storage_paths.base / "checkpoint.safetensors"
        cp_path.write_bytes(b"test")

        cp = CheckpointRecord(
            job_id="job-1",
            step=100,
            loss=0.5,
            timestamp=datetime.utcnow(),
            file_path=str(cp_path),
        )
        store.add_checkpoint(cp)
        store.delete_checkpoint(str(cp_path))

        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 0
        assert not cp_path.exists()


class TestFileSystemStoreEvaluations:
    """Test FileSystemStore evaluation operations."""

    def test_list_evaluations_empty(self, store):
        """list_evaluations should return empty list when no evaluations exist."""
        result = store.list_evaluations(limit=10)
        assert result == []

    def test_save_evaluation(self, store, sample_evaluation):
        """save_evaluation should persist evaluation to storage."""
        store.save_evaluation(sample_evaluation)
        evaluations = store.list_evaluations(limit=10)

        assert len(evaluations) == 1
        assert evaluations[0].id == sample_evaluation.id

    def test_get_evaluation(self, store, sample_evaluation):
        """get_evaluation should retrieve evaluation by id."""
        store.save_evaluation(sample_evaluation)
        result = store.get_evaluation(sample_evaluation.id)

        assert result is not None
        assert result.id == sample_evaluation.id

    def test_get_evaluation_not_found(self, store):
        """get_evaluation should return None for non-existent evaluation."""
        result = store.get_evaluation("non-existent")
        assert result is None

    def test_list_evaluations_respects_limit(self, store):
        """list_evaluations should respect limit parameter."""
        for i in range(5):
            ev = EvaluationResult(
                id=f"eval-{i}",
                model_path="/path",
                model_name="model",
                dataset_path="/path",
                dataset_name="dataset",
                average_loss=0.5,
                perplexity=1.5,
                sample_count=100,
                timestamp=datetime(2025, 1, i + 1),
                config={},
                sample_results=[],
            )
            store.save_evaluation(ev)

        result = store.list_evaluations(limit=3)
        assert len(result) == 3

    def test_list_evaluations_sorted_by_timestamp(self, store):
        """list_evaluations should return newest first."""
        for i in range(3):
            ev = EvaluationResult(
                id=f"eval-{i}",
                model_path="/path",
                model_name="model",
                dataset_path="/path",
                dataset_name="dataset",
                average_loss=0.5,
                perplexity=1.5,
                sample_count=100,
                timestamp=datetime(2025, 1, i + 1),
                config={},
                sample_results=[],
            )
            store.save_evaluation(ev)

        result = store.list_evaluations(limit=10)
        # Should be sorted newest first (eval-2 -> eval-1 -> eval-0)
        assert result[0].id == "eval-2"
        assert result[2].id == "eval-0"


class TestFileSystemStoreSessions:
    """Test FileSystemStore comparison session operations."""

    def test_list_sessions_empty(self, store):
        """list_sessions should return empty list when no sessions exist."""
        result = store.list_sessions(limit=10)
        assert result == []

    def test_save_session(self, store):
        """save_session should persist session to storage."""
        session = CompareSession(
            id="session-123",
            created_at=datetime.utcnow(),
            prompt="Test prompt",
            config={"status": "active"},
            checkpoints=[],
        )
        store.save_session(session)
        sessions = store.list_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0].id == session.id

    def test_get_session(self, store):
        """get_session should retrieve session by id."""
        session = CompareSession(
            id="session-123",
            created_at=datetime.utcnow(),
            prompt="Test prompt",
            config={},
            checkpoints=[],
        )
        store.save_session(session)
        result = store.get_session("session-123")

        assert result is not None
        assert result.id == "session-123"

    def test_get_session_not_found(self, store):
        """get_session should return None for non-existent session."""
        result = store.get_session("non-existent")
        assert result is None

    def test_list_sessions_filter_by_status(self, store):
        """list_sessions should filter by status in config."""
        session1 = CompareSession(
            id="session-1",
            created_at=datetime.utcnow(),
            prompt="Test",
            config={"status": "active"},
            checkpoints=[],
        )
        session2 = CompareSession(
            id="session-2",
            created_at=datetime.utcnow(),
            prompt="Test",
            config={"status": "completed"},
            checkpoints=[],
        )
        store.save_session(session1)
        store.save_session(session2)

        result = store.list_sessions(limit=10, status="active")
        assert len(result) == 1
        assert result[0].id == "session-1"


class TestFileSystemStoreJsonOperations:
    """Test FileSystemStore JSON read/write operations."""

    def test_read_json_nonexistent_file(self, store, temp_storage_paths):
        """_read_json should return None for non-existent file."""
        result = store._read_json(temp_storage_paths.base / "nonexistent.json")
        assert result is None

    def test_write_json_creates_parent_dirs(self, store, temp_storage_paths):
        """_write_json should create parent directories."""
        path = temp_storage_paths.base / "subdir" / "nested" / "file.json"
        store._write_json(path, {"key": "value"})

        assert path.exists()
        assert json.loads(path.read_text()) == {"key": "value"}

    def test_write_json_atomic(self, store, temp_storage_paths):
        """_write_json should write atomically (no partial writes)."""
        path = temp_storage_paths.base / "atomic_test.json"
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        store._write_json(path, data)

        # Verify no temp files left behind
        temp_files = list(temp_storage_paths.base.glob(".*.tmp"))
        assert len(temp_files) == 0

        # Verify content is complete
        assert json.loads(path.read_text()) == data


class TestFileSystemStoreDatetimeSerialization:
    """Test datetime serialization/deserialization."""

    def test_to_iso_with_datetime(self, store):
        """_to_iso should convert datetime to ISO string."""
        dt = datetime(2025, 1, 15, 10, 30, 0)
        result = store._to_iso(dt)
        assert result == "2025-01-15T10:30:00"

    def test_to_iso_with_none(self, store):
        """_to_iso should return None for None input."""
        result = store._to_iso(None)
        assert result is None

    def test_from_iso_with_string(self, store):
        """_from_iso should parse ISO string to datetime."""
        result = store._from_iso("2025-01-15T10:30:00")
        assert result == datetime(2025, 1, 15, 10, 30, 0)

    def test_from_iso_with_none(self, store):
        """_from_iso should return None for None input."""
        result = store._from_iso(None)
        assert result is None
