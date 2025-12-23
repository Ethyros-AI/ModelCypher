from __future__ import annotations

from datetime import datetime

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import (
    CheckpointRecord,
    DatasetInfo,
    EvaluationResult,
    ModelInfo,
)


def test_atomic_write_cleans_temp_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    store = FileSystemStore()
    store.register_model(
        ModelInfo(
            id="demo",
            alias="demo",
            architecture="llama",
            format="safetensors",
            path=str(tmp_path / "model"),
            size_bytes=0,
            parameter_count=None,
            is_default_chat=False,
            created_at=datetime.utcnow(),
        )
    )

    tmp_files = list(store.paths.base.glob(".models.json.*.tmp"))
    assert tmp_files == []
    assert store.paths.base.joinpath("models.json.lock").exists()


def test_register_dataset_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    store = FileSystemStore()
    dataset = DatasetInfo(
        id="ds1",
        name="ds1",
        path=str(tmp_path / "data"),
        size_bytes=100,
        example_count=10,
        created_at=datetime.utcnow(),
    )
    store.register_dataset(dataset)
    
    assert store.paths.datasets.exists()
    assert store.paths.base.joinpath("datasets.json.lock").exists()


def test_add_checkpoint_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    store = FileSystemStore()
    checkpoint = CheckpointRecord(
        job_id="job1",
        step=100,
        loss=0.5,
        timestamp=datetime.utcnow(),
        file_path=str(tmp_path / "ckpt.safetensors"),
    )
    store.add_checkpoint(checkpoint)
    
    assert store.paths.checkpoints.exists()
    assert store.paths.base.joinpath("checkpoints.json.lock").exists()


def test_save_evaluation_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "mc_home"))
    store = FileSystemStore()
    eval_result = EvaluationResult(
        id="eval1",
        model_path="path",
        model_name="model",
        dataset_path="data",
        dataset_name="ds",
        average_loss=0.2,
        perplexity=1.5,
        sample_count=5,
        timestamp=datetime.utcnow(),
        config={},
        sample_results=[],
    )
    store.save_evaluation(eval_result)

    assert store.paths.evaluations.exists()
    assert store.paths.base.joinpath("evaluations.json.lock").exists()
