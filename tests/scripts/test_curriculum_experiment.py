# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_script_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "curriculum_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "curriculum_experiment_script",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_prompt_and_expected_prefers_answer_start():
    script = _load_script_module()
    item = {"text": "What is 2 + 2?\n4", "answer_start": 14}

    prompt, expected = script._extract_prompt_and_expected(item)

    assert prompt == "What is 2 + 2?\n"
    assert expected == "4"


def test_extract_prompt_and_expected_uses_answer_marker():
    script = _load_script_module()
    item = {"text": "Question: Why?\nAnswer: Because."}

    prompt, expected = script._extract_prompt_and_expected(item)

    assert prompt == "Question: Why?\nAnswer:"
    assert expected == "Because."


def test_extract_prompt_and_expected_falls_back_to_last_token():
    script = _load_script_module()
    item = {"text": "sky blue"}

    prompt, expected = script._extract_prompt_and_expected(item)

    assert prompt == "sky"
    assert expected == "blue"


def test_merge_eval_files_concatenates_all_shards(tmp_path, monkeypatch):
    script = _load_script_module()
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    shard_a.write_text('{"text":"a1"}\n{"text":"a2"}\n')
    shard_b.write_text('{"text":"b1"}\n')
    skill = SimpleNamespace(name="logic_skill", eval_files=("a.jsonl", "b.jsonl"))

    path_map = {
        "a.jsonl": shard_a,
        "b.jsonl": shard_b,
    }
    monkeypatch.setattr(script, "_resolve_data_path", lambda rel_path: path_map[rel_path])

    merged = script._merge_eval_files(skill)

    assert merged.read_text().splitlines() == [
        '{"text":"a1"}',
        '{"text":"a2"}',
        '{"text":"b1"}',
    ]


def test_parse_training_envelope_extracts_agent_envelope_result():
    script = _load_script_module()
    envelope = {
        "status": "ok",
        "result": {"final_loss": 1.23, "adapter_path": "/tmp/adapter"},
        "metadata": {"adapter_path": "/tmp/metadata_adapter"},
    }

    parsed_envelope, training_result = script._parse_training_envelope(json.dumps(envelope))

    assert parsed_envelope == envelope
    assert training_result == envelope["result"]


def test_parse_training_envelope_accepts_flat_payload():
    script = _load_script_module()
    payload = {"final_loss": 1.23, "adapter_path": "/tmp/adapter"}

    parsed_envelope, training_result = script._parse_training_envelope(json.dumps(payload))

    assert parsed_envelope == payload
    assert training_result == payload


def test_resolve_adapter_path_uses_expected_fallback_order():
    script = _load_script_module()

    assert script._resolve_adapter_path(
        {"adapter_path": "/tmp/from_result"},
        {"metadata": {"adapter_path": "/tmp/from_metadata"}},
        "/tmp/default",
    ) == "/tmp/from_result"
    assert script._resolve_adapter_path(
        {},
        {"metadata": {"adapter_path": "/tmp/from_metadata"}},
        "/tmp/default",
    ) == "/tmp/from_metadata"
    assert script._resolve_adapter_path(None, None, "/tmp/default") == "/tmp/default"


def test_main_uses_merged_eval_path_and_saves_full_envelope(tmp_path, monkeypatch):
    script = _load_script_module()
    output_dir = tmp_path / "out"
    merged_eval = tmp_path / "merged_eval.jsonl"
    merged_eval.write_text('{"text":"eval"}\n')
    prepared_train = tmp_path / "train.jsonl"
    prepared_train.write_text('{"text":"train"}\n')

    model_path = "/tmp/model"
    args = argparse.Namespace(
        model=model_path,
        skill="logic_skill",
        output_dir=output_dir,
        dry_run=False,
        samples=2,
    )
    skill = SimpleNamespace(
        name="logic_skill",
        formal_statement="If p then q; p; therefore q.",
        answer_mode="exact",
        branch="logic",
        prerequisites=(),
        train_files=("train.jsonl",),
        eval_files=("eval_a.jsonl", "eval_b.jsonl"),
    )
    baseline = SimpleNamespace(
        accuracy=0.45,
        n_correct=45,
        n_total=100,
        ci_lower=0.322,
        ci_upper=0.583,
        regime="reinforce",
    )
    post = SimpleNamespace(
        accuracy=0.56,
        n_correct=56,
        n_total=100,
        ci_lower=0.427,
        ci_upper=0.687,
        regime="reinforce",
    )
    envelope = {
        "status": "ok",
        "result": {
            "train_iters": 10,
            "baseline_loss": 4.2,
            "final_loss": 1.0,
            "post_loss": 3.7,
            "min_cka": 0.899,
            "spectral_bounds_ok": True,
            "pipeline_gate_passed": True,
            "adapter_path": str(output_dir / "logic_skill_adapter"),
        },
        "metadata": {"adapter_path": str(output_dir / "logic_skill_adapter")},
        "diagnostics": {"summary": "synthetic diagnostics"},
    }
    calls: list[tuple[str, str | None]] = []
    train_cmd: list[str] = []

    monkeypatch.setattr(script.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(script, "_find_skill_node", lambda skill_name: skill if skill_name == args.skill else None)
    monkeypatch.setattr(script, "_prepare_training_data", lambda _: prepared_train)
    monkeypatch.setattr(script, "_merge_eval_files", lambda _: merged_eval)

    def _fake_run_mastery_eval(model, skill_node, adapter_path=None):
        calls.append((model, adapter_path))
        assert skill_node is skill
        if adapter_path is None:
            return baseline
        return post

    monkeypatch.setattr(script, "_run_mastery_eval", _fake_run_mastery_eval)
    monkeypatch.setattr(script, "_run_sample_inference", lambda *args, **kwargs: None)

    def _fake_subprocess_run(cmd, capture_output, text, cwd):
        _ = (capture_output, text, cwd)
        train_cmd[:] = cmd
        adapter_dir = Path(envelope["result"]["adapter_path"])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr(script.subprocess, "run", _fake_subprocess_run)

    script.main()

    assert train_cmd[0:5] == ["poetry", "run", "mc", "train", "run"]
    assert train_cmd[train_cmd.index("--eval-data") + 1] == str(merged_eval)
    assert calls == [
        (model_path, None),
        (model_path, str(output_dir / "logic_skill_adapter")),
    ]

    result_path = output_dir / "logic_skill_training_result.json"
    summary_path = output_dir / "logic_skill_experiment_summary.json"
    assert json.loads(result_path.read_text()) == envelope

    summary = json.loads(summary_path.read_text())
    assert summary["baseline_accuracy"] == baseline.accuracy
    assert summary["post_accuracy"] == post.accuracy
    assert summary["training_result"] == envelope["result"]
