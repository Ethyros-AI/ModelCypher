from __future__ import annotations

import json

from modelcypher.core.domain.dataset_validation import DatasetContentFormat
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_preview_and_get_row_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "data.jsonl"
    long_text = "x" * 9000
    line = json.dumps({"text": long_text}, ensure_ascii=True)
    _write_lines(dataset_path, [line])

    service = DatasetEditorService()
    preview = service.preview(str(dataset_path), limit=1)
    assert preview.rows
    row = preview.rows[0]
    assert row.raw_truncated is True
    assert "truncated" in row.raw

    fetched = service.get_row(str(dataset_path), 1)
    assert fetched.raw_truncated is True
    assert "truncated" in fetched.raw
    assert fetched.fields_truncated == ["text"]


def test_add_update_delete_row(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "data.jsonl"
    _write_lines(dataset_path, [json.dumps({"text": "hello"}, ensure_ascii=True)])

    service = DatasetEditorService()
    add_result = service.add_row(
        str(dataset_path),
        DatasetContentFormat.text,
        {"text": "world"},
    )
    assert add_result.line_number == 2

    update_result = service.update_row(str(dataset_path), 1, {"text": "updated"})
    assert update_result.status == "updated"
    assert update_result.row is not None
    assert update_result.row.fields["text"] == "updated"

    delete_result = service.delete_row(str(dataset_path), 2)
    assert delete_result.status == "deleted"

    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "updated"


def test_convert_dataset_text_to_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCYPHER_HOME", str(tmp_path / "home"))
    dataset_path = tmp_path / "data.jsonl"
    _write_lines(dataset_path, [json.dumps({"text": "hello"}, ensure_ascii=True)])

    output_path = tmp_path / "out.jsonl"
    service = DatasetEditorService()
    result = service.convert_dataset(
        str(dataset_path),
        DatasetContentFormat.chat,
        str(output_path),
    )
    assert result.line_count == 1

    converted = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert converted["messages"][0]["role"] == "user"
    assert converted["messages"][0]["content"] == "hello"
