# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.dataset_loading import load_jsonl_dataset


def test_load_jsonl_valid(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "hello world"}\n{"text": "foo bar"}\n', encoding="utf-8")

    samples = load_jsonl_dataset(path)

    assert len(samples) == 2
    assert samples[0]["text"] == "hello world"


def test_load_jsonl_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n', encoding="utf-8")

    samples = load_jsonl_dataset(path)

    assert len(samples) == 2


def test_load_jsonl_skips_non_text(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n{"label": "b"}\n', encoding="utf-8")

    samples = load_jsonl_dataset(path)

    assert len(samples) == 1
    assert samples[0]["text"] == "a"


def test_load_jsonl_empty_raises(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"label": "no text"}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        load_jsonl_dataset(path)


def test_load_jsonl_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_jsonl_dataset("/nonexistent/path.jsonl")
