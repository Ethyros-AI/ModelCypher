# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

from modelcypher.core.domain.safety.training_sample import TrainingSample


def test_extract_text_handles_blank_plain_and_non_dict_json() -> None:
    assert TrainingSample._extract_text("   ") == ""
    assert TrainingSample._extract_text("  plain line  ") == "plain line"
    assert TrainingSample._extract_text('["a", "b"]') == '["a", "b"]'


def test_extract_text_prefers_text_and_completion_formats() -> None:
    assert TrainingSample._extract_text('{"text": "  hello  "}') == "hello"

    assert TrainingSample._extract_text('{"completion": " done "}') == "done"
    assert (
        TrainingSample._extract_text('{"prompt": "Q", "completion": "A"}')
        == "Q\nA"
    )


def test_extract_text_messages_and_fallbacks() -> None:
    messages = (
        '{"messages": ['
        '{"role": "system", "content": "  rule  "}, '
        '{"role": "user", "content": " hi "}, '
        '{"role": "assistant", "content": 42}'
        ']} '
    )
    assert TrainingSample._extract_text(messages) == "rule\nhi"

    unknown = '{"foo": "bar"}'
    assert TrainingSample._extract_text(unknown) == unknown


def test_training_sample_from_line_and_str() -> None:
    source = Path("/tmp/dataset.jsonl")
    sample = TrainingSample.from_line('{"text": "example"}', source)

    assert sample.raw == '{"text": "example"}'
    assert sample.text == "example"
    assert sample.source_path == source
    assert str(sample) == "example"
