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

"""Tests for data preparation domain and service.

Pure-domain tests — no GPU, no MLX, no model loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelcypher.core.domain.data_preparation import (
    LengthStats,
    compute_length_stats,
    detect_source_format,
    parse_conversation_json,
    parse_csv_to_samples,
    validate_jsonl_lines,
)
from modelcypher.core.use_cases.data_preparation_service import (
    DataPreparationService,
)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectSourceFormat:
    def test_jsonl_extension(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.touch()
        assert detect_source_format(str(f)) == "jsonl"

    def test_csv_extension(self, tmp_path):
        f = tmp_path / "data.csv"
        f.touch()
        assert detect_source_format(str(f)) == "csv"

    def test_json_extension(self, tmp_path):
        f = tmp_path / "data.json"
        f.touch()
        assert detect_source_format(str(f)) == "conversation_json"

    def test_txt_extension(self, tmp_path):
        f = tmp_path / "data.txt"
        f.touch()
        assert detect_source_format(str(f)) == "txt"

    def test_parquet_extension(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.touch()
        assert detect_source_format(str(f)) == "parquet"

    def test_huggingface_simple_name(self):
        assert detect_source_format("gsm8k") == "huggingface"

    def test_huggingface_org_name(self):
        assert detect_source_format("openassistant/oasst1") == "huggingface"

    def test_existing_file_without_extension(self, tmp_path):
        f = tmp_path / "mydata"
        f.write_text("hello")
        assert detect_source_format(str(f)) == "txt"


# ---------------------------------------------------------------------------
# JSONL validation
# ---------------------------------------------------------------------------


class TestValidateJsonlLines:
    def test_valid_text_samples(self):
        lines = [
            '{"text": "Hello world"}',
            '{"text": "Another sample"}',
        ]
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 2
        assert not warnings

    def test_valid_messages_samples(self):
        lines = [
            '{"messages": [{"role": "user", "content": "hi"}]}',
        ]
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 1

    def test_skips_empty_lines(self):
        lines = ['{"text": "one"}', "", '{"text": "two"}', "  "]
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 2
        assert any("empty" in w.lower() for w in warnings)

    def test_skips_invalid_json(self):
        lines = ['{"text": "ok"}', "not json at all"]
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 1
        assert any("invalid JSON" in w for w in warnings)

    def test_skips_missing_fields(self):
        lines = ['{"text": "ok"}', '{"label": "only a label"}']
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 1
        assert any("missing" in w.lower() for w in warnings)

    def test_removes_duplicates(self):
        lines = ['{"text": "same"}', '{"text": "same"}', '{"text": "different"}']
        valid, warnings = validate_jsonl_lines(lines)
        assert len(valid) == 2
        assert any("duplicate" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


class TestParseCsvToSamples:
    def test_explicit_column(self):
        rows = [{"text": "a", "label": "1"}, {"text": "b", "label": "2"}]
        samples, warnings = parse_csv_to_samples(rows, text_column="text")
        assert len(samples) == 2
        assert samples[0]["text"] == "a"

    def test_auto_detect_column(self):
        rows = [{"content": "hello", "id": "1"}]
        samples, warnings = parse_csv_to_samples(rows)
        assert len(samples) == 1
        assert samples[0]["text"] == "hello"

    def test_fallback_to_first_column(self):
        rows = [{"custom_field": "data", "other": "x"}]
        samples, warnings = parse_csv_to_samples(rows)
        assert len(samples) == 1
        assert any("first column" in w.lower() for w in warnings)

    def test_missing_column_error(self):
        rows = [{"a": "1"}]
        samples, warnings = parse_csv_to_samples(rows, text_column="nonexistent")
        assert len(samples) == 0
        assert any("not found" in w.lower() for w in warnings)

    def test_skips_empty_rows(self):
        rows = [{"text": "valid"}, {"text": ""}, {"text": "also valid"}]
        samples, warnings = parse_csv_to_samples(rows, text_column="text")
        assert len(samples) == 2

    def test_empty_csv(self):
        samples, warnings = parse_csv_to_samples([])
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# Conversation JSON parsing
# ---------------------------------------------------------------------------


class TestParseConversationJson:
    def test_single_conversation(self):
        data = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        samples, warnings = parse_conversation_json(data)
        assert len(samples) == 1
        assert samples[0]["messages"][0]["role"] == "user"

    def test_multiple_conversations(self):
        data = [
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
            [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        ]
        samples, warnings = parse_conversation_json(data)
        assert len(samples) == 2

    def test_dict_with_messages_key(self):
        data = {"messages": [{"role": "user", "content": "test"}]}
        samples, warnings = parse_conversation_json(data)
        assert len(samples) == 1

    def test_list_of_dicts_with_messages(self):
        data = [
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]},
        ]
        samples, warnings = parse_conversation_json(data)
        assert len(samples) == 2

    def test_empty_array(self):
        samples, warnings = parse_conversation_json([])
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# Length statistics
# ---------------------------------------------------------------------------


class TestComputeLengthStats:
    def test_basic_stats(self):
        stats = compute_length_stats([10, 20, 30, 40, 50])
        assert stats is not None
        assert stats.min == 10
        assert stats.max == 50
        assert stats.mean == 30.0
        assert stats.median == 30.0

    def test_empty_returns_none(self):
        assert compute_length_stats([]) is None

    def test_single_value(self):
        stats = compute_length_stats([42])
        assert stats is not None
        assert stats.min == 42
        assert stats.max == 42
        assert stats.mean == 42.0

    def test_to_dict(self):
        stats = compute_length_stats([100, 200, 300])
        d = stats.to_dict()
        assert "min" in d
        assert "p95" in d


# ---------------------------------------------------------------------------
# Service integration tests (filesystem only, no GPU)
# ---------------------------------------------------------------------------


class TestDataPreparationService:
    def test_prepare_jsonl(self, tmp_path):
        source = tmp_path / "data.jsonl"
        source.write_text(
            '{"text": "sample one"}\n'
            '{"text": "sample two"}\n'
            '{"text": "sample three"}\n'
        )
        output = tmp_path / "out.jsonl"

        service = DataPreparationService()
        result = service.prepare(str(source), output=output)

        assert result.statistics.n_samples == 3
        assert result.statistics.format_detected == "jsonl"
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_prepare_csv(self, tmp_path):
        source = tmp_path / "data.csv"
        source.write_text("text,label\nhello world,positive\ngoodbye,negative\n")
        output = tmp_path / "out.jsonl"

        service = DataPreparationService()
        result = service.prepare(str(source), output=output)

        assert result.statistics.n_samples == 2
        assert result.statistics.format_detected == "csv"
        assert output.exists()

    def test_prepare_conversation_json(self, tmp_path):
        source = tmp_path / "data.json"
        conversations = [
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        ]
        source.write_text(json.dumps(conversations))
        output = tmp_path / "out.jsonl"

        service = DataPreparationService()
        result = service.prepare(str(source), output=output)

        assert result.statistics.n_samples == 1
        assert result.statistics.format_detected == "conversation_json"

    def test_prepare_text(self, tmp_path):
        source = tmp_path / "data.txt"
        source.write_text("First paragraph.\n\nSecond paragraph.\n\nThird one.")
        output = tmp_path / "out.jsonl"

        service = DataPreparationService()
        result = service.prepare(str(source), output=output)

        assert result.statistics.n_samples == 3
        assert result.statistics.format_detected == "txt"

    def test_auto_derived_output_path(self, tmp_path):
        source = tmp_path / "mydata.jsonl"
        source.write_text('{"text": "hello"}\n')

        service = DataPreparationService()
        result = service.prepare(str(source))

        expected = tmp_path / "mydata.prepared.jsonl"
        assert result.statistics.output_path == str(expected)
        assert expected.exists()

    def test_suggested_command_with_model(self, tmp_path):
        source = tmp_path / "data.jsonl"
        source.write_text('{"text": "sample"}\n')

        service = DataPreparationService()
        result = service.prepare(
            str(source),
            model_path=Path("/path/to/model"),
        )

        assert result.suggested_command is not None
        assert "/path/to/model" in result.suggested_command

    def test_envelope_success(self, tmp_path):
        source = tmp_path / "data.jsonl"
        source.write_text('{"text": "sample"}\n')

        service = DataPreparationService()
        result = service.prepare(str(source), model_path=Path("/model"))
        envelope = service.make_envelope(result, model_path="/model")

        assert envelope.status == "success"
        assert envelope.command == "mc data prepare"
        d = envelope.to_dict()
        assert "diagnostics" in d
        assert len(d["diagnostics"]["recommendations"]) > 0

    def test_envelope_warnings(self, tmp_path):
        source = tmp_path / "data.jsonl"
        source.write_text('{"text": "valid"}\nnot json\n')

        service = DataPreparationService()
        result = service.prepare(str(source))
        envelope = service.make_envelope(result)

        assert envelope.status == "partial"
        assert "warning" in envelope.diagnostics.summary.lower()

    def test_empty_file(self, tmp_path):
        source = tmp_path / "data.txt"
        source.write_text("")

        service = DataPreparationService()
        result = service.prepare(str(source))
        envelope = service.make_envelope(result)

        assert envelope.status == "failure"
        assert result.statistics.n_samples == 0
