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

"""Global CLI behavior contract tests.

Covers output-format and envelope invariants for top-level options:
- --output json behavior is position-independent
- --json/--text shorthands preserve contract
- --ai default output behavior
- --trace-id propagation in structured errors
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


def _invoke(args: list[str], *, env: dict[str, str] | None = None):
    return runner.invoke(app, args, env=env)


def _json_payload(output: str) -> dict:
    return json.loads(output)


def test_output_json_is_position_independent() -> None:
    front = _invoke(["--output", "json", "analyze", "verification-depth-profile"])
    tail = _invoke(["analyze", "verification-depth-profile", "--output", "json"])

    assert front.exit_code != 0
    assert tail.exit_code != 0

    front_payload = _json_payload(front.stdout)
    tail_payload = _json_payload(tail.stdout)

    assert front_payload["error"]["code"] == "MC-3072"
    assert tail_payload["error"]["code"] == "MC-3072"


def test_json_shorthand_matches_output_json() -> None:
    explicit = _invoke(["--output", "json", "analyze", "verification-depth-profile"])
    shorthand = _invoke(["--json", "analyze", "verification-depth-profile"])

    assert explicit.exit_code != 0
    assert shorthand.exit_code != 0

    explicit_payload = _json_payload(explicit.stdout)
    shorthand_payload = _json_payload(shorthand.stdout)

    assert explicit_payload["error"]["code"] == "MC-3072"
    assert shorthand_payload["error"]["code"] == "MC-3072"
    assert explicit_payload["error"]["title"] == shorthand_payload["error"]["title"]


def test_text_shorthand_forces_text_output_shape() -> None:
    result = _invoke(["--text", "analyze", "verification-depth-profile"])
    assert result.exit_code != 0

    # Text mode prints pretty JSON for structured payloads.
    assert result.stdout.startswith("{\n")
    assert '  "error"' in result.stdout
    payload = _json_payload(result.stdout)
    assert payload["error"]["code"] == "MC-3072"


def test_ai_flag_defaults_to_json_when_ai_disabled_in_env() -> None:
    text_mode = _invoke(
        ["analyze", "verification-depth-profile"],
        env={"MC_NO_AI": "1"},
    )
    ai_mode = _invoke(
        ["--ai", "analyze", "verification-depth-profile"],
        env={"MC_NO_AI": "1"},
    )

    assert text_mode.exit_code != 0
    assert ai_mode.exit_code != 0

    # When AI mode is disabled via env, default is text (pretty JSON).
    assert text_mode.stdout.startswith("{\n")
    # --ai overrides env and forces compact JSON.
    assert ai_mode.stdout.startswith('{"error":')

    assert _json_payload(text_mode.stdout)["error"]["code"] == "MC-3072"
    assert _json_payload(ai_mode.stdout)["error"]["code"] == "MC-3072"


def test_trace_id_propagates_into_error_envelope() -> None:
    result = _invoke(
        [
            "--trace-id",
            "trace-123",
            "--output",
            "json",
            "analyze",
            "verification-depth-profile",
        ]
    )
    assert result.exit_code != 0

    payload = _json_payload(result.stdout)
    assert payload["error"]["code"] == "MC-3072"
    assert payload["error"]["traceId"] == "trace-123"


def test_invalid_global_option_is_rejected() -> None:
    result = _invoke(["--bogus-global", "analyze", "verification-depth-profile"])
    assert result.exit_code != 0
    assert "No such option" in result.stdout
