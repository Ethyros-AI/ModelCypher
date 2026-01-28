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

"""Benchmark CLI smoke tests using local fixture."""

from __future__ import annotations

import json
import sys
import types

from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


def test_benchmark_run_local_smoke(tmp_path, monkeypatch):
    """Runs benchmark CLI against local fixture and writes outputs."""

    def dummy_generate(_model, _tokenizer, prompt, max_tokens=50, verbose=False):
        return "yes" if "sky" in prompt.lower() else "no"

    dummy_module = types.SimpleNamespace(
        load=lambda _path: (object(), object()),
        generate=dummy_generate,
    )

    monkeypatch.setitem(sys.modules, "mlx_lm", dummy_module)

    results_path = tmp_path / "results.json"
    failures_path = tmp_path / "failures.tsv"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "run",
            "--model",
            "dummy",
            "--suite",
            "local_smoke",
            "--results-path",
            str(results_path),
            "--failures-path",
            str(failures_path),
            "--no-geometry",
        ],
    )

    assert result.exit_code == 0
    assert results_path.exists()
    assert failures_path.exists()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["suite"] == "local_smoke"
    assert data["benchmarks"][0]["total"] == 2
