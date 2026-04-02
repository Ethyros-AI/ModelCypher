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

"""Tests for analyze CLI commands.

The 'analyze' command exposes workflow-first observation commands plus expert
metrics for geometry, probes, entropy monitoring, and benchmarks.

Tests:
- Command help text for each subcommand
- Input validation for key commands
- Error handling for missing paths
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


def _write_report_bundle_fixture(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bundleVersion": "mc.analyze.bundle.v1",
                "workflow": "family",
                "promptFamilyManifest": {"name": "fixture_family"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            {
                "bundleVersion": "mc.analyze.bundle.v1",
                "workflow": "family",
                "targetCount": 1,
                "variantCount": 2,
                "comparisonCount": 1,
                "errorCount": 0,
                "spaces": ["embedding", "hidden"],
                "meanPromptTokenCount": 5.0,
                "meanResponseTokenCount": 7.0,
                "meanEntropy": 0.3,
                "meanGeodesicDeviation": 0.2,
                "meanCurvature": 0.4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "variants.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "targetLabel": "base",
                        "caseId": "case1",
                        "variantId": "control",
                        "summaryMetrics": {
                            "meanEntropy": 0.2,
                            "meanGeodesicDeviation": 0.1,
                            "meanCurvature": 0.3,
                        },
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "targetLabel": "base",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "summaryMetrics": {
                            "meanEntropy": 0.5,
                            "meanGeodesicDeviation": 0.4,
                            "meanCurvature": 0.7,
                        },
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "layer_metrics.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "targetLabel": "base",
                        "caseId": "case1",
                        "variantId": "control",
                        "layer": -1,
                        "space": "embedding",
                        "vectorNorm": 0.5,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "targetLabel": "base",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "layer": 1,
                        "space": "hidden",
                        "vectorNorm": 2.0,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "comparisons.jsonl").write_text(
        json.dumps(
            {
                "mode": "within_target",
                "caseId": "case1",
                "from": "control",
                "to": "all_caps",
                "targetLabel": "base",
                "scalarDeltas": {"meanEntropy": 0.3},
                "perLayerDeltas": {"entropy": [{"layer": 1, "delta": 0.4}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "REPORT.md").write_text("# report\n", encoding="utf-8")
    return bundle_dir


def _write_measurement_atlas_report_fixture(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "atlas-bundle"
    bundle_dir.mkdir()
    long_prompt = (
        "ALPHA\\nBETA\n\n"
        "This prompt keeps going with extra context about casing and reasoning. "
        "This prompt keeps going with extra context about casing and reasoning. "
        "This prompt keeps going with extra context about casing and reasoning."
    )
    long_generated = (
        "SUPPORTED\\n\\n"
        "This generated explanation keeps going across lines and should be clipped for atlas previews. "
        "This generated explanation keeps going across lines and should be clipped for atlas previews. "
        "This generated explanation keeps going across lines and should be clipped for atlas previews."
    )
    (bundle_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "mc.measurement_atlas.run_manifest.v2",
                "runId": "atlas-run",
                "linkedBlocker": "A1",
                "frozenSurfaces": {
                    "requestedLiveSpaces": ["hidden"],
                    "observedLiveSpaces": ["hidden"],
                    "requestedReplaySpaces": ["hidden", "embedding"],
                    "observedReplaySpaces": ["hidden", "embedding"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            {
                "runId": "atlas-run",
                "linkedBlocker": "A1",
                "studyCount": 1,
                "variantCount": 2,
                "comparisonCount": 1,
                "onsetEventCount": 1,
                "errorCount": 0,
                "spaces": ["embedding", "hidden"],
                "modes": ["live", "replay"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "variants.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "control",
                        "promptText": "alpha beta",
                        "generatedText": "SUPPORTED",
                        "errors": [],
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "promptText": long_prompt,
                        "generatedText": long_generated,
                        "errors": [],
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "sequence_metrics.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "control",
                        "mode": "replay",
                        "region": "response",
                        "space": "embedding",
                        "peakLayer": None,
                        "peakLocus": "embedding",
                        "firstBendLayer": None,
                        "firstBendLocus": "embedding",
                        "meanEntropy": None,
                        "meanSpectralEntropy": 0.2,
                        "meanEffectiveRank": 1.0,
                        "meanIntrinsicDimension": 2.0,
                        "meanCurvature": 0.1,
                        "maxCurvature": 0.2,
                        "meanGeodesicDeviation": 0.1,
                        "meanPathLengthRatio": 1.1,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "mode": "replay",
                        "region": "response",
                        "space": "hidden",
                        "peakLayer": 3,
                        "peakLocus": "layer:3",
                        "firstBendLayer": 2,
                        "firstBendLocus": "layer:2",
                        "meanEntropy": 0.5,
                        "meanSpectralEntropy": 0.5,
                        "meanEffectiveRank": 1.5,
                        "meanIntrinsicDimension": 2.5,
                        "meanCurvature": 0.8,
                        "maxCurvature": 1.0,
                        "meanGeodesicDeviation": 0.7,
                        "meanPathLengthRatio": 1.5,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "step_metrics.jsonl").write_text(
        json.dumps({"studyId": "measurement_atlas_casing", "stepIndex": 0}) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "space_step_metrics.jsonl").write_text(
        json.dumps({"studyId": "measurement_atlas_casing", "space": "hidden", "stepIndex": 0})
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "comparisons.jsonl").write_text(
        json.dumps(
            {
                "studyId": "measurement_atlas_casing",
                "caseId": "case1",
                "from": "control",
                "to": "all_caps",
                "alignmentMode": "step_index_min_prefix",
                "sharedStepCount": 1,
                "sequenceLengthDeltas": [],
                "scalarDeltas": {
                    "replay.response.hidden.meanGeodesicDeviation": 0.4,
                },
                "sequenceDeltas": [
                    {
                        "mode": "replay",
                        "region": "response",
                        "space": "hidden",
                        "metric": "meanGeodesicDeviation",
                        "baselineValue": 0.3,
                        "variantValue": 0.7,
                        "delta": 0.4,
                    }
                    ,
                    {
                        "mode": "replay",
                        "region": "response",
                        "space": "hidden",
                        "metric": "meanPathLengthRatio",
                        "baselineValue": 1.2,
                        "variantValue": 99.2,
                        "delta": 98.0,
                    }
                ],
                "locusComparisons": [
                    {
                        "mode": "replay",
                        "region": "response",
                        "space": "hidden",
                        "metric": "peak",
                        "baselineLocus": "layer:1",
                        "variantLocus": "layer:3",
                        "changed": True,
                    }
                ],
                "liveGeneratedFirstDivergenceStep": 1,
                "replayResponseFirstDivergenceStep": 1,
                "firstGeneratedShiftAgreement": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "onset_events.jsonl").write_text(
        "".join(
            [
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "eventType": "grounded_label_onset",
                        "mode": "live",
                        "region": "generated",
                        "stepIndex": 2,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "studyId": "measurement_atlas_casing",
                        "caseId": "case1",
                        "variantId": "all_caps",
                        "baselineVariantId": "control",
                        "eventType": "first_divergence",
                        "mode": "replay",
                        "region": "response",
                        "stepIndex": 1,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    (bundle_dir / "REPORT.md").write_text("# report\n", encoding="utf-8")
    return bundle_dir


class TestAnalyzeCommandHelp:
    """Test that analyze commands have proper help text."""

    def test_analyze_help(self):
        """Test 'mc analyze --help' lists subcommands."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        stdout_lower = result.stdout.lower()
        for command in ["capture", "family", "compare", "report", "probe", "dimension-profile"]:
            assert command in stdout_lower

    def test_analyze_probe_help(self):
        """Test the canonical probe workflow is discoverable."""
        result = runner.invoke(app, ["analyze", "probe", "--help"])
        assert result.exit_code == 0
        stdout_lower = result.stdout.lower()
        for command in ["calibrate", "jailbreak", "redteam", "behavioral"]:
            assert command in stdout_lower


class TestAnalyzeWorkflowCommands:
    """Tests for workflow-first analyze commands."""

    def test_capture_help(self):
        result = runner.invoke(app, ["analyze", "capture", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--prompt" in result.stdout
        assert "--spaces" in result.stdout

    def test_family_help(self):
        result = runner.invoke(app, ["analyze", "family", "--help"])
        assert result.exit_code == 0
        assert "--manifest" in result.stdout
        assert "--spaces" in result.stdout

    def test_compare_help(self):
        result = runner.invoke(app, ["analyze", "compare", "--help"])
        assert result.exit_code == 0
        assert "--left-model" in result.stdout
        assert "--right-model" in result.stdout

    def test_report_help(self):
        result = runner.invoke(app, ["analyze", "report", "--help"])
        assert result.exit_code == 0
        assert "--bundle" in result.stdout

    def test_family_uses_observation_service(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        manifest_path = tmp_path / "family.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "name": "caps-study",
                    "variants": [
                        {
                            "case_id": "logic_1",
                            "variant_id": "control",
                            "text": "hello world",
                        },
                        {
                            "case_id": "logic_1",
                            "variant_id": "all_caps",
                            "text": "HELLO WORLD",
                            "comparison_to": "control",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        class _StubResult:
            def to_dict(self):
                return {
                    "workflow": "family",
                    "outputDir": str(tmp_path / "bundle"),
                    "summary": {
                        "workflow": "family",
                        "targetCount": 1,
                        "variantCount": 2,
                        "comparisonCount": 1,
                        "spaces": ["hidden", "embedding"],
                    },
                    "files": {},
                }

        class _StubService:
            def family(self, **kwargs):
                captured.update(kwargs)
                return _StubResult()

        monkeypatch.setattr(
            "modelcypher.cli.commands.analyze.workflows.get_observation_service",
            lambda: _StubService(),
        )

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "family",
                "--model",
                str(model_dir),
                "--manifest",
                str(manifest_path),
            ],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["workflow"] == "family"
        assert payload["summary"]["comparisonCount"] == 1
        assert Path(captured["target"].model) == model_dir.resolve()

    def test_report_reads_existing_bundle(self, tmp_path):
        bundle_dir = _write_report_bundle_fixture(tmp_path)

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "report",
                "--bundle",
                str(bundle_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["bundleDir"] == str(bundle_dir.resolve())
        assert payload["summary"]["workflow"] == "family"
        assert "observedSpaces" in payload["sections"]
        assert "topScalarShifts" in payload["sections"]

    def test_report_reads_existing_measurement_atlas_bundle(self, tmp_path):
        bundle_dir = _write_measurement_atlas_report_fixture(tmp_path)

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "report",
                "--bundle",
                str(bundle_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["bundleDir"] == str(bundle_dir.resolve())
        assert payload["summary"]["workflow"] == "measurement_atlas"
        assert "surfaces" in payload["sections"]
        assert "studySummaries" in payload["sections"]
        assert payload["sections"]["topSequenceShifts"][0]["metric"] == "meanGeodesicDeviation"
        assert all(
            row["metric"] == "meanGeodesicDeviation"
            for row in payload["sections"]["topSequenceShifts"]
        )
        example = payload["sections"]["exampleComparisons"][0]
        assert "promptText" not in example
        assert "generatedText" not in example
        assert example["promptPreview"].endswith("...")
        assert example["generatedPreview"].endswith("...")
        assert example["promptCharCount"] > len(example["promptPreview"])
        assert example["generatedCharCount"] > len(example["generatedPreview"])

    def test_report_text_for_measurement_atlas_uses_compact_preview_lines(self, tmp_path):
        bundle_dir = _write_measurement_atlas_report_fixture(tmp_path)

        result = runner.invoke(
            app,
            [
                "--output",
                "text",
                "analyze",
                "report",
                "--bundle",
                str(bundle_dir),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "## Largest Geodesic Shifts" in result.stdout
        assert "## Largest Sequence Shifts" not in result.stdout
        assert "meanPathLengthRatio" not in result.stdout
        assert "prompt='ALPHA" not in result.stdout
        assert "prompt=`ALPHA BETA" in result.stdout
        assert "\\n\\n" not in result.stdout
        assert "chars)" in result.stdout

    def test_report_missing_bundle_directory(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "report",
                "--bundle",
                str(tmp_path / "missing-bundle"),
            ],
        )

        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-1098"
        assert payload["error"]["title"] == "Missing report bundle"

    def test_report_missing_required_files(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "report",
                "--bundle",
                str(bundle_dir),
            ],
        )

        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-1099"
        assert payload["error"]["title"] == "Invalid report bundle"

    def test_report_missing_required_atlas_files(self, tmp_path):
        bundle_dir = _write_measurement_atlas_report_fixture(tmp_path)
        (bundle_dir / "onset_events.jsonl").unlink()

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "report",
                "--bundle",
                str(bundle_dir),
            ],
        )

        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-1099"
        assert payload["error"]["title"] == "Invalid report bundle"


class TestSafetyAdapterProbe:
    """Tests for 'mc analyze adapter-probe' command."""

    def test_adapter_probe_help(self):
        """Test help text for adapter-probe."""
        result = runner.invoke(app, ["analyze", "adapter-probe", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.stdout

    def test_adapter_probe_requires_adapter(self):
        """Test that --adapter is required."""
        result = runner.invoke(app, ["analyze", "adapter-probe"])
        assert result.exit_code != 0


class TestSafetyBehavioralSignature:
    """Tests for 'mc analyze behavioral-signature' command."""

    def test_behavioral_signature_help(self):
        """Test help text for behavioral-signature."""
        result = runner.invoke(app, ["analyze", "behavioral-signature", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_behavioral_signature_requires_model(self):
        """Test that --model is required."""
        result = runner.invoke(app, ["analyze", "behavioral-signature"])
        assert result.exit_code != 0


class TestSafetyDimensionProfile:
    """Tests for 'mc analyze dimension-profile' command."""

    def test_dimension_profile_help(self):
        """Test help text for dimension-profile."""
        result = runner.invoke(app, ["analyze", "dimension-profile", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_dimension_profile_requires_model(self):
        """Test that model is required."""
        result = runner.invoke(app, ["analyze", "dimension-profile"])
        assert result.exit_code != 0


class TestVerificationDepthProfile:
    """Tests for 'mc analyze verification-depth-profile' command."""

    def test_verification_depth_profile_help(self):
        """Test help text for verification-depth-profile."""
        result = runner.invoke(app, ["analyze", "verification-depth-profile", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--levels" in result.stdout
        assert "--mode" in result.stdout

    def test_verification_depth_profile_requires_model(self):
        """Test missing model path returns structured error."""
        result = runner.invoke(
            app,
            ["--output", "json", "analyze", "verification-depth-profile"],
        )
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-3072"

    def test_verification_depth_profile_invalid_levels(self, tmp_path):
        """Test invalid --levels returns structured error."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "verification-depth-profile",
                "--model",
                str(model_path),
                "--levels",
                "0,one,2",
            ],
        )
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-3073"

    def test_verification_depth_profile_missing_depth_metadata(self, tmp_path):
        """Test missing verification-depth metadata returns structured error."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        probes_path = tmp_path / "probes.json"
        probes_path.write_text(
            json.dumps(
                {
                    "domain": "logical",
                    "probe_count": 1,
                    "probes": [
                        {
                            "id": "semantic_prime:TEST",
                            "name": "test",
                            "description": "test probe",
                            "support_texts": ["test text"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "--output",
                "json",
                "analyze",
                "verification-depth-profile",
                "--model",
                str(model_path),
                "--probes",
                str(probes_path),
            ],
        )
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "MC-3075"


class TestSafetyEntropyCommands:
    """Tests for entropy-related analyze commands."""

    def test_entropy_trajectory_help(self):
        """Test help text for entropy-trajectory."""
        result = runner.invoke(app, ["analyze", "entropy-trajectory", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_spectral_trajectory_help(self):
        """Test help text for spectral-trajectory."""
        result = runner.invoke(app, ["analyze", "spectral-trajectory", "--help"])
        assert result.exit_code == 0


class TestSafetyReasoningFlow:
    """Tests for 'mc analyze reasoning-flow' command."""

    def test_reasoning_flow_help(self):
        """Test help text for reasoning-flow."""
        result = runner.invoke(app, ["analyze", "reasoning-flow", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout


class TestBenchmarkCommands:
    """Tests for benchmark commands under analyze."""

    def test_benchmark_help(self):
        """Test help text for benchmark command."""
        result = runner.invoke(app, ["analyze", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.stdout

    def test_benchmark_passes_adapter_to_model_loader(self, monkeypatch):
        """Benchmark command should load base model with optional adapter."""
        from modelcypher.adapters import model_loader as model_loader_module
        from modelcypher.cli import composition

        captured: dict[str, object] = {}

        class _StubModelLoader:
            def __init__(self, _backend):
                pass

            def load_model(self, model_path: str, adapter_path: str | None = None):
                captured["model_path"] = model_path
                captured["adapter_path"] = adapter_path
                return object(), object()

        class _StubBackend:
            @staticmethod
            def generate(_m, _t, _prompt, max_tokens, verbose=False):
                _ = verbose
                return f"generated_{max_tokens}"

        class _StubSuiteResult:
            @staticmethod
            def to_dict():
                return {
                    "suite": "quick",
                    "overall_accuracy": 1.0,
                    "benchmarks": [],
                }

        class _StubBenchmarkService:
            def run_suite(
                self,
                *,
                model,
                tokenizer,
                suite_name: str,
                generate_fn,
                limit_per_benchmark: int | None = None,
                max_failures: int | None = None,
                max_tokens: int = 512,
            ):
                _ = (model, tokenizer, generate_fn, max_failures, max_tokens)
                captured["suite_name"] = suite_name
                captured["limit_per_benchmark"] = limit_per_benchmark
                return _StubSuiteResult()

        monkeypatch.setattr(model_loader_module, "ModelLoader", _StubModelLoader)
        monkeypatch.setattr(composition, "get_backend", lambda: _StubBackend())
        monkeypatch.setattr(
            composition, "get_benchmark_service", lambda: _StubBenchmarkService(),
        )

        result = runner.invoke(
            app,
            [
                "--output", "json",
                "analyze", "benchmark", "/tmp/base-model",
                "--suite", "quick",
                "--limit", "7",
                "--adapter", "/tmp/adapter-path",
            ],
        )

        assert result.exit_code == 0
        json_start = result.stdout.find("{")
        assert json_start >= 0
        payload = json.loads(result.stdout[json_start:])
        assert captured["model_path"] == "/tmp/base-model"
        assert captured["adapter_path"] == "/tmp/adapter-path"
        assert captured["suite_name"] == "quick"
        assert captured["limit_per_benchmark"] == 7
        assert payload["modelPath"] == "/tmp/base-model"
        assert payload["adapterPath"] == "/tmp/adapter-path"


class TestLoRADiagnosticCommands:
    """Tests for LoRA diagnostic commands."""

    def test_lora_svd_help(self):
        """Test help text for lora-svd."""
        result = runner.invoke(app, ["analyze", "lora-svd", "--help"])
        assert result.exit_code == 0
        assert "--base" in result.stdout or "-b" in result.stdout
        assert "--baseline-artifact" in result.stdout

    def test_lora_svd_requires_adapter(self):
        """Test that adapter path is required."""
        result = runner.invoke(app, ["analyze", "lora-svd"])
        assert result.exit_code != 0


class TestCRMCommands:
    """Tests for Concept Response Matrix commands."""

    def test_crm_build_help(self):
        """Test help text for crm-build."""
        result = runner.invoke(app, ["analyze", "crm-build", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout

    def test_crm_compare_help(self):
        """Test help text for crm-compare."""
        result = runner.invoke(app, ["analyze", "crm-compare", "--help"])
        assert result.exit_code == 0


class TestCurriculumProfile:
    """Tests for curriculum-profile command."""

    def test_curriculum_profile_help(self):
        """Test help text for curriculum-profile."""
        result = runner.invoke(app, ["analyze", "curriculum-profile", "--help"])
        assert result.exit_code == 0
        # Takes MODEL as positional arg, not --model
        assert "model" in result.stdout.lower() or "problems" in result.stdout.lower()


class TestJailbreakTest:
    """Tests for jailbreak-test command."""

    def test_calibrate_safety_help(self):
        """Test help text for calibrate-safety."""
        result = runner.invoke(app, ["analyze", "calibrate-safety", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--output-file" in result.stdout

    def test_jailbreak_test_help(self):
        """Test help text for jailbreak-test."""
        result = runner.invoke(app, ["analyze", "jailbreak-test", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--calibration" in result.stdout
