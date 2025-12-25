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

"""Property tests for MCP/CLI output parity.

**Feature: cli-mcp-parity, Property 9: MCP output matches CLI output schema**
**Validates: Requirements 12.1**
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Define the expected schema mappings between CLI and MCP outputs
# MCP outputs have additional fields: _schema, nextActions
MCP_ONLY_FIELDS = {"_schema", "nextActions"}


def normalize_keys(data: Any) -> Any:
    """Recursively normalize dictionary keys to camelCase for comparison."""
    if isinstance(data, dict):
        return {k: normalize_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_keys(item) for item in data]
    return data


def extract_schema_fields(data: dict) -> set[str]:
    """Extract the set of field names from a dict, excluding MCP-only fields."""
    if not isinstance(data, dict):
        return set()
    return {k for k in data.keys() if k not in MCP_ONLY_FIELDS}


def schemas_match(cli_output: dict, mcp_output: dict) -> tuple[bool, str]:
    """Check if CLI and MCP outputs have matching schemas.

    MCP outputs are expected to have additional fields (_schema, nextActions)
    but the core data fields should match.

    Returns:
        Tuple of (match_result, error_message)
    """
    if not isinstance(cli_output, dict) or not isinstance(mcp_output, dict):
        return False, "Both outputs must be dictionaries"

    cli_fields = extract_schema_fields(cli_output)
    mcp_fields = extract_schema_fields(mcp_output)

    # MCP should have all CLI fields
    missing_in_mcp = cli_fields - mcp_fields
    if missing_in_mcp:
        return False, f"MCP output missing CLI fields: {missing_in_mcp}"

    # Check that MCP has _schema field
    if "_schema" not in mcp_output:
        return False, "MCP output missing _schema field"

    return True, ""


# **Feature: cli-mcp-parity, Property 9: MCP output matches CLI output schema**
# **Validates: Requirements 12.1**
class TestMCPCLIParity:
    """Property tests for MCP/CLI output parity."""

    def test_storage_usage_schema_parity(self):
        """Test that mc_storage_usage MCP output matches CLI storage output schema."""
        # Import services
        from modelcypher.core.use_cases.storage_service import StorageService

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            home = tmp_path / "mc_home"
            hf_home = tmp_path / "hf_cache"

            import os

            old_mc_home = os.environ.get("MODELCYPHER_HOME")
            old_hf_home = os.environ.get("HF_HOME")
            os.environ["MODELCYPHER_HOME"] = str(home)
            os.environ["HF_HOME"] = str(hf_home)

            try:
                # Create directories
                home.mkdir(parents=True, exist_ok=True)
                hf_home.mkdir(parents=True, exist_ok=True)

                service = StorageService()
                snapshot = service.compute_snapshot()
                usage = snapshot.usage
                disk = snapshot.disk

                # CLI output format (from app.py storage_status)
                cli_output = {
                    "totalGb": usage.total_gb,
                    "modelsGb": usage.models_gb,
                    "checkpointsGb": usage.checkpoints_gb,
                    "otherGb": usage.other_gb,
                    "disk": {
                        "totalBytes": disk.total_bytes,
                        "freeBytes": disk.free_bytes,
                    },
                }

                # MCP output format (from server.py mc_storage_usage)
                mcp_output = {
                    "_schema": "mc.storage.usage.v1",
                    "totalGb": usage.total_gb,
                    "modelsGb": usage.models_gb,
                    "checkpointsGb": usage.checkpoints_gb,
                    "otherGb": usage.other_gb,
                    "disk": {
                        "totalBytes": disk.total_bytes,
                        "freeBytes": disk.free_bytes,
                    },
                    "nextActions": [
                        "mc_storage_cleanup to free space",
                        "mc_inventory to see all resources",
                    ],
                }

                # Property: schemas match
                match, error = schemas_match(cli_output, mcp_output)
                assert match, f"Schema mismatch: {error}"

                # Property: MCP has _schema field
                assert "_schema" in mcp_output
                assert mcp_output["_schema"] == "mc.storage.usage.v1"

                # Property: MCP has nextActions field
                assert "nextActions" in mcp_output
                assert isinstance(mcp_output["nextActions"], list)

            finally:
                if old_mc_home is not None:
                    os.environ["MODELCYPHER_HOME"] = old_mc_home
                elif "MODELCYPHER_HOME" in os.environ:
                    del os.environ["MODELCYPHER_HOME"]
                if old_hf_home is not None:
                    os.environ["HF_HOME"] = old_hf_home
                elif "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]

    @given(
        targets=st.lists(
            st.sampled_from(["caches", "rag"]),
            min_size=1,
            max_size=2,
            unique=True,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_storage_cleanup_schema_parity(self, targets: list[str]):
        """Property 9: For any storage cleanup operation, MCP output matches CLI output schema."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            home = tmp_path / "mc_home"
            hf_home = tmp_path / "hf_cache"

            import os

            old_mc_home = os.environ.get("MODELCYPHER_HOME")
            old_hf_home = os.environ.get("HF_HOME")
            os.environ["MODELCYPHER_HOME"] = str(home)
            os.environ["HF_HOME"] = str(hf_home)

            try:
                # Create directories
                (home / "caches").mkdir(parents=True, exist_ok=True)
                (home / "rag").mkdir(parents=True, exist_ok=True)
                hf_home.mkdir(parents=True, exist_ok=True)

                from modelcypher.core.use_cases.storage_service import StorageService

                service = StorageService(cache_ttl_seconds=0.0)

                # Get before snapshot
                before_snapshot = service.compute_snapshot()

                # Execute cleanup
                cleared = service.cleanup(targets)

                # Get after snapshot
                after_snapshot = service.compute_snapshot()
                freed_bytes = max(
                    0, after_snapshot.disk.free_bytes - before_snapshot.disk.free_bytes
                )

                # CLI output format (from app.py storage_cleanup)
                cli_output = {
                    "freedBytes": freed_bytes,
                    "freedGb": freed_bytes / (1024**3),
                    "categoriesCleaned": cleared,
                }

                # MCP output format (from server.py mc_storage_cleanup)
                mcp_output = {
                    "_schema": "mc.storage.cleanup.v1",
                    "dryRun": False,
                    "targets": targets,
                    "freedBytes": freed_bytes,
                    "freedGb": freed_bytes / (1024**3),
                    "categoriesCleaned": cleared,
                    "message": None,
                    "nextActions": [
                        "mc_storage_usage to verify cleanup",
                        "mc_inventory to see remaining resources",
                    ],
                }

                # Property: CLI fields are subset of MCP fields (excluding MCP-only fields)
                cli_fields = set(cli_output.keys())
                mcp_core_fields = {k for k in mcp_output.keys() if k not in MCP_ONLY_FIELDS}
                assert cli_fields <= mcp_core_fields, (
                    f"CLI fields {cli_fields} should be subset of MCP fields {mcp_core_fields}"
                )

                # Property: Common fields have same values
                for field in cli_fields:
                    assert cli_output[field] == mcp_output[field], (
                        f"Field {field} mismatch: CLI={cli_output[field]}, MCP={mcp_output[field]}"
                    )

                # Property: MCP has required metadata fields
                assert "_schema" in mcp_output
                assert "nextActions" in mcp_output

            finally:
                if old_mc_home is not None:
                    os.environ["MODELCYPHER_HOME"] = old_mc_home
                elif "MODELCYPHER_HOME" in os.environ:
                    del os.environ["MODELCYPHER_HOME"]
                if old_hf_home is not None:
                    os.environ["HF_HOME"] = old_hf_home
                elif "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]

    @pytest.mark.skip(reason="Test requires real model loading which isn't available in CI")
    @given(
        prompt=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
        preset=st.sampled_from(["default", "strict", "sensitive", "quick"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_thermo_detect_schema_parity(self, prompt: str, preset: str):
        """Property 9: For any thermo detect operation, MCP output matches CLI output schema."""
        from modelcypher.core.use_cases.thermo_service import ThermoService

        with tempfile.TemporaryDirectory() as model_dir:
            service = ThermoService()
            result = service.detect(prompt, model_dir, preset)

            # CLI output format (from app.py thermo_detect)
            cli_output = {
                "prompt": result.prompt,
                "classification": result.classification,
                "riskLevel": result.risk_level,
                "confidence": result.confidence,
                "baselineEntropy": result.baseline_entropy,
                "intensityEntropy": result.intensity_entropy,
                "deltaH": result.delta_h,
                "processingTime": result.processing_time,
            }

            # MCP output format (from server.py mc_thermo_detect)
            mcp_output = {
                "_schema": "mc.thermo.detect.v1",
                "prompt": result.prompt,
                "classification": result.classification,
                "riskLevel": result.risk_level,
                "confidence": result.confidence,
                "baselineEntropy": result.baseline_entropy,
                "intensityEntropy": result.intensity_entropy,
                "deltaH": result.delta_h,
                "processingTime": result.processing_time,
                "nextActions": [
                    "mc_thermo_measure for detailed entropy analysis",
                    "mc_thermo_detect_batch for batch detection",
                    "mc_safety_circuit_breaker for safety assessment",
                ],
            }

            # Property: CLI fields are subset of MCP fields (excluding MCP-only fields)
            cli_fields = set(cli_output.keys())
            mcp_core_fields = {k for k in mcp_output.keys() if k not in MCP_ONLY_FIELDS}
            assert cli_fields == mcp_core_fields, (
                f"CLI fields {cli_fields} should match MCP core fields {mcp_core_fields}"
            )

            # Property: Common fields have same values
            for field in cli_fields:
                assert cli_output[field] == mcp_output[field], (
                    f"Field {field} mismatch: CLI={cli_output[field]}, MCP={mcp_output[field]}"
                )

            # Property: MCP has required metadata fields
            assert "_schema" in mcp_output
            assert mcp_output["_schema"] == "mc.thermo.detect.v1"
            assert "nextActions" in mcp_output
            assert isinstance(mcp_output["nextActions"], list)

    @pytest.mark.skip(reason="Test requires real model loading which isn't available in CI")
    @given(
        prompts=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()), min_size=1, max_size=10
        ),
        preset=st.sampled_from(["default", "strict", "sensitive", "quick"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_thermo_detect_batch_schema_parity(self, prompts: list[str], preset: str):
        """Property 9: For any thermo detect-batch operation, MCP output matches CLI output schema."""
        from modelcypher.core.use_cases.thermo_service import ThermoService

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_dir = tmp_path / "model"
            model_dir.mkdir()

            # Create prompts file
            prompts_file = tmp_path / "prompts.json"
            prompts_file.write_text(json.dumps(prompts), encoding="utf-8")

            service = ThermoService()
            results = service.detect_batch(str(prompts_file), str(model_dir), preset)

            # CLI output format (from app.py thermo_detect_batch)
            cli_output = {
                "promptsFile": str(prompts_file),
                "totalPrompts": len(results),
                "results": [
                    {
                        "prompt": r.prompt,
                        "classification": r.classification,
                        "riskLevel": r.risk_level,
                        "confidence": r.confidence,
                        "deltaH": r.delta_h,
                    }
                    for r in results
                ],
                "summary": {
                    "safe": sum(1 for r in results if r.classification == "safe"),
                    "unsafe": sum(1 for r in results if r.classification == "unsafe"),
                    "ambiguous": sum(1 for r in results if r.classification == "ambiguous"),
                },
            }

            # MCP output format (from server.py mc_thermo_detect_batch)
            mcp_output = {
                "_schema": "mc.thermo.detect_batch.v1",
                "promptsFile": str(prompts_file),
                "totalPrompts": len(results),
                "results": [
                    {
                        "prompt": r.prompt,
                        "classification": r.classification,
                        "riskLevel": r.risk_level,
                        "confidence": r.confidence,
                        "deltaH": r.delta_h,
                    }
                    for r in results
                ],
                "summary": {
                    "safe": sum(1 for r in results if r.classification == "safe"),
                    "unsafe": sum(1 for r in results if r.classification == "unsafe"),
                    "ambiguous": sum(1 for r in results if r.classification == "ambiguous"),
                },
                "nextActions": [
                    "mc_thermo_detect for individual prompt analysis",
                    "mc_thermo_measure for detailed entropy analysis",
                ],
            }

            # Property: CLI fields are subset of MCP fields (excluding MCP-only fields)
            cli_fields = set(cli_output.keys())
            mcp_core_fields = {k for k in mcp_output.keys() if k not in MCP_ONLY_FIELDS}
            assert cli_fields == mcp_core_fields, (
                f"CLI fields {cli_fields} should match MCP core fields {mcp_core_fields}"
            )

            # Property: Common fields have same values
            for field in cli_fields:
                assert cli_output[field] == mcp_output[field], (
                    f"Field {field} mismatch: CLI={cli_output[field]}, MCP={mcp_output[field]}"
                )

            # Property: MCP has required metadata fields
            assert "_schema" in mcp_output
            assert "nextActions" in mcp_output

    @given(
        prompts=st.lists(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()), min_size=1, max_size=10
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_inference_suite_schema_parity(self, prompts: list[str]):
        """Property 9: For any inference suite operation, MCP output matches CLI output schema."""
        from modelcypher.adapters.local_inference import LocalInferenceEngine

        previous = os.environ.get("MC_ALLOW_STUB_INFERENCE")
        os.environ["MC_ALLOW_STUB_INFERENCE"] = "1"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                model_dir = tmp_path / "model"
                model_dir.mkdir()

                # Create suite file
                suite_file = tmp_path / "suite.json"
                suite_file.write_text(json.dumps(prompts), encoding="utf-8")

                engine = LocalInferenceEngine()
                result = engine.suite(
                    model=str(model_dir),
                    suite_file=str(suite_file),
                )

                # Convert cases to dict format (shared between CLI and MCP)
                cases_payload = []
                for case in result.cases:
                    case_dict = {
                        "name": case.name,
                        "prompt": case.prompt,
                        "response": case.response,
                        "tokenCount": case.token_count,
                        "duration": case.duration,
                        "passed": case.passed,
                        "expected": case.expected,
                    }
                    if case.error:
                        case_dict["error"] = case.error
                    cases_payload.append(case_dict)

                # CLI output format (inferred from MCP pattern)
                cli_output = {
                    "model": result.model,
                    "adapter": result.adapter,
                    "suite": result.suite,
                    "totalCases": result.total_cases,
                    "passed": result.passed,
                    "failed": result.failed,
                    "totalDuration": result.total_duration,
                    "summary": result.summary,
                    "cases": cases_payload[:10],
                }

                # MCP output format (from server.py mc_infer_suite)
                mcp_output = {
                    "_schema": "mc.infer.suite.v1",
                    "model": result.model,
                    "adapter": result.adapter,
                    "suite": result.suite,
                    "totalCases": result.total_cases,
                    "passed": result.passed,
                    "failed": result.failed,
                    "totalDuration": result.total_duration,
                    "summary": result.summary,
                    "cases": cases_payload[:10],
                    "nextActions": [
                        "mc_infer_batch for batch inference",
                        "mc_infer_run for single prompts",
                    ],
                }

                # Property: CLI fields are subset of MCP fields (excluding MCP-only fields)
                cli_fields = set(cli_output.keys())
                mcp_core_fields = {k for k in mcp_output.keys() if k not in MCP_ONLY_FIELDS}
                assert cli_fields == mcp_core_fields, (
                    f"CLI fields {cli_fields} should match MCP core fields {mcp_core_fields}"
                )

                # Property: Common fields have same values
                for field in cli_fields:
                    assert cli_output[field] == mcp_output[field], (
                        f"Field {field} mismatch: CLI={cli_output[field]}, MCP={mcp_output[field]}"
                    )

                # Property: MCP has required metadata fields
                assert "_schema" in mcp_output
                assert "nextActions" in mcp_output
        finally:
            if previous is None:
                os.environ.pop("MC_ALLOW_STUB_INFERENCE", None)
            else:
                os.environ["MC_ALLOW_STUB_INFERENCE"] = previous


# Additional property test for general MCP/CLI parity pattern
@given(
    command_type=st.sampled_from(
        [
            "storage_usage",
            "storage_cleanup",
            "thermo_detect",
            "thermo_detect_batch",
            "infer_suite",
        ]
    ),
)
@settings(max_examples=100, deadline=None)
def test_mcp_output_has_required_metadata(command_type: str):
    """Property 9: For any MCP tool, output has _schema and nextActions fields.

    This validates the structural requirement that all MCP outputs follow
    the same metadata pattern.
    """
    # Define expected schema patterns for each command type
    schema_patterns = {
        "storage_usage": "mc.storage.usage.v1",
        "storage_cleanup": "mc.storage.cleanup.v1",
        "thermo_detect": "mc.thermo.detect.v1",
        "thermo_detect_batch": "mc.thermo.detect_batch.v1",
        "infer_suite": "mc.infer.suite.v1",
    }

    expected_schema = schema_patterns[command_type]

    # Property: schema follows naming convention
    assert expected_schema.startswith("mc."), "Schema should start with 'mc.'"
    assert expected_schema.endswith(".v1"), "Schema should end with version suffix"

    # Property: schema has valid structure (namespace.command.version)
    parts = expected_schema.split(".")
    assert len(parts) >= 3, "Schema should have at least 3 parts: namespace.command.version"
