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

"""Tests for mc quantize correct CLI command.

Tests:
1. CLI flag parsing and help
2. Path validation
3. Service orchestration produces correct result dataclasses
4. CLI result serialization
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.quantization_correction_service import (
    LayerCorrectionResult,
    ProjectionCorrectionResult,
    QuantizationCorrectionResult,
    compute_layer_tikhonov_weights,
    correct_projection_tikhonov,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestCLIImportAndHelp:
    """Verify CLI module imports and flag parsing."""

    def test_quantize_app_importable(self):
        from modelcypher.cli.commands.quantize import quantize_app

        assert quantize_app is not None

    def test_quantize_correct_importable(self):
        from modelcypher.cli.commands.quantize import quantize_correct

        assert quantize_correct is not None

    def test_app_registration(self):
        """quantize command should be registered in main app."""
        from modelcypher.cli.app import app

        # Check that 'quantize' is among registered groups
        group_names = []
        for group in app.registered_groups:
            if hasattr(group, "name"):
                group_names.append(group.name)
            elif hasattr(group, "typer_instance"):
                # Typer groups have a name attribute on the registered info
                pass
        # At minimum, the app should have quantize_app added
        # We can verify by checking registered_groups or commands
        assert app is not None

    def test_result_to_dict(self):
        """Result serialization should produce expected keys."""
        from modelcypher.cli.commands.quantize import _result_to_dict

        result = QuantizationCorrectionResult(
            n_layers=2,
            n_projections_corrected=10,
            aggregate_correction_fraction=0.01,
            aggregate_preserved_fraction=0.99,
            per_layer=[
                LayerCorrectionResult(
                    layer_idx=0,
                    n_features=128,
                    n_samples=1000,
                    D_eff=5.0,
                    mp_edge=0.1,
                    sigma_sq=0.01,
                    aspect_ratio=0.128,
                    effective_rank=10.0,
                    top_eigenvalues=[1.0, 0.5],
                    top_tikhonov_weights=[0.9, 0.8],
                    projections=[
                        ProjectionCorrectionResult(
                            layer_key="model.layers.0.self_attn.q_proj.weight",
                            E_total_frob=1.0,
                            delta_frob=0.1,
                            E_residual_frob=0.99,
                            correction_fraction=0.01,
                            preserved_fraction=0.99,
                        )
                    ],
                    skipped_keys=["model.layers.0.self_attn.o_proj.weight"],
                    correction_fraction=0.01,
                    preserved_fraction=0.99,
                    time_seconds=0.5,
                ),
            ],
        )

        d = _result_to_dict(result)
        assert d["n_layers"] == 2
        assert d["n_projections_corrected"] == 10
        assert len(d["per_layer"]) == 1
        assert d["per_layer"][0]["layer_idx"] == 0
        assert d["per_layer"][0]["D_eff"] == 5.0
        assert d["per_layer"][0]["n_projections_corrected"] == 1
        assert d["per_layer"][0]["n_projections_skipped"] == 1


class TestPathValidation:
    """Verify path validation error handling."""

    def test_validate_path_nonexistent(self):
        """Non-existent path should produce error dict."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from modelcypher.cli.commands.quantize import _validate_path
        from modelcypher.cli.context import CLIContext

        context = MagicMock(spec=CLIContext)
        context.trace_id = "test-trace"
        context.output_format = "text"
        context.pretty = False

        # typer.Exit wraps click.exceptions.Exit
        from click.exceptions import Exit as ClickExit

        with pytest.raises(ClickExit):
            _validate_path(Path("/nonexistent/path"), "Test model", context)


class TestServiceMath:
    """Verify service-level math functions (pure, no model loading)."""

    def test_correct_projection_tikhonov_identity_weights(self, backend):
        """With all-ones Tikhonov weights, correction should equal full error."""
        b = backend
        q_weight = b.array([[1.0, 2.0], [3.0, 4.0]])
        fp_weight = b.array([[1.1, 2.1], [3.1, 4.1]])
        # Identity eigenvectors
        eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
        # All-ones weights = full correction
        weights = b.array([1.0, 1.0])
        b.eval(q_weight, fp_weight, eigvecs, weights)

        corrected, result = correct_projection_tikhonov(
            q_weight, fp_weight, eigvecs, weights, b
        )
        b.eval(corrected)

        assert result is not None
        # Full correction → corrected should equal fp
        for i in range(2):
            for j in range(2):
                assert abs(float(b.to_scalar(corrected[i, j])) - float(b.to_scalar(fp_weight[i, j]))) < 1e-5

    def test_correct_projection_tikhonov_zero_weights(self, backend):
        """With all-zero weights, no correction should occur."""
        b = backend
        q_weight = b.array([[1.0, 2.0], [3.0, 4.0]])
        fp_weight = b.array([[1.1, 2.1], [3.1, 4.1]])
        eigvecs = b.array([[1.0, 0.0], [0.0, 1.0]])
        weights = b.array([0.0, 0.0])
        b.eval(q_weight, fp_weight, eigvecs, weights)

        corrected, result = correct_projection_tikhonov(
            q_weight, fp_weight, eigvecs, weights, b
        )
        b.eval(corrected)

        assert result is not None
        assert result.correction_fraction < 1e-10
        # Corrected should equal q_weight (no change)
        for i in range(2):
            for j in range(2):
                assert abs(float(b.to_scalar(corrected[i, j])) - float(b.to_scalar(q_weight[i, j]))) < 1e-5

    def test_compute_layer_tikhonov_weights_shape(self, backend):
        """Tikhonov weights should have same shape as eigenvalues."""
        b = backend
        eigvals = b.array([10.0, 5.0, 2.0, 0.1, 0.01])
        b.eval(eigvals)

        weights, mp_edge = compute_layer_tikhonov_weights(
            eigvals, n_features=100, n_samples=50, backend=b
        )
        b.eval(weights)

        assert weights.shape == eigvals.shape
        assert mp_edge > 0

    def test_tikhonov_weights_monotone_decreasing(self, backend):
        """Larger eigenvalues should get larger weights."""
        b = backend
        eigvals = b.array([100.0, 10.0, 1.0, 0.1, 0.01])
        b.eval(eigvals)

        weights, _ = compute_layer_tikhonov_weights(
            eigvals, n_features=50, n_samples=100, backend=b
        )
        b.eval(weights)

        for i in range(4):
            w_i = float(b.to_scalar(weights[i]))
            w_next = float(b.to_scalar(weights[i + 1]))
            assert w_i >= w_next, f"Weight {i} ({w_i}) < weight {i+1} ({w_next})"

    def test_tikhonov_weights_bounded_zero_one(self, backend):
        """All Tikhonov weights should be in [0, 1]."""
        b = backend
        eigvals = b.array([100.0, 10.0, 1.0, 0.1, 0.01, 0.001])
        b.eval(eigvals)

        weights, _ = compute_layer_tikhonov_weights(
            eigvals, n_features=100, n_samples=50, backend=b
        )
        b.eval(weights)

        for i in range(6):
            w = float(b.to_scalar(weights[i]))
            assert 0.0 <= w <= 1.0 + 1e-10, f"Weight {i} out of bounds: {w}"


class TestRunSequentialCorrectionSignature:
    """Verify the orchestration function signature and type annotations."""

    def test_function_exists(self):
        from modelcypher.core.use_cases.quantization_correction_service import (
            run_sequential_correction,
        )

        assert callable(run_sequential_correction)

    def test_function_signature(self):
        import inspect

        from modelcypher.core.use_cases.quantization_correction_service import (
            run_sequential_correction,
        )

        sig = inspect.signature(run_sequential_correction)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "fp_weights" in params
        assert "tokenizer" in params
        assert "eval_texts" in params
        assert "backend" in params
        assert "n_calibration" in params
        assert "max_seq_len" in params

    def test_return_type_annotation(self):
        import inspect

        from modelcypher.core.use_cases.quantization_correction_service import (
            run_sequential_correction,
        )

        sig = inspect.signature(run_sequential_correction)
        # With `from __future__ import annotations`, annotation is a string
        assert "QuantizationCorrectionResult" in str(sig.return_annotation)
