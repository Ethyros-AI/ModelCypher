"""Property tests for AdapterService merge functionality."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from safetensors.numpy import save_file

from modelcypher.core.use_cases.adapter_service import AdapterMergeResult, AdapterService


def _create_adapter(
    path: Path,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
    weight_keys: list[str] | None = None,
    weight_shape: tuple[int, int] = (64, 8),
) -> None:
    """Create a mock adapter directory with safetensors and config."""
    path.mkdir(parents=True, exist_ok=True)

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    if weight_keys is None:
        weight_keys = ["layer.0.lora_A", "layer.0.lora_B"]

    # Create adapter config
    config = {
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    # Create adapter weights
    rng = np.random.default_rng(42)
    weights = {}
    for key in weight_keys:
        weights[key] = rng.standard_normal(weight_shape).astype(np.float32)

    save_file(weights, path / "adapter_model.safetensors")


# **Feature: cli-mcp-parity, Property 7: Adapter merge produces valid output**
# **Validates: Requirements 13.1**
@given(
    num_adapters=st.integers(min_value=2, max_value=5),
    strategy=st.sampled_from(["ties", "dare-ties"]),
    ties_topk=st.floats(min_value=0.1, max_value=1.0),
    recommend_ensemble=st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_adapter_merge_produces_valid_output(
    num_adapters: int,
    strategy: str,
    ties_topk: float,
    recommend_ensemble: bool,
):
    """Property 7: For any set of 2+ adapter paths, merge() creates an output
    directory containing adapter_model.safetensors and adapter_config.json.
    """
    service = AdapterService()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create adapter directories with common weight keys
        adapter_paths = []
        weight_keys = ["layer.0.lora_A", "layer.0.lora_B", "layer.1.lora_A", "layer.1.lora_B"]

        for i in range(num_adapters):
            adapter_path = tmp_path / f"adapter_{i}"
            _create_adapter(
                adapter_path,
                rank=8,
                alpha=16.0,
                weight_keys=weight_keys,
            )
            adapter_paths.append(str(adapter_path))

        output_dir = tmp_path / "merged_adapter"

        # Execute merge
        result = service.merge(
            adapter_paths=adapter_paths,
            output_dir=str(output_dir),
            strategy=strategy,
            ties_topk=ties_topk,
            recommend_ensemble=recommend_ensemble,
        )

        # Property: result is correct type
        assert isinstance(result, AdapterMergeResult)

        # Property: output directory exists
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Property: adapter_model.safetensors exists
        safetensors_path = output_dir / "adapter_model.safetensors"
        assert safetensors_path.exists(), "adapter_model.safetensors must exist"

        # Property: adapter_config.json exists
        config_path = output_dir / "adapter_config.json"
        assert config_path.exists(), "adapter_config.json must exist"

        # Property: config is valid JSON
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert isinstance(config, dict)

        # Property: result fields are valid
        # Use resolve() to handle macOS symlink /var -> /private/var
        assert Path(result.output_path).resolve() == output_dir.resolve()
        assert result.strategy == strategy.lower()
        assert result.merged_modules > 0

        # Property: ensemble recommendation is present if requested
        if recommend_ensemble:
            assert result.ensemble_recommendation is not None
            assert isinstance(result.ensemble_recommendation, dict)
        else:
            assert result.ensemble_recommendation is None


@given(
    strategy=st.sampled_from(["ties", "dare-ties"]),
)
@settings(max_examples=50, deadline=None)
def test_adapter_merge_with_minimum_adapters(strategy: str):
    """Test merge works with exactly 2 adapters (minimum required)."""
    service = AdapterService()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create exactly 2 adapters
        weight_keys = ["layer.0.lora_A", "layer.0.lora_B"]
        adapter_paths = []

        for i in range(2):
            adapter_path = tmp_path / f"adapter_{i}"
            _create_adapter(adapter_path, weight_keys=weight_keys)
            adapter_paths.append(str(adapter_path))

        output_dir = tmp_path / "merged"

        result = service.merge(
            adapter_paths=adapter_paths,
            output_dir=str(output_dir),
            strategy=strategy,
        )

        # Property: output files exist
        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

        # Property: merged modules count matches common keys
        assert result.merged_modules == len(weight_keys)


@given(
    drop_rate=st.floats(min_value=0.0, max_value=0.9),
)
@settings(max_examples=50, deadline=None)
def test_adapter_merge_dare_ties_with_drop_rate(drop_rate: float):
    """Test DARE-TIES merge with various drop rates."""
    service = AdapterService()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        weight_keys = ["layer.0.lora_A", "layer.0.lora_B"]
        adapter_paths = []

        for i in range(3):
            adapter_path = tmp_path / f"adapter_{i}"
            _create_adapter(adapter_path, weight_keys=weight_keys)
            adapter_paths.append(str(adapter_path))

        output_dir = tmp_path / "merged"

        result = service.merge(
            adapter_paths=adapter_paths,
            output_dir=str(output_dir),
            strategy="dare-ties",
            drop_rate=drop_rate,
        )

        # Property: output files exist
        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

        # Property: strategy is recorded correctly
        assert result.strategy == "dare-ties"
