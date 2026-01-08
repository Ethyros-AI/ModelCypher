#!/usr/bin/env python3
"""Quick test for density stage debugging.

Creates mock activations with shapes matching real models and tests density stage.

Usage:
    poetry run python scripts/test_density_stage.py
"""

import logging
import sys
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_density_with_mock_data():
    """Test density stage with mock activations matching LFM2-350M shapes."""
    from modelcypher.core.domain._backend import set_default_backend, get_default_backend
    from modelcypher.backends import get_backend
    from modelcypher.core.use_cases.merge.stages.density import stage_density

    set_default_backend(get_backend("mlx"))
    backend = get_default_backend()

    # LFM2-350M parameters - use fewer probes to avoid memory issues
    n_probes = 256  # Fewer probes for fast testing
    target_hidden = 1024  # LFM2-350M hidden dim
    source_hidden = 2048  # Qwen2.5-3B hidden dim
    n_layers = 4  # Test with fewer layers first

    logger.info("Creating mock activations: %d layers, %d probes", n_layers, n_probes)
    logger.info("Source hidden: %d, Target hidden: %d", source_hidden, target_hidden)

    # Create random activations as 1D vectors
    source_activations = {}
    target_activations = {}

    for layer_idx in range(n_layers):
        # Random vectors - using backend operations
        source_activations[layer_idx] = [
            backend.random_normal(shape=(source_hidden,)) for _ in range(n_probes)
        ]
        target_activations[layer_idx] = [
            backend.random_normal(shape=(target_hidden,)) for _ in range(n_probes)
        ]

    probe_ids = [f"probe_{i}" for i in range(n_probes)]
    probe_domains = ["test"] * n_probes
    layers = list(range(n_layers))

    # Check activation shapes
    logger.info("Source activation shape: %s", source_activations[0][0].shape)
    logger.info("Target activation shape: %s", target_activations[0][0].shape)

    logger.info("Running density stage...")
    try:
        result = stage_density(
            source_activations=source_activations,
            target_activations=target_activations,
            probe_ids=probe_ids,
            probe_domains=probe_domains,
            layers=layers,
            backend=backend,
        )
        logger.info("SUCCESS! Graft mask has %d entries", len(result.graft_mask))
        logger.info("Metrics: %s", result.metrics)
        return 0
    except Exception as e:
        logger.error("FAILED: %s", e)
        traceback.print_exc()
        return 1


def main():
    return test_density_with_mock_data()


if __name__ == "__main__":
    sys.exit(main())
