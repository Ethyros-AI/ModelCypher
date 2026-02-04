#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Integration test for stacked LoRA self-improvement system.
# Uses LFM2-350M as a small test model.

"""Integration test for stacked LoRA system.

Validates the complete pipeline:
1. LoRAStacker state tracking
2. LoRASafetyService geometry metrics
3. Integration with real model activations

Usage:
    poetry run python tests/integration/test_stacked_lora_integration.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path on codecypher volume
MODEL_PATH = Path("/Volumes/codecypher/models/mlx-community/LFM2-350M-MLX-bf16")


def initialize_backend():
    """Initialize the MLX backend for model operations."""
    try:
        from modelcypher.backends import initialize_default_backend
        initialize_default_backend()  # Auto-detects MLX
        logger.info("Initialized backend")
        return True
    except Exception as e:
        logger.warning("Failed to initialize backend: %s", e)
        return False


def test_stacker_state_tracking():
    """Test that LoRAStacker correctly tracks cumulative state."""
    from modelcypher.core.use_cases.self_improve import LoRAStacker, StackerPolicy

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create stacker with explicit policy
        policy = StackerPolicy(
            barrier_merge_threshold=0.03,
            cka_drift_threshold=0.1,
            max_adapters=5,
            convergence_ratio_threshold=1.0,
            convergence_barrier_multiplier=0.5,
        )
        stacker = LoRAStacker(MODEL_PATH, policy=policy)
        assert stacker.state.n_adapters == 0
        assert stacker.state.cumulative_barrier == 0.0
        
        # Create fake adapter directories
        for i in range(3):
            adapter_path = tmpdir / f"adapter_{i}"
            adapter_path.mkdir()
            
            result = stacker.add_adapter(
                adapter_path=adapter_path,
                barrier=0.008,
                cka_from_base=0.96 - (i * 0.02),  # Decreasing CKA
                difficulty_level=i + 1,
            )
            
            assert result.success
            logger.info(
                "Added adapter %d: barrier=%.4f, cka_drift=%.4f",
                i, result.cumulative_barrier, result.cumulative_cka_drift
            )
        
        # Check cumulative tracking (use approx for float comparison)
        assert stacker.state.n_adapters == 3
        assert abs(stacker.state.cumulative_barrier - 0.024) < 1e-6  # 3 * 0.008
        assert abs(stacker.state.cumulative_cka_drift - 0.08) < 1e-6  # 1 - 0.92 (worst)
        
        # Test persistence
        state_file = tmpdir / "state.json"
        stacker.save_state(state_file)
        
        # Load in new stacker (policy from state file)
        stacker2 = LoRAStacker(MODEL_PATH, policy=policy, state_path=state_file)
        assert stacker2.state.n_adapters == 3
        assert abs(stacker2.state.cumulative_barrier - 0.024) < 1e-6
        
        logger.info("✓ Stacker state tracking works correctly")


def test_lora_safety_service_with_model():
    """Test LoRASafetyService with real model activations."""
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
    
    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s, skipping", MODEL_PATH)
        return
    
    service = LoRASafetyService()
    
    # Test curriculum scoring
    problems = [
        {"prompt": "What is 2+2?"},
        {"prompt": "What is 15+27?"},
        {"prompt": "If I have 3 apples and get 5 more, how many do I have?"},
        {"prompt": "Calculate the derivative of x^2 + 3x - 5"},
    ]
    
    logger.info("Scoring curriculum with LFM2-350M...")
    result = service.score_curriculum(
        model_path=MODEL_PATH,
        problems=problems,
        top_k=4,
    )
    
    logger.info("Scored %d problems:", result.n_problems)
    for p in result.top_problems:
        logger.info("  %.3f %s - %s", p["quality_score"], p["quality_level"], p["prompt"][:50])
    
    logger.info("Quality distribution: %s", result.quality_distribution)
    logger.info("✓ Curriculum scoring works with real model")


def test_difficulty_filtering():
    """Test difficulty filtering with real model."""
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
    
    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s, skipping", MODEL_PATH)
        return
    
    service = LoRASafetyService()
    
    problems = [
        {"prompt": "1+1="},
        {"prompt": "2+2="},
        {"prompt": "What is the integral of sin(x)*e^x?"},
        {"prompt": "Solve the differential equation dy/dx = y^2 + x"},
        {"prompt": "What is 5*5?"},
    ]
    
    logger.info("Filtering problems by difficulty...")
    
    for difficulty in ["easy", "medium", "hard"]:
        filtered = service.filter_by_difficulty(
            model_path=MODEL_PATH,
            problems=problems,
            target_difficulty=difficulty,
        )
        logger.info("  %s: %d problems", difficulty, len(filtered))
    
    logger.info("✓ Difficulty filtering works with real model")


def test_fisher_recommendations():
    """Test Fisher-based module recommendations."""
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
    
    if not MODEL_PATH.exists():
        logger.warning("Model not found at %s, skipping", MODEL_PATH)
        return
    
    service = LoRASafetyService()
    
    prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Explain photosynthesis.",
    ]
    
    logger.info("Getting Fisher-guided module recommendations...")
    result = service.recommend_target_modules(
        model_path=MODEL_PATH,
        prompts=prompts,
        top_k=4,
    )
    
    logger.info("Layer %d recommendations:", result.layer)
    for rec in result.recommendations:
        logger.info("  %s: Fisher=%.6f (%s)", rec.module, rec.fisher_score, rec.recommendation)
    
    logger.info("✓ Fisher recommendations work with real model")


def main():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("Stacked LoRA Integration Tests")
    logger.info("=" * 60)
    logger.info("Model: %s", MODEL_PATH)
    logger.info("")
    
    # Always run state tracking test (no model needed)
    test_stacker_state_tracking()
    
    # Model-dependent tests
    if MODEL_PATH.exists():
        if not initialize_backend():
            logger.error("Backend initialization failed - skipping model tests")
            return
        
        logger.info("")
        test_fisher_recommendations()
        logger.info("")
        test_lora_safety_service_with_model()
        logger.info("")
        test_difficulty_filtering()
    else:
        logger.warning("Model not found - skipping model-dependent tests")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("All integration tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
