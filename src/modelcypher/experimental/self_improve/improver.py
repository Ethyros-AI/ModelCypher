#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2024 ModelCypher
"""Autonomous self-improvement orchestrator.

The AutonomousSelfImprover runs the complete self-improvement loop:

1. SCAN: Identify capability states (WORKING/DISCONNECTED/TRUE_GAP)
2. BRIDGE: For DISCONNECTED capabilities, record the prime that works
3. GENERATE: For TRUE_GAP capabilities, generate verified training data
4. SPEC: Output LoRA training specification

This loop runs without human intervention. All learned knowledge is
verified by the oracle, ensuring the model learns facts, not nonsense.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelcypher.adapters.self_improve.oracle import VerificationOracle
from modelcypher.adapters.self_improve.scanner import CapabilityScanner
from .self_play_generator import SafeSelfPlayGenerator
from .geometric_training_data import augment_training_data_with_geometry
from .types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    ImprovementAction,
    ImprovementLog,
    SelfImprovementConfig,
    VerifiedSample,
)

logger = logging.getLogger(__name__)


class AutonomousSelfImprover:
    """Complete autonomous self-improvement loop.

    This class orchestrates the full self-improvement pipeline:
    - Uses CapabilityScanner to identify gaps
    - Uses VerificationOracle to validate learning
    - Uses SafeSelfPlayGenerator to create training data
    - Outputs LoRA training specifications

    Example:
        >>> improver = AutonomousSelfImprover(model, tokenizer)
        >>> capabilities = [
        ...     Capability.from_lists("arithmetic", [...], [...]),
        ...     Capability.from_lists("word_problems", [...], [...]),
        ... ]
        >>> log = improver.improve(capabilities)
        >>> print(f"Bridged: {log.capabilities_bridged}")
        >>> print(f"True gaps: {log.true_gaps}")
    """

    def __init__(self, model, tokenizer):
        """Initialize improver.

        Args:
            model: The language model
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

        # Initialize components
        self.scanner = CapabilityScanner(model, tokenizer)
        self.oracle = VerificationOracle(model, tokenizer)
        self.generator = SafeSelfPlayGenerator(self.oracle)

    def improve(
        self,
        capabilities: List[Capability],
        training_data_path: Optional[Path] = None,
        n_training_samples: int = 500,
    ) -> ImprovementLog:
        """Run the self-improvement loop.

        This is the main entry point for autonomous improvement.

        Args:
            capabilities: List of capabilities to analyze and improve
            training_data_path: Path to save generated training data
            n_training_samples: Number of training samples to generate per gap

        Returns:
            ImprovementLog with full details of what was done
        """
        log = ImprovementLog()

        # Phase 1: SCAN
        logger.info("PHASE 1: SCAN - Identifying capabilities")
        analyses: List[CapabilityAnalysis] = []
        for cap in capabilities:
            analysis = self.scanner.scan(cap)
            analyses.append(analysis)
            log.capabilities_scanned.append(cap.name)

            status_icon = (
                "✓" if analysis.status == CapabilityStatus.WORKING
                else "⚡" if analysis.status == CapabilityStatus.DISCONNECTED
                else "✗"
            )
            logger.info(
                f"  {status_icon} {cap.name}: {analysis.status.value.upper()} "
                f"(raw={analysis.accuracy_raw:.0%}, "
                f"primed={analysis.accuracy_primed:.0%})"
            )

        # Phase 2: CLASSIFY
        logger.info("PHASE 2: CLASSIFY - Categorizing by status")
        for analysis in analyses:
            if analysis.status == CapabilityStatus.WORKING:
                log.capabilities_working.append(analysis.capability.name)
            elif analysis.status == CapabilityStatus.DISCONNECTED:
                log.capabilities_bridged.append(analysis.capability.name)
                log.actions.append(
                    ImprovementAction(
                        capability=analysis.capability.name,
                        action_type="apply_prime",
                        details={
                            "prime": analysis.best_prime,
                            "accuracy_improvement": (
                                analysis.accuracy_primed - analysis.accuracy_raw
                            ),
                        },
                    )
                )
            else:  # TRUE_GAP
                log.true_gaps.append(analysis.capability.name)

        logger.info(f"  Working: {len(log.capabilities_working)}")
        logger.info(f"  Disconnected (bridged): {len(log.capabilities_bridged)}")
        logger.info(f"  True gaps: {len(log.true_gaps)}")

        # Phase 3: GENERATE (for true gaps)
        if log.true_gaps:
            logger.info("PHASE 3: GENERATE - Creating verified training data")

            # Calibrate oracle first
            calibration_tests = VerificationOracle.default_calibration_tests()
            accuracy, _ = self.oracle.calibrate(calibration_tests)
            logger.info(f"  Oracle calibration: {accuracy:.0%}")

            # Derive minimum required accuracy based on statistical confidence intervals.
            # Convert configured error budget into the required Z-score confidence bound.
            # e.g., error_tolerance = 0.05 -> 95% confidence -> Z ≈ 1.96
            error_tolerance = getattr(config, "oracle_error_tolerance", 0.05) if "config" in locals() else 0.05
            
            from modelcypher.core.domain.geometry.numerical_stability import erfinv_scalar
            # Z = sqrt(2) * erfinv(1 - error_tolerance)
            z_score = 1.41421356 * erfinv_scalar(1.0 - error_tolerance)
            z_sq = z_score ** 2
            
            n_tests = len(calibration_tests)
            p_hat = accuracy
            min_bound = (p_hat + z_sq / (2 * n_tests) - z_score * ((p_hat * (1 - p_hat) + z_sq / (4 * n_tests)) / n_tests) ** 0.5) / (1 + z_sq / n_tests)
            
            # If the lower bound of our confidence interval is <= 50%, the oracle is unreliable.
            if min_bound <= 0.5:
                logger.warning("  Oracle calibration too low, skipping generation")
            else:
                # Generate training data
                samples = self.generator.generate_verified(
                    n_training_samples,
                )
                stats = self.generator.get_statistics(samples)

                logger.info(f"  Generated {stats['total']} verified samples")
                logger.info(f"    Addition: {stats['addition']}")
                logger.info(f"    Subtraction: {stats['subtraction']}")

                # Save if path provided
                if training_data_path:
                    self.generator.save_jsonl(samples, training_data_path)
                    log.training_data_path = str(training_data_path)
                    logger.info(f"  Saved to: {training_data_path}")

                # Create training spec
                log.training_spec = self.create_training_spec(
                    gap_names=log.true_gaps,
                    data_path=str(training_data_path) if training_data_path else None,
                    n_samples=len(samples),
                )

                for gap in log.true_gaps:
                    log.actions.append(
                        ImprovementAction(
                            capability=gap,
                            action_type="generate_training",
                            details={
                                "samples_generated": len(samples),
                                "verification_rate": 1.0,  # All samples verified
                            },
                        )
                    )

        log.iterations = 1
        return log

    def create_training_spec(
        self,
        gap_names: List[str],
        data_path: Optional[str] = None,
        n_samples: int = 500,
    ) -> Dict[str, Any]:
        """Create LoRA training specification for true gaps.

        Args:
            gap_names: Names of capabilities with true gaps
            data_path: Path to training data
            n_samples: Number of training samples

        Returns:
            Dictionary with training specification
        """
        return {
            "target_capabilities": gap_names,
            "adapter": {
                "type": "lora",
                "rank": 8,
                "alpha": 16,
                "target_layers": "early",  # Early layers for parsing
            },
            "training": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "warmup_steps": 100,
            },
            "freeze": {
                "late_layers": True,  # Preserve arithmetic capability
            },
            "data": {
                "path": data_path,
                "samples": n_samples,
                "verified": True,
            },
            "rationale": (
                "LoRA on early layers because: "
                "1) Parsing happens in early layers (language understanding), "
                "2) Arithmetic is in later layers (computation), "
                "3) Small rank (8) because gap is narrow, "
                "4) Don't touch arithmetic capability."
            ),
        }

    def save_log(self, log: ImprovementLog, path: Path) -> None:
        """Save improvement log to JSON file.

        Args:
            log: The improvement log
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(log.to_dict(), f, indent=2)

    def improve_iterative(
        self,
        capabilities: List[Capability],
        output_dir: Path,
        config: Optional[SelfImprovementConfig] = None,
        stacker: Optional["LoRAStacker"] = None,
    ) -> Dict[str, Any]:
        """Run iterative self-improvement with stacked LoRA.

        This is the main entry point for cumulative self-improvement.
        Each round:
        1. Scan capabilities for TRUE_GAPs
        2. Generate training data targeting gaps
        3. AUGMENT: Add geometric context to each sample (if enabled)
        4. TRAIN: With loop preservation loss (if enabled)
        5. Add to stack, check cumulative geometry
        6. If merge needed: consolidate adapters
        7. Increase difficulty, repeat

        Args:
            capabilities: List of capabilities to analyze and improve
            output_dir: Directory for training data and adapters
            config: Self-improvement configuration. If None, uses defaults.
            stacker: Optional LoRAStacker (creates new if not provided)

        Returns:
            Summary dict with rounds completed, adapters trained, etc.
        """
        from .lora_stacker import LoRAStacker

        # Use default config if not provided
        if config is None:
            config = SelfImprovementConfig()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stacker if not provided
        if stacker is None:
            logger.warning(
                "No stacker provided - iterative improvement requires "
                "external stacker with base model path"
            )
            return {
                "success": False,
                "error": "stacker required for iterative improvement",
                "rounds_completed": 0,
            }

        rounds_completed = 0
        adapters_trained = 0
        merges_performed = 0
        all_logs: List[ImprovementLog] = []

        # Pre-compute geometric config if enabled
        loop_config = None
        highway_layer = 0
        base_delta_entropy = 0.0

        if config.loop_preservation or config.geometric_self_awareness:
            logger.info("Computing geometric configuration...")
            from modelcypher.core.domain.training.loop_preservation import (
                detect_highway_layer,
                compute_base_entropy_trajectory,
            )

            # Use a diverse set of probe prompts
            probe_prompts = [
                "What is 2 + 2?",
                "Calculate: 15 - 7",
                "Explain the concept of addition.",
                "If I have 5 apples and get 3 more, how many do I have?",
            ]

            highway_layer = detect_highway_layer(
                self.model, self.tokenizer, probe_prompts
            )
            base_delta_entropy = compute_base_entropy_trajectory(
                self.model, self.tokenizer, probe_prompts, highway_layer
            )

            if config.loop_preservation:
                from modelcypher.core.domain.training.loop_preservation import (
                    LoopPreservationConfig,
                )
                from modelcypher.core.domain.training.geometric_lora import (
                    analyze_model_geometry,
                )

                # Get sigma_max from model geometry
                geometries = analyze_model_geometry(self.model)
                if geometries:
                    first_geom = next(iter(geometries.values()))
                    sigma_max = first_geom.sigma_max
                else:
                    sigma_max = 1.0

                loop_config = LoopPreservationConfig(
                    highway_layer=highway_layer,
                    base_delta_entropy=base_delta_entropy,
                    lambda_scale=1.0 / max(sigma_max, 1e-8),
                )
                logger.info(
                    "Loop preservation enabled: highway=%d, base_ΔH=%.4f, λ=%.6f",
                    loop_config.highway_layer,
                    loop_config.base_delta_entropy,
                    loop_config.lambda_scale,
                )

        for round_idx in range(config.max_rounds):
            logger.info(f"=== ROUND {round_idx + 1}/{config.max_rounds} ===")

            # Run single improvement round
            training_path = output_dir / f"round{round_idx + 1}_training.jsonl"
            log = self.improve(
                capabilities=capabilities,
                training_data_path=training_path,
                n_training_samples=config.n_samples_per_round,
            )
            all_logs.append(log)
            rounds_completed += 1

            # Check if we found gaps to train on
            if not log.true_gaps:
                logger.info("No true gaps found - self-improvement converged!")
                break

            # Check if training data was generated
            if not log.training_data_path:
                logger.info("No training data generated (oracle calibration low?)")
                continue

            # ===== AUGMENT: Add geometric context =====
            training_samples: List[Dict[str, str]] = []
            with open(log.training_data_path, "r") as f:
                for line in f:
                    training_samples.append(json.loads(line))

            if config.geometric_self_awareness:
                logger.info("Augmenting training data with geometric context...")
                training_samples = augment_training_data_with_geometry(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    training_samples=training_samples,
                    highway_layer=highway_layer,
                    base_delta_entropy=base_delta_entropy,
                )

                # Save augmented data
                augmented_path = output_dir / f"round{round_idx + 1}_augmented.jsonl"
                with open(augmented_path, "w") as f:
                    for sample in training_samples:
                        f.write(json.dumps(sample) + "\n")
                logger.info(f"Saved augmented data to {augmented_path}")

            # ===== TRAIN LORA ADAPTER =====
            from modelcypher.core.use_cases.lora_training_service import (
                LoRATrainingService,
            )

            training_service = LoRATrainingService()
            adapter_path = output_dir / f"adapter_round{round_idx + 1}"

            logger.info(f"Training LoRA adapter for gaps: {log.true_gaps}")

            # Get training spec params
            spec = log.training_spec or {}
            training_config = spec.get("training", {})
            adapter_config = spec.get("adapter", {})

            training_result = training_service.train_lora(
                model_path=stacker.state.base_model_path,
                training_data_path=Path(log.training_data_path),
                output_path=adapter_path,
                epochs=training_config.get("epochs", 3),
                batch_size=training_config.get("batch_size", 4),
                learning_rate=training_config.get("learning_rate", 1e-4),
                rank=adapter_config.get("rank"),
                loop_config=loop_config,  # Pass loop preservation config
            )

            if not training_result.success:
                logger.error(
                    "Training failed: %s", training_result.error
                )
                continue

            logger.info(
                "Training complete: loss=%.4f, barrier=%.4f, cka=%.4f",
                training_result.final_loss,
                training_result.barrier_to_base,
                training_result.cka_from_base,
            )

            # Add to stacker
            stack_result = stacker.add_adapter(
                adapter_path=training_result.adapter_path,
                barrier=training_result.barrier_to_base,
                cka_from_base=training_result.cka_from_base,
                difficulty_level=round_idx + 1,
                training_samples=training_result.samples_used,
                target_modules=training_result.target_modules,
            )

            adapters_trained += 1

            # Check if merge needed
            if stack_result.should_merge:
                logger.info(
                    "Merge triggered: %s", stack_result.merge_reason
                )
                merged_path = output_dir / f"merged_round{round_idx + 1}"
                merge_result = stacker.merge_stack(merged_path)

                if merge_result.success:
                    logger.info(
                        "Merged %d adapters into %s",
                        merge_result.adapters_merged,
                        merge_result.merged_path,
                    )
                    merges_performed += 1
                else:
                    logger.warning("Merge failed: %s", merge_result.message)

            # Log stacker status
            status = stacker.get_status()
            logger.info(f"Stacker status: {status['n_adapters']} adapters, "
                       f"barrier={status['cumulative_barrier']:.4f}")

        return {
            "success": True,
            "rounds_completed": rounds_completed,
            "adapters_trained": adapters_trained,
            "merges_performed": merges_performed,
            "loop_preservation_enabled": config.loop_preservation,
            "geometric_self_awareness_enabled": config.geometric_self_awareness,
            "final_stacker_status": stacker.get_status(),
            "logs": [log.to_dict() for log in all_logs],
        }


# Avoid circular import - use string annotation above
if __name__ != "__main__":
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .lora_stacker import LoRAStacker


__all__ = ["AutonomousSelfImprover"]
