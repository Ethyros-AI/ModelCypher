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

from .generator import SafeSelfPlayGenerator
from .oracle import VerificationOracle
from .scanner import CapabilityScanner
from .types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    ImprovementAction,
    ImprovementLog,
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

            if accuracy < 0.9:
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


__all__ = ["AutonomousSelfImprover"]
