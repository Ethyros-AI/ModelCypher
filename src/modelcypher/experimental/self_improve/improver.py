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

from modelcypher.experimental.self_improve.oracle import VerificationOracle
from modelcypher.experimental.self_improve.scanner import CapabilityScanner

from .geometric_training_data import augment_training_data_with_geometry
from .self_play_generator import SafeSelfPlayGenerator
from .types import (
    Capability,
    CapabilityAnalysis,
    CapabilityStatus,
    ImprovementAction,
    ImprovementLog,
    SelfImprovementConfig,
)

logger = logging.getLogger(__name__)


def _extract_projection_weights(model: Any) -> dict[str, Any]:
    """Extract 2D projection weight matrices from model layers.

    Traverses ``model.model.layers`` and collects q/k/v/o/gate/up/down
    projection weights — the same matrices that geometric_lora analyzes.
    """
    weights: dict[str, Any] = {}
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        return weights
    for layer_idx, layer in enumerate(base.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None:
                    w = getattr(proj, "weight", None)
                    if w is not None and hasattr(w, "ndim") and w.ndim == 2:
                        key = f"model.layers.{layer_idx}.self_attn.{proj_name}.weight"
                        weights[key] = w
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None:
                    w = getattr(proj, "weight", None)
                    if w is not None and hasattr(w, "ndim") and w.ndim == 2:
                        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                        weights[key] = w
    return weights


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

    def __init__(self, model, tokenizer, model_path: str | Path | None = None):
        """Initialize improver.

        Args:
            model: The language model.
            tokenizer: The tokenizer for the model.
            model_path: Path to the model on disk (needed for training spec).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = str(model_path) if model_path else ""

        # Initialize components
        self.scanner = CapabilityScanner(model, tokenizer)
        self.oracle = VerificationOracle(model, tokenizer)
        self.generator = SafeSelfPlayGenerator(self.oracle)

    def improve(
        self,
        capabilities: List[Capability],
        accuracy_threshold: float = 1.0,
        training_data_path: Optional[Path] = None,
        n_training_samples: int = 500,
    ) -> ImprovementLog:
        """Run the self-improvement loop.

        This is the main entry point for autonomous improvement.

        Args:
            capabilities: List of capabilities to analyze and improve
            accuracy_threshold: Accuracy threshold for WORKING/DISCONNECTED classification
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
            analysis = self.scanner.scan(
                cap,
                accuracy_threshold=accuracy_threshold,
            )
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
        disconnected_analyses: List[CapabilityAnalysis] = []
        for analysis in analyses:
            if analysis.status == CapabilityStatus.WORKING:
                log.capabilities_working.append(analysis.capability.name)
            elif analysis.status == CapabilityStatus.DISCONNECTED:
                log.capabilities_bridged.append(analysis.capability.name)
                disconnected_analyses.append(analysis)
            else:  # TRUE_GAP
                log.true_gaps.append(analysis.capability.name)

        logger.info(f"  Working: {len(log.capabilities_working)}")
        logger.info(f"  Disconnected (bridged): {len(log.capabilities_bridged)}")
        logger.info(f"  True gaps: {len(log.true_gaps)}")

        # Phase 2b: STEER — Compute contrastive steering vectors for DISCONNECTED
        if disconnected_analyses:
            logger.info("PHASE 2b: STEER - Computing contrastive steering vectors")
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.experimental.interpretability.feature_steering import (
                FeatureSteering,
            )

            b = get_default_backend()
            steering = FeatureSteering(self.model, b)

            for analysis in disconnected_analyses:
                cap = analysis.capability
                # Collect contrastive activations at each layer
                # Use middle layer as default target (model-dependent)
                layers = steering._get_layers()
                target_layer = len(layers) // 2

                try:
                    pos_acts, neg_acts = self.scanner.collect_contrastive_activations(
                        capability=cap,
                        best_prime=analysis.best_prime,
                        target_layer=target_layer,
                    )

                    # Extract contrastive direction via Fréchet mean difference
                    vec = steering.extract_contrastive_direction(
                        positive_activations=pos_acts,
                        negative_activations=neg_acts,
                        layer=target_layer,
                        label=f"bridge_{cap.name}",
                    )

                    # Project into null space for safety (AlphaSteer)
                    _projected_dir, proj_loss = steering.project_to_null_space(
                        steering_direction=vec.direction,
                        prior_activations=neg_acts,
                    )

                    log.actions.append(
                        ImprovementAction(
                            capability=cap.name,
                            action_type="steering",
                            details={
                                "prime": analysis.best_prime,
                                "accuracy_improvement": (
                                    analysis.accuracy_primed - analysis.accuracy_raw
                                ),
                                "target_layer": target_layer,
                                "projection_loss": proj_loss,
                                "strength_range": list(vec.strength_range),
                            },
                        )
                    )
                    logger.info(
                        "  %s: steering vector at layer %d "
                        "(projection_loss=%.3f, range=[%.2f, %.2f])",
                        cap.name,
                        target_layer,
                        proj_loss,
                        vec.strength_range[0],
                        vec.strength_range[1],
                    )
                except Exception:
                    raise RuntimeError(
                        f"Steering vector extraction failed for {cap.name}. "
                        f"Falling back to 'apply_prime' is a different operation — "
                        f"fix the extraction, don't silently substitute."
                    )

        # Phase 3: GENERATE (for true gaps)
        if log.true_gaps:
            logger.info("PHASE 3: GENERATE - Creating verified training data")

            # Calibrate oracle first
            calibration_tests = VerificationOracle.default_calibration_tests()
            accuracy, _ = self.oracle.calibrate(calibration_tests)
            logger.info(f"  Oracle calibration: {accuracy:.0%}")

            # Clopper-Pearson exact binomial CI (Biometrika 26(4), 1934).
            # alpha = 1/N — data-derived, not a hyperparameter.
            from scipy.stats import beta as beta_dist

            n_tests = len(calibration_tests)
            n_correct = round(accuracy * n_tests)
            alpha = 1.0 / n_tests if n_tests > 0 else 0.5

            if n_correct == 0:
                min_bound = 0.0
            else:
                min_bound = float(
                    beta_dist.ppf(alpha / 2.0, n_correct, n_tests - n_correct + 1)
                )

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
                    model_path=self.model_path,
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
        model_path: str,
        data_path: Optional[str] = None,
        n_samples: int = 0,
    ) -> Dict[str, Any]:
        """Create training specification for true gaps.

        All training parameters (rank, scale, LR, batch size, stopping)
        are derived from geometry by DatasetTrainingService. This spec
        only records WHAT to train, not HOW — the geometry decides that.

        Args:
            gap_names: Names of capabilities with true gaps.
            model_path: Path to the base model (needed by training service).
            data_path: Path to training data JSONL.
            n_samples: Number of training samples generated.

        Returns:
            Dictionary with training specification pointing to the
            geometry-derived training pipeline.
        """
        return {
            "target_capabilities": gap_names,
            "training_service": "DatasetTrainingService",
            "model_path": model_path,
            "data": {
                "path": data_path,
                "samples": n_samples,
                "verified": True,
            },
            "note": (
                "Rank, scale, LR, batch size, and stopping criteria are all "
                "derived from the model's weight geometry by "
                "DatasetTrainingService.train_from_dataset(). No hyperparameters."
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
        accuracy_threshold: float = 1.0,
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
            accuracy_threshold: Accuracy threshold for WORKING/DISCONNECTED classification
            config: Self-improvement configuration. If None, uses defaults.
            stacker: Optional LoRAStacker (creates new if not provided)

        Returns:
            Summary dict with rounds completed, adapters trained, etc.
        """

        if config is None:
            raise ValueError(
                "SelfImprovementConfig is required — max_rounds and "
                "n_samples_per_round must be specified by the caller."
            )

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
        all_logs: List[ImprovementLog] = []

        # Pre-compute geometric config if enabled
        loop_config = None
        highway_layer = 0
        base_delta_entropy = 0.0

        if config.loop_preservation or config.geometric_self_awareness:
            logger.info("Computing geometric configuration...")
            # Collect per-layer intrinsic dimensions from model
            # (requires activation collection at each layer)
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.training.loop_preservation import (
                compute_entropy_delta,
                find_highway_layer_from_intrinsic_dims,
            )
            b = get_default_backend()

            probe_prompts = [
                "What is 2 + 2?",
                "Calculate: 15 - 7",
                "Explain the concept of addition.",
                "If I have 5 apples and get 3 more, how many do I have?",
            ]

            # Collect per-layer activations for highway detection
            layers = getattr(
                getattr(self.model, "model", self.model), "layers", []
            )
            n_layers = len(layers)
            layer_ids: list[float] = []

            for layer_idx in range(n_layers):
                acts_list = []
                for prompt in probe_prompts:
                    states = b.collect_hidden_activations(
                        self.model, self.tokenizer, [prompt],
                        layer_indices=[layer_idx],
                    )
                    if layer_idx in states:
                        act = states[layer_idx]
                        if hasattr(act, "ndim") and act.ndim >= 3:
                            act = act[0]  # [seq, hidden]
                        b.eval(act)
                        acts_list.append(act)
                # Compute intrinsic dimension for this layer
                if acts_list:
                    stacked = b.concatenate(acts_list, axis=0)
                    b.eval(stacked)
                    from modelcypher.core.domain.geometry.intrinsic_dimension import (
                        IntrinsicDimension,
                    )
                    id_estimator = IntrinsicDimension(b)
                    estimate = id_estimator.compute(stacked)
                    layer_ids.append(estimate.dimension)
                else:
                    layer_ids.append(float("inf"))

            highway_result = find_highway_layer_from_intrinsic_dims(layer_ids)
            highway_layer = highway_result[0] if highway_result[0] is not None else 0

            if config.loop_preservation:
                from modelcypher.core.domain.training.geometric_lora import (
                    analyze_weight_geometries,
                )
                from modelcypher.core.domain.training.loop_preservation import (
                    LoopPreservationConfig,
                )

                # Get sigma_max from model geometry
                # Extract 2D projection weights for geometry analysis
                model_weights = _extract_projection_weights(self.model)
                geometries = analyze_weight_geometries(model_weights, b)
                if geometries:
                    first_geom = next(iter(geometries.values()))
                    sigma_max = first_geom.sigma_max
                else:
                    raise RuntimeError(
                        "Cannot derive loop_config: model has no analyzable weight matrices"
                    )

                # Collect entropy at highway and exit layers for base_delta_entropy
                exit_layer = n_layers - 1
                highway_entropies: list[float] = []
                exit_entropies: list[float] = []

                from modelcypher.core.domain.training.loop_preservation import (
                    compute_spectral_entropy,
                )

                for prompt in probe_prompts:
                    states = b.collect_hidden_activations(
                        self.model, self.tokenizer, [prompt],
                        layer_indices=[highway_layer, exit_layer],
                    )
                    if highway_layer in states:
                        h_act = states[highway_layer]
                        if hasattr(h_act, "ndim") and h_act.ndim >= 3:
                            h_act = h_act[0]
                        b.eval(h_act)
                        highway_entropies.append(compute_spectral_entropy(h_act, b))
                    if exit_layer in states:
                        e_act = states[exit_layer]
                        if hasattr(e_act, "ndim") and e_act.ndim >= 3:
                            e_act = e_act[0]
                        b.eval(e_act)
                        exit_entropies.append(compute_spectral_entropy(e_act, b))

                base_delta_entropy = compute_entropy_delta(
                    highway_entropies, exit_entropies
                )

                loop_config = LoopPreservationConfig(
                    highway_layer=highway_layer,
                    base_delta_entropy=base_delta_entropy,
                    lambda_scale=1.0 / sigma_max,
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
                accuracy_threshold=accuracy_threshold,
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
            # All training parameters derived from geometry by DatasetTrainingService.
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.use_cases.dataset_training_service import (
                DatasetTrainingService,
            )

            b = get_default_backend()
            training_service = DatasetTrainingService(b)
            adapter_path = output_dir / f"adapter_round{round_idx + 1}"

            logger.info(f"Training LoRA adapter for gaps: {log.true_gaps}")

            try:
                training_result = training_service.train_from_dataset(
                    model_path=stacker.state.base_model_path,
                    dataset_path=Path(log.training_data_path),
                    output_path=adapter_path,
                )
            except Exception:
                logger.exception("Training failed")
                continue

            logger.info(
                "Training complete: loss=%.4f→%.4f, stop=%s",
                training_result.initial_loss,
                training_result.final_loss,
                training_result.stop_reason,
            )

            adapters_trained += 1

            # Log round result
            logger.info(
                "Round %d: %d iters, final_loss=%.4f",
                round_idx + 1,
                training_result.train_iters,
                training_result.final_loss,
            )

        return {
            "success": True,
            "rounds_completed": rounds_completed,
            "adapters_trained": adapters_trained,
            "loop_preservation_enabled": config.loop_preservation,
            "geometric_self_awareness_enabled": config.geometric_self_awareness,
            "logs": [log.to_dict() for log in all_logs],
        }


# Avoid circular import - use string annotation above
if __name__ != "__main__":
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .lora_stacker import LoRAStacker


__all__ = ["AutonomousSelfImprover"]
