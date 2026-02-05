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

"""Semantic Probe Service.

High-level orchestration for semantic verification of LoRA adapter transfers.
Integrates with the LoRA transfer pipeline to provide behavioral validation.

This service:
1. Loads models and adapters
2. Runs semantic probes on source (pre-transfer) and target (post-transfer)
3. Computes KL divergence between probability distributions
4. Reports semantic drift metrics

Usage:
    from modelcypher.core.use_cases.semantic_probe_service import (
        SemanticProbeService,
        SemanticVerificationConfig,
    )

    service = SemanticProbeService()
    result = service.verify_transfer(
        source_model_path=Path("./llama-base"),
        source_adapter_path=Path("./llama-adapter"),
        target_model_path=Path("./mistral-base"),
        target_adapter_path=Path("./mistral-transferred-adapter"),
    )
    if result.passed:
        print("Transfer preserved semantics!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.semantic_probe_verifier import (
    KL_DIVERGENCE_THRESHOLD,
    SemanticDriftResult,
    SemanticProbe,
    SemanticProbeVerifier,
    get_default_probes,
    load_semantic_probes,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Default Probe Path
# =============================================================================

DEFAULT_PROBE_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "probes" / "semantic_transfer_probes.json"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SemanticVerificationConfig:
    """Configuration for semantic verification.

    Attributes:
        probe_path: Path to JSON file with probe definitions.
            If None, uses built-in default probes.
        kl_threshold: KL divergence threshold for pass/fail.
            Default is ln(2) ≈ 0.693 (1 bit of information loss).
        require_all_pass: If True, all probes must pass for overall pass.
            If False, uses mean KL threshold.
        domains: Optional list of domains to filter probes.
            If None, uses all probes.
        max_probes: Maximum number of probes to run (for quick checks).
            If None, runs all probes.
    """

    probe_path: Path | None = None
    kl_threshold: float = KL_DIVERGENCE_THRESHOLD
    require_all_pass: bool = False
    domains: list[str] | None = None
    max_probes: int | None = None


# =============================================================================
# Semantic Probe Service
# =============================================================================


class SemanticProbeService:
    """High-level service for semantic verification of LoRA transfers.

    Orchestrates the loading of models, adapters, and probes to provide
    a simple API for verifying that adapter transfers preserve behavior.

    Example:
        service = SemanticProbeService()

        # Verify a transfer
        result = service.verify_transfer(
            source_model_path=Path("./source-base"),
            source_adapter_path=Path("./source-adapter"),
            target_model_path=Path("./target-base"),
            target_adapter_path=Path("./target-adapter"),
        )

        print(f"Passed: {result.passed}")
        print(f"Mean KL: {result.mean_kl_divergence:.4f}")
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        config: SemanticVerificationConfig | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            backend: Compute backend. Uses default if None.
            config: Verification configuration. Uses defaults if None.
        """
        self._backend = backend or get_default_backend()
        self._config = config or SemanticVerificationConfig()
        self._verifier: SemanticProbeVerifier | None = None
        self._probes: list[SemanticProbe] | None = None

    def _get_probes(self) -> list[SemanticProbe]:
        """Load probes based on configuration."""
        if self._probes is not None:
            return self._probes

        config = self._config

        # Load probes from file or use defaults
        if config.probe_path is not None and config.probe_path.exists():
            probes = load_semantic_probes(config.probe_path)
            logger.info("Loaded %d probes from %s", len(probes), config.probe_path)
        elif DEFAULT_PROBE_PATH.exists():
            probes = load_semantic_probes(DEFAULT_PROBE_PATH)
            logger.info("Loaded %d probes from default path", len(probes))
        else:
            probes = get_default_probes()
            logger.info("Using %d built-in default probes", len(probes))

        # Filter by domain if specified
        if config.domains:
            probes = [p for p in probes if p.domain in config.domains]
            logger.debug("Filtered to %d probes for domains: %s", len(probes), config.domains)

        # Limit number of probes if specified
        if config.max_probes is not None and len(probes) > config.max_probes:
            probes = probes[: config.max_probes]
            logger.debug("Limited to %d probes", len(probes))

        self._probes = probes
        return probes

    def verify_transfer(
        self,
        source_model_path: Path,
        target_model_path: Path,
        source_adapter_path: Path | None = None,
        target_adapter_path: Path | None = None,
        tokenizer: Any = None,
    ) -> SemanticDriftResult:
        """Verify that a LoRA transfer preserves semantic behavior.

        Runs semantic probes on both source and target configurations
        and measures KL divergence between their probability distributions.

        Args:
            source_model_path: Path to source base model.
            target_model_path: Path to target base model.
            source_adapter_path: Path to source LoRA adapter (optional).
            target_adapter_path: Path to target (transferred) LoRA adapter (optional).
            tokenizer: Tokenizer for encoding text. If None, loads from source model.

        Returns:
            SemanticDriftResult with verification metrics.

        Note:
            This is a simplified implementation. Full implementation would:
            1. Load models via InferenceEngine
            2. Load adapters and apply them properly
            3. Handle tokenizer loading from model configs
        """
        logger.info(
            "Verifying transfer: %s -> %s",
            source_model_path.name,
            target_model_path.name,
        )

        # Load probes
        probes = self._get_probes()
        if not probes:
            logger.warning("No probes available for verification")
            return SemanticDriftResult(
                mean_kl_divergence=0.0,
                max_kl_divergence=0.0,
                probes_passed=0,
                probes_total=0,
                rank_preservation_rate=0.0,
                top_prediction_rate=0.0,
                probe_results=[],
            )

        # Load tokenizer if not provided
        if tokenizer is None:
            tokenizer = self._load_tokenizer(source_model_path)

        # Initialize verifier
        verifier = SemanticProbeVerifier(
            backend=self._backend,
            tokenizer=tokenizer,
        )

        # Load models
        source_model = self._load_model(source_model_path)
        target_model = self._load_model(target_model_path)

        # Load adapters if provided
        source_adapter = None
        target_adapter = None
        if source_adapter_path is not None:
            source_adapter = self._load_adapter(source_adapter_path)
        if target_adapter_path is not None:
            target_adapter = self._load_adapter(target_adapter_path)

        # Run verification
        result = verifier.verify_transfer(
            source_model=source_model,
            source_adapter=source_adapter,
            target_model=target_model,
            target_adapter=target_adapter,
            probes=probes,
        )

        logger.info(
            "Verification complete: passed=%s, mean_kl=%.4f, max_kl=%.4f",
            result.passed,
            result.mean_kl_divergence,
            result.max_kl_divergence,
        )

        return result

    def verify_same_model(
        self,
        model_path: Path,
        tokenizer: Any = None,
    ) -> SemanticDriftResult:
        """Verify that a model produces consistent outputs (sanity check).

        Runs the same probes twice on the same model to ensure KL ≈ 0.
        This is useful for validating the probe system itself.

        Args:
            model_path: Path to model.
            tokenizer: Tokenizer for encoding text.

        Returns:
            SemanticDriftResult (should have near-zero KL).
        """
        return self.verify_transfer(
            source_model_path=model_path,
            target_model_path=model_path,
            source_adapter_path=None,
            target_adapter_path=None,
            tokenizer=tokenizer,
        )

    def _load_model(self, model_path: Path) -> Any:
        """Load a model from path.

        This is a placeholder - full implementation would use
        InferenceEngine or framework-specific loading.
        """
        logger.debug("Loading model from %s", model_path)

        # Try to use the inference engine if available
        try:
            from modelcypher.adapters.inference_engine import load_model_and_tokenizer

            model, _ = load_model_and_tokenizer(model_path)
            return model
        except ImportError:
            logger.warning("InferenceEngine not available, returning placeholder")
            return None
        except Exception as e:
            logger.warning("Failed to load model: %s", e)
            return None

    def _load_tokenizer(self, model_path: Path) -> Any:
        """Load tokenizer from model path."""
        logger.debug("Loading tokenizer from %s", model_path)

        try:
            from modelcypher.adapters.inference_engine import load_model_and_tokenizer

            _, tokenizer = load_model_and_tokenizer(model_path)
            return tokenizer
        except ImportError:
            logger.warning("InferenceEngine not available for tokenizer")
            return None
        except Exception as e:
            logger.warning("Failed to load tokenizer: %s", e)
            return None

    def _load_adapter(self, adapter_path: Path) -> dict[str, Any] | None:
        """Load adapter weights from path."""
        logger.debug("Loading adapter from %s", adapter_path)

        try:
            adapter_file = adapter_path / "adapter_model.safetensors"
            if adapter_file.exists():
                tensors = self._backend.load_safetensors(str(adapter_file))
                return tensors
        except Exception as e:
            logger.warning("Failed to load adapter: %s", e)

        return None


__all__ = [
    "SemanticProbeService",
    "SemanticVerificationConfig",
]
