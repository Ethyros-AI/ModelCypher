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

"""
Token Labeling Service.

Provides CLI-consumable operations for SAE-based token labeling,
implementing the pipeline from arXiv:2601.21571v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from modelcypher.experimental.interpretability.sae import (
    SAEConfig,
    SAEWeights,
    SparseAutoencoder,
)
from modelcypher.experimental.interpretability.token_labeling import (
    LatentActivationStats,
    SAETokenLabeler,
    TokenLabelingConfig,
    TokenLabelResult,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class LabelRunResult:
    """Result of a token labeling run.

    Attributes
    ----------
    total_tokens : int
        Total number of tokens processed.
    positive_tokens : int
        Number of tokens labeled positive.
    positive_rate : float
        Fraction of tokens labeled positive.
    texts_processed : int
        Number of texts processed.
    output_path : str | None
        Path where results were saved, if any.
    """

    total_tokens: int
    positive_tokens: int
    positive_rate: float
    texts_processed: int
    output_path: str | None


@dataclass(frozen=True)
class CalibrationResult:
    """Result of threshold calibration.

    Attributes
    ----------
    calibrated_sigma : float
        Calibrated sigma threshold.
    achieved_positive_rate : float
        Actual positive rate achieved at calibrated threshold.
    target_positive_rate : float
        Target positive rate.
    sample_count : int
        Number of tokens used for calibration.
    """

    calibrated_sigma: float
    achieved_positive_rate: float
    target_positive_rate: float
    sample_count: int


class TokenLabelingService:
    """Service for SAE-based token labeling operations.

    Provides methods for:
    - Running token labeling on text data
    - Calibrating thresholds for target positive rates
    - Loading/saving SAE weights and latent indices
    """

    def __init__(self, backend: "Backend") -> None:
        """Initialize service.

        Parameters
        ----------
        backend : Backend
            Computation backend.
        """
        self._backend = backend

    def run_labeling(
        self,
        sae_activations: "Array",
        domain_latent_indices: list[int],
        text_lengths: list[int],
        stats: LatentActivationStats | None = None,
        config: TokenLabelingConfig | None = None,
        output_path: str | None = None,
    ) -> tuple[LabelRunResult, TokenLabelResult]:
        """Run token labeling on pre-computed SAE activations.

        Parameters
        ----------
        sae_activations : Array
            SAE latent activations. Shape: [total_tokens, latent_dim].
        domain_latent_indices : list[int]
            Indices of latents associated with target domain.
        text_lengths : list[int]
            Length of each text in tokens.
        stats : LatentActivationStats, optional
            Pre-computed activation statistics. Computed if None.
        config : TokenLabelingConfig, optional
            Labeling configuration. Uses defaults if None.
        output_path : str, optional
            Path to save results as JSONL.

        Returns
        -------
        tuple[LabelRunResult, TokenLabelResult]
            Summary and detailed results.
        """
        b = self._backend

        labeler = SAETokenLabeler(config=config, backend=b)

        # Compute stats if not provided
        if stats is None:
            stats = labeler.compute_activation_stats(sae_activations)

        # Run labeling
        result = labeler.label_tokens(
            activations=sae_activations,
            domain_latent_indices=domain_latent_indices,
            stats=stats,
            text_lengths=text_lengths,
        )

        # Compute summary stats
        labels_sum = b.sum(result.labels)
        b.eval(labels_sum)
        positive_count = int(b.to_scalar(labels_sum))
        total_tokens = int(result.labels.shape[0])
        positive_rate = positive_count / max(total_tokens, 1)

        # Save results if output path provided
        if output_path:
            self._save_results(result, text_lengths, output_path)

        summary = LabelRunResult(
            total_tokens=total_tokens,
            positive_tokens=positive_count,
            positive_rate=positive_rate,
            texts_processed=len(text_lengths),
            output_path=output_path,
        )

        return summary, result

    def calibrate(
        self,
        sae_activations: "Array",
        domain_latent_indices: list[int],
        target_positive_rate: float = 0.1,
        config: TokenLabelingConfig | None = None,
    ) -> CalibrationResult:
        """Calibrate threshold to achieve target positive rate.

        Parameters
        ----------
        sae_activations : Array
            SAE latent activations. Shape: [total_tokens, latent_dim].
        domain_latent_indices : list[int]
            Indices of latents associated with target domain.
        target_positive_rate : float
            Target fraction of tokens to label positive.
        config : TokenLabelingConfig, optional
            Base configuration. Uses defaults if None.

        Returns
        -------
        CalibrationResult
            Calibration results.
        """
        b = self._backend
        labeler = SAETokenLabeler(config=config, backend=b)

        calibrated_sigma = labeler.calibrate_threshold(
            activations=sae_activations,
            domain_latent_indices=domain_latent_indices,
            target_positive_rate=target_positive_rate,
        )

        # Verify achieved rate
        calibrated_config = TokenLabelingConfig(
            min_active_latents=labeler.config.min_active_latents,
            activation_threshold_sigma=calibrated_sigma,
            expand_adjacent=False,  # Don't expand for rate calculation
            expansion_radius=labeler.config.expansion_radius,
        )
        verify_labeler = SAETokenLabeler(config=calibrated_config, backend=b)
        stats = verify_labeler.compute_activation_stats(sae_activations)

        # Create dummy text_lengths (one big text)
        n_tokens = int(sae_activations.shape[0])
        result = verify_labeler.label_tokens(
            activations=sae_activations,
            domain_latent_indices=domain_latent_indices,
            stats=stats,
            text_lengths=[n_tokens],
        )

        labels_sum = b.sum(result.labels)
        b.eval(labels_sum)
        positive_count = int(b.to_scalar(labels_sum))
        achieved_rate = positive_count / max(n_tokens, 1)

        return CalibrationResult(
            calibrated_sigma=calibrated_sigma,
            achieved_positive_rate=achieved_rate,
            target_positive_rate=target_positive_rate,
            sample_count=n_tokens,
        )

    def encode_texts_with_sae(
        self,
        model_activations: "Array",
        sae_weights: SAEWeights,
    ) -> "Array":
        """Encode model activations through an SAE.

        Parameters
        ----------
        model_activations : Array
            Model hidden state activations. Shape: [total_tokens, hidden_dim].
        sae_weights : SAEWeights
            Trained SAE weights.

        Returns
        -------
        Array
            SAE latent activations. Shape: [total_tokens, latent_dim].
        """
        sae = SparseAutoencoder(config=sae_weights.config, backend=self._backend)
        result = sae.encode(model_activations, sae_weights)
        return result.sparse_codes

    def load_sae_weights(self, path: str) -> SAEWeights:
        """Load SAE weights from file.

        Parameters
        ----------
        path : str
            Path to SAE weights JSON file.

        Returns
        -------
        SAEWeights
            Loaded weights.
        """
        b = self._backend
        data = json.loads(Path(path).read_text())

        config = SAEConfig(
            hidden_dim=data["config"]["hidden_dim"],
            expansion_factor=data["config"]["expansion_factor"],
            sparsity_coefficient=data["config"].get("sparsity_coefficient"),
            normalize_decoder=data["config"].get("normalize_decoder", True),
            tied_weights=data["config"].get("tied_weights", False),
        )

        return SAEWeights(
            W_enc=b.array(data["W_enc"]),
            b_enc=b.array(data["b_enc"]),
            W_dec=b.array(data["W_dec"]),
            b_dec=b.array(data["b_dec"]),
            config=config,
        )

    def load_domain_latents(self, path: str) -> list[int]:
        """Load domain latent indices from file.

        Parameters
        ----------
        path : str
            Path to JSON file containing latent indices.

        Returns
        -------
        list[int]
            Domain latent indices.
        """
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            return [int(x) for x in data]
        if isinstance(data, dict) and "indices" in data:
            return [int(x) for x in data["indices"]]
        raise ValueError(f"Invalid domain latents file format: {path}")

    def save_calibration(self, result: CalibrationResult, path: str) -> None:
        """Save calibration results to file.

        Parameters
        ----------
        result : CalibrationResult
            Calibration results.
        path : str
            Output path.
        """
        data = {
            "calibrated_sigma": result.calibrated_sigma,
            "achieved_positive_rate": result.achieved_positive_rate,
            "target_positive_rate": result.target_positive_rate,
            "sample_count": result.sample_count,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def _save_results(
        self,
        result: TokenLabelResult,
        text_lengths: list[int],
        output_path: str,
    ) -> None:
        """Save labeling results to JSONL file.

        Each line is a JSON object for one text with its token labels.
        """
        b = self._backend
        b.eval(result.labels, result.confidence_scores, result.active_latent_counts)

        labels_list = [int(x) for x in b.tolist(result.labels)]
        confidence_list = [float(x) for x in b.tolist(result.confidence_scores)]
        counts_list = [int(x) for x in b.tolist(result.active_latent_counts)]

        with open(output_path, "w") as f:
            offset = 0
            for text_idx, length in enumerate(text_lengths):
                text_labels = labels_list[offset : offset + length]
                text_confidence = confidence_list[offset : offset + length]
                text_counts = counts_list[offset : offset + length]

                record = {
                    "text_index": text_idx,
                    "token_count": length,
                    "positive_count": sum(text_labels),
                    "labels": text_labels,
                    "confidence": text_confidence,
                    "active_latent_counts": text_counts,
                }
                f.write(json.dumps(record) + "\n")
                offset += length

    @staticmethod
    def label_run_payload(result: LabelRunResult) -> dict:
        """Convert label run result to CLI payload."""
        return {
            "totalTokens": result.total_tokens,
            "positiveTokens": result.positive_tokens,
            "positiveRate": result.positive_rate,
            "textsProcessed": result.texts_processed,
            "outputPath": result.output_path,
        }

    @staticmethod
    def calibration_payload(result: CalibrationResult) -> dict:
        """Convert calibration result to CLI payload."""
        return {
            "calibratedSigma": result.calibrated_sigma,
            "achievedPositiveRate": result.achieved_positive_rate,
            "targetPositiveRate": result.target_positive_rate,
            "sampleCount": result.sample_count,
        }


__all__ = [
    "TokenLabelingService",
    "LabelRunResult",
    "CalibrationResult",
]
