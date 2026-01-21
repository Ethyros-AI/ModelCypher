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

"""Consolidation Service - Orchestrate manifold completion and weight updates.

This module implements the "dreaming" phase of continual learning, where the
model consolidates experiences by filling sparse manifold regions.

The consolidation process:
    1. Collect sparse regions from the EntropyLearningBridge sparsity queue
    2. Generate probe embeddings covering the sparse regions
    3. Run ManifoldCompletion to fill in gaps
    4. Apply weight updates via KnowledgeEncoder
    5. Clear the sparsity queue

This is analogous to sleep-time compute in biological systems - the model
processes and consolidates information during idle periods.

Architecture:
    ┌─────────────────────────┐
    │  EntropyLearningBridge  │
    │  (Sparsity Queue)       │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │  ConsolidationService   │
    │  - collect_sparse()     │
    │  - generate_probes()    │
    │  - consolidate()        │
    └───────────┬─────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│ Manifold      │ │ Knowledge     │
│ Completion    │ │ Encoder       │
└───────────────┘ └───────────────┘

References:
    - Sleep-Time Compute (Letta 2025)
    - NeuroDream (Dec 2024) - 38% less forgetting via replay
    - TITANS (Google Dec 2024) - Test-time memorization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.continual.knowledge_encoder import KnowledgeEncoder
from modelcypher.core.domain.continual.manifold_completion import (
    CompletionStep,
    ManifoldCompletion,
    RetrievalFunction,
)
from modelcypher.core.domain.continual.null_space_tracker import NullSpaceTracker

if TYPE_CHECKING:
    from modelcypher.core.use_cases.entropy_learning_bridge import (
        EntropyLearningBridge,
        SparsityEvent,
    )
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.consolidation_service")


class ConsolidationStatus(str, Enum):
    """Status of consolidation process."""

    idle = "idle"
    collecting = "collecting"
    probing = "probing"
    completing = "completing"
    encoding = "encoding"
    done = "done"
    failed = "failed"


@dataclass
class ConsolidationStats:
    """Statistics from a consolidation run.

    Attributes
    ----------
    status : ConsolidationStatus
        Final status of the consolidation.
    sparsity_events_processed : int
        Number of sparsity events from the queue.
    probes_generated : int
        Number of probe embeddings generated.
    completion_steps : int
        Number of ManifoldCompletion iterations.
    encodings_applied : int
        Number of weight updates applied.
    mean_entropy_before : float
        Mean entropy before consolidation.
    mean_entropy_after : float
        Mean entropy after consolidation.
    entropy_reduction : float
        Total entropy reduction achieved.
    mean_preserved_fraction : float
        Mean preserved fraction for encodings.
    error_message : str | None
        Error message if failed.
    """

    status: ConsolidationStatus = ConsolidationStatus.idle
    sparsity_events_processed: int = 0
    probes_generated: int = 0
    completion_steps: int = 0
    encodings_applied: int = 0
    mean_entropy_before: float = 0.0
    mean_entropy_after: float = 0.0
    entropy_reduction: float = 0.0
    mean_preserved_fraction: float = 0.0
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "sparsity_events_processed": self.sparsity_events_processed,
            "probes_generated": self.probes_generated,
            "completion_steps": self.completion_steps,
            "encodings_applied": self.encodings_applied,
            "mean_entropy_before": self.mean_entropy_before,
            "mean_entropy_after": self.mean_entropy_after,
            "entropy_reduction": self.entropy_reduction,
            "mean_preserved_fraction": self.mean_preserved_fraction,
            "error_message": self.error_message,
        }


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation process.

    Attributes
    ----------
    max_probes : int
        Maximum number of probe embeddings to generate.
    max_completion_steps : int
        Maximum ManifoldCompletion iterations.
    clear_queue_after : bool
        Whether to clear sparsity queue after consolidation.
    """

    max_probes: int = 100
    max_completion_steps: int = 50
    clear_queue_after: bool = True


class ConsolidationService:
    """Orchestrates manifold completion and knowledge encoding.

    The consolidation service is the "dreaming" phase - it processes
    accumulated sparsity events and fills in manifold gaps.

    Parameters
    ----------
    model : Any
        The language model to consolidate.
    null_space_tracker : NullSpaceTracker
        Tracker for null-space availability.
    backend : Backend, optional
        Compute backend.

    Examples
    --------
    Basic usage after inference:

        # During inference, sparsity events accumulate in bridge
        bridge = EntropyLearningBridge(hidden_dim=576)
        # ... inference loop ...

        # Run consolidation
        service = ConsolidationService(model, tracker)
        stats = service.consolidate_from_bridge(bridge)

        print(f"Entropy reduced by {stats.entropy_reduction:.3f}")
        print(f"Applied {stats.encodings_applied} weight updates")

    Manual consolidation with custom probes:

        service = ConsolidationService(model, tracker)

        # Generate probes from a specific domain
        probes = service.generate_probes_from_text(
            texts=["Paris is the capital of France", "Tokyo is in Japan"],
            tokenizer=tokenizer,
        )

        # Run consolidation
        for step in service.consolidate_stream(probes):
            print(f"Step {step.iteration}: entropy {step.final_entropy:.3f}")
    """

    def __init__(
        self,
        model: Any,
        null_space_tracker: NullSpaceTracker,
        backend: "Backend",
        knowledge_retrieval_fn: "RetrievalFunction | None" = None,
    ) -> None:
        self._backend = backend
        self._model = model
        self._tracker = null_space_tracker
        self._retrieval_fn = knowledge_retrieval_fn

        # Initialize sub-components
        self._encoder = KnowledgeEncoder(
            model=model,
            null_space_tracker=null_space_tracker,
            backend=self._backend,
        )

        # ManifoldCompletion requires encoder
        # Pass through knowledge retrieval function for external knowledge injection
        self._completion = ManifoldCompletion(
            model=model,
            null_space_tracker=null_space_tracker,
            knowledge_encoder=self._encoder,
            backend=self._backend,
            knowledge_retrieval_fn=knowledge_retrieval_fn,
        )

        # State
        self._status = ConsolidationStatus.idle
        self._last_stats = ConsolidationStats()

    def consolidate_from_bridge(
        self,
        bridge: "EntropyLearningBridge",
        config: ConsolidationConfig | None = None,
    ) -> ConsolidationStats:
        """Run consolidation using sparsity events from the bridge.

        Parameters
        ----------
        bridge : EntropyLearningBridge
            Bridge containing sparsity events to process.
        config : ConsolidationConfig, optional
            Configuration for the consolidation.

        Returns
        -------
        ConsolidationStats
            Statistics from the consolidation run.
        """
        config = config or ConsolidationConfig()
        stats = ConsolidationStats(status=ConsolidationStatus.collecting)

        try:
            # Collect sparsity events
            events = bridge.get_sparsity_queue()
            stats.sparsity_events_processed = len(events)

            if not events:
                stats.status = ConsolidationStatus.done
                logger.info("No sparsity events to process")
                return stats

            logger.info("Processing %d sparsity events", len(events))

            # Generate probe embeddings from sparsity events
            stats.status = ConsolidationStatus.probing
            probes = self._generate_probes_from_events(events, config.max_probes)
            stats.probes_generated = int(probes.shape[0])

            if stats.probes_generated == 0:
                stats.status = ConsolidationStatus.done
                logger.info("No probes generated")
                return stats

            logger.info("Generated %d probes", stats.probes_generated)

            # Compute initial entropy
            coverage_before = self._completion.estimate_completion_coverage(probes)
            stats.mean_entropy_before = 1.0 - coverage_before["mean_density"]

            # Run manifold completion
            stats.status = ConsolidationStatus.completing
            total_preserved = 0.0
            encodings = 0

            for step in self._completion.complete(probes):
                stats.completion_steps += 1

                if step.encoding_applied:
                    encodings += 1
                    # Track preservation (proxy from entropy reduction)
                    if step.entropy_reduction > 0:
                        total_preserved += step.entropy_reduction

                if step.converged:
                    logger.info(
                        "Converged at step %d, entropy %.4f",
                        step.iteration,
                        step.final_entropy,
                    )
                    break

                if stats.completion_steps >= config.max_completion_steps:
                    logger.info(
                        "Reached max steps %d", config.max_completion_steps
                    )
                    break

            stats.encodings_applied = encodings

            # Compute final entropy
            coverage_after = self._completion.estimate_completion_coverage(probes)
            stats.mean_entropy_after = 1.0 - coverage_after["mean_density"]
            stats.entropy_reduction = (
                stats.mean_entropy_before - stats.mean_entropy_after
            )

            # Compute mean preserved fraction
            if encodings > 0:
                encoder_stats = self._encoder.get_stats()
                stats.mean_preserved_fraction = encoder_stats.get(
                    "average_preserved_fraction", 0.0
                )

            # Clear queue if configured
            if config.clear_queue_after:
                cleared = bridge.clear_sparsity_queue()
                logger.info("Cleared %d events from sparsity queue", cleared)

            stats.status = ConsolidationStatus.done
            logger.info(
                "Consolidation complete: %d steps, %d encodings, entropy %.4f -> %.4f",
                stats.completion_steps,
                stats.encodings_applied,
                stats.mean_entropy_before,
                stats.mean_entropy_after,
            )

        except Exception as e:
            stats.status = ConsolidationStatus.failed
            stats.error_message = str(e)
            logger.error("Consolidation failed: %s", e)

        self._last_stats = stats
        self._status = stats.status
        return stats

    def consolidate_stream(
        self,
        probe_embeddings: "Array",
        max_steps: int | None = None,
    ) -> Iterator[CompletionStep]:
        """Run consolidation as a streaming iterator.

        Parameters
        ----------
        probe_embeddings : Array
            Probe embeddings [n_probes, hidden_dim].
        max_steps : int, optional
            Maximum completion steps.

        Yields
        ------
        CompletionStep
            Each step of the manifold completion.
        """
        self._status = ConsolidationStatus.completing

        step_count = 0
        for step in self._completion.complete(probe_embeddings):
            yield step
            step_count += 1

            if step.converged:
                break

            if max_steps is not None and step_count >= max_steps:
                break

        self._status = ConsolidationStatus.done

    def _generate_probes_from_events(
        self,
        events: list["SparsityEvent"],
        max_probes: int,
    ) -> "Array":
        """Generate probe embeddings from sparsity events.

        Uses the model to generate embeddings for synthetic probes
        around the sparse regions.
        """
        b = self._backend

        if not events:
            return b.zeros((0, self._tracker.hidden_dim))

        # Collect unique hidden state hashes to avoid duplicates
        seen_hashes: set[int] = set()
        unique_events: list["SparsityEvent"] = []

        for event in events:
            if event.hidden_state_hash not in seen_hashes:
                seen_hashes.add(event.hidden_state_hash)
                unique_events.append(event)
                if len(unique_events) >= max_probes:
                    break

        # Generate embeddings
        # For now, use random perturbations around the eigenscore values
        # This is a simplified approach - production would use actual hidden states
        n_probes = min(len(unique_events), max_probes)
        hidden_dim = self._tracker.hidden_dim

        # Create probe embeddings with structure based on eigenscore
        probes_list = []
        for i, event in enumerate(unique_events[:n_probes]):
            # Base embedding: random with variance proportional to eigenscore
            # Higher eigenscore = more sparse = more variance
            scale = max(0.1, event.eigenscore)
            probe = b.random_normal((hidden_dim,)) * scale

            # Add structure based on refusal projection
            # This helps the completion algorithm understand the region type
            if event.refusal_projection > 0.5:
                # High refusal - add signal in early dimensions
                adjustment = b.zeros((hidden_dim,))
                # Set first 10% of dimensions
                n_refusal_dims = max(1, hidden_dim // 10)
                refusal_signal = b.ones((n_refusal_dims,)) * event.refusal_projection
                probe = probe + b.concatenate(
                    [refusal_signal, b.zeros((hidden_dim - n_refusal_dims,))],
                    axis=0,
                )

            b.eval(probe)
            probes_list.append(probe)

        if not probes_list:
            return b.zeros((0, hidden_dim))

        probes = b.stack(probes_list, axis=0)
        b.eval(probes)
        return probes

    def generate_probes_from_activations(
        self,
        activations: "Array",
        n_augment: int = 0,
    ) -> "Array":
        """Generate probes from actual activation vectors.

        Parameters
        ----------
        activations : Array
            Activation vectors [n_samples, hidden_dim].
        n_augment : int
            Number of augmented probes to add via interpolation.

        Returns
        -------
        Array
            Probe embeddings [n_probes, hidden_dim].
        """
        b = self._backend
        n = int(activations.shape[0])

        if n_augment == 0:
            return activations

        # Add interpolated probes between existing points
        augmented = [activations]

        for _ in range(n_augment):
            # Random pairs
            idx1 = b.randint(0, n, (1,))
            idx2 = b.randint(0, n, (1,))
            b.eval(idx1, idx2)

            p1 = b.take(activations, idx1, axis=0)
            p2 = b.take(activations, idx2, axis=0)

            # Random interpolation factor
            alpha = b.random_uniform((1, 1))
            b.eval(alpha)
            interp = p1 * alpha + p2 * (1.0 - alpha)
            b.eval(interp)
            augmented.append(interp)

        result = b.concatenate(augmented, axis=0)
        b.eval(result)
        return result

    def get_status(self) -> ConsolidationStatus:
        """Get current consolidation status."""
        return self._status

    def get_last_stats(self) -> ConsolidationStats:
        """Get statistics from last consolidation run."""
        return self._last_stats

    def get_null_space_summary(self) -> dict[str, Any]:
        """Get summary of null-space availability across layers."""
        model_state = self._tracker.get_model_state()
        return {
            "total_layers": self._tracker.n_layers,
            "hidden_dim": self._tracker.hidden_dim,
            "average_null_rank": model_state.null_rank,
            "average_used_rank": model_state.used_rank,
            "capacity_fraction": model_state.capacity_fraction,
            "total_variance": model_state.total_variance,
            "null_variance": model_state.null_variance,
        }

    def reset(self) -> None:
        """Reset consolidation state."""
        self._status = ConsolidationStatus.idle
        self._completion.reset()
        self._encoder.reset_stats()


def create_consolidation_service(
    model: Any,
    n_layers: int,
    hidden_dim: int,
    backend: Backend,
    knowledge_retrieval_fn: RetrievalFunction | None = None,
) -> ConsolidationService:
    """Create a consolidation service for a model.

    Parameters
    ----------
    model : Any
        The language model.
    n_layers : int
        Number of transformer layers.
    hidden_dim : int
        Hidden dimension.
    backend : Backend
        Compute backend.
    knowledge_retrieval_fn : RetrievalFunction, optional
        Function to query external knowledge sources during consolidation.
        When provided, consolidation blends local geometry with external attractors.
        Signature: (sparse_embedding, neighbor_indices) -> (attractor, confidence) | None

    Returns
    -------
    ConsolidationService
        Configured consolidation service.
    """
    tracker = NullSpaceTracker(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        backend=backend,
    )
    return ConsolidationService(
        model=model,
        null_space_tracker=tracker,
        backend=backend,
        knowledge_retrieval_fn=knowledge_retrieval_fn,
    )
