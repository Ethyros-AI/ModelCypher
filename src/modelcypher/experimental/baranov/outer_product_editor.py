"""Naive outer-product weight editor for Baranov replication.

EXPERIMENTAL: Not validated for production use.

Implements ``EditApplicator`` via rank-1 outer-product updates to MLP
projection weights.  For each fact, collects the subject's hidden
representation at the target layer and adds an outer-product update
that maps the subject key toward the object embedding.

This is a minimal first implementation for testing Track B/C
infrastructure.  It is NOT a faithful MEMIT implementation -- that
requires constrained least-squares across layers with a Woodbury
factorization, which is deferred to a future patchset.

Original weights are stored for rollback.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from modelcypher.experimental.baranov.models import (
    EditState,
    EditStatus,
    FactTriple,
)

logger = logging.getLogger(__name__)


def _navigate_to_module(model: Any, path: str) -> Any:
    """Navigate a dotted path on a model to reach a sub-module.

    Handles numeric segments as list indices (e.g. ``layers.0``).
    """
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


class OuterProductEditor:
    """Naive rank-1 outer-product weight editor.

    For each target layer, computes a rank-1 update to the MLP down_proj
    (or specified projection) that pushes the subject's hidden
    representation toward the object's embedding.

    Satisfies the ``EditApplicator`` protocol.

    Parameters
    ----------
    backend:
        Backend instance (for hidden activation collection and tensor ops).
    tokenizer:
        Tokenizer for encoding object strings to embeddings.
    projection:
        Which MLP projection to edit within each layer.  Typically
        ``"down_proj"`` for Llama-style architectures.
    """

    def __init__(
        self,
        backend: Any,
        tokenizer: Any,
        projection: str = "down_proj",
    ) -> None:
        self._backend = backend
        self._tokenizer = tokenizer
        self._projection = projection
        # edit_id -> {layer_key: original_weight_array}
        self._snapshots: dict[str, dict[str, Any]] = {}

    def _get_weight_key(self, layer_id: int) -> str:
        """Build the dotted path to the target weight matrix."""
        return f"model.layers.{layer_id}.mlp.{self._projection}"

    def _get_weight(self, model: Any, layer_id: int) -> Any:
        """Read the current weight matrix for a layer."""
        module = _navigate_to_module(model, self._get_weight_key(layer_id))
        return module.weight

    def _snapshot_weights(
        self,
        model: Any,
        layer_ids: list[int],
    ) -> dict[str, Any]:
        """Copy current weights for the target layers (for rollback)."""
        snapshots: dict[str, Any] = {}
        for lid in layer_ids:
            key = self._get_weight_key(lid)
            weight = self._get_weight(model, lid)
            # In MLX, arrays are immutable -- ``weight + delta`` creates a
            # new array, so storing the current reference is a safe snapshot.
            snapshots[key] = weight
        return snapshots

    def _restore_weights(
        self,
        model: Any,
        snapshots: dict[str, Any],
    ) -> None:
        """Restore weights from a snapshot."""
        for key, original_weight in snapshots.items():
            module = _navigate_to_module(model, key)
            module.weight = original_weight

    def _collect_subject_keys(
        self,
        model: Any,
        facts: list[FactTriple],
        layer_ids: list[int],
    ) -> dict[int, Any]:
        """Collect last-token hidden states for fact subjects at target layers.

        Returns ``{layer_id: key_matrix}`` where ``key_matrix`` has shape
        ``[n_facts, hidden_dim]``.
        """
        prompts = [f"{fact.subject} {fact.relation}" for fact in facts]
        activations = self._backend.collect_hidden_activations(
            model,
            self._tokenizer,
            prompts,
            layer_indices=layer_ids,
        )
        # activations[layer_id] has shape [n_prompts, seq_len, hidden_dim]
        # Take last token: [:, -1, :]
        keys: dict[int, Any] = {}
        for lid in layer_ids:
            act = activations[lid]
            # last token across the batch
            keys[lid] = act[:, -1, :]
        return keys

    def _compute_object_targets(
        self,
        model: Any,
        facts: list[FactTriple],
    ) -> Any:
        """Compute target vectors for fact objects.

        Uses the model's embedding layer to get the object token embedding.
        For multi-token objects, averages the token embeddings.

        Returns array of shape ``[n_facts, hidden_dim]``.
        """
        embeddings = []
        for fact in facts:
            token_ids = self._tokenizer.encode(fact.object)
            if not token_ids:
                token_ids = self._tokenizer.encode(" " + fact.object)

            # Get embedding matrix from model
            base = getattr(model, "model", model)
            embed_layer = getattr(base, "embed_tokens", None)
            if embed_layer is None:
                raise RuntimeError(
                    "Cannot find embed_tokens on model — unsupported architecture",
                )

            # Look up embeddings for each token
            token_embeddings = []
            for tid in token_ids:
                token_embeddings.append(embed_layer.weight[tid])

            # Average across tokens
            stacked = self._backend.stack(token_embeddings)
            avg = self._backend.mean(stacked, axis=0)
            embeddings.append(avg)

        return self._backend.stack(embeddings)

    def apply_edit(
        self,
        facts: list[FactTriple],
        layer_ids: list[int],
        model: Any,
    ) -> EditState:
        """Apply outer-product edits to target layers.

        For each layer, computes:
            ΔW = Σ_i (v_i ⊗ k_i) / (k_i · k_i)

        where ``k_i`` is the subject's hidden state at that layer and
        ``v_i`` is the object's embedding target.

        Returns an ``EditState`` with status ``applied`` on success
        or ``failed`` on error.
        """
        edit_id = f"ope-{uuid.uuid4().hex[:12]}"
        fact_ids = tuple(f.fact_id for f in facts)
        layer_ids_tuple = tuple(layer_ids)

        try:
            # Snapshot for rollback
            snapshots = self._snapshot_weights(model, layer_ids)

            # Collect subject keys and object targets
            keys_by_layer = self._collect_subject_keys(
                model, facts, layer_ids,
            )
            targets = self._compute_object_targets(model, facts)

            metrics: dict[str, float] = {}

            for lid in layer_ids:
                keys = keys_by_layer[lid]  # [n_facts, hidden_dim]
                weight = self._get_weight(model, lid)

                # Compute ΔW = Σ_i v_i ⊗ k_i / (k_i · k_i)
                # targets: [n_facts, embed_dim], keys: [n_facts, hidden_dim]
                # For down_proj: weight shape is [out_dim, in_dim]
                # We want ΔW such that ΔW @ k ≈ v for each (k, v) pair
                # ΔW = V^T @ K @ (K^T @ K)^{-1} ... but rank-1 is simpler:
                # Just sum outer products with normalization
                delta = self._backend.zeros(weight.shape, dtype=weight.dtype)
                for i in range(len(facts)):
                    k_i = keys[i]  # [hidden_dim]
                    v_i = targets[i]  # [embed_dim]
                    k_dot_k = self._backend.sum(k_i * k_i)
                    # outer product: v_i[:, None] @ k_i[None, :]
                    v_col = self._backend.reshape(v_i, (-1, 1))
                    k_row = self._backend.reshape(k_i, (1, -1))
                    outer = self._backend.matmul(v_col, k_row)
                    # Normalize by ||k||^2 to make the update scale-invariant
                    delta = delta + outer / k_dot_k

                # Apply the update
                new_weight = weight + delta
                module = _navigate_to_module(
                    model, self._get_weight_key(lid),
                )
                module.weight = new_weight

                # Record per-layer metric: relative edit magnitude
                delta_norm = float(
                    self._backend.to_scalar(
                        self._backend.norm(delta),
                    ),
                )
                weight_norm = float(
                    self._backend.to_scalar(
                        self._backend.norm(weight),
                    ),
                )
                if weight_norm > 0:
                    metrics[f"relative_edit_norm_layer_{lid}"] = (
                        delta_norm / weight_norm
                    )
                else:
                    metrics[f"relative_edit_norm_layer_{lid}"] = 0.0

            # Store snapshot for rollback
            self._snapshots[edit_id] = snapshots

            logger.info(
                "Applied outer-product edit %s: %d facts across %d layers",
                edit_id,
                len(facts),
                len(layer_ids),
            )

            return EditState.from_metrics_dict(
                edit_id=edit_id,
                fact_ids=fact_ids,
                layer_ids=layer_ids_tuple,
                status=EditStatus.applied,
                metrics_dict=metrics,
            )

        except Exception:
            logger.exception("Edit application failed for %s", edit_id)
            return EditState.from_metrics_dict(
                edit_id=edit_id,
                fact_ids=fact_ids,
                layer_ids=layer_ids_tuple,
                status=EditStatus.failed,
                metrics_dict={},
            )

    def rollback_edit(
        self,
        edit_state: EditState,
        model: Any,
    ) -> EditState:
        """Rollback a previously applied edit by restoring original weights.

        Returns an ``EditState`` with status ``rolled_back``.

        Raises
        ------
        ValueError
            If no snapshot is stored for the given edit_id.
        """
        snapshots = self._snapshots.pop(edit_state.edit_id, None)
        if snapshots is None:
            raise ValueError(
                f"No snapshot found for edit {edit_state.edit_id}. "
                "Cannot rollback.",
            )

        self._restore_weights(model, snapshots)
        logger.info("Rolled back edit %s", edit_state.edit_id)
        return edit_state.transition_to(EditStatus.rolled_back)


__all__ = ["OuterProductEditor"]
