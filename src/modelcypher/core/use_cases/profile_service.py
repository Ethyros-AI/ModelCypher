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

"""Service for computing and managing geometric profiles.

**Profile once, merge many.**

This service computes the geometric profile of a model by running probe
inference and measuring activation geometry. The profile is stored alongside
the model and used by the merge pipeline.

Usage:
    service = ProfileService(backend, model_loader, activation_provider)

    # Compute profile for a model
    profile = service.compute_profile("/path/to/model")

    # Load existing profile
    profile = service.load_profile("/path/to/model")

    # Check if profile exists and is valid
    if service.profile_exists("/path/to/model"):
        profile = service.load_profile("/path/to/model")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


from modelcypher.core.domain.profile import (
    ConvergenceMetrics,
    GeometricProfile,
    GeometricProfileStore,
    LayerGeometricProfile,
    ProfileActivations,
    compute_weights_hash,
    load_activations,
    save_activations,
)

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Result of profile computation."""

    profile: GeometricProfile
    profile_dir: Path
    layers_profiled: int
    probes_processed: int
    probes_failed: int
    from_cache: bool = False


class ProfileService:
    """Service for computing and managing geometric profiles.

    This service is the single entry point for all profiling operations:
    - Compute new profiles via probe inference
    - Load existing profiles from storage
    - Validate profile freshness against model weights
    """

    def __init__(
        self,
        backend: "Backend",
        model_loader: "ModelLoaderPort | None" = None,
        activation_provider: "ActivationProvider | None" = None,
        store: GeometricProfileStore | None = None,
    ) -> None:
        """Initialize the profile service.

        Args:
            backend: Compute backend
            model_loader: Model loader port for loading models
            activation_provider: Activation provider for collecting activations
            store: Profile store (defaults to standard paths)
        """
        self._backend = backend
        self._model_loader = model_loader
        self._activation_provider = activation_provider
        self._store = store or GeometricProfileStore()

    def profile_exists(self, model_path: str | Path) -> bool:
        """Check if a valid profile exists for a model."""
        return self._store.exists(model_path)

    def load_profile(self, model_path: str | Path) -> GeometricProfile | None:
        """Load an existing profile for a model.

        Returns None if no valid profile exists.
        """
        return self._store.load(model_path)

    def load_activations(self, model_path: str | Path) -> ProfileActivations:
        """Load all activation types for a model.

        Args:
            model_path: Path to model directory

        Returns:
            ProfileActivations containing all stored activation types

        Raises:
            FileNotFoundError: If no profile or activations exist
        """
        profile = self.load_profile(model_path)
        if profile is None:
            raise FileNotFoundError(f"No profile found for {model_path}")

        if not profile.has_activations:
            raise FileNotFoundError(f"Profile exists but has no activations for {model_path}")

        # Determine profile directory
        profile_dir = self._store.profile_dir_for_model(model_path)
        if not (profile_dir / profile.activations_file).exists():
            profile_dir = self._store.central_profile_dir(model_path)

        return load_activations(profile_dir, self._backend)

    def compute_profile(
        self,
        model_path: str | Path,
        force: bool = False,
        probe_mode: str = "atlas",
        max_batches: int | None = None,
        full: bool = True,
    ) -> ProfileResult:
        """Compute a geometric profile for a model using trajectory-based manifold mapping.

        This is the main entry point for profiling. It:
        1. Checks for existing valid profile (unless force=True)
        2. Loads the model and tokenizer
        3. Uses ManifoldMapper with domain-stratified sampling
        4. Runs until rank saturation (geometric termination)
        5. Optionally collects intermediate/gate activations (full=True)
        6. Saves the profile and activations

        The key improvement over per-probe profiling:
        - A 100-token text yields 199 samples (100 positions + 99 velocities) vs 1
        - Domain-stratified sampling ensures coverage of all 15 atlas domains
        - Rank saturation detection provides geometric termination
        - 20x more samples per forward pass

        Args:
            model_path: Path to model directory
            force: If True, recompute even if valid profile exists
            probe_mode: Probe mode ("atlas" or "atlas_full")
            max_batches: Optional maximum batches for testing (None = no limit)
            full: If True, collect ALL activation types (intermediate, gate, embedding)
                  for profile-based merging. Default False for backward compatibility.

        Returns:
            ProfileResult with computed profile

        Raises:
            ValueError: If model_path is invalid
            RuntimeError: If profile computation fails
        """
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Check for existing profile
        if not force:
            existing = self._store.load(model_path)
            if existing is not None:
                logger.info("Using existing profile for %s", model_path)
                profile_dir = self._store.profile_dir_for_model(model_path)
                return ProfileResult(
                    profile=existing,
                    profile_dir=profile_dir,
                    layers_profiled=len(existing.layer_profiles),
                    probes_processed=existing.probe_count,
                    probes_failed=existing.convergence.probes_failed,
                    from_cache=True,
                )

        # Ensure we have required components
        if self._model_loader is None:
            raise RuntimeError("ProfileService requires model_loader for compute_profile")
        if self._activation_provider is None:
            raise RuntimeError("ProfileService requires activation_provider for compute_profile")

        logger.info("PROFILE: Computing trajectory-based manifold map for %s", model_path)

        # Load model and tokenizer
        model, tokenizer = self._model_loader.load_model_for_training(str(model_path))

        # Load atlas probes for domain-stratified sampling
        from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory

        probes = UnifiedAtlasInventory.all_probes()
        unique_domains = len({p.domain for p in probes})
        logger.info("PROFILE: Loaded %d atlas probes across %d domains", len(probes), unique_domains)

        # Create ManifoldMapper and run trajectory-based profiling
        from modelcypher.core.use_cases.manifold_mapper import ManifoldMapper

        mapper = ManifoldMapper(
            backend=self._backend,
            activation_provider=self._activation_provider,
        )

        map_result = mapper.map_manifold(
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            max_batches=max_batches,
        )

        # Convert ManifoldMapResult to GeometricProfile
        profile = self._convert_map_result_to_profile(
            map_result=map_result,
            model_path=str(model_path),
            model=model,
        )

        # Save profile
        profile_dir = self._store.save(profile, model_path)

        # Collect additional activation types if full profile requested
        intermediate_activations: dict[int, Any] = {}
        gate_activations: dict[int, Any] = {}
        embedding_activations: Any | None = None

        if full:
            logger.info("PROFILE: Collecting full activation types (intermediate, gate, embedding)...")
            intermediate_activations, gate_activations, embedding_activations = (
                self._collect_full_activations(
                    model=model,
                    tokenizer=tokenizer,
                    probes=probes,
                    max_probes=min(len(probes), 500),  # Limit for memory efficiency
                )
            )

        # Save activations (hidden + optional intermediate/gate/embedding + mean_pooled)
        save_activations(
            map_result.positions,
            embedding_activations,
            profile_dir,
            self._backend,
            intermediate_activations=intermediate_activations,
            gate_activations=gate_activations,
            mean_pooled_activations=map_result.mean_pooled,
        )

        # Update profile to indicate which activations are saved
        profile.has_activations = True
        profile.has_hidden = True
        profile.has_intermediate = bool(intermediate_activations)
        profile.has_gate = bool(gate_activations)
        profile.has_embedding = embedding_activations is not None

        # Store probe metadata for merge consistency
        # This ensures profile-based merges produce identical results to probe-based
        profile.probe_ids = map_result.probe_ids
        profile.probe_domains = map_result.probe_domains

        profile.save(profile_dir)

        logger.info(
            "PROFILE COMPLETE: %d layers, %d probes, %d batches, %s saturated, saved to %s",
            len(profile.layer_profiles),
            map_result.total_probes_processed,
            map_result.total_batches,
            "all" if map_result.all_layers_saturated else "partial",
            profile_dir,
        )

        return ProfileResult(
            profile=profile,
            profile_dir=profile_dir,
            layers_profiled=len(profile.layer_profiles),
            probes_processed=map_result.total_probes_processed,
            probes_failed=0,  # ManifoldMapper handles failures internally
            from_cache=False,
        )

    def _convert_map_result_to_profile(
        self,
        map_result: Any,  # ManifoldMapResult
        model_path: str,
        model: Any,
    ) -> GeometricProfile:
        """Convert ManifoldMapResult to GeometricProfile.

        This bridges the new trajectory-based mapping to the existing profile format.
        """
        from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
            compute_numerical_rank,
        )
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
        from modelcypher.core.use_cases.manifold_mapper import ManifoldMapResult

        assert isinstance(map_result, ManifoldMapResult)

        # Extract model dimensions from config
        hidden_dim = 0
        intermediate_dim = 0
        vocab_size = 0
        num_attention_heads = 0
        num_kv_heads = 0

        if hasattr(model, "config"):
            config = model.config
            hidden_dim = getattr(config, "hidden_size", 0)
            intermediate_dim = getattr(config, "intermediate_size", 0)
            vocab_size = getattr(config, "vocab_size", 0)
            num_attention_heads = getattr(config, "num_attention_heads", 0)
            num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # Convert ManifoldMapper profiles to LayerGeometricProfile
        layer_profiles: dict[int, LayerGeometricProfile] = {}
        for layer_idx, mp in map_result.profiles.items():
            layer_profiles[layer_idx] = LayerGeometricProfile(
                layer_idx=layer_idx,
                activation_rank=mp.activation_rank,
                trajectory_rank=mp.trajectory_rank,
                gram_condition=mp.gram_condition,
                signal_rank=mp.activation_rank,  # Upper bound (includes noise dims)
                hidden_dim=mp.hidden_dim,
                n_probes=mp.probes_processed,
                null_rank=mp.null_rank,
                weight_rank_o_proj=mp.weight_rank_o_proj,
                weight_rank_down_proj=mp.weight_rank_down_proj,
                structural_capacity=mp.structural_capacity,
                trajectory_samples=mp.total_samples,
                position_samples=mp.position_samples,
                velocity_samples=mp.velocity_samples,
                domains_sampled=list(mp.domains_sampled),
                batches_to_saturation=mp.batches_to_saturation,
                saturated=mp.saturated,
            )

            # Log probe-limited layers
            if layer_profiles[layer_idx].is_probe_limited:
                logger.info(
                    "PROFILE: Layer %d is PROBE-LIMITED: activation=%d < capacity=%d",
                    layer_idx,
                    mp.activation_rank,
                    mp.structural_capacity,
                )

            # Update hidden_dim from actual profile if not set
            if hidden_dim == 0:
                hidden_dim = mp.hidden_dim

        # Build convergence metrics
        convergence = ConvergenceMetrics(
            probes_processed=map_result.total_probes_processed,
            probes_failed=0,
            total_batches=map_result.total_batches,
            all_layers_saturated=map_result.all_layers_saturated,
            domains_covered=list(map_result.domains_covered),
        )

        # Populate per-layer convergence metrics
        for layer_idx, mp in map_result.profiles.items():
            convergence.final_rank[layer_idx] = mp.activation_rank
            convergence.trajectory_rank[layer_idx] = mp.trajectory_rank
            convergence.ceiling_achieved[layer_idx] = mp.saturated
            convergence.batches_to_saturation[layer_idx] = mp.batches_to_saturation

        # Compute embedding geometry from trajectory data
        embedding_rank = 0
        embedding_gram_condition = 0.0
        embedding_n_probes = 0

        b = self._backend
        # Prefer embedding_positions (full trajectory) if available, else use mean_pooled
        emb_data = map_result.embedding_positions
        if emb_data is None and map_result.embedding_mean_pooled:
            # Stack mean-pooled embeddings into matrix
            emb_data = b.stack(map_result.embedding_mean_pooled, axis=0)
            b.eval(emb_data)

        if emb_data is not None:
            embedding_n_probes = int(b.shape(emb_data)[0])
            embedding_rank, _ = compute_numerical_rank(emb_data, b)

            # Compute Gram condition for embeddings
            try:
                emb_gram = b.matmul(emb_data, b.transpose(emb_data))
                b.eval(emb_gram)
                emb_s = b.svd(emb_gram, full_matrices=False)[1]
                b.eval(emb_s)
                emb_s_max = b.max(emb_s)
                emb_s_min = b.min(emb_s)
                b.eval(emb_s_max, emb_s_min)
                eps = machine_epsilon(b, emb_data)
                emb_s_min_safe = emb_s_min + eps
                emb_cond_arr = emb_s_max / emb_s_min_safe
                b.eval(emb_cond_arr)
                embedding_gram_condition = float(b.to_scalar(emb_cond_arr))
            except Exception:
                embedding_gram_condition = float("inf")

            logger.info(
                "PROFILE: Embedding trajectory - rank=%d, condition=%.2e, samples=%d",
                embedding_rank,
                embedding_gram_condition,
                embedding_n_probes,
            )

        return GeometricProfile(
            model_path=model_path,
            weights_hash=compute_weights_hash(model_path),
            probe_count=map_result.total_probes_processed,
            probe_ids=[],  # Trajectory-based doesn't track individual probe IDs
            probe_domains=list(map_result.domains_covered),
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=len(layer_profiles),
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_profiles=layer_profiles,
            embedding_rank=embedding_rank,
            embedding_gram_condition=embedding_gram_condition,
            embedding_n_probes=embedding_n_probes,
            convergence=convergence,
        )

    def _collect_full_activations(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[Any],
        max_probes: int = 500,
    ) -> tuple[dict[int, "Array"], dict[int, "Array"], "Array | None"]:
        """Collect intermediate, gate, and embedding activations for full profile.

        This uses the same activation collection infrastructure as the merge pipeline
        to ensure consistency. We limit to max_probes for memory efficiency.

        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            probes: List of probe objects
            max_probes: Maximum number of probes to process

        Returns:
            Tuple of (intermediate_activations, gate_activations, embedding_activations)
        """
        assert self._activation_provider is not None

        # Limit probes for memory efficiency
        selected_probes = probes[:max_probes]
        probe_texts = [p.text if hasattr(p, "text") else str(p) for p in selected_probes]
        valid_probes = [(p, text) for p, text in zip(selected_probes, probe_texts)]

        if not valid_probes:
            return {}, {}, None

        logger.info(
            "PROFILE FULL: Collecting intermediate/gate/embedding from %d probes...",
            len(valid_probes),
        )

        # Use single model probe inference from merge pipeline
        from modelcypher.experimental.merge.stages.probe_inference import (
            run_single_model_probe_inference,
        )

        result = run_single_model_probe_inference(
            valid_probes=valid_probes,
            model=model,
            tokenizer=tokenizer,
            activation_provider=self._activation_provider,
            backend=self._backend,
            model_label="profile",
        )

        # Stack embedding activations if available
        embedding_stacked: "Array | None" = None
        if result.embedding:
            embedding_stacked = self._backend.stack(result.embedding, axis=0)
            self._backend.eval(embedding_stacked)

        logger.info(
            "PROFILE FULL: Collected intermediate=%d, gate=%d, embedding=%s",
            len(result.intermediate),
            len(result.gate),
            embedding_stacked is not None,
        )

        return result.intermediate, result.gate, embedding_stacked

    def _run_probe_inference(
        self,
        model: Any,
        tokenizer: Any,
        valid_probes: list[tuple[Any, str]],
        layer_activations: dict[int, list["Array"]],
        embedding_activations: list["Array"],
    ) -> tuple[int, int]:
        """Run probe inference to collect activations.

        This is a simplified version of the merge probe inference
        that only collects hidden and embedding activations.
        """
        assert self._activation_provider is not None

        probes_processed = 0
        probes_failed = 0
        total_probes = len(valid_probes)

        if total_probes == 0:
            return 0, 0

        logger.info("PROFILE: Running %d probes through model...", total_probes)

        for i, (probe, probe_text) in enumerate(valid_probes):
            try:
                # Collect hidden activations
                acts = self._activation_provider.collect_hidden_activations(
                    model=model,
                    tokenizer=tokenizer,
                    text=probe_text,
                )

                # Store layer activations
                for layer_idx, act in acts.items():
                    if layer_idx not in layer_activations:
                        layer_activations[layer_idx] = []
                    self._backend.eval(act)
                    layer_activations[layer_idx].append(act)

                # Collect embedding activation if available
                if hasattr(self._activation_provider, "collect_embedding_activation"):
                    emb_act = self._activation_provider.collect_embedding_activation(
                        model=model,
                        tokenizer=tokenizer,
                        text=probe_text,
                    )
                    if emb_act is not None:
                        self._backend.eval(emb_act)
                        embedding_activations.append(emb_act)

                probes_processed += 1

                if (i + 1) % 100 == 0:
                    logger.info("PROFILE: Processed %d/%d probes...", i + 1, total_probes)

            except Exception as e:
                logger.warning("PROFILE: Failed probe %d: %s", i, e)
                probes_failed += 1

            # Periodic cleanup
            if (i + 1) % 50 == 0:
                try:
                    self._backend.clear_cache()
                except Exception:
                    pass

        return probes_processed, probes_failed

    def _compute_profile_from_activations(
        self,
        model_path: str,
        layer_activations: dict[int, list["Array"]],
        embedding_activations: list["Array"],
        valid_probes: list[tuple[Any, str]],
        probes_processed: int,
        probes_failed: int,
        model: Any,
    ) -> GeometricProfile:
        """Compute geometric profile from collected activations."""
        from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
            compute_numerical_rank,
        )
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        b = self._backend

        # Extract model dimensions from weights/config
        hidden_dim = 0
        intermediate_dim = 0
        num_layers = len(layer_activations)
        vocab_size = 0
        num_attention_heads = 0
        num_kv_heads = 0

        # Try to get dimensions from model config
        if hasattr(model, "config"):
            config = model.config
            hidden_dim = getattr(config, "hidden_size", 0)
            intermediate_dim = getattr(config, "intermediate_size", 0)
            vocab_size = getattr(config, "vocab_size", 0)
            num_attention_heads = getattr(config, "num_attention_heads", 0)
            num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # Compute per-layer geometry
        layer_profiles: dict[int, LayerGeometricProfile] = {}
        convergence = ConvergenceMetrics(
            probes_processed=probes_processed,
            probes_failed=probes_failed,
        )

        for layer_idx, acts in layer_activations.items():
            if not acts:
                continue

            # Stack activations
            stacked = b.stack(acts, axis=0)
            b.eval(stacked)

            n_probes, layer_hidden_dim = b.shape(stacked)

            # Compute numerical rank
            activation_rank, _ = compute_numerical_rank(stacked, b)

            # Compute Gram matrix condition number
            gram = b.matmul(stacked, b.transpose(stacked))
            b.eval(gram)

            # SVD for condition number
            try:
                singular_vals = b.svd(gram, full_matrices=False)[1]
                b.eval(singular_vals)

                s_max = b.max(singular_vals)
                s_min = b.min(singular_vals)
                b.eval(s_max, s_min)

                eps = machine_epsilon(b, stacked)
                s_min_safe = s_min + eps
                condition_arr = s_max / s_min_safe
                b.eval(condition_arr)
                gram_condition = float(b.to_scalar(condition_arr))
            except Exception as e:
                logger.warning("Failed to compute condition number for layer %d: %s", layer_idx, e)
                gram_condition = float("inf")

            # Update hidden_dim from actual activations if not set
            if hidden_dim == 0:
                hidden_dim = layer_hidden_dim

            # For trajectory rank, use activation_rank as a proxy
            # (full trajectory analysis requires more probes)
            trajectory_rank = activation_rank
            null_rank = layer_hidden_dim - trajectory_rank

            layer_profiles[layer_idx] = LayerGeometricProfile(
                layer_idx=layer_idx,
                activation_rank=activation_rank,
                trajectory_rank=trajectory_rank,
                gram_condition=gram_condition,
                signal_rank=activation_rank,  # Upper bound (includes noise dims)
                hidden_dim=layer_hidden_dim,
                n_probes=n_probes,
                null_rank=null_rank,
            )

            # Update convergence metrics
            convergence.final_rank[layer_idx] = activation_rank
            convergence.trajectory_rank[layer_idx] = trajectory_rank
            convergence.ceiling_achieved[layer_idx] = activation_rank >= trajectory_rank

        # Compute embedding geometry
        embedding_rank = 0
        embedding_gram_condition = 0.0
        embedding_n_probes = 0

        if embedding_activations:
            emb_stacked = b.stack(embedding_activations, axis=0)
            b.eval(emb_stacked)

            embedding_n_probes = b.shape(emb_stacked)[0]
            embedding_rank, _ = compute_numerical_rank(emb_stacked, b)

            # Gram condition for embeddings
            try:
                emb_gram = b.matmul(emb_stacked, b.transpose(emb_stacked))
                b.eval(emb_gram)
                emb_s = b.svd(emb_gram, full_matrices=False)[1]
                b.eval(emb_s)
                emb_s_max = b.max(emb_s)
                emb_s_min = b.min(emb_s)
                b.eval(emb_s_max, emb_s_min)
                eps = machine_epsilon(b, emb_stacked)
                emb_s_min_safe = emb_s_min + eps
                emb_cond_arr = emb_s_max / emb_s_min_safe
                b.eval(emb_cond_arr)
                embedding_gram_condition = float(b.to_scalar(emb_cond_arr))
            except Exception:
                embedding_gram_condition = float("inf")

        # Build profile
        probe_ids = [p.probe_id for p, _ in valid_probes]
        probe_domains = [p.domain.value if hasattr(p.domain, "value") else str(p.domain) for p, _ in valid_probes]

        return GeometricProfile(
            model_path=model_path,
            weights_hash=compute_weights_hash(model_path),
            probe_count=probes_processed,
            probe_ids=probe_ids,
            probe_domains=list(set(probe_domains)),
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            layer_profiles=layer_profiles,
            embedding_rank=embedding_rank,
            embedding_gram_condition=embedding_gram_condition,
            embedding_n_probes=embedding_n_probes,
            convergence=convergence,
        )


__all__ = [
    "ProfileService",
    "ProfileResult",
]
