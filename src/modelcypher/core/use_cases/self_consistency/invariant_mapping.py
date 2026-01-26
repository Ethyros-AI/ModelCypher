# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
"""Invariant Structure Mapping - Find where the constants live.

Inspired by: bioRxiv 2026.01.03.697478v1 showing that language models
encode universal structural patterns that transfer across domains.

The hypothesis: fundamental constants (π/e, φ, √2, etc.) appear in specific
regions of the model's weight and activation space. By mapping WHERE these
constants appear, we can:
1. Understand what the model "knows" geometrically
2. Find gaps where the invariant structure is incomplete
3. Guide learning toward filling those gaps

This is exploratory - we're trying to understand the geometry before
we try to improve it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


CONSTANTS = {
    "pi/e": 1.1557,
    "e/pi": 0.8653,
    "phi": 1.6180,
    "sqrt2": 1.4142,
    "e": 2.7183,
    "pi": 3.1416,
    "1/phi": 0.6180,
    "ln2": 0.6931,
    "sqrt3": 1.7320,
}


@dataclass
class ConstantMatch:
    """A match between an SVD ratio and a fundamental constant."""
    layer_idx: int
    component_type: str  # 'weight' or 'activation'
    ratio_i: int
    ratio_j: int
    ratio_value: float
    matched_constant: str
    constant_value: float
    error_percent: float


@dataclass
class LayerGeometry:
    """Geometric structure of a single layer."""
    layer_idx: int

    # Weight matrix geometry
    weight_matches: List[ConstantMatch]
    weight_n_matches: int
    weight_dominant_ratio: float
    weight_spectral_entropy: float

    # Activation geometry (per probe)
    activation_matches: Dict[str, List[ConstantMatch]]  # probe -> matches
    activation_mean_matches: float

    # Coverage: which constants appear in this layer?
    constants_found: Dict[str, int]  # constant_name -> count


@dataclass
class ModelInvariantMap:
    """Complete map of invariant structure in a model."""
    model_path: str
    n_layers: int

    # Per-layer geometry
    layers: Dict[int, LayerGeometry]

    # Aggregate statistics
    total_weight_matches: int
    total_activation_matches: int

    # Which constants are represented across the model?
    constant_distribution: Dict[str, Dict[int, int]]  # constant -> {layer -> count}

    # Gaps: layers where we expect constants but don't find them
    underrepresented_layers: List[int]
    underrepresented_constants: List[str]


class InvariantMapper:
    """Map the invariant geometric structure of a model."""

    def __init__(
        self,
        model,
        tokenizer,
        match_threshold: float = 5.0,  # % error for match
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.match_threshold = match_threshold
        self.n_layers = len(model.model.layers)

    def _get_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get all weight matrices for a layer."""
        import mlx.core as mx

        layer = self.model.model.layers[layer_idx]
        weights = {}

        # Attention weights
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, name):
                    w = getattr(attn, name).weight
                    mx.eval(w)
                    weights[f'attn_{name}'] = np.array(w.tolist(), dtype=np.float32)

        # MLP weights
        if hasattr(layer, 'feed_forward'):
            mlp = layer.feed_forward
        else:
            mlp = layer.mlp

        for name in ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']:
            if hasattr(mlp, name):
                w = getattr(mlp, name).weight
                mx.eval(w)
                weights[f'mlp_{name}'] = np.array(w.tolist(), dtype=np.float32)

        return weights

    def _get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations for a layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        captured = {}

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured['output'])
            return np.array(captured['output'][0].tolist(), dtype=np.float32)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def _analyze_svd(
        self,
        matrix: np.ndarray,
        layer_idx: int,
        component_type: str,
    ) -> Tuple[List[ConstantMatch], float, float]:
        """Analyze SVD ratios of a matrix.

        Returns:
            (matches, dominant_ratio, spectral_entropy)
        """
        from scipy.linalg import svd

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        try:
            _, S, _ = svd(matrix, full_matrices=False)
        except:
            return [], 0.0, 0.0

        if len(S) < 2:
            return [], S[0] if len(S) > 0 else 0.0, 0.0

        matches = []

        # Check all nearby ratios
        for i in range(min(len(S) - 1, 20)):
            for j in range(i + 1, min(len(S), i + 6)):
                if S[j] > 1e-10:
                    ratio = S[i] / S[j]

                    # Check against all constants
                    for const_name, const_val in CONSTANTS.items():
                        error = abs(ratio - const_val) / const_val * 100
                        if error < self.match_threshold:
                            matches.append(ConstantMatch(
                                layer_idx=layer_idx,
                                component_type=component_type,
                                ratio_i=i,
                                ratio_j=j,
                                ratio_value=float(ratio),
                                matched_constant=const_name,
                                constant_value=const_val,
                                error_percent=float(error),
                            ))

        # Dominant ratio
        dominant_ratio = S[0] / S[1] if S[1] > 1e-10 else 0.0

        # Spectral entropy
        S_sum = S.sum()
        if S_sum > 1e-10:
            S_norm = S / S_sum
            entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
        else:
            entropy = 0.0

        return matches, float(dominant_ratio), entropy

    def map_layer(
        self,
        layer_idx: int,
        probes: List[str],
    ) -> LayerGeometry:
        """Map the invariant structure of a single layer."""

        logger.info(f"  Mapping layer {layer_idx}...")

        # Analyze weight matrices
        weights = self._get_weights(layer_idx)
        all_weight_matches = []
        weight_dominant_ratio = 0.0
        weight_entropy = 0.0

        for name, W in weights.items():
            matches, dom_ratio, entropy = self._analyze_svd(
                W, layer_idx, f'weight_{name}'
            )
            all_weight_matches.extend(matches)
            # Use MLP gate_proj as representative
            if 'gate_proj' in name or 'w1' in name:
                weight_dominant_ratio = dom_ratio
                weight_entropy = entropy

        # Analyze activations for each probe
        activation_matches = {}
        total_act_matches = 0

        for probe in probes:
            act = self._get_activations(probe, layer_idx)
            matches, _, _ = self._analyze_svd(act, layer_idx, 'activation')
            activation_matches[probe] = matches
            total_act_matches += len(matches)

        mean_act_matches = total_act_matches / len(probes) if probes else 0.0

        # Count which constants appear
        constants_found = {name: 0 for name in CONSTANTS}
        for match in all_weight_matches:
            constants_found[match.matched_constant] += 1
        for probe_matches in activation_matches.values():
            for match in probe_matches:
                constants_found[match.matched_constant] += 1

        return LayerGeometry(
            layer_idx=layer_idx,
            weight_matches=all_weight_matches,
            weight_n_matches=len(all_weight_matches),
            weight_dominant_ratio=weight_dominant_ratio,
            weight_spectral_entropy=weight_entropy,
            activation_matches=activation_matches,
            activation_mean_matches=mean_act_matches,
            constants_found=constants_found,
        )

    def map_model(
        self,
        probes: List[str],
        layer_indices: Optional[List[int]] = None,
    ) -> ModelInvariantMap:
        """Map the complete invariant structure of a model."""

        if layer_indices is None:
            layer_indices = list(range(self.n_layers))

        logger.info(f"\nMapping invariant structure across {len(layer_indices)} layers...")

        layers = {}
        constant_distribution = {name: {} for name in CONSTANTS}

        for layer_idx in layer_indices:
            geom = self.map_layer(layer_idx, probes)
            layers[layer_idx] = geom

            # Aggregate constant distribution
            for const_name, count in geom.constants_found.items():
                constant_distribution[const_name][layer_idx] = count

        # Compute aggregates
        total_weight_matches = sum(l.weight_n_matches for l in layers.values())
        total_act_matches = sum(
            sum(len(m) for m in l.activation_matches.values())
            for l in layers.values()
        )

        # Find gaps
        mean_matches = total_weight_matches / len(layers) if layers else 0
        underrepresented_layers = [
            idx for idx, l in layers.items()
            if l.weight_n_matches < mean_matches * 0.5
        ]

        # Find underrepresented constants
        const_totals = {
            name: sum(dist.values())
            for name, dist in constant_distribution.items()
        }
        mean_const = sum(const_totals.values()) / len(const_totals) if const_totals else 0
        underrepresented_constants = [
            name for name, total in const_totals.items()
            if total < mean_const * 0.3
        ]

        return ModelInvariantMap(
            model_path="",  # Fill in later
            n_layers=self.n_layers,
            layers=layers,
            total_weight_matches=total_weight_matches,
            total_activation_matches=total_act_matches,
            constant_distribution=constant_distribution,
            underrepresented_layers=underrepresented_layers,
            underrepresented_constants=underrepresented_constants,
        )

    def print_summary(self, inv_map: ModelInvariantMap):
        """Print a summary of the invariant map."""

        print("\n" + "="*70)
        print("INVARIANT STRUCTURE MAP")
        print("="*70)

        print(f"\nLayers analyzed: {len(inv_map.layers)}")
        print(f"Total weight matches: {inv_map.total_weight_matches}")
        print(f"Total activation matches: {inv_map.total_activation_matches}")

        print("\n--- Constant Distribution Across Layers ---")
        for const_name in sorted(CONSTANTS.keys()):
            dist = inv_map.constant_distribution.get(const_name, {})
            total = sum(dist.values())
            layers_present = len([v for v in dist.values() if v > 0])
            print(f"  {const_name:10s}: {total:4d} matches in {layers_present:2d} layers")

        print("\n--- Layer-by-Layer Summary ---")
        for layer_idx in sorted(inv_map.layers.keys()):
            geom = inv_map.layers[layer_idx]
            print(f"  Layer {layer_idx:2d}: {geom.weight_n_matches:3d} weight matches, "
                  f"{geom.activation_mean_matches:.1f} mean act matches, "
                  f"entropy={geom.weight_spectral_entropy:.3f}")

        if inv_map.underrepresented_layers:
            print(f"\nUnderrepresented layers: {inv_map.underrepresented_layers}")
        if inv_map.underrepresented_constants:
            print(f"Underrepresented constants: {inv_map.underrepresented_constants}")


__all__ = ["InvariantMapper", "ModelInvariantMap", "LayerGeometry", "ConstantMatch"]
