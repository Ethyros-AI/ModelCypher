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
Activation Patching for Causal Intervention.

Measures the causal effect of activations at specific locations by replacing
them with values from another run (clean/corrupt) and measuring downstream
impact.

Key concepts:
    - Clean run: Normal forward pass on baseline input
    - Corrupt run: Forward pass on modified input
    - Patching: Replace activation at (layer, position) and measure effect

This enables:
    - Circuit discovery: Which components matter for specific behaviors
    - Attribution: Which inputs cause which outputs
    - Localization: Where is information processed

Uses the _StreamingLayerWrapper pattern from activation_stream.py for
backend-compatible layer interception.

References:
    - "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)
    - "Interpretability in the Wild" (Wang et al., 2022)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    regularization_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class PatchComponent(str, Enum):
    """Component type to patch."""

    residual = "residual"
    attention = "attention"
    mlp = "mlp"
    attention_output = "attention_output"
    mlp_output = "mlp_output"


@dataclass(frozen=True)
class PatchSpec:
    """Specification for an activation patch.

    Attributes
    ----------
    layer : int
        Layer index to patch.
    position : int | slice | None
        Token position(s) to patch. None = all positions.
    component : PatchComponent
        Which component to patch (residual stream, attention, MLP).
    patch_value : Array | None
        Value to patch in. If None, will use value from clean run.
    """

    layer: int
    position: int | slice | None = None
    component: PatchComponent = PatchComponent.residual
    patch_value: Any = None


@dataclass(frozen=True)
class PatchingResult:
    """Result of activation patching experiment.

    Attributes
    ----------
    original_logits : Array
        Logits from clean run. Shape: [seq, vocab].
    patched_logits : Array
        Logits after patching. Shape: [seq, vocab].
    logit_diff : float
        Geodesic distance between original and patched logits.
    kl_divergence : float
        KL divergence from original to patched distribution.
    causal_effect : float
        Normalized causal effect magnitude (0-1 scale).
    original_probs : Array
        Probabilities from clean run. Shape: [seq, vocab].
    patched_probs : Array
        Probabilities after patching. Shape: [seq, vocab].
    patch_spec : PatchSpec
        The patch specification used.
    """

    original_logits: Any
    patched_logits: Any
    logit_diff: float
    kl_divergence: float
    causal_effect: float
    original_probs: Any
    patched_probs: Any
    patch_spec: PatchSpec


@dataclass(frozen=True)
class PathPatchingResult:
    """Result of path patching (tracing through multiple layers).

    Attributes
    ----------
    layer_effects : dict[int, float]
        Causal effect at each layer.
    component_effects : dict[tuple[int, PatchComponent], float]
        Causal effect for each (layer, component) pair.
    peak_layer : int
        Layer with maximum causal effect.
    peak_component : PatchComponent
        Component with maximum causal effect at peak layer.
    total_effect : float
        Total causal effect across all patches.
    """

    layer_effects: dict[int, float]
    component_effects: dict[tuple[int, PatchComponent], float]
    peak_layer: int
    peak_component: PatchComponent
    total_effect: float


@dataclass
class CapturedActivations:
    """Activations captured during a forward pass.

    Attributes
    ----------
    layer_outputs : dict[int, Any]
        Output tensor for each layer. Keys are layer indices.
    attention_outputs : dict[int, Any]
        Attention output for each layer (if captured).
    mlp_outputs : dict[int, Any]
        MLP output for each layer (if captured).
    final_logits : Any
        Final model logits.
    """

    layer_outputs: dict[int, Any] = field(default_factory=dict)
    attention_outputs: dict[int, Any] = field(default_factory=dict)
    mlp_outputs: dict[int, Any] = field(default_factory=dict)
    final_logits: Any = None


class ActivationPatcher:
    """Performs activation patching experiments.

    Example
    -------
    >>> patcher = ActivationPatcher(model)
    >>> result = patcher.patch(
    ...     clean_input=clean_ids,
    ...     corrupt_input=corrupt_ids,
    ...     patch_spec=PatchSpec(layer=10, position=-1),
    ... )
    >>> # result.causal_effect shows importance of layer 10, last token
    """

    def __init__(self, model: Any, backend: "Backend | None" = None) -> None:
        """Initialize patcher.

        Parameters
        ----------
        model : Any
            Model to patch. Must have accessible layers.
        backend : Backend, optional
            Computation backend.
        """
        self._model = model
        self._backend = backend or get_default_backend()
        self._layers = self._get_layers()

    def _get_layers(self) -> list[Any]:
        """Get model layers."""
        base_model = getattr(self._model, "model", self._model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Model does not expose transformer layers. "
                "Expected model.layers or model.model.layers."
            )
        return layers

    def capture_clean_run(
        self,
        input_ids: Any,
        layers_to_capture: set[int] | None = None,
    ) -> CapturedActivations:
        """Run clean forward pass and capture activations.

        Parameters
        ----------
        input_ids : Array
            Input token IDs. Shape: [batch, seq] or [seq].
        layers_to_capture : set[int], optional
            Layers to capture. None = all layers.

        Returns
        -------
        CapturedActivations
            Captured activations for patching.
        """
        b = self._backend
        input_ids = b.array(input_ids) if not hasattr(input_ids, "shape") else input_ids
        b.eval(input_ids)

        captured = CapturedActivations()

        # Determine which layers to capture
        n_layers = len(self._layers)
        if layers_to_capture is None:
            layers_to_capture = set(range(n_layers))

        # Capture callback
        def capture_callback(layer_idx: int, output: Any) -> None:
            if layer_idx in layers_to_capture:
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                b.eval(hidden)
                captured.layer_outputs[layer_idx] = hidden

        # Wrap layers and run forward pass
        with _CaptureContext(self._layers, capture_callback, layers_to_capture):
            logits = self._model(input_ids)
            b.eval(logits)
            captured.final_logits = logits

        return captured

    def patch(
        self,
        clean_input: Any,
        corrupt_input: Any,
        patch_spec: PatchSpec,
    ) -> PatchingResult:
        """Perform activation patching.

        Runs clean forward pass, captures activations at patch location,
        runs corrupt forward pass with patch applied, measures effect.

        Parameters
        ----------
        clean_input : Array
            Clean input token IDs.
        corrupt_input : Array
            Corrupt input token IDs.
        patch_spec : PatchSpec
            Specification of where and what to patch.

        Returns
        -------
        PatchingResult
            Patching experiment result.
        """
        b = self._backend
        clean_input = b.array(clean_input) if not hasattr(clean_input, "shape") else clean_input
        corrupt_input = (
            b.array(corrupt_input) if not hasattr(corrupt_input, "shape") else corrupt_input
        )
        b.eval(clean_input, corrupt_input)

        # Run clean forward pass
        clean_captured = self.capture_clean_run(
            clean_input, layers_to_capture={patch_spec.layer}
        )

        # Determine patch value
        patch_value = patch_spec.patch_value
        if patch_value is None:
            clean_activation = clean_captured.layer_outputs.get(patch_spec.layer)
            if clean_activation is None:
                raise ValueError(f"Layer {patch_spec.layer} was not captured in clean run")
            patch_value = clean_activation

        # Run corrupt forward pass with patching
        patched_logits = self._run_with_patch(corrupt_input, patch_spec, patch_value)
        b.eval(patched_logits)

        # Run corrupt forward pass without patching (for comparison)
        corrupt_logits = self._model(corrupt_input)
        b.eval(corrupt_logits)

        # Compute metrics
        return self._compute_metrics(
            clean_captured.final_logits,
            patched_logits,
            patch_spec,
        )

    def patch_from_cached(
        self,
        corrupt_input: Any,
        patch_spec: PatchSpec,
        cached_activations: CapturedActivations,
    ) -> PatchingResult:
        """Perform patching using pre-cached clean activations.

        More efficient when running multiple patches from same clean run.

        Parameters
        ----------
        corrupt_input : Array
            Corrupt input token IDs.
        patch_spec : PatchSpec
            Specification of where to patch.
        cached_activations : CapturedActivations
            Pre-captured clean run activations.

        Returns
        -------
        PatchingResult
            Patching experiment result.
        """
        b = self._backend
        corrupt_input = (
            b.array(corrupt_input) if not hasattr(corrupt_input, "shape") else corrupt_input
        )
        b.eval(corrupt_input)

        # Get patch value from cache
        patch_value = patch_spec.patch_value
        if patch_value is None:
            patch_value = cached_activations.layer_outputs.get(patch_spec.layer)
            if patch_value is None:
                raise ValueError(
                    f"Layer {patch_spec.layer} not in cached activations"
                )

        # Run with patch
        patched_logits = self._run_with_patch(corrupt_input, patch_spec, patch_value)
        b.eval(patched_logits)

        return self._compute_metrics(
            cached_activations.final_logits,
            patched_logits,
            patch_spec,
        )

    def path_patching(
        self,
        clean_input: Any,
        corrupt_input: Any,
        layers: list[int] | None = None,
        components: list[PatchComponent] | None = None,
    ) -> PathPatchingResult:
        """Trace causal effects through the network.

        Patches each (layer, component) combination and measures effect,
        building a map of where information flows.

        Parameters
        ----------
        clean_input : Array
            Clean input token IDs.
        corrupt_input : Array
            Corrupt input token IDs.
        layers : list[int], optional
            Layers to test. None = all layers.
        components : list[PatchComponent], optional
            Components to test. None = just residual.

        Returns
        -------
        PathPatchingResult
            Map of causal effects across network.
        """
        b = self._backend
        n_layers = len(self._layers)

        if layers is None:
            layers = list(range(n_layers))
        if components is None:
            components = [PatchComponent.residual]

        # First, capture all clean activations
        clean_captured = self.capture_clean_run(
            clean_input, layers_to_capture=set(layers)
        )

        layer_effects: dict[int, float] = {}
        component_effects: dict[tuple[int, PatchComponent], float] = {}

        max_effect = 0.0
        peak_layer = 0
        peak_component = PatchComponent.residual

        for layer in layers:
            layer_max = 0.0
            for component in components:
                spec = PatchSpec(layer=layer, component=component)
                result = self.patch_from_cached(
                    corrupt_input, spec, clean_captured
                )
                effect = result.causal_effect
                component_effects[(layer, component)] = effect

                if effect > layer_max:
                    layer_max = effect
                if effect > max_effect:
                    max_effect = effect
                    peak_layer = layer
                    peak_component = component

            layer_effects[layer] = layer_max

        # Compute total effect (sum of individual effects)
        total_effect = sum(component_effects.values())

        return PathPatchingResult(
            layer_effects=layer_effects,
            component_effects=component_effects,
            peak_layer=peak_layer,
            peak_component=peak_component,
            total_effect=total_effect,
        )

    def _run_with_patch(
        self,
        input_ids: Any,
        patch_spec: PatchSpec,
        patch_value: Any,
    ) -> Any:
        """Run forward pass with activation patching."""
        b = self._backend

        def patch_callback(layer_idx: int, output: Any) -> Any:
            if layer_idx != patch_spec.layer:
                return output

            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Apply patch
            patched = self._apply_patch(hidden, patch_value, patch_spec.position)
            b.eval(patched)

            if rest is not None:
                return (patched,) + rest
            return patched

        with _PatchContext(self._layers, patch_callback, {patch_spec.layer}):
            logits = self._model(input_ids)
            b.eval(logits)

        return logits

    def _apply_patch(
        self,
        target: Any,
        patch_value: Any,
        position: int | slice | None,
    ) -> Any:
        """Apply patch value to target activation."""
        b = self._backend

        if position is None:
            # Patch all positions
            return patch_value

        # Handle different tensor shapes
        # Target shape: [batch, seq, hidden] or [seq, hidden]
        target_shape = b.shape(target)
        patch_shape = b.shape(patch_value)

        if len(target_shape) == 3:
            # [batch, seq, hidden]
            if isinstance(position, int):
                # Create a mask for the position
                seq_len = int(target_shape[1])
                pos = position if position >= 0 else seq_len + position

                # Build indices for position
                indices = b.arange(seq_len)
                mask = indices == pos
                mask = b.reshape(mask, (1, seq_len, 1))

                # Get patch for this position
                if len(patch_shape) == 3:
                    patch_at_pos = patch_value[:, pos : pos + 1, :]
                else:
                    patch_at_pos = b.reshape(patch_value[pos, :], (1, 1, -1))

                result = b.where(mask, patch_at_pos, target)
            else:
                # Slice - just replace
                result = patch_value
        elif len(target_shape) == 2:
            # [seq, hidden]
            if isinstance(position, int):
                seq_len = int(target_shape[0])
                pos = position if position >= 0 else seq_len + position

                indices = b.arange(seq_len)
                mask = indices == pos
                mask = b.reshape(mask, (seq_len, 1))

                if len(patch_shape) == 2:
                    patch_at_pos = patch_value[pos : pos + 1, :]
                else:
                    patch_at_pos = b.reshape(patch_value, (1, -1))

                result = b.where(mask, patch_at_pos, target)
            else:
                result = patch_value
        else:
            # 1D or other - just replace
            result = patch_value

        b.eval(result)
        return result

    def _compute_metrics(
        self,
        original_logits: Any,
        patched_logits: Any,
        patch_spec: PatchSpec,
    ) -> PatchingResult:
        """Compute patching metrics."""
        b = self._backend

        # Ensure 2D: [seq, vocab]
        if len(b.shape(original_logits)) == 3:
            original_logits = original_logits[0]
        if len(b.shape(patched_logits)) == 3:
            patched_logits = patched_logits[0]

        original_logits = b.astype(original_logits, "float32")
        patched_logits = b.astype(patched_logits, "float32")
        b.eval(original_logits, patched_logits)

        # Geodesic distance between logit distributions
        diff = original_logits - patched_logits
        norms = geodesic_norms(diff, b)
        b.eval(norms)
        logit_diff = float(b.to_scalar(b.mean(norms)))

        # Softmax to get probabilities
        original_probs = self._softmax(original_logits)
        patched_probs = self._softmax(patched_logits)
        b.eval(original_probs, patched_probs)

        # KL divergence
        kl_div = self._kl_divergence(original_probs, patched_probs)

        # Causal effect: normalized logit difference
        # Scale by typical logit magnitude
        logit_scale = b.mean(b.abs(original_logits))
        b.eval(logit_scale)
        scale_val = float(b.to_scalar(logit_scale))

        eps = regularization_epsilon(b, original_logits)
        if scale_val > eps:
            causal_effect = logit_diff / scale_val
        else:
            causal_effect = logit_diff

        # Clamp to reasonable range
        causal_effect = min(causal_effect, 1.0)

        return PatchingResult(
            original_logits=original_logits,
            patched_logits=patched_logits,
            logit_diff=logit_diff,
            kl_divergence=kl_div,
            causal_effect=causal_effect,
            original_probs=original_probs,
            patched_probs=patched_probs,
            patch_spec=patch_spec,
        )

    def _softmax(self, logits: Any) -> Any:
        """Compute softmax probabilities."""
        b = self._backend
        # Subtract max for numerical stability
        max_logits = b.max(logits, axis=-1, keepdims=True)
        exp_logits = b.exp(logits - max_logits)
        sum_exp = b.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp
        b.eval(probs)
        return probs

    def _kl_divergence(self, p: Any, q: Any) -> float:
        """Compute KL divergence D_KL(p || q)."""
        b = self._backend
        eps = regularization_epsilon(b, p)

        # KL = sum(p * log(p / q))
        log_ratio = b.log((p + eps) / (q + eps))
        kl = b.sum(p * log_ratio, axis=-1)
        mean_kl = b.mean(kl)
        b.eval(mean_kl)
        return float(b.to_scalar(mean_kl))


class _CaptureContext:
    """Context manager for capturing layer outputs."""

    def __init__(
        self,
        layers: list[Any],
        capture: Callable[[int, Any], None],
        target_layers: set[int],
    ) -> None:
        self._layers = layers
        self._capture = capture
        self._target_layers = target_layers
        self._original: list[Any] | None = None

    def __enter__(self) -> "_CaptureContext":
        self._original = list(self._layers)
        wrapped = []
        for idx, layer in enumerate(self._layers):
            if idx in self._target_layers:
                wrapped.append(_CaptureWrapper(layer, idx, self._capture))
            else:
                wrapped.append(layer)
        self._layers[:] = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original is not None:
            self._layers[:] = self._original


class _CaptureWrapper:
    """Wrapper that captures output without modifying it."""

    __slots__ = ("_layer", "_idx", "_capture")

    def __init__(
        self,
        layer: Any,
        idx: int,
        capture: Callable[[int, Any], None],
    ) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_capture", capture)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._layer(*args, **kwargs)
        self._capture(self._idx, output)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


class _PatchContext:
    """Context manager for patching layer outputs."""

    def __init__(
        self,
        layers: list[Any],
        patch_fn: Callable[[int, Any], Any],
        target_layers: set[int],
    ) -> None:
        self._layers = layers
        self._patch_fn = patch_fn
        self._target_layers = target_layers
        self._original: list[Any] | None = None

    def __enter__(self) -> "_PatchContext":
        self._original = list(self._layers)
        wrapped = []
        for idx, layer in enumerate(self._layers):
            if idx in self._target_layers:
                wrapped.append(_PatchWrapper(layer, idx, self._patch_fn))
            else:
                wrapped.append(layer)
        self._layers[:] = wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original is not None:
            self._layers[:] = self._original


class _PatchWrapper:
    """Wrapper that patches output."""

    __slots__ = ("_layer", "_idx", "_patch_fn")

    def __init__(
        self,
        layer: Any,
        idx: int,
        patch_fn: Callable[[int, Any], Any],
    ) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_idx", idx)
        object.__setattr__(self, "_patch_fn", patch_fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._layer(*args, **kwargs)
        return self._patch_fn(self._idx, output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


__all__ = [
    "PatchComponent",
    "PatchSpec",
    "PatchingResult",
    "PathPatchingResult",
    "CapturedActivations",
    "ActivationPatcher",
]
