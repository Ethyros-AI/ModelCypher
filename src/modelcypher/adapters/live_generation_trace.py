from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelcypher.adapters.model_backbone import (
    resolve_model_backbone,
)
from modelcypher.core.use_cases.generation_trace_service import (
    LiveGenerationTraceResult,
    LiveTraceRunner,
    LiveTraceStep,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


def build_live_generation_trace_runner(backend: "Backend") -> LiveTraceRunner:
    """Build the default live generation trace runner for greedy decode."""

    def _runner(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int,
    ) -> LiveGenerationTraceResult:
        backbone = resolve_model_backbone(model, getattr(model, "model_type", None))
        if backbone is None:
            raise ValueError("Unable to resolve model backbone for live tracing.")
        _embed_tokens, layers, _norm = backbone

        prompt_ids = list(backend.encode_tokens(tokenizer, prompt))
        current_ids = backend.array([prompt_ids])
        generated_ids: list[int] = []
        steps: list[LiveTraceStep] = []
        stop_reason = "max_tokens"

        for step_index in range(max_tokens):
            captured: dict[int, Any] = {}

            class CaptureWrapper:
                def __init__(wrapper_self, layer: Any, layer_idx: int) -> None:
                    wrapper_self._layer = layer
                    wrapper_self._layer_idx = layer_idx

                def __call__(wrapper_self, *args: Any, **kwargs: Any) -> Any:
                    output = wrapper_self._layer(*args, **kwargs)
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[wrapper_self._layer_idx] = hidden
                    return output

                def __getattr__(wrapper_self, name: str) -> Any:
                    return getattr(wrapper_self._layer, name)

            original_layers = list(layers)
            try:
                for index in range(len(layers)):
                    layers[index] = CaptureWrapper(original_layers[index], index)
                logits = _forward_with_model(model, current_ids)
                backend.eval(logits)
            finally:
                for index, layer in enumerate(original_layers):
                    layers[index] = layer

            last_hidden: dict[int, Any] = {}
            for layer_idx, hidden in captured.items():
                if getattr(hidden, "ndim", 0) == 3:
                    last_hidden[layer_idx] = hidden[0, -1, :]
                else:
                    last_hidden[layer_idx] = hidden[-1, :]
            if last_hidden:
                backend.eval(*list(last_hidden.values()))

            if getattr(logits, "ndim", 0) == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]
            backend.eval(last_logits)
            probs = backend.softmax(last_logits, axis=-1)
            backend.eval(probs)
            log_probs = backend.log(backend.clip(probs, 1e-12, None))
            backend.eval(log_probs)
            logit_entropy = float(
                backend.to_scalar(-backend.sum(probs * log_probs))
            )
            sorted_logits = backend.sort(last_logits)
            backend.eval(sorted_logits)
            top_value = float(backend.to_scalar(sorted_logits[-1]))
            second_value = float(backend.to_scalar(sorted_logits[-2])) if sorted_logits.shape[0] > 1 else top_value
            logit_margin = top_value - second_value

            next_token = int(backend.to_scalar(backend.argmax(last_logits)))
            generated_ids.append(next_token)
            try:
                token_text = backend.decode_tokens(tokenizer, [next_token])
            except Exception:
                token_text = f"<{next_token}>"

            steps.append(
                LiveTraceStep(
                    step_index=step_index,
                    token_id=next_token,
                    token_text=token_text,
                    hidden_by_layer=last_hidden,
                    logit_entropy=logit_entropy,
                    logit_margin=logit_margin,
                )
            )

            current_ids = backend.concatenate(
                [current_ids, backend.array([[next_token]])],
                axis=1,
            )
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token == eos_id:
                stop_reason = "eos"
                break

        generated_text = backend.decode_tokens(tokenizer, generated_ids) if generated_ids else ""
        return LiveGenerationTraceResult(
            prompt_token_ids=tuple(prompt_ids),
            generated_token_ids=tuple(generated_ids),
            generated_text=generated_text,
            stop_reason=stop_reason,
            steps=tuple(steps),
        )

    return _runner


def _forward_with_model(model: Any, input_ids: Any) -> Any:
    output = model(input_ids)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output
