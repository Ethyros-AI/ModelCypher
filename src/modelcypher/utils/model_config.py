"""Helpers for extracting model config fields across architectures."""

from __future__ import annotations

from typing import Any, Iterable


_NESTED_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "language_model",
    "llm_config",
    "model_config",
)


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _iter_nested_configs(
    config: dict[str, Any], nested_keys: Iterable[str]
) -> Iterable[dict[str, Any]]:
    for key in nested_keys:
        nested = config.get(key)
        if isinstance(nested, dict):
            yield nested


def _resolve_field(
    config: dict[str, Any],
    field_names: Iterable[str],
    nested_keys: Iterable[str] = _NESTED_CONFIG_KEYS,
) -> int:
    for name in field_names:
        value = config.get(name)
        if value is not None:
            coerced = _coerce_int(value)
            if coerced:
                return coerced
    for nested in _iter_nested_configs(config, nested_keys):
        for name in field_names:
            value = nested.get(name)
            if value is not None:
                coerced = _coerce_int(value)
                if coerced:
                    return coerced
    return 0


def resolve_vocab_size(config: dict[str, Any]) -> int:
    return _resolve_field(config, ("vocab_size",))


def resolve_hidden_size(config: dict[str, Any]) -> int:
    return _resolve_field(config, ("hidden_size", "n_embd", "d_model"))


def resolve_num_attention_heads(config: dict[str, Any]) -> int:
    return _resolve_field(
        config,
        ("num_attention_heads", "n_head", "num_heads", "n_heads"),
    )


def resolve_num_hidden_layers(config: dict[str, Any]) -> int:
    return _resolve_field(
        config,
        ("num_hidden_layers", "n_layer", "num_layers"),
    )
