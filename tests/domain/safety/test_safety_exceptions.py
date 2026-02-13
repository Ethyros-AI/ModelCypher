# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.safety.exceptions import (
    LayerAlphaMissingError,
    ModerationUnavailableError,
    SafetyValidationError,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_safety_validation_error_importable(self) -> None:
        assert SafetyValidationError is not None

    def test_moderation_unavailable_error_importable(self) -> None:
        assert ModerationUnavailableError is not None

    def test_layer_alpha_missing_error_importable(self) -> None:
        assert LayerAlphaMissingError is not None


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_safety_validation_error_with_all_args(self) -> None:
        ctx = {"layer": "self_attn.q_proj", "rank": 64}
        err = SafetyValidationError(stage="alignment", message="bad rank", context=ctx)
        assert err.stage == "alignment"
        assert err.context == ctx
        assert "[alignment] bad rank" in str(err)

    def test_moderation_unavailable_error_with_all_args(self) -> None:
        ctx = {"error_type": "timeout", "error_message": "10s exceeded"}
        err = ModerationUnavailableError(
            stage="openai_moderation",
            message="API unreachable",
            context=ctx,
        )
        assert err.stage == "openai_moderation"
        assert err.context == ctx

    def test_layer_alpha_missing_error_with_all_args(self) -> None:
        ctx = {"layer": "layers.5.mlp", "available_layers": ["layers.0.mlp"]}
        err = LayerAlphaMissingError(
            stage="apply_alpha",
            message="missing layer alpha",
            context=ctx,
        )
        assert err.stage == "apply_alpha"
        assert err.context == ctx


# ---------------------------------------------------------------------------
# Context defaults to {} when None
# ---------------------------------------------------------------------------


class TestContextDefault:
    def test_context_defaults_to_empty_dict_when_none(self) -> None:
        err = SafetyValidationError(stage="test", message="msg", context=None)
        assert err.context == {}

    def test_context_defaults_to_empty_dict_when_omitted(self) -> None:
        err = SafetyValidationError(stage="test", message="msg")
        assert err.context == {}

    def test_subclass_context_defaults_to_empty_dict(self) -> None:
        err = ModerationUnavailableError(stage="mod", message="down")
        assert err.context == {}

        err2 = LayerAlphaMissingError(stage="alpha", message="missing")
        assert err2.context == {}


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_includes_stage(self) -> None:
        err = SafetyValidationError(stage="regex_scan", message="found PII")
        r = repr(err)
        assert "regex_scan" in r
        assert "SafetyValidationError" in r

    def test_repr_includes_context(self) -> None:
        ctx = {"detail": "value"}
        err = SafetyValidationError(stage="s", message="m", context=ctx)
        r = repr(err)
        assert "detail" in r

    def test_subclass_repr_uses_own_class_name(self) -> None:
        err = ModerationUnavailableError(stage="api", message="timeout")
        assert "ModerationUnavailableError" in repr(err)

        err2 = LayerAlphaMissingError(stage="alpha", message="gone")
        assert "LayerAlphaMissingError" in repr(err2)


# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_moderation_unavailable_is_safety_validation_error(self) -> None:
        assert issubclass(ModerationUnavailableError, SafetyValidationError)

    def test_layer_alpha_missing_is_safety_validation_error(self) -> None:
        assert issubclass(LayerAlphaMissingError, SafetyValidationError)

    def test_safety_validation_error_is_exception(self) -> None:
        assert issubclass(SafetyValidationError, Exception)

    def test_moderation_unavailable_is_exception(self) -> None:
        assert issubclass(ModerationUnavailableError, Exception)

    def test_layer_alpha_missing_is_exception(self) -> None:
        assert issubclass(LayerAlphaMissingError, Exception)

    def test_isinstance_checks(self) -> None:
        err = ModerationUnavailableError(stage="s", message="m")
        assert isinstance(err, SafetyValidationError)
        assert isinstance(err, Exception)

        err2 = LayerAlphaMissingError(stage="s", message="m")
        assert isinstance(err2, SafetyValidationError)
        assert isinstance(err2, Exception)


# ---------------------------------------------------------------------------
# Can be raised and caught
# ---------------------------------------------------------------------------


class TestRaiseAndCatch:
    def test_raise_safety_validation_error(self) -> None:
        with pytest.raises(SafetyValidationError, match="alignment"):
            raise SafetyValidationError(stage="alignment", message="failed")

    def test_raise_moderation_unavailable_error(self) -> None:
        with pytest.raises(ModerationUnavailableError):
            raise ModerationUnavailableError(stage="api", message="down")

    def test_raise_layer_alpha_missing_error(self) -> None:
        with pytest.raises(LayerAlphaMissingError):
            raise LayerAlphaMissingError(stage="alpha", message="missing")

    def test_catch_subclass_as_base(self) -> None:
        with pytest.raises(SafetyValidationError):
            raise ModerationUnavailableError(stage="api", message="timeout")

    def test_catch_subclass_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise LayerAlphaMissingError(stage="alpha", message="gone")

    def test_raised_exception_preserves_attributes(self) -> None:
        ctx = {"layer": "layers.3"}
        try:
            raise LayerAlphaMissingError(
                stage="apply_alpha", message="not found", context=ctx
            )
        except SafetyValidationError as exc:
            assert exc.stage == "apply_alpha"
            assert exc.context == ctx
            assert "not found" in str(exc)
