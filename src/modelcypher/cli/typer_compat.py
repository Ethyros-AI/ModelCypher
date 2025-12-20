from __future__ import annotations

import inspect
from typing import Any, Callable

import click
import typer
from click.core import UNSET


def apply_typer_compat() -> None:
    _patch_option_flag_value()
    _patch_make_metavar()


def _patch_option_flag_value() -> None:
    original_option: Callable[..., Any] = typer.Option

    def patched_option(*args: Any, **kwargs: Any) -> Any:
        if "flag_value" not in kwargs:
            # Click 8.3 treats flag_value=None as a boolean flag, so opt out by default.
            kwargs["flag_value"] = UNSET
        return original_option(*args, **kwargs)

    typer.Option = patched_option


def _patch_make_metavar() -> None:
    signature = inspect.signature(click.core.Parameter.make_metavar)
    ctx_param = signature.parameters.get("ctx")
    if ctx_param is None or ctx_param.default is not inspect._empty:
        return

    original = click.core.Parameter.make_metavar

    def patched_make_metavar(self: click.core.Parameter, ctx: click.Context | None = None) -> str:
        return original(self, ctx)

    click.core.Parameter.make_metavar = patched_make_metavar
