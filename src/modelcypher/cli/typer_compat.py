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

    try:
        from typer.main import TyperArgument
    except ImportError:
        TyperArgument = None

    if TyperArgument is not None:
        argument_signature = inspect.signature(TyperArgument.make_metavar)
        if "ctx" not in argument_signature.parameters:
            # TyperArgument.make_metavar doesn't accept ctx, but it calls
            # self.type.get_metavar(ctx) internally. We need to patch it to
            # accept ctx and pass it through.
            def patched_argument_make_metavar(self: TyperArgument, ctx: click.Context | None = None) -> str:
                # Call the parent class method which we already patched
                return click.core.Argument.make_metavar(self, ctx)

            TyperArgument.make_metavar = patched_argument_make_metavar
