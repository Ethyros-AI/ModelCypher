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

"""Backward-compatible shim for vocab Frechet utilities."""

from modelcypher.core.use_cases.merge.stages.vocab import frechet_ops as _stage

__all__ = [name for name in dir(_stage) if not name.startswith("_")]


def __getattr__(name: str):
    return getattr(_stage, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_stage)))
