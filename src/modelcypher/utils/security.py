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

import logging
import os
from typing import Mapping

TRUST_REMOTE_CODE_ENV = "MC_TRUST_REMOTE_CODE"

_TRUST_REMOTE_CODE_VALUES = {"1", "true", "yes", "on"}


def trust_remote_code_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether trust_remote_code is explicitly enabled."""
    env = environment or os.environ
    value = env.get(TRUST_REMOTE_CODE_ENV, "")
    return value.strip().lower() in _TRUST_REMOTE_CODE_VALUES


def warn_trust_remote_code(logger: logging.Logger | None = None) -> None:
    """Log a warning if trust_remote_code is enabled."""
    if not trust_remote_code_enabled():
        return
    (logger or logging.getLogger(__name__)).warning(
        "trust_remote_code is enabled via %s. Loading models/tokenizers may execute arbitrary code "
        "from model repositories. Use only trusted sources.",
        TRUST_REMOTE_CODE_ENV,
    )
