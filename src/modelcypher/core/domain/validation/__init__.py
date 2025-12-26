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

"""Validation domain modules.

This package contains modules for training data validation and quality scoring.
"""

from modelcypher.core.domain.validation.auto_fix_engine import (
    AutoFixEngine,
    AutoFixResult,
    Fix,
    FixType,
    UnfixableLine,
)
from modelcypher.core.domain.validation.intrinsic_identity_linter import (
    DatasetIdentityScanner,
    IdentityFinding,
    IntrinsicIdentityLinter,
)

__all__ = [
    # Auto-fix
    "AutoFixEngine",
    "AutoFixResult",
    "Fix",
    "FixType",
    "UnfixableLine",
    # Identity linter
    "DatasetIdentityScanner",
    "IdentityFinding",
    "IntrinsicIdentityLinter",
]
