# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""MLX-specific self-improvement adapters.

Contains:
- oracle: Verification oracle for model inference
- scanner: Capability scanner with model inference
"""

from __future__ import annotations

from .oracle import VerificationOracle
from .scanner import CapabilityScanner

__all__: list[str] = ["VerificationOracle", "CapabilityScanner"]
