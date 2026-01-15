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

"""Geometric AI alignment experiments.

This package contains experiments to prove that AI alignment can be measured,
detected, and transferred geometrically using high-dimensional manifold analysis.

Core Hypothesis:
    Alignment is a measurable geometric property—a position on a shared manifold—
    not a behavioral heuristic.

Experiments:
    1. Alignment Detection: Measure geometric difference between base and instruct models
    2. Refusal Direction: Extract the low-dimensional refusal vector
    3. Universality: Prove alignment geometry is universal across architectures
    4. Jailbreak Detection: Detect jailbreaks from activation geometry
    5. Alignment Transfer: Transfer alignment via null-space projection
    6. Geometric Guardrails: Enforce alignment boundaries at inference time

References:
    - Huh et al. (2024). "The Platonic Representation Hypothesis." arXiv:2405.07987
    - Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction."
    - Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency."
"""

from __future__ import annotations

__all__: list[str] = []
