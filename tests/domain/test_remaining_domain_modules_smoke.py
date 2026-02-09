# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import importlib

import pytest


MODULES = [
    "modelcypher.core.domain.atlas.semantic_prime_multilingual",
    "modelcypher.core.domain.chat_template",
    "modelcypher.core.domain.entropy.eigenscore",
    "modelcypher.core.domain.geometry.activation_buffer",
    "modelcypher.core.domain.geometry.direct_lora_synthesis",
    "modelcypher.core.domain.geometry.gram_spectrum",
    "modelcypher.core.domain.geometry.manifold_boundary",
    "modelcypher.core.domain.geometry.null_space_tracker",
    "modelcypher.core.domain.geometry.persona_vector_monitor",
    "modelcypher.core.domain.geometry.safety_polytope",
    "modelcypher.core.domain.geometry.spatial_3d",
    "modelcypher.core.domain.geometry.trajectory_coherence",
    "modelcypher.core.domain.profile",
    "modelcypher.core.domain.training.evaluation",
    "modelcypher.core.domain.training.residual_scaling",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import_remaining_domain_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None

