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

"""Test fixtures package.

Contains reusable fixtures for testing:
- synthetic_manifolds: Manifolds with known ground truth properties
- models: Real model fixtures for integration testing
"""

from .synthetic_manifolds import (
    ManifoldSample,
    random_orthogonal_matrix,
    sample_flat_torus,
    sample_hyperbolic_paraboloid,
    sample_linear_subspace,
    sample_sphere,
    sample_swiss_roll,
)
from .models import (
    MODELS_CACHE_DIR,
    SMOL_LM_135M,
    collect_real_activations,
    ensure_model,
    get_atlas_probes,
    load_model_and_tokenizer,
    load_model_weights,
)

__all__ = [
    # Synthetic manifolds
    "ManifoldSample",
    "random_orthogonal_matrix",
    "sample_flat_torus",
    "sample_hyperbolic_paraboloid",
    "sample_linear_subspace",
    "sample_sphere",
    "sample_swiss_roll",
    # Real model fixtures
    "MODELS_CACHE_DIR",
    "SMOL_LM_135M",
    "collect_real_activations",
    "ensure_model",
    "get_atlas_probes",
    "load_model_and_tokenizer",
    "load_model_weights",
]
