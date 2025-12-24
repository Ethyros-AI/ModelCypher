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


import sys
import os
import mlx.core as mx

# Setup path
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("."))

print(f"Running tests with python {sys.version}")

try:
    from tests import test_permutation_aligner_mlx
    from tests import test_entropy_dynamics
    
    print("Alignment Tests:")
    test_permutation_aligner_mlx.test_permutation_alignment_identity()
    print(" - test_permutation_alignment_identity passed")
    test_permutation_aligner_mlx.test_permutation_alignment_permuted()
    print(" - test_permutation_alignment_permuted passed")
    test_permutation_aligner_mlx.test_permutation_alignment_sign_flip()
    print(" - test_permutation_alignment_sign_flip passed")
    
    print("\nEntropy Tests:")
    test_entropy_dynamics.test_logit_entropy_calculator()
    print(" - test_logit_entropy_calculator passed")
    test_entropy_dynamics.test_logit_divergence_calculator()
    print(" - test_logit_divergence_calculator passed")
    test_entropy_dynamics.test_entropy_delta_tracker_anomaly()
    test_entropy_dynamics.test_entropy_delta_tracker_anomaly()
    print(" - test_entropy_delta_tracker_anomaly passed")

    print("\nAdvanced Geometry Tests:")
    from tests import test_advanced_geometry
    test_advanced_geometry.test_intrinsic_dimension_estimator_mle()
    print(" - test_intrinsic_dimension_estimator_mle passed")
    test_advanced_geometry.test_manifold_clusterer_simple()
    print(" - test_manifold_clusterer_simple passed")
    test_advanced_geometry.test_manifold_clusterer_noise()
    print(" - test_manifold_clusterer_noise passed")
    
    print("\nAll verification tests passed!")
    
except Exception as e:
    print(f"\nTEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
