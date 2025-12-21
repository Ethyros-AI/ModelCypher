
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
    print(" - test_entropy_delta_tracker_anomaly passed")
    
    print("\nAll verification tests passed!")
    
except Exception as e:
    print(f"\nTEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
