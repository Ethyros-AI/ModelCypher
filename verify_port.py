
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    print("Importing geometry...")
    import modelcypher.core.domain.geometry as geom
    print("Geometry imported.")
    
    print("Importing permutation_aligner...")
    from modelcypher.core.domain.geometry.permutation_aligner import PermutationAligner
    print("PermutationAligner imported.")
    
    print("Importing entropy_dynamics...")
    from modelcypher.core.domain.inference.entropy_dynamics import EntropyDeltaTracker
    print("EntropyDynamics imported.")
    
    print("Importing advanced geometry...")
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator
    from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer, ManifoldPoint
    print("Advanced geometry imported.")
    
    print("Verification Successful: All modules imported.")
    
except Exception as e:
    print(f"Verification Failed: {e}")
    sys.exit(1)
