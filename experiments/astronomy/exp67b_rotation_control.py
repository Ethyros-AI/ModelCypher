"""
Experiment 67b: Rotation Control (Sanity Check)

We found that Wow! segments rotate by exactly pi (3.1416 rad).
We need to prove this isn't a mathematical artifact of the SVD/Procrustes method.

Hypothesis:
- Random high-dim vectors should be roughly orthogonal (pi/2 rotation).
- Only specific, anti-correlated vectors should show pi rotation.

Method:
1. Generate random segments of the same shape (82, 12).
2. Apply the EXACT same rotation calculation as exp67.
3. Check the distribution of angles.
"""

import numpy as np
from scipy import linalg

def compute_rotation_angle(R):
    """Same function as exp67."""
    U, _, Vh = linalg.svd(R)
    R_ortho = U @ Vh
    eigvals = linalg.eigvals(R_ortho)
    angles = np.angle(eigvals)
    pos_angles = np.sort(angles[angles > 1e-5])
    return pos_angles

def main():
    print("=" * 60)
    print("Experiment 67b: Rotation Code Sanity Check")
    print("=" * 60)
    
    n_trials = 100
    angles = []
    
    print(f"Running {n_trials} random trials (Shape 82x12)...")
    
    for _ in range(n_trials):
        # 1. Generate two random segments
        A = np.random.randn(82, 12)
        B = np.random.randn(82, 12)
        
        # 2. Compute Rotation (Same logic as exp67)
        # We treated them as (82, 12) blocks rotating in 82-dim frequency space?
        # exp67 used: M_freq = B @ A.T
        
        M_freq = B @ A.T
        U, _, Vh = linalg.svd(M_freq)
        R_freq = U @ Vh
        
        # 3. Compute Angle
        computed_angles = compute_rotation_angle(R_freq)
        primary = computed_angles[-1] if len(computed_angles) > 0 else 0
        angles.append(primary)

    # Statistics
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    
    print("\nResults:")
    print(f"Mean Rotation Angle: {mean_angle:.4f} rad ({np.degrees(mean_angle):.1f} deg)")
    print(f"Std Deviation:       {std_angle:.4f} rad")
    print(f"Expected (Random):   {np.pi/2:.4f} rad (90.0 deg)")
    print(f"Observed (Wow!):     {np.pi:.4f} rad (180.0 deg)")
    
    print("\n" + "="*60)
    if abs(mean_angle - np.pi/2) < 0.2:
        print("PASS: Random data rotates by ~90 degrees.")
        print("The 180 degree rotation in Wow! is REAL and UNIQUE.")
    elif abs(mean_angle - np.pi) < 0.2:
        print("FAIL: Random data also rotates by ~180 degrees.")
        print("The finding is an ARTIFACT of the math.")
    else:
        print(f"UNCERTAIN: Random data rotates by {np.degrees(mean_angle):.1f} degrees.")

if __name__ == "__main__":
    main()
