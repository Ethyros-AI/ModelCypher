"""
Experiment 76b: Berry Phase Control

We found a Winding Number of exactly 11.0000.
We must prove that the code does not 'force' integers.

Method:
1. Generate random signals (Gaussian noise).
2. Generate continuous rotation signals (Sine waves with random freq).
3. Run the EXACT same compute_2d_berry_phase function.
4. Check if the results are integers.
"""

import numpy as np
from scipy import linalg

def compute_2d_berry_phase(signal, window_size=4, step=1):
    n_freq, n_time = signal.shape
    accumulated_phase = 0.0
    
    seg = signal[:, 0:window_size]
    U, _, _ = linalg.svd(seg, full_matrices=False)
    P_prev = U[:, :2]
    
    for t in range(step, n_time - window_size, step):
        seg = signal[:, t:t+window_size]
        U, _, _ = linalg.svd(seg, full_matrices=False)
        P_curr = U[:, :2]
        
        M = np.dot(P_prev.T, P_curr)
        det_M = linalg.det(M)
        d_gamma = np.angle(det_M)
        
        accumulated_phase += d_gamma
        P_prev = P_curr
        
    return accumulated_phase

def main():
    print("=" * 60)
    print("Experiment 76b: Berry Phase Control")
    print("=" * 60)
    
    # 1. Random Noise Control
    print("\n1. Testing Random Noise (10 trials)...")
    for i in range(10):
        noise = np.random.randn(82, 50) # Same shape as Wow!
        gamma = compute_2d_berry_phase(noise, window_size=6, step=1)
        w = gamma / (2 * np.pi)
        remainder = abs(w - round(w))
        print(f"   Trial {i}: W = {w:.4f} (Deviation: {remainder:.4f})")
        
    # 2. Random Sine Wave Control (Continuous rotation)
    print("\n2. Testing Random Sine Waves...")
    t = np.linspace(0, 10, 50)
    for i in range(5):
        # Create a signal that rotates phase
        freq = 1.0 + np.random.rand() # Random frequency
        sig = np.outer(np.random.randn(82), np.sin(freq * t))
        sig += 0.1 * np.random.randn(82, 50) # Add noise
        
        gamma = compute_2d_berry_phase(sig, window_size=6, step=1)
        w = gamma / (2 * np.pi)
        print(f"   Freq {freq:.2f}: W = {w:.4f}")

if __name__ == "__main__":
    main()
