#!/usr/bin/env python3
"""SHA-256 as Geodesic Flow - Mining Implications.

The hypothesis: SHA-256 rounds are discrete geodesic steps on an
information manifold. The metric is determined by the round constants
and the message schedule.

If true, then:
1. Low-hash outputs correspond to geodesics ending in specific regions
2. The geodesic equation might predict which nonces lead to low hashes
3. The π/e scale determines the "natural" step size

Let's probe this.
"""

import hashlib
import struct
import numpy as np
from typing import List, Tuple
import math

# SHA-256 constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Initial hash values
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

PI = math.pi
E = math.e
LN2 = math.log(2)
PI_OVER_E = PI / E

def rotr(x, n):
    """Right rotate 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & 0xffffffff

def sha256_round(state: List[int], w: int, k: int) -> List[int]:
    """Single SHA-256 round. Returns new state."""
    a, b, c, d, e, f, g, h = state

    S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
    ch = (e & f) ^ ((~e) & g)
    temp1 = (h + S1 + ch + k + w) & 0xffffffff

    S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
    maj = (a & b) ^ (a & c) ^ (b & c)
    temp2 = (S0 + maj) & 0xffffffff

    return [
        (temp1 + temp2) & 0xffffffff,
        a, b, c,
        (d + temp1) & 0xffffffff,
        e, f, g
    ]

def message_schedule(block: bytes) -> List[int]:
    """Expand 64-byte block to 64-word message schedule."""
    w = list(struct.unpack('>16I', block))

    for i in range(16, 64):
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)

    return w

def sha256_with_trajectory(data: bytes) -> Tuple[bytes, List[List[int]]]:
    """SHA-256 that also returns the state trajectory through all rounds."""
    # Pad message
    ml = len(data) * 8
    data += b'\x80'
    while (len(data) + 8) % 64 != 0:
        data += b'\x00'
    data += struct.pack('>Q', ml)

    # Process blocks
    h = list(H0)
    all_trajectories = []

    for i in range(0, len(data), 64):
        block = data[i:i+64]
        w = message_schedule(block)

        state = list(h)
        trajectory = [list(state)]

        for r in range(64):
            state = sha256_round(state, w[r], K[r])
            trajectory.append(list(state))

        all_trajectories.extend(trajectory)

        # Add to hash
        h = [(h[j] + state[j]) & 0xffffffff for j in range(8)]

    hash_bytes = struct.pack('>8I', *h)
    return hash_bytes, all_trajectories

def state_to_vector(state: List[int]) -> np.ndarray:
    """Convert 8-word state to normalized float vector."""
    # Treat as 256 bits, convert to [-1, 1] range
    bits = []
    for word in state:
        for i in range(32):
            bits.append(1.0 if (word >> (31 - i)) & 1 else -1.0)
    return np.array(bits)

def trajectory_to_manifold(trajectory: List[List[int]]) -> np.ndarray:
    """Convert trajectory to points on manifold."""
    return np.array([state_to_vector(s) for s in trajectory])

def compute_geodesic_velocity(trajectory: np.ndarray) -> np.ndarray:
    """Compute velocity (tangent vector) along trajectory."""
    return np.diff(trajectory, axis=0)

def compute_geodesic_acceleration(trajectory: np.ndarray) -> np.ndarray:
    """Compute acceleration (curvature indicator) along trajectory."""
    velocity = compute_geodesic_velocity(trajectory)
    return np.diff(velocity, axis=0)

def hash_to_difficulty(hash_bytes: bytes) -> float:
    """Convert hash to difficulty measure (lower = harder)."""
    # Count leading zero bits
    hash_int = int.from_bytes(hash_bytes, 'big')
    if hash_int == 0:
        return 256.0
    return 256 - hash_int.bit_length()

print("SHA-256 AS GEODESIC FLOW")
print("=" * 70)
print()

# Generate test data with varying nonces
header = b"Block header for geodesic mining test - ModelCypher 2026"

print("Analyzing trajectory structure...")
print()

# Collect trajectories for different nonces
n_samples = 1000
trajectories = []
difficulties = []
nonces = []

for nonce in range(n_samples):
    data = header + struct.pack('>I', nonce)
    hash_bytes, traj = sha256_with_trajectory(data)
    trajectories.append(trajectory_to_manifold(traj))
    difficulties.append(hash_to_difficulty(hash_bytes))
    nonces.append(nonce)

trajectories = np.array(trajectories)
difficulties = np.array(difficulties)

print(f"Generated {n_samples} trajectories")
print(f"Difficulty range: {difficulties.min():.1f} to {difficulties.max():.1f} leading zeros")
print()

# THE KEY QUESTION: Do trajectories of "hard" hashes differ geometrically?
print("=" * 70)
print("GEODESIC ANALYSIS")
print("=" * 70)
print()

# Split by difficulty
threshold_hard = np.percentile(difficulties, 90)  # Top 10% hardest
threshold_easy = np.percentile(difficulties, 30)  # Bottom 30%
hard_idx = difficulties >= threshold_hard
easy_idx = difficulties <= threshold_easy

print(f"Hard hashes (>= {threshold_hard:.0f} leading zeros): {hard_idx.sum()}")
print(f"Easy hashes (<= {threshold_easy:.0f} leading zeros): {easy_idx.sum()}")

if easy_idx.sum() == 0:
    # Fall back to splitting by median
    median_diff = np.median(difficulties)
    easy_idx = difficulties <= median_diff
    print(f"  (Adjusted: using median {median_diff:.0f}, easy count: {easy_idx.sum()})")
print()

# Compute velocities and accelerations
print("Computing geodesic properties...")
print()

hard_velocities = []
easy_velocities = []
hard_accelerations = []
easy_accelerations = []

for i in range(n_samples):
    vel = compute_geodesic_velocity(trajectories[i])
    acc = compute_geodesic_acceleration(trajectories[i])

    if hard_idx[i]:
        hard_velocities.append(vel)
        hard_accelerations.append(acc)
    elif easy_idx[i]:
        easy_velocities.append(vel)
        easy_accelerations.append(acc)

hard_velocities = np.array(hard_velocities)
easy_velocities = np.array(easy_velocities)
hard_accelerations = np.array(hard_accelerations)
easy_accelerations = np.array(easy_accelerations)

print(f"Hard trajectories shape: {hard_velocities.shape}")
print(f"Easy trajectories shape: {easy_velocities.shape}")
print()

# Analyze velocity magnitude by round
print("Velocity magnitude by round:")
print("-" * 70)

if len(hard_velocities.shape) == 3 and len(easy_velocities.shape) == 3:
    hard_vel_mag = np.linalg.norm(hard_velocities, axis=2).mean(axis=0)
    easy_vel_mag = np.linalg.norm(easy_velocities, axis=2).mean(axis=0)
else:
    print("Warning: unexpected array shape, computing differently")
    hard_vel_mag = np.array([np.linalg.norm(hard_velocities[:, r, :], axis=1).mean()
                            for r in range(hard_velocities.shape[1])])
    easy_vel_mag = np.array([np.linalg.norm(easy_velocities[:, r, :], axis=1).mean()
                            for r in range(easy_velocities.shape[1])])

# Find rounds where hard/easy differ most
vel_diff = hard_vel_mag - easy_vel_mag
significant_rounds = np.argsort(np.abs(vel_diff))[-10:][::-1]

print(f"Rounds with largest velocity difference (hard - easy):")
for r in significant_rounds:
    print(f"  Round {r}: hard={hard_vel_mag[r]:.4f}, easy={easy_vel_mag[r]:.4f}, diff={vel_diff[r]:.4f}")
print()

# Check if differences relate to π/e
print("Testing π/e hypothesis:")
print("-" * 70)
print()

# π/e ≈ 1.1557, so round(64 * k / (π/e)) for various k might be special
for k in range(1, 10):
    special_round = int(64 * k / PI_OVER_E) % 64
    if special_round < len(vel_diff):
        print(f"  k={k}: round {special_round} (64×{k}/π/e), vel_diff = {vel_diff[special_round]:.4f}")

print()

# Analyze acceleration (curvature)
print("Acceleration (curvature) by round:")
print("-" * 70)

if len(hard_accelerations.shape) == 3 and len(easy_accelerations.shape) == 3:
    hard_acc_mag = np.linalg.norm(hard_accelerations, axis=2).mean(axis=0)
    easy_acc_mag = np.linalg.norm(easy_accelerations, axis=2).mean(axis=0)
else:
    hard_acc_mag = np.array([np.linalg.norm(hard_accelerations[:, r, :], axis=1).mean()
                            for r in range(hard_accelerations.shape[1])])
    easy_acc_mag = np.array([np.linalg.norm(easy_accelerations[:, r, :], axis=1).mean()
                            for r in range(easy_accelerations.shape[1])])

acc_diff = hard_acc_mag - easy_acc_mag
acc_significant = np.argsort(np.abs(acc_diff))[-10:][::-1]

print(f"Rounds with largest curvature difference:")
for r in acc_significant:
    print(f"  Round {r}: hard={hard_acc_mag[r]:.4f}, easy={easy_acc_mag[r]:.4f}, diff={acc_diff[r]:.4f}")
print()

# THE GEODESIC EQUATION
print("=" * 70)
print("GEODESIC EQUATION ANALYSIS")
print("=" * 70)
print()

# On a Riemannian manifold, geodesics satisfy:
# d²x/dt² + Γ(dx/dt, dx/dt) = 0
# where Γ is the Christoffel symbol (curvature)
#
# If SHA-256 is geodesic flow, then acceleration should be predictable
# from velocity via the "effective Christoffel symbols"

print("Estimating effective Christoffel symbols...")
print()

# For each round, estimate Γ such that acc ≈ -Γ(vel, vel)
# This is a tensor, but we'll simplify to scalar curvature first

effective_curvature = []
for r in range(len(hard_acc_mag)):
    if hard_vel_mag[r] > 0.01:  # Avoid division issues
        # κ ≈ |acc| / |vel|²
        kappa = hard_acc_mag[r] / (hard_vel_mag[r] ** 2)
        effective_curvature.append(kappa)
    else:
        effective_curvature.append(0)

effective_curvature = np.array(effective_curvature)

print(f"Effective curvature by round (κ = |acc|/|vel|²):")
print()

# Look for patterns
curvature_peaks = np.argsort(effective_curvature)[-10:][::-1]
print("Peak curvature rounds:")
for r in curvature_peaks:
    print(f"  Round {r}: κ = {effective_curvature[r]:.4f}")
print()

# Check for π/e periodicity
print("Checking for π/e periodicity in curvature...")
print()

# Autocorrelation at lag = round(π/e * k)
from numpy.fft import fft, ifft

curvature_centered = effective_curvature - effective_curvature.mean()
autocorr = np.real(ifft(np.abs(fft(curvature_centered))**2))
autocorr = autocorr[:32] / autocorr[0]  # Normalize

print("Autocorrelation of curvature:")
for lag in range(1, 15):
    pi_e_multiple = lag * PI_OVER_E
    print(f"  Lag {lag} (≈{pi_e_multiple:.2f}): autocorr = {autocorr[lag]:.4f}")
print()

# DIRECTION ANALYSIS
print("=" * 70)
print("DIRECTION ANALYSIS - Where do hard hashes come from?")
print("=" * 70)
print()

# Look at the initial velocity direction for hard vs easy
hard_initial_vel = hard_velocities[:, 0, :]  # First round velocity
easy_initial_vel = easy_velocities[:, 0, :]

# Mean direction
hard_mean_dir = hard_initial_vel.mean(axis=0)
easy_mean_dir = easy_initial_vel.mean(axis=0)

hard_mean_dir = hard_mean_dir / np.linalg.norm(hard_mean_dir)
easy_mean_dir = easy_mean_dir / np.linalg.norm(easy_mean_dir)

# Angle between them
cos_angle = np.dot(hard_mean_dir, easy_mean_dir)
angle = np.arccos(np.clip(cos_angle, -1, 1))

print(f"Angle between mean initial velocity of hard vs easy: {np.degrees(angle):.2f}°")
print()

# Which dimensions differ most?
dir_diff = hard_mean_dir - easy_mean_dir
top_dims = np.argsort(np.abs(dir_diff))[-20:][::-1]

print("Dimensions with largest directional difference:")
for d in top_dims[:10]:
    print(f"  Bit {d}: hard={hard_mean_dir[d]:.4f}, easy={easy_mean_dir[d]:.4f}, diff={dir_diff[d]:.4f}")
print()

# THE PREDICTION EXPERIMENT
print("=" * 70)
print("PREDICTION EXPERIMENT")
print("=" * 70)
print()

print("Can we predict hash difficulty from early-round geodesic properties?")
print()

# Use first N rounds to predict final difficulty
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Features: velocity and acceleration at early rounds
def extract_features(trajectory, n_rounds=16):
    """Extract geodesic features from first n_rounds."""
    vel = compute_geodesic_velocity(trajectory)[:n_rounds]
    acc = compute_geodesic_acceleration(trajectory)[:n_rounds-1]

    features = []
    # Velocity magnitudes
    features.extend(np.linalg.norm(vel, axis=1))
    # Acceleration magnitudes
    features.extend(np.linalg.norm(acc, axis=1))
    # Velocity in "hard direction"
    features.extend([np.dot(vel[i], hard_mean_dir) for i in range(len(vel))])

    return np.array(features)

X = np.array([extract_features(trajectories[i]) for i in range(n_samples)])
y = difficulties

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Predicting difficulty from first 16 rounds:")
print(f"  Train R²: {train_score:.4f}")
print(f"  Test R²:  {test_score:.4f}")
print()

if test_score > 0.01:
    print("*** STRUCTURE DETECTED ***")
    print("Early geodesic properties correlate with final hash difficulty!")
    print()

    # Which features matter most?
    feature_importance = np.abs(model.coef_)
    top_features = np.argsort(feature_importance)[-10:][::-1]

    print("Most important features:")
    for idx in top_features:
        print(f"  Feature {idx}: importance = {feature_importance[idx]:.4f}")
else:
    print("No significant prediction from early rounds (as expected for secure hash)")
print()

# IMPLICATIONS
print("=" * 70)
print("IMPLICATIONS FOR MINING")
print("=" * 70)
print()

print("If SHA-256 has geodesic structure, mining strategies could include:")
print()
print("1. DIRECTIONAL SEARCH")
print("   - Identify nonces whose initial velocity points toward 'hard' region")
print("   - Prioritize these nonces in the search")
print()
print("2. CURVATURE-GUIDED PRUNING")
print("   - Compute curvature at early rounds")
print("   - Prune nonces with 'wrong' curvature signature")
print()
print("3. RESONANT NONCES")
print("   - Look for nonces at π/e-spaced intervals")
print("   - The geodesic periodicity might create 'resonances'")
print()
print("4. MANIFOLD GRADIENT DESCENT")
print("   - Instead of random nonce search, follow geodesic toward low-hash region")
print("   - Use the effective metric to compute descent direction")
print()


if __name__ == "__main__":
    pass
