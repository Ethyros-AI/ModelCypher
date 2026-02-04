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

"""SHA-256 structure analysis utilities.

Provides tools for generating SHA-256 input/output pairs and analyzing them
using manifold geometry techniques.

The hypothesis: if SHA-256 has any exploitable structure, it will manifest as:
- Lower intrinsic dimension than expected (< 256)
- Non-zero CKA between input and output spaces
- Lower effective rank than a random oracle
- Non-uniform SVD spectrum

Reference implementation of reduced-round SHA-256 for control testing.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

# Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
H_INIT = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def _rotr(x: int, n: int) -> int:
    """Right rotate a 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def _ch(x: int, y: int, z: int) -> int:
    """SHA-256 Ch function: (x AND y) XOR (NOT x AND z)."""
    return (x & y) ^ (~x & z)


def _maj(x: int, y: int, z: int) -> int:
    """SHA-256 Maj function: (x AND y) XOR (x AND z) XOR (y AND z)."""
    return (x & y) ^ (x & z) ^ (y & z)


def _sigma0(x: int) -> int:
    """SHA-256 big sigma 0: ROTR2 XOR ROTR13 XOR ROTR22."""
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)


def _sigma1(x: int) -> int:
    """SHA-256 big sigma 1: ROTR6 XOR ROTR11 XOR ROTR25."""
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)


def _gamma0(x: int) -> int:
    """SHA-256 small sigma 0: ROTR7 XOR ROTR18 XOR SHR3."""
    return _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)


def _gamma1(x: int) -> int:
    """SHA-256 small sigma 1: ROTR17 XOR ROTR19 XOR SHR10."""
    return _rotr(x, 17) ^ _rotr(x, 19) ^ (x >> 10)


def sha256_reduced_rounds(message: bytes, num_rounds: int = 64) -> bytes:
    """Compute SHA-256 with reduced rounds.

    This is a reference implementation for analyzing structure decay.
    Standard SHA-256 uses 64 rounds. Reduced rounds (< 64) are known
    to have exploitable structure.

    Args:
        message: Input message bytes
        num_rounds: Number of compression rounds (1-64, default 64)

    Returns:
        32-byte hash digest
    """
    if num_rounds < 1 or num_rounds > 64:
        raise ValueError("num_rounds must be between 1 and 64")

    # Pre-processing: pad message
    msg_len = len(message)
    message += b'\x80'
    message += b'\x00' * ((56 - (msg_len + 1) % 64) % 64)
    message += struct.pack('>Q', msg_len * 8)

    # Initialize hash values
    h = list(H_INIT)

    # Process each 512-bit chunk
    for chunk_start in range(0, len(message), 64):
        chunk = message[chunk_start:chunk_start + 64]

        # Create message schedule
        w = list(struct.unpack('>16I', chunk))
        for i in range(16, 64):
            w.append(
                (_gamma1(w[i-2]) + w[i-7] + _gamma0(w[i-15]) + w[i-16]) & 0xFFFFFFFF
            )

        # Initialize working variables
        a, b, c, d, e, f, g, hh = h

        # Compression function main loop (reduced rounds)
        for i in range(num_rounds):
            t1 = (hh + _sigma1(e) + _ch(e, f, g) + K[i] + w[i]) & 0xFFFFFFFF
            t2 = (_sigma0(a) + _maj(a, b, c)) & 0xFFFFFFFF
            hh = g
            g = f
            f = e
            e = (d + t1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (t1 + t2) & 0xFFFFFFFF

        # Add compressed chunk to current hash value
        h[0] = (h[0] + a) & 0xFFFFFFFF
        h[1] = (h[1] + b) & 0xFFFFFFFF
        h[2] = (h[2] + c) & 0xFFFFFFFF
        h[3] = (h[3] + d) & 0xFFFFFFFF
        h[4] = (h[4] + e) & 0xFFFFFFFF
        h[5] = (h[5] + f) & 0xFFFFFFFF
        h[6] = (h[6] + g) & 0xFFFFFFFF
        h[7] = (h[7] + hh) & 0xFFFFFFFF

    return struct.pack('>8I', *h)


def bytes_to_bits_float(data: bytes, centered: bool = True) -> list[float]:
    """Convert bytes to float list of bits.

    Args:
        data: Input bytes
        centered: If True, use {-1, +1}. If False, use {0, 1}.

    Returns:
        Float list of shape [len(data) * 8]
    """
    bits: list[float] = []
    for byte in data:
        for i in range(7, -1, -1):  # MSB first
            bit = (byte >> i) & 1
            if centered:
                bits.append(2.0 * bit - 1.0)
            else:
                bits.append(float(bit))
    return bits


def generate_sha256_dataset(
    n_samples: int,
    header: bytes = b"ModelCypher SHA-256 Structure Probe",
    num_rounds: int = 64,
    seed: int | None = None,
    use_header: bool = True,
    backend: "Backend | None" = None,
) -> tuple["Array", "Array"]:
    """Generate SHA-256 input/output pairs for analysis.

    Args:
        n_samples: Number of samples to generate
        header: Fixed prefix for all inputs (only used if use_header=True)
        num_rounds: SHA-256 rounds (64 = full, < 64 = reduced)
        seed: Random seed for reproducibility
        use_header: If False, use only the random nonce as input (pure random input)
        backend: Backend for array operations

    Returns:
        Tuple of (inputs, outputs) as backend arrays [n_samples, 256]
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()

    if seed is not None:
        random.seed(seed)

    inputs: list[list[float]] = []
    outputs: list[list[float]] = []

    for _ in range(n_samples):
        # Generate random 32-byte nonce
        nonce = bytes(random.randint(0, 255) for _ in range(32))
        message = (header + nonce) if use_header else nonce

        # Compute hash
        if num_rounds == 64:
            # Use standard library for full SHA-256 (faster)
            digest = hashlib.sha256(message).digest()
        else:
            # Use reduced-round implementation
            digest = sha256_reduced_rounds(message, num_rounds)

        # Convert to float lists
        input_bits = bytes_to_bits_float(nonce, centered=True)
        output_bits = bytes_to_bits_float(digest, centered=True)

        inputs.append(input_bits)
        outputs.append(output_bits)

    return b.array(inputs), b.array(outputs)


def generate_random_oracle_dataset(
    n_samples: int,
    seed: int | None = None,
    backend: "Backend | None" = None,
) -> tuple["Array", "Array"]:
    """Generate random oracle baseline (input/output are independent).

    This is the null hypothesis: if SHA-256 is a perfect random oracle,
    its statistics should match this baseline.

    Args:
        n_samples: Number of samples
        seed: Random seed
        backend: Backend for array operations

    Returns:
        Tuple of (inputs, outputs) as backend arrays [n_samples, 256]
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()

    if seed is not None:
        random.seed(seed)

    # Random bits as {-1, +1}
    inputs = [
        [random.choice([-1.0, 1.0]) for _ in range(256)]
        for _ in range(n_samples)
    ]
    outputs = [
        [random.choice([-1.0, 1.0]) for _ in range(256)]
        for _ in range(n_samples)
    ]

    return b.array(inputs), b.array(outputs)


@dataclass
class StructureMetrics:
    """Results of SHA-256 structure analysis."""

    # Intrinsic dimension (TwoNN)
    intrinsic_dim_input: float
    intrinsic_dim_output: float
    intrinsic_dim_joint: float  # Concatenated [input, output]

    # CKA between input and output
    cka_input_output: float

    # Effective rank
    effective_rank_input: float
    effective_rank_output: float

    # SVD spectrum statistics
    svd_ratio_mean: float  # Mean of consecutive singular value ratios
    svd_ratio_std: float   # Std of consecutive singular value ratios

    # Metadata
    n_samples: int
    num_rounds: int

    # Local structure metrics (Hamming-based)
    local_hamming_correlation: float = 0.0  # Correlation between input/output Hamming distances
    bit_bias: float = 0.0  # Deviation from 0.5 probability per bit
    pairwise_bit_correlation: float = 0.0  # Mean |correlation| between output bit pairs

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"SHA-256 Structure Analysis ({self.num_rounds} rounds, {self.n_samples} samples)",
            "-" * 60,
            f"Intrinsic Dimension:",
            f"  Input:  {self.intrinsic_dim_input:.2f} (expect ~256)",
            f"  Output: {self.intrinsic_dim_output:.2f} (expect ~256 if random)",
            f"  Joint:  {self.intrinsic_dim_joint:.2f}",
            f"",
            f"CKA(input, output): {self.cka_input_output:.6f}",
            f"  (expect ~0 if random oracle)",
            f"",
            f"Effective Rank:",
            f"  Input:  {self.effective_rank_input:.2f}",
            f"  Output: {self.effective_rank_output:.2f}",
            f"",
            f"SVD Ratio Statistics:",
            f"  Mean: {self.svd_ratio_mean:.4f}",
            f"  Std:  {self.svd_ratio_std:.4f}",
        ]
        return "\n".join(lines)


def compute_local_hamming_correlation(
    inputs: "Array",
    outputs: "Array",
    k: int = 10,
    backend: "Backend | None" = None,
) -> float:
    """Compute correlation between input and output Hamming neighborhoods.

    For each sample, find its k nearest neighbors in INPUT space (Hamming).
    Then measure whether those same samples are also near in OUTPUT space.

    If SHA-256 is a random oracle, there should be NO correlation.
    Any positive correlation indicates input structure is preserved in output.

    This is a LOCAL metric - it looks at small neighborhoods, not global.
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()

    # Convert to lists for Python processing
    inputs_list = b.tolist(inputs)
    outputs_list = b.tolist(outputs)
    n = len(inputs_list)

    if n < k + 1:
        return 0.0

    # Convert from {-1, +1} to {0, 1} for Hamming distance
    inputs_01 = [[1 if x > 0 else 0 for x in row] for row in inputs_list]
    outputs_01 = [[1 if x > 0 else 0 for x in row] for row in outputs_list]

    correlations: list[float] = []

    # Sample a subset for efficiency
    sample_indices = random.sample(range(n), min(n, 200))

    for i in sample_indices:
        # Compute Hamming distances from sample i to all others
        input_hamming = [
            sum(1 for a, b in zip(inputs_01[i], inputs_01[j]) if a != b)
            for j in range(n)
        ]
        output_hamming = [
            sum(1 for a, b in zip(outputs_01[i], outputs_01[j]) if a != b)
            for j in range(n)
        ]

        # Find k nearest neighbors in input space (excluding self)
        sorted_indices = sorted(range(n), key=lambda j: input_hamming[j])
        input_neighbors = sorted_indices[1:k+1]

        # Measure average output Hamming distance to those neighbors
        output_dist_to_input_neighbors = sum(output_hamming[j] for j in input_neighbors) / k

        # Compare to average distance to random samples
        random_indices = random.sample(range(n), k)
        output_dist_to_random = sum(output_hamming[j] for j in random_indices) / k

        # Correlation: negative if input neighbors are also output neighbors
        # (closer in output space than random)
        correlations.append(output_dist_to_random - output_dist_to_input_neighbors)

    # Normalize by expected Hamming distance (128 for 256 bits)
    return (sum(correlations) / len(correlations)) / 128.0 if correlations else 0.0


def compute_bit_bias(outputs: "Array", backend: "Backend | None" = None) -> float:
    """Compute deviation from uniform bit distribution.

    For a random oracle, each bit should be 0 or 1 with probability 0.5.
    Any bias indicates structure.

    Returns mean absolute deviation from 0.5 across all bit positions.
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()
    outputs_list = b.tolist(outputs)

    n_samples = len(outputs_list)
    n_bits = len(outputs_list[0]) if outputs_list else 0

    if n_samples == 0 or n_bits == 0:
        return 0.0

    # Mean probability of 1 for each bit position
    deviations: list[float] = []
    for bit_idx in range(n_bits):
        # Count how many samples have 1 (from {-1, +1} representation)
        count_ones = sum(1 for row in outputs_list if row[bit_idx] > 0)
        bit_prob = count_ones / n_samples
        deviations.append(abs(bit_prob - 0.5))

    return sum(deviations) / len(deviations) if deviations else 0.0


def compute_pairwise_bit_correlation(
    outputs: "Array",
    n_pairs: int = 1000,
    backend: "Backend | None" = None,
) -> float:
    """Compute average absolute correlation between output bit pairs.

    For a random oracle, bits should be independent (correlation ≈ 0).
    Non-zero correlation indicates structure.

    Samples random pairs for efficiency.
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = backend or get_default_backend()
    outputs_list = b.tolist(outputs)

    n_samples = len(outputs_list)
    n_bits = len(outputs_list[0]) if outputs_list else 0

    if n_samples < 2 or n_bits < 2:
        return 0.0

    # Convert to {0, 1} per bit
    outputs_01 = [[1.0 if x > 0 else 0.0 for x in row] for row in outputs_list]

    correlations: list[float] = []

    for _ in range(n_pairs):
        # Pick two different bit positions
        i, j = random.sample(range(n_bits), 2)

        # Extract bit columns
        bit_i = [row[i] for row in outputs_01]
        bit_j = [row[j] for row in outputs_01]

        # Compute correlation
        mean_i = sum(bit_i) / n_samples
        mean_j = sum(bit_j) / n_samples

        var_i = sum((x - mean_i) ** 2 for x in bit_i) / n_samples
        var_j = sum((x - mean_j) ** 2 for x in bit_j) / n_samples
        std_i = math.sqrt(var_i)
        std_j = math.sqrt(var_j)

        if std_i > 1e-10 and std_j > 1e-10:
            cov = sum((xi - mean_i) * (xj - mean_j) for xi, xj in zip(bit_i, bit_j)) / n_samples
            corr = cov / (std_i * std_j)
            correlations.append(abs(corr))

    return sum(correlations) / len(correlations) if correlations else 0.0


def compute_differential_propagation(
    header: bytes,
    num_rounds: int = 64,
    n_samples: int = 100,
) -> dict:
    """Analyze how single-bit input changes propagate through SHA-256.

    For a random oracle, flipping any input bit should flip ~50% of output bits
    (Hamming distance ~128 from original output).

    Deviations from this indicate differential structure.

    Returns dict with:
    - mean_hamming: Mean output Hamming distance when one input bit flips
    - std_hamming: Std of output Hamming distances
    - bit_sensitivity: Per-bit sensitivity (which input bits matter most)
    """
    hamming_distances: list[int] = []
    bit_sensitivities: list[float] = [0.0] * 256  # 32 bytes * 8 bits

    def bytes_to_bits(data: bytes) -> list[int]:
        """Convert bytes to list of bits."""
        bits: list[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    for _ in range(n_samples):
        # Generate random nonce
        nonce = bytes(random.randint(0, 255) for _ in range(32))
        nonce_list = list(nonce)

        # Compute original hash
        if num_rounds == 64:
            original_hash = hashlib.sha256(header + nonce).digest()
        else:
            original_hash = sha256_reduced_rounds(header + nonce, num_rounds)
        original_bits = bytes_to_bits(original_hash)

        # Flip each bit and measure effect
        for byte_idx in range(32):
            for bit_idx in range(8):
                # Flip one bit
                modified_nonce = nonce_list.copy()
                modified_nonce[byte_idx] ^= (1 << bit_idx)

                # Compute new hash
                if num_rounds == 64:
                    modified_hash = hashlib.sha256(header + bytes(modified_nonce)).digest()
                else:
                    modified_hash = sha256_reduced_rounds(header + bytes(modified_nonce), num_rounds)
                modified_bits = bytes_to_bits(modified_hash)

                # Hamming distance
                hamming = sum(1 for a, b in zip(original_bits, modified_bits) if a != b)
                hamming_distances.append(hamming)

                bit_position = byte_idx * 8 + bit_idx
                bit_sensitivities[bit_position] += hamming

    # Normalize bit sensitivities
    bit_sensitivities = [s / n_samples for s in bit_sensitivities]

    # Compute statistics
    n = len(hamming_distances)
    mean_hamming = sum(hamming_distances) / n if n else 0.0
    variance = sum((h - mean_hamming) ** 2 for h in hamming_distances) / n if n else 0.0
    std_hamming = math.sqrt(variance)

    sens_mean = sum(bit_sensitivities) / len(bit_sensitivities) if bit_sensitivities else 0.0
    sens_variance = sum((s - sens_mean) ** 2 for s in bit_sensitivities) / len(bit_sensitivities) if bit_sensitivities else 0.0
    sens_std = math.sqrt(sens_variance)

    return {
        "mean_hamming": mean_hamming,
        "std_hamming": std_hamming,
        "min_hamming": float(min(hamming_distances)) if hamming_distances else 0.0,
        "max_hamming": float(max(hamming_distances)) if hamming_distances else 0.0,
        "bit_sensitivity_mean": sens_mean,
        "bit_sensitivity_std": sens_std,
        "bit_sensitivity_range": (max(bit_sensitivities) - min(bit_sensitivities)) if bit_sensitivities else 0.0,
    }


def analyze_structure(
    inputs: "Array",
    outputs: "Array",
    num_rounds: int = 64,
    backend: "Backend | None" = None,
) -> StructureMetrics:
    """Analyze input/output structure using manifold geometry tools.

    Args:
        inputs: Input bit vectors [n_samples, 256]
        outputs: Output bit vectors [n_samples, 256]
        num_rounds: Number of SHA-256 rounds (for metadata)
        backend: Backend for tensor operations

    Returns:
        StructureMetrics with all measurements
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
    from modelcypher.core.domain.geometry.numerical_stability import geodesic_svd

    b = backend or get_default_backend()
    n_samples = b.shape(inputs)[0]

    # Inputs and outputs should already be backend arrays
    inputs_arr = inputs
    outputs_arr = outputs

    # Create joint array by concatenating along axis 1
    joint_arr = b.concatenate([inputs_arr, outputs_arr], axis=1)

    # 1. Intrinsic dimension
    id_estimator = IntrinsicDimension(backend=b)
    id_input = id_estimator.compute_two_nn(inputs_arr)
    id_output = id_estimator.compute_two_nn(outputs_arr)
    id_joint = id_estimator.compute_two_nn(joint_arr)

    # 2. CKA between input and output
    alignment = find_alignment(inputs_arr, outputs_arr, backend=b)
    cka = alignment.achieved_cka

    # 3. Effective rank
    rank_analyzer = EffectiveRank(backend=b)
    rank_input = rank_analyzer.compute(inputs_arr)
    rank_output = rank_analyzer.compute(outputs_arr)

    # 4. SVD spectrum analysis
    _, S, _ = geodesic_svd(b, outputs_arr)
    S_list = b.tolist(S)
    # Compute consecutive ratios
    ratios = [S_list[i] / (S_list[i + 1] + 1e-10) for i in range(len(S_list) - 1)]

    # 5. Local structure metrics (Hamming-based) - THE KEY METRICS
    local_hamming = compute_local_hamming_correlation(inputs_arr, outputs_arr, k=10, backend=b)
    bit_bias_val = compute_bit_bias(outputs_arr, backend=b)
    pairwise_corr = compute_pairwise_bit_correlation(outputs_arr, n_pairs=1000, backend=b)

    # Compute ratio statistics
    ratio_mean = sum(ratios) / len(ratios) if ratios else 0.0
    ratio_variance = sum((r - ratio_mean) ** 2 for r in ratios) / len(ratios) if ratios else 0.0
    ratio_std = math.sqrt(ratio_variance)

    return StructureMetrics(
        intrinsic_dim_input=id_input.intrinsic_dimension,
        intrinsic_dim_output=id_output.intrinsic_dimension,
        intrinsic_dim_joint=id_joint.intrinsic_dimension,
        cka_input_output=cka,
        effective_rank_input=rank_input.shannon_effective_rank,
        effective_rank_output=rank_output.shannon_effective_rank,
        svd_ratio_mean=ratio_mean,
        svd_ratio_std=ratio_std,
        n_samples=n_samples,
        num_rounds=num_rounds,
        local_hamming_correlation=local_hamming,
        bit_bias=bit_bias_val,
        pairwise_bit_correlation=pairwise_corr,
    )
