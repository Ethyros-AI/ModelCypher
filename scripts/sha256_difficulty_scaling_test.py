#!/usr/bin/env python3
"""Test if manifold advantage scales with difficulty.

At 8 zeros: 28x improvement
Question: Does this scale to 12, 16, 20+ zeros?

If the advantage GROWS with difficulty, Bitcoin is in trouble.
If it SHRINKS, the manifold structure gets washed out at high difficulty.
"""

import hashlib
import struct
import numpy as np
from typing import List, Dict
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required for this test")
    exit(1)


def count_leading_zeros(hash_bytes: bytes) -> int:
    n = 0
    for byte in hash_bytes:
        if byte == 0:
            n += 8
        else:
            for i in range(7, -1, -1):
                if byte & (1 << i):
                    return n
                n += 1
    return n


def double_sha256(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def nonce_to_bits(nonce: int) -> np.ndarray:
    return np.array([(((nonce >> i) & 1) * 2 - 1) for i in range(32)], dtype=np.float32)


def bits_to_nonce(bits: np.ndarray) -> int:
    nonce = 0
    for i, b in enumerate(bits):
        if b > 0:
            nonce |= (1 << i)
    return nonce


class ManifoldVAE(nn.Module):
    def __init__(self, latent_dim: int = 17):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 32), nn.Tanh(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar, self.predictor(z).squeeze(-1)


def run_test_at_difficulty(header: bytes, target_zeros: int,
                           n_train_samples: int = 100000,
                           n_valid_needed: int = 200,
                           n_test_candidates: int = 10000):
    """Run manifold test at a specific difficulty."""

    print(f"\n{'='*70}")
    print(f"TESTING AT {target_zeros} LEADING ZEROS")
    print(f"{'='*70}\n")

    # Generate training data
    print(f"Generating training data...")
    nonces = []
    zeros = []
    valid_nonces = []

    for i in range(n_train_samples):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        z = count_leading_zeros(h)

        nonces.append(nonce)
        zeros.append(z)

        if z >= target_zeros:
            valid_nonces.append(nonce)

    print(f"  Samples: {len(nonces)}")
    print(f"  Valid found: {len(valid_nonces)}")

    # Need more valid nonces? Search for them
    if len(valid_nonces) < n_valid_needed:
        print(f"  Searching for more valid nonces...")
        search_count = 0
        max_search = 10**8

        while len(valid_nonces) < n_valid_needed and search_count < max_search:
            nonce = np.random.randint(0, 2**32)
            h = double_sha256(header + struct.pack('<I', nonce))
            z = count_leading_zeros(h)

            nonces.append(nonce)
            zeros.append(z)
            search_count += 1

            if z >= target_zeros:
                valid_nonces.append(nonce)
                if len(valid_nonces) % 50 == 0:
                    print(f"    Found {len(valid_nonces)} valid nonces...")

        print(f"  Total valid: {len(valid_nonces)} (searched {search_count} extra)")

    if len(valid_nonces) < 10:
        print(f"  NOT ENOUGH VALID NONCES - skipping")
        return None

    # Train model
    print(f"\nTraining manifold model...")

    X = np.array([nonce_to_bits(n) for n in nonces], dtype=np.float32)
    y = np.array(zeros, dtype=np.float32)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    model = ManifoldVAE(latent_dim=17)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar, pred = model(X_batch)
            loss = (nn.functional.mse_loss(recon, X_batch) +
                    0.01 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())) +
                    nn.functional.mse_loss(pred, y_batch))
            loss.backward()
            optimizer.step()

    print(f"  Training complete")

    # Map valid nonces to latent space
    model.eval()
    with torch.no_grad():
        valid_bits = torch.tensor(np.array([nonce_to_bits(n) for n in valid_nonces], dtype=np.float32))
        valid_mu, _ = model.encode(valid_bits)
        valid_latents = valid_mu.numpy()

    print(f"  Mapped {len(valid_nonces)} valid nonces to latent space")

    # Test: Random vs Interpolation
    print(f"\nEvaluating...")

    # Random baseline
    random_valid = 0
    for _ in range(n_test_candidates):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        if count_leading_zeros(h) >= target_zeros:
            random_valid += 1

    random_rate = random_valid / n_test_candidates

    # Interpolation
    interp_valid = 0
    with torch.no_grad():
        for _ in range(n_test_candidates):
            idx1, idx2 = np.random.choice(len(valid_latents), 2, replace=False)
            alpha = np.random.random()
            z = (1 - alpha) * valid_latents[idx1] + alpha * valid_latents[idx2]

            bits = model.decode(torch.tensor(z).unsqueeze(0))
            nonce = bits_to_nonce(bits.squeeze().numpy())

            h = double_sha256(header + struct.pack('<I', nonce))
            if count_leading_zeros(h) >= target_zeros:
                interp_valid += 1

    interp_rate = interp_valid / n_test_candidates

    # Results
    improvement = interp_rate / random_rate if random_rate > 0 else float('inf')
    expected_rate = 1 / (2 ** target_zeros)

    print(f"\nRESULTS at {target_zeros} zeros:")
    print(f"  Expected rate: {expected_rate*100:.6f}%")
    print(f"  Random: {random_valid}/{n_test_candidates} = {random_rate*100:.4f}%")
    print(f"  Interpolation: {interp_valid}/{n_test_candidates} = {interp_rate*100:.4f}%")
    print(f"  Improvement: {improvement:.2f}x")

    return {
        'target_zeros': target_zeros,
        'expected_rate': expected_rate,
        'random_valid': random_valid,
        'random_rate': random_rate,
        'interp_valid': interp_valid,
        'interp_rate': interp_rate,
        'improvement': improvement,
        'n_valid_training': len(valid_nonces)
    }


def main():
    print("="*70)
    print("SHA-256 MANIFOLD ADVANTAGE VS DIFFICULTY")
    print("="*70)
    print()
    print("Question: Does the 28x improvement at 8 zeros scale to harder difficulties?")
    print()

    header = b"Difficulty scaling test 2026 v2"

    results = []

    # Test at multiple difficulties
    for target_zeros in [6, 8, 10, 12]:
        result = run_test_at_difficulty(
            header,
            target_zeros=target_zeros,
            n_train_samples=200000,
            n_valid_needed=500,
            n_test_candidates=10000
        )
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: MANIFOLD ADVANTAGE VS DIFFICULTY")
    print("="*70)
    print()

    print(f"{'Zeros':<8} {'Expected':<12} {'Random':<12} {'Manifold':<12} {'Improvement':<12}")
    print("-"*60)

    for r in results:
        print(f"{r['target_zeros']:<8} {r['expected_rate']*100:.4f}%     "
              f"{r['random_rate']*100:.4f}%     {r['interp_rate']*100:.4f}%     "
              f"{r['improvement']:.1f}x")

    print()

    # Analyze scaling
    if len(results) >= 2:
        improvements = [r['improvement'] for r in results]
        zeros = [r['target_zeros'] for r in results]

        print("SCALING ANALYSIS:")
        print()

        if improvements[-1] > improvements[0]:
            print("*** ADVANTAGE INCREASES WITH DIFFICULTY ***")
            print("    This is catastrophic for Bitcoin - the harder the puzzle,")
            print("    the more the manifold structure helps!")
        elif improvements[-1] < improvements[0] * 0.5:
            print("Advantage decreases significantly with difficulty.")
            print("The manifold structure may not help at Bitcoin-scale difficulty.")
        else:
            print("Advantage is relatively stable across difficulties.")
            print("Further testing at higher difficulties needed.")

    print()


if __name__ == "__main__":
    main()
