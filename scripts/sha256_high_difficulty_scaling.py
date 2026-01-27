#!/usr/bin/env python3
"""SHA-256 High Difficulty Scaling Test.

Test the manifold advantage at 12, 14, 16+ zeros.

At 10 zeros: 42.5x improvement
Question: Does this continue to scale?

If improvement grows exponentially with difficulty → Bitcoin is vulnerable
If improvement plateaus or decreases → Structure washes out at high difficulty
"""

import hashlib
import struct
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from functools import partial

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required")
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


def search_for_valid_nonces(header: bytes, target_zeros: int, n_needed: int,
                            max_search: int = 10**9) -> Tuple[List[int], List[int], List[int]]:
    """
    Search for valid nonces, collecting all samples along the way.
    Returns (all_nonces, all_zeros, valid_nonces)
    """
    all_nonces = []
    all_zeros = []
    valid_nonces = []

    searched = 0
    start_time = time.time()
    last_report = start_time

    while len(valid_nonces) < n_needed and searched < max_search:
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        z = count_leading_zeros(h)

        all_nonces.append(nonce)
        all_zeros.append(z)
        searched += 1

        if z >= target_zeros:
            valid_nonces.append(nonce)

        # Progress report every 10 seconds
        now = time.time()
        if now - last_report > 10:
            elapsed = now - start_time
            rate = searched / elapsed
            expected_per_valid = 2 ** target_zeros
            eta = (n_needed - len(valid_nonces)) * expected_per_valid / rate if rate > 0 else float('inf')
            print(f"    Searched {searched:,}, found {len(valid_nonces)}/{n_needed}, "
                  f"rate: {rate:,.0f}/s, ETA: {eta/60:.1f}min")
            last_report = now

    return all_nonces, all_zeros, valid_nonces


class ManifoldVAE(nn.Module):
    """Improved VAE with larger capacity."""
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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


def train_model(nonces: List[int], zeros: List[int], epochs: int = 100,
                latent_dim: int = 32, device: str = 'cpu') -> ManifoldVAE:
    """Train the VAE on collected data."""
    X = np.array([nonce_to_bits(n) for n in nonces], dtype=np.float32)
    y = np.array(zeros, dtype=np.float32)

    X_tensor = torch.tensor(X).to(device)
    y_tensor = torch.tensor(y).to(device)

    model = ManifoldVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    train_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            recon, mu, logvar, pred = model(X_batch)

            recon_loss = nn.functional.mse_loss(recon, X_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = nn.functional.mse_loss(pred, y_batch)

            # Weight prediction loss more heavily
            loss = recon_loss + 0.01 * kl_loss + 2.0 * pred_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

    return model


def evaluate_strategies(model: ManifoldVAE, valid_nonces: List[int],
                       header: bytes, target_zeros: int,
                       n_test: int = 10000, device: str = 'cpu') -> Dict:
    """Evaluate random vs manifold-guided search."""
    model.eval()

    # Get valid latent codes
    with torch.no_grad():
        valid_bits = torch.tensor(
            np.array([nonce_to_bits(n) for n in valid_nonces], dtype=np.float32)
        ).to(device)
        valid_mu, _ = model.encode(valid_bits)
        valid_latents = valid_mu.cpu().numpy()

    results = {}

    # Random baseline
    print(f"    Testing random search ({n_test} candidates)...")
    random_valid = 0
    for _ in range(n_test):
        nonce = np.random.randint(0, 2**32)
        h = double_sha256(header + struct.pack('<I', nonce))
        if count_leading_zeros(h) >= target_zeros:
            random_valid += 1

    results['random'] = {
        'valid': random_valid,
        'total': n_test,
        'rate': random_valid / n_test
    }

    # Manifold interpolation
    print(f"    Testing manifold interpolation ({n_test} candidates)...")
    interp_valid = 0

    with torch.no_grad():
        batch_size = 1000
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_n = batch_end - batch_start

            # Generate interpolations
            idx1 = np.random.choice(len(valid_latents), batch_n)
            idx2 = np.random.choice(len(valid_latents), batch_n)
            alphas = np.random.random(batch_n).reshape(-1, 1)

            z_interp = (1 - alphas) * valid_latents[idx1] + alphas * valid_latents[idx2]
            z_tensor = torch.tensor(z_interp.astype(np.float32)).to(device)

            # Decode
            bits = model.decode(z_tensor).cpu().numpy()

            # Check each
            for b in bits:
                nonce = bits_to_nonce(b)
                h = double_sha256(header + struct.pack('<I', nonce))
                if count_leading_zeros(h) >= target_zeros:
                    interp_valid += 1

    results['interpolation'] = {
        'valid': interp_valid,
        'total': n_test,
        'rate': interp_valid / n_test
    }

    # Compute improvement
    if results['random']['rate'] > 0:
        results['improvement'] = results['interpolation']['rate'] / results['random']['rate']
    else:
        results['improvement'] = float('inf') if results['interpolation']['valid'] > 0 else 1.0

    return results


def run_difficulty_test(target_zeros: int, header: bytes,
                       n_valid_needed: int = 500,
                       n_test_candidates: int = 10000,
                       epochs: int = 100) -> Optional[Dict]:
    """Run complete test at a specific difficulty."""

    print(f"\n{'='*70}")
    print(f"TESTING AT {target_zeros} LEADING ZEROS")
    print(f"{'='*70}")
    print()

    # Estimate time
    expected_per_valid = 2 ** target_zeros
    print(f"  Expected hashes per valid nonce: {expected_per_valid:,}")
    print(f"  Need {n_valid_needed} valid nonces for training")
    print()

    # Search for valid nonces
    print("  Searching for valid nonces...")
    start = time.time()
    all_nonces, all_zeros, valid_nonces = search_for_valid_nonces(
        header, target_zeros, n_valid_needed
    )
    search_time = time.time() - start

    print(f"  Found {len(valid_nonces)} valid nonces in {search_time:.1f}s")
    print(f"  Total samples collected: {len(all_nonces):,}")
    print()

    if len(valid_nonces) < 50:
        print("  NOT ENOUGH VALID NONCES - need at least 50")
        return None

    # Determine device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Train model
    print("  Training manifold model...")
    start = time.time()
    model = train_model(all_nonces, all_zeros, epochs=epochs, device=device)
    train_time = time.time() - start
    print(f"  Training completed in {train_time:.1f}s")
    print()

    # Evaluate
    print("  Evaluating strategies...")
    results = evaluate_strategies(
        model, valid_nonces, header, target_zeros,
        n_test=n_test_candidates, device=device
    )

    # Print results
    print()
    print(f"  RESULTS at {target_zeros} zeros:")
    print(f"    Expected rate: {100/2**target_zeros:.6f}%")
    print(f"    Random: {results['random']['valid']}/{results['random']['total']} = {results['random']['rate']*100:.4f}%")
    print(f"    Interpolation: {results['interpolation']['valid']}/{results['interpolation']['total']} = {results['interpolation']['rate']*100:.4f}%")
    print(f"    Improvement: {results['improvement']:.1f}x")

    return {
        'target_zeros': target_zeros,
        'n_valid_training': len(valid_nonces),
        'n_total_samples': len(all_nonces),
        'search_time': search_time,
        'train_time': train_time,
        'expected_rate': 1 / 2**target_zeros,
        **results
    }


def main():
    print("="*70)
    print("SHA-256 HIGH DIFFICULTY MANIFOLD SCALING TEST")
    print("="*70)
    print()
    print("Previous results:")
    print("  6 zeros: 0.9x")
    print("  8 zeros: 1.1x")
    print("  10 zeros: 42.5x")
    print()
    print("Question: Does the advantage continue to grow at 12, 14, 16 zeros?")
    print()

    header = b"High difficulty manifold scaling test 2026"

    results = []

    # Start with 10 zeros to confirm baseline, then go higher
    difficulties = [10, 12, 14]

    for target_zeros in difficulties:
        result = run_difficulty_test(
            target_zeros=target_zeros,
            header=header,
            n_valid_needed=300,  # Fewer needed for higher difficulty (takes longer)
            n_test_candidates=5000,
            epochs=100
        )
        if result:
            results.append(result)

            # Save intermediate results
            with open('sha256_scaling_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: MANIFOLD ADVANTAGE VS DIFFICULTY")
    print("="*70)
    print()

    print(f"{'Zeros':<8} {'Expected':<12} {'Random':<12} {'Manifold':<12} {'Improvement':<12}")
    print("-"*70)

    for r in results:
        print(f"{r['target_zeros']:<8} {r['expected_rate']*100:.6f}%   "
              f"{r['random']['rate']*100:.4f}%     {r['interpolation']['rate']*100:.4f}%     "
              f"{r['improvement']:.1f}x")

    print()

    # Analyze scaling
    if len(results) >= 2:
        improvements = [r['improvement'] for r in results]
        zeros_list = [r['target_zeros'] for r in results]

        print("SCALING ANALYSIS:")
        print()

        # Compute growth rate
        if len(improvements) >= 2 and all(i > 0 for i in improvements):
            # Log-linear fit
            log_imp = np.log(improvements)
            slope = (log_imp[-1] - log_imp[0]) / (zeros_list[-1] - zeros_list[0])
            growth_per_zero = np.exp(slope)

            print(f"  Growth factor per zero: {growth_per_zero:.2f}x")

            if growth_per_zero > 1.5:
                print()
                print("*** EXPONENTIAL SCALING DETECTED ***")
                print(f"  At this rate, improvement at 20 zeros would be: {improvements[0] * growth_per_zero**(20-zeros_list[0]):.0f}x")
                print(f"  At this rate, improvement at 32 zeros would be: {improvements[0] * growth_per_zero**(32-zeros_list[0]):.0f}x")
                print()
                print("  THIS IS CATASTROPHIC FOR BITCOIN!")
            elif growth_per_zero > 1.0:
                print(f"  Advantage grows with difficulty but sub-exponentially")
            else:
                print(f"  Advantage decreases with difficulty")

    print()

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sha256_scaling_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
