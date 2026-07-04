
import json
from pathlib import Path

import mlx.core as mx

from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend


def main():
    initialize_default_backend()
    backend = get_default_backend()

    # Paths
    model_path = Path('/Volumes/codecypher/models/mlx-community/Qwen3-8B-bf16')
    adapter_path = Path('/Volumes/codecypher/adapters/ensemble-characters/michael_kl_v12')

    # Load Base Weight W
    print("Loading base weight W...")
    index_file = model_path / 'model.safetensors.index.json'
    with open(index_file) as f:
        weight_map = json.load(f).get('weight_map', {})

    # Analyze Layer 0 q_proj
    base_key = 'model.layers.0.self_attn.q_proj.weight'
    shard = model_path / weight_map[base_key]
    base_weights = mx.load(str(shard))
    W = base_weights[base_key].astype(mx.float32)

    # Load Trained LoRA
    print("Loading trained adapter...")
    loader = AutoAdapterWeightsLoader()
    adapter_file = adapter_path / 'adapters.safetensors'
    adapter_weights = loader.load(adapter_file, backend)
    lora_a = adapter_weights['model.layers.0.self_attn.q_proj.lora_a'].astype(mx.float32)
    lora_b = adapter_weights['model.layers.0.self_attn.q_proj.lora_b'].astype(mx.float32)
    delta_trained = mx.matmul(lora_a, lora_b)

    # Baselines
    print("Generating baselines...")
    frob_norm = float(mx.sqrt(mx.sum(delta_trained**2)))

    # 1. Random Rank-16 (Gaussian A, Gaussian B) scaling to same norm
    A_rand = mx.random.normal((4096, 16))
    B_rand = mx.random.normal((16, 4096))
    delta_rand = mx.matmul(A_rand, B_rand)
    delta_rand = delta_rand * (frob_norm / float(mx.sqrt(mx.sum(delta_rand**2))))

    # 2. Untrained Initialized (Gaussian A, Zero B) - Simulate slight training start?
    # Actually, standard init is Kaiming A, Zero B -> Delta is ZERO.
    # So "Untrained" is not useful for geometry unless we assume some random initialization steps.
    # Let's use Random Rank-16 as the "unstructured" baseline.

    # SVD of W
    print("Computing SVD of W (this may take a moment)...")
    U, S, Vt = mx.linalg.svd(W, stream=mx.cpu)

    # --- EXPERIMENT 1: The 2x Signal (Subspace Alignment) ---
    print("\n=== EXPERIMENT 1: Decoding the 2x Weyl Signal ===")

    def analyze_subspace_alignment(delta, name):
        # Project delta onto U_k (Left Singular Vectors of W)
        # We want to see how much energy is in the top-k vs bottom-k

        # Energy in Top-K
        k_values = [16, 32, 64, 128, 256, 512, 1024]
        print(f"\n{name} Alignment with W's Left Singular Vectors (U):")
        total_energy = float(mx.sum(delta**2))

        for k in k_values:
            U_k = U[:, :k]
            # Projection: P = U_k @ U_k.T @ delta
            # Norm of projection: ||U_k.T @ delta||_F
            proj = mx.matmul(U_k.T, delta)
            energy = float(mx.sum(proj**2))
            ratio = energy / total_energy
            print(f"  Top-{k:<4}: {ratio*100:.3f}% energy")

        # Alignment with V (Right Singular Vectors) - Input side
        print(f"{name} Alignment with W's Right Singular Vectors (V):")
        for k in k_values:
            V_k = Vt[:k, :] # Vt rows are V.T columns -> V columns
            # Projection: delta @ V_k.T
            proj = mx.matmul(delta, V_k.T)
            energy = float(mx.sum(proj**2))
            ratio = energy / total_energy
            print(f"  Top-{k:<4}: {ratio*100:.3f}% energy")

    analyze_subspace_alignment(delta_trained, "Trained LoRA")
    analyze_subspace_alignment(delta_rand, "Random LoRA")

    # --- EXPERIMENT 2: Activation Geometry ---
    print("\n=== EXPERIMENT 2: Activation Geometry (Data-Dependent) ===")
    # Generate synthetic "data" that matches W's expected input distribution?
    # W expects inputs aligned with its V vectors (weighted by S?)
    # A better proxy for real activations: Inputs are not isotropic Gaussian.
    # In Transformers, inputs X often align with the dominant directions of the previous layer.
    # But locally, let's assume inputs aligned with W's Right Singular Vectors (V) are "strong" features.

    # Test inputs: Directions of W's V vectors (Features W cares about)
    # x_k = Vt[k]

    print("Measuring effect on specific input directions (Right Singular Vectors of W):")
    print(f"{'Input Dir':<10} | {'Trained ||ΔWx||':<15} | {'Random ||ΔWx||':<15} | {'Ratio (T/R)':<10}")

    def measure_directional_amplification(delta, direction):
        # direction shape (4096,) unit vector
        # output change = delta @ direction
        out = mx.matmul(delta, direction)
        return float(mx.linalg.norm(out))

    # Expanded sweep to see spectrum of interaction
    k_range = list(range(0, 32)) + [50, 64, 100, 128, 256, 512, 1000, 2000, 4000]
    for k in k_range:
        v_k = Vt[k, :] # k-th right singular vector
        amp_trained = measure_directional_amplification(delta_trained, v_k)
        amp_rand = measure_directional_amplification(delta_rand, v_k)
        ratio = amp_trained / amp_rand if amp_rand > 0 else 0
        print(f"v_{k:<8} | {amp_trained:.6f}        | {amp_rand:.6f}        | {ratio:.4f}")

    # Test inputs: Random direction (Isotropic)
    rand_dir = mx.random.normal((4096,))
    rand_dir = rand_dir / mx.linalg.norm(rand_dir)
    amp_trained = measure_directional_amplification(delta_trained, rand_dir)
    amp_rand = measure_directional_amplification(delta_rand, rand_dir)
    print(f"{'Random':<10} | {amp_trained:.6f}        | {amp_rand:.6f}        | {amp_trained/amp_rand:.4f}")

    print("\nInterpretation:")
    print("If Trained LoRA suppresses effects on top V vectors (v_0..v_16) compared to Random,")
    print("it means it is actively avoiding disrupting the 'core features' of W.")

if __name__ == "__main__":
    main()
