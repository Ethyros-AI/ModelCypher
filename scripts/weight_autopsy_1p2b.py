#!/usr/bin/env python3
"""Weight Autopsy: Per-layer analysis of 1.2B NB-LoRA perturbation.

Loads base weights and trained adapter. For each of 18 adapted layers:
1. SVD(Δ) — perturbation spectral structure and effective rank
2. SVD(W) vs SVD(W+Δ) — singular value shifts
3. Principal angles between W and Δ subspaces
4. Per-direction analysis (U^T Δ V — projection into W's singular basis)
5. Energy redistribution

Uses MLX for loading (bf16 support) and SVD (CPU stream).
"""

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx

# ── Configuration ────────────────────────────────────────────────────────────

BASE_MODEL = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
ADAPTER_PATH = "/Volumes/CodeCypher/experiments/1p2b-cayley-v1"
OUTPUT_DIR = "/Volumes/CodeCypher/experiments/1p2b-weight-autopsy"

# Attention layers in LFM2-1.2B
ATTN_LAYERS = [2, 5, 8, 10, 12, 14]
PROJECTIONS = ["q_proj", "k_proj", "v_proj"]

_EPS_F32 = math.ldexp(1.0, -23)
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class LayerAutopsy:
    """Per-projection weight perturbation analysis."""

    layer_idx: int
    proj_name: str
    W_shape: list[int]
    rank_adapter: int  # rank of lora_a/lora_b

    # Base weight W
    W_spectral_norm: float
    W_frobenius_norm: float
    W_effective_rank: float  # Shannon effective rank
    W_numerical_rank: int  # rank at sqrt(eps) * sigma_max
    W_sigma_max: float
    W_sigma_k: float  # sigma at Shannon boundary
    W_sigma_min: float

    # Delta Δ = (lora_a @ lora_b)^T
    delta_spectral_norm: float
    delta_frobenius_norm: float
    delta_effective_rank: float
    delta_rank_50pct: int  # SVs needed for 50% energy
    delta_rank_90pct: int  # SVs needed for 90% energy
    delta_rank_99pct: int  # SVs needed for 99% energy
    delta_top1_energy_frac: float  # fraction of energy in largest SV

    # Combined W + Δ
    combined_spectral_norm: float
    combined_effective_rank: float
    effective_rank_change: float  # combined - base

    # Budget (perturbation relative to base)
    budget_ratio: float  # ||Δ||_2 / sigma_k(W)
    frob_ratio: float  # ||Δ||_F / ||W||_F

    # Per-direction analysis: U^T Δ V
    # How much does Δ affect each singular direction of W?
    max_diagonal_ratio: float  # max |diag(U^T Δ V)_i| / σ_i(W)
    mean_diagonal_ratio: float
    n_directions_affected: int  # count where |diag| > sqrt(eps) * sigma_max

    # Off-diagonal energy: how much of Δ leaks between W's directions?
    off_diagonal_energy_frac: float  # ||off-diag(U^T Δ V)||_F / ||U^T Δ V||_F

    # Principal angles between leading subspaces of W and Δ
    # Small angle = Δ is aligned WITH W's principal space (leaking in)
    # Large angle (near 90°) = Δ is in the null space (correct behavior)
    min_principal_angle_deg: float  # smallest angle (most aligned direction)
    mean_principal_angle_deg: float
    median_principal_angle_deg: float

    # Singular value shifts: how much did each of W's SVs change?
    max_sv_shift: float  # max |σ_i(W+Δ) - σ_i(W)|
    mean_sv_shift: float
    sv_shift_at_boundary: float  # shift at Shannon rank boundary


# ── Computation Helpers ──────────────────────────────────────────────────────


def shannon_effective_rank(sigmas: list[float]) -> float:
    """Shannon effective rank: exp(H(p)) where p_i = σ_i² / Σσ_j²."""
    total = sum(s * s for s in sigmas)
    if total == 0:
        return 0.0
    entropy = 0.0
    for s in sigmas:
        p = (s * s) / total
        if p > 0:
            entropy -= p * math.log(p)
    return math.exp(entropy)


def numerical_rank(sigmas: list[float], threshold_factor: float = _SQRT_EPS_F32) -> int:
    """Count singular values above threshold_factor * sigma_max."""
    if not sigmas or sigmas[0] == 0:
        return 0
    threshold = threshold_factor * sigmas[0]
    return sum(1 for s in sigmas if s > threshold)


def energy_rank(sigmas: list[float], fraction: float) -> int:
    """Number of SVs needed to capture `fraction` of total energy."""
    total = sum(s * s for s in sigmas)
    if total == 0:
        return 0
    cumulative = 0.0
    for i, s in enumerate(sigmas):
        cumulative += s * s
        if cumulative / total >= fraction:
            return i + 1
    return len(sigmas)


def svd_safe(W: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """SVD on CPU stream. Returns U, S, Vt."""
    W_f32 = W.astype(mx.float32)
    mx.eval(W_f32)
    U, S, Vt = mx.linalg.svd(W_f32, stream=mx.cpu)
    mx.eval(U, S, Vt)
    return U, S, Vt


def to_list(arr: mx.array) -> list[float]:
    """Convert MLX array to Python list."""
    mx.eval(arr)
    return arr.tolist()


# ── Per-Layer Autopsy ────────────────────────────────────────────────────────


def autopsy_layer(
    W: mx.array,
    lora_a: mx.array,
    lora_b: mx.array,
    layer_idx: int,
    proj_name: str,
) -> LayerAutopsy:
    """Full spectral autopsy of one adapted projection."""

    t0 = time.time()
    rank = lora_a.shape[1]

    # Cast to f32
    W_f32 = W.astype(mx.float32)
    lora_a_f32 = lora_a.astype(mx.float32)
    lora_b_f32 = lora_b.astype(mx.float32)
    mx.eval(W_f32, lora_a_f32, lora_b_f32)

    # Compute delta in [out, in] space (same as W)
    # MLX LoRA: y = x @ W^T + x @ lora_a @ lora_b
    # delta_W = (lora_a @ lora_b)^T  [out, in]
    delta_product = mx.matmul(lora_a_f32, lora_b_f32)  # [in, out]
    delta_W = mx.transpose(delta_product)  # [out, in]
    mx.eval(delta_W)

    # Combined weight
    W_adapted = W_f32 + delta_W
    mx.eval(W_adapted)

    # ── SVD of base weight W ──
    print(f"    SVD(W) [{W.shape[0]}x{W.shape[1]}]...", end=" ", flush=True)
    U_w, S_w, Vt_w = svd_safe(W_f32)
    s_w = to_list(S_w)
    print(f"done ({time.time()-t0:.1f}s)")

    w_eff_rank = shannon_effective_rank(s_w)
    w_num_rank = numerical_rank(s_w)
    w_sigma_max = s_w[0] if s_w else 0.0
    w_sigma_min = s_w[-1] if s_w else 0.0
    # sigma_k at Shannon boundary
    k_idx = max(0, int(round(w_eff_rank)) - 1)
    w_sigma_k = s_w[k_idx] if k_idx < len(s_w) else w_sigma_min

    # ── SVD of delta Δ ──
    print("    SVD(Δ)...", end=" ", flush=True)
    _, S_d, _ = svd_safe(delta_W)
    s_d = to_list(S_d)
    print(f"done ({time.time()-t0:.1f}s)")

    d_eff_rank = shannon_effective_rank(s_d)
    d_total_energy = sum(x * x for x in s_d)
    d_top1_frac = (s_d[0] ** 2 / d_total_energy) if d_total_energy > 0 else 0.0

    # ── SVD of combined W + Δ ──
    print("    SVD(W+Δ)...", end=" ", flush=True)
    _, S_c, _ = svd_safe(W_adapted)
    s_c = to_list(S_c)
    print(f"done ({time.time()-t0:.1f}s)")

    c_eff_rank = shannon_effective_rank(s_c)

    # ── Singular value shifts ──
    n_compare = min(len(s_w), len(s_c))
    sv_shifts = [abs(s_c[i] - s_w[i]) for i in range(n_compare)]
    max_sv_shift = max(sv_shifts) if sv_shifts else 0.0
    mean_sv_shift = sum(sv_shifts) / len(sv_shifts) if sv_shifts else 0.0
    boundary_idx = max(0, int(round(w_eff_rank)) - 1)
    sv_shift_boundary = sv_shifts[boundary_idx] if boundary_idx < len(sv_shifts) else 0.0

    # ── Per-direction analysis: U^T Δ V ──
    # Project delta into W's singular basis
    print("    Per-direction analysis...", end=" ", flush=True)
    # U_w: [out, min(out,in)], Vt_w: [min(out,in), in]
    # delta_W: [out, in]
    # projected = U_w^T @ delta_W @ V_w^T^T = U_w^T @ delta_W @ V_w
    V_w = mx.transpose(Vt_w)  # [in, min(out,in)]
    projected = mx.matmul(mx.matmul(mx.transpose(U_w), delta_W), V_w)
    mx.eval(projected)

    min_dim = min(projected.shape)
    diag_vals = [abs(float(projected[i, i])) for i in range(min_dim)]

    # Per-direction ratios: |projected_ii| / σ_i(W)
    dir_ratios = []
    for i in range(min(len(diag_vals), len(s_w))):
        if s_w[i] > _SQRT_EPS_F32 * w_sigma_max:
            dir_ratios.append(diag_vals[i] / s_w[i])
        else:
            break  # below numerical rank

    # Off-diagonal energy
    proj_frob_sq = float(mx.sum(projected * projected))
    diag_energy = sum(d * d for d in diag_vals)
    off_diag_frac = 1.0 - diag_energy / proj_frob_sq if proj_frob_sq > 0 else 0.0

    n_affected = sum(1 for d in diag_vals if d > _SQRT_EPS_F32 * w_sigma_max)
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Principal angles between leading subspaces ──
    # Use the leading r singular vectors of W and Δ
    print("    Principal angles...", end=" ", flush=True)
    _, _, Vt_d = svd_safe(delta_W)

    # Compare right singular vectors (input-space directions)
    # Use min(rank, w_num_rank) dimensions for comparison
    n_compare_vecs = min(rank, w_num_rank, 64)  # cap at 64 for speed
    V_w_lead = Vt_w[:n_compare_vecs, :]  # [k, in]
    V_d_lead = Vt_d[:n_compare_vecs, :]  # [k, in]

    # Principal angles = arccos(singular values of V_w_lead @ V_d_lead^T)
    cross = mx.matmul(V_w_lead, mx.transpose(V_d_lead))  # [k, k]
    mx.eval(cross)
    _, cross_svs, _ = svd_safe(cross)
    cross_sv_list = to_list(cross_svs)

    # Clamp to [0, 1] for arccos
    angles_rad = [math.acos(min(1.0, max(0.0, s))) for s in cross_sv_list if s > 0]
    angles_deg = [math.degrees(a) for a in angles_rad]

    min_angle = min(angles_deg) if angles_deg else 90.0
    mean_angle = sum(angles_deg) / len(angles_deg) if angles_deg else 90.0
    sorted_angles = sorted(angles_deg)
    median_angle = sorted_angles[len(sorted_angles) // 2] if sorted_angles else 90.0
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Norms ──
    w_frob = float(mx.sqrt(mx.sum(W_f32 * W_f32)))
    d_frob = float(mx.sqrt(mx.sum(delta_W * delta_W)))

    return LayerAutopsy(
        layer_idx=layer_idx,
        proj_name=proj_name,
        W_shape=list(W.shape),
        rank_adapter=rank,
        # Base weight
        W_spectral_norm=w_sigma_max,
        W_frobenius_norm=w_frob,
        W_effective_rank=w_eff_rank,
        W_numerical_rank=w_num_rank,
        W_sigma_max=w_sigma_max,
        W_sigma_k=w_sigma_k,
        W_sigma_min=w_sigma_min,
        # Delta
        delta_spectral_norm=s_d[0] if s_d else 0.0,
        delta_frobenius_norm=d_frob,
        delta_effective_rank=d_eff_rank,
        delta_rank_50pct=energy_rank(s_d, 0.50),
        delta_rank_90pct=energy_rank(s_d, 0.90),
        delta_rank_99pct=energy_rank(s_d, 0.99),
        delta_top1_energy_frac=d_top1_frac,
        # Combined
        combined_spectral_norm=s_c[0] if s_c else 0.0,
        combined_effective_rank=c_eff_rank,
        effective_rank_change=c_eff_rank - w_eff_rank,
        # Budget
        budget_ratio=s_d[0] / w_sigma_k if w_sigma_k > 0 else float("inf"),
        frob_ratio=d_frob / w_frob if w_frob > 0 else 0.0,
        # Per-direction
        max_diagonal_ratio=max(dir_ratios) if dir_ratios else 0.0,
        mean_diagonal_ratio=sum(dir_ratios) / len(dir_ratios) if dir_ratios else 0.0,
        n_directions_affected=n_affected,
        off_diagonal_energy_frac=off_diag_frac,
        # Principal angles
        min_principal_angle_deg=min_angle,
        mean_principal_angle_deg=mean_angle,
        median_principal_angle_deg=median_angle,
        # SV shifts
        max_sv_shift=max_sv_shift,
        mean_sv_shift=mean_sv_shift,
        sv_shift_at_boundary=sv_shift_boundary,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("WEIGHT AUTOPSY: LFM2-1.2B NB-LoRA Perturbation Analysis")
    print("=" * 70)

    # Verify volume
    if not Path(BASE_MODEL).exists():
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        sys.exit(1)
    if not Path(ADAPTER_PATH).exists():
        print(f"ERROR: Adapter not found at {ADAPTER_PATH}")
        sys.exit(1)

    # Create output dir
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Load base model weights (only the adapted projections)
    print("\nLoading base model weights...")
    base_weights = mx.load(f"{BASE_MODEL}/model.safetensors")

    # Load adapter weights
    print("Loading adapter weights...")
    adapter_weights = mx.load(f"{ADAPTER_PATH}/adapters.safetensors")

    results: list[LayerAutopsy] = []

    for layer_idx in ATTN_LAYERS:
        for proj in PROJECTIONS:
            print(f"\n── Layer {layer_idx} / {proj} ──")

            w_key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            a_key = f"model.layers.{layer_idx}.self_attn.{proj}.lora_a"
            b_key = f"model.layers.{layer_idx}.self_attn.{proj}.lora_b"

            if w_key not in base_weights:
                print(f"  SKIP: {w_key} not in base model")
                continue
            if a_key not in adapter_weights:
                print(f"  SKIP: {a_key} not in adapter")
                continue

            W = base_weights[w_key]
            lora_a = adapter_weights[a_key]
            lora_b = adapter_weights[b_key]

            print(f"  W: {W.shape} {W.dtype}")
            print(f"  lora_a: {lora_a.shape}, lora_b: {lora_b.shape}")

            result = autopsy_layer(W, lora_a, lora_b, layer_idx, proj)
            results.append(result)

            # Print summary for this layer
            print("  Results:")
            print(f"    W eff_rank: {result.W_effective_rank:.1f}, σ_max: {result.W_sigma_max:.4f}, σ_k: {result.W_sigma_k:.6f}")
            print(f"    Δ eff_rank: {result.delta_effective_rank:.1f}, ||Δ||₂: {result.delta_spectral_norm:.6f}")
            print(f"    Δ rank for 50%/90%/99% energy: {result.delta_rank_50pct}/{result.delta_rank_90pct}/{result.delta_rank_99pct}")
            print(f"    Δ top-1 SV energy fraction: {result.delta_top1_energy_frac:.3f}")
            print(f"    Budget ||Δ||₂/σ_k: {result.budget_ratio:.4f}")
            print(f"    ||Δ||_F/||W||_F: {result.frob_ratio:.4f}")
            print(f"    Eff rank change: {result.effective_rank_change:+.2f}")
            print(f"    Per-direction: max_ratio={result.max_diagonal_ratio:.4f}, mean={result.mean_diagonal_ratio:.6f}, n_affected={result.n_directions_affected}")
            print(f"    Off-diagonal energy: {result.off_diagonal_energy_frac:.3f}")
            print(f"    Principal angles (deg): min={result.min_principal_angle_deg:.1f}, mean={result.mean_principal_angle_deg:.1f}, median={result.median_principal_angle_deg:.1f}")
            print(f"    SV shifts: max={result.max_sv_shift:.6f}, mean={result.mean_sv_shift:.6f}, at_boundary={result.sv_shift_at_boundary:.6f}")

    # ── Summary Report ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Group by projection type
    for proj in PROJECTIONS:
        proj_results = [r for r in results if r.proj_name == proj]
        if not proj_results:
            continue

        print(f"\n{proj}:")
        print(f"  {'Layer':>5} {'W_effRk':>8} {'Δ_effRk':>8} {'Δ_top1%':>8} {'budget':>8} {'maxDirR':>8} {'offDiag':>8} {'minAngl':>8} {'ΔeffRk':>8}")
        for r in proj_results:
            print(
                f"  {r.layer_idx:>5} "
                f"{r.W_effective_rank:>8.1f} "
                f"{r.delta_effective_rank:>8.1f} "
                f"{r.delta_top1_energy_frac:>8.3f} "
                f"{r.budget_ratio:>8.4f} "
                f"{r.max_diagonal_ratio:>8.4f} "
                f"{r.off_diagonal_energy_frac:>8.3f} "
                f"{r.min_principal_angle_deg:>8.1f} "
                f"{r.effective_rank_change:>+8.2f}"
            )

    # Aggregate findings
    print("\n── Key Findings ──")

    # Is perturbation rank-1?
    avg_top1 = sum(r.delta_top1_energy_frac for r in results) / len(results)
    print(f"\nAvg delta top-1 SV energy: {avg_top1:.3f}")
    if avg_top1 > 0.5:
        print("  → Perturbation is RANK-1 DOMINATED. Single attractor direction.")
    elif avg_top1 > 0.2:
        print("  → Perturbation is LOW-RANK. Few dominant directions.")
    else:
        print("  → Perturbation is DISTRIBUTED across null space.")

    # Is perturbation leaking into active subspace?
    avg_min_angle = sum(r.min_principal_angle_deg for r in results) / len(results)
    print(f"\nAvg min principal angle: {avg_min_angle:.1f}°")
    if avg_min_angle < 10:
        print("  → Perturbation is ALIGNED with W's principal space. LEAKING into active subspace.")
    elif avg_min_angle < 45:
        print("  → Perturbation has PARTIAL overlap with active subspace.")
    else:
        print("  → Perturbation is approximately ORTHOGONAL to active subspace (in null space).")

    # Effective rank changes
    avg_rank_change = sum(r.effective_rank_change for r in results) / len(results)
    print(f"\nAvg effective rank change: {avg_rank_change:+.2f}")
    if avg_rank_change < -1:
        print("  → Training COMPRESSED the weight spectrum. Lost representational capacity.")
    elif avg_rank_change > 1:
        print("  → Training EXPANDED the weight spectrum. Added new directions.")
    else:
        print("  → Effective rank approximately preserved.")

    # Off-diagonal energy
    avg_offdiag = sum(r.off_diagonal_energy_frac for r in results) / len(results)
    print(f"\nAvg off-diagonal energy fraction: {avg_offdiag:.3f}")
    if avg_offdiag > 0.5:
        print("  → Perturbation has strong CROSS-DIRECTION coupling. Mixing W's singular modes.")
    else:
        print("  → Perturbation is mostly ALIGNED with W's singular directions.")

    # Budget
    avg_budget = sum(r.budget_ratio for r in results) / len(results)
    max_budget = max(r.budget_ratio for r in results)
    print(f"\nBudget: avg={avg_budget:.4f}, max={max_budget:.4f}")

    # Save JSON
    report = {
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
        "n_layers": len(results),
        "layers": [asdict(r) for r in results],
        "summary": {
            "avg_delta_top1_energy_frac": avg_top1,
            "avg_min_principal_angle_deg": avg_min_angle,
            "avg_effective_rank_change": avg_rank_change,
            "avg_off_diagonal_energy_frac": avg_offdiag,
            "avg_budget_ratio": avg_budget,
            "max_budget_ratio": max_budget,
        },
    }

    report_path = out / "weight_autopsy.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
