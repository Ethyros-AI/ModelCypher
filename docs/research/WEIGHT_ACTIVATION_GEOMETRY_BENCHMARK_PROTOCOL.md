# Weight vs Activation Geometry Benchmark Protocol

**Updated:** 2026-02-23

## Purpose

Determine, with falsifiable controls, whether:

1. Weight space itself should be treated as a curved manifold.
2. Curvature is instead an objective-space (loss/Hessian) property.
3. Activation space requires geodesic measurement.

This protocol rejects assumptions and records raw measurements only.

## Registered Hypotheses

- `H_weight_curved`: Weight-space geometry requires non-Euclidean geodesic metrics for core diagnostics.
- `H_objective_curved`: Curvature near weights is best measured via Hessian/Fisher spectral structure.
- `H_activation_curved`: Activation-space geometry is curved enough that chord metrics alter density/comparison outcomes.

## Controls

- `C_gaussian`: Matched-dimensional Gaussian point clouds for geodesic/chord distortion baseline.
- `C_k_sweep`: Fixed `k` sweep (`k in {4, 8, 16, 32, 64, 96, n-1}`) to test graph-construction sensitivity.
- `C_interpolation`: Linear vs geodesic interpolation path-loss equivalence in weight space.
- `C_sectional_sanity`: Flat plane vs unit sphere for sectional estimator validity.
- `C_hessian_quadratic`: Analytic quadratic with known Hessian eigenvalues and trace.

## Required Commands

Run from repository root:

```bash
poetry run mc model capacity /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16 --output json > results/weight_geometry/lfm2_350m_capacity.json
```

```bash
poetry run python - <<'PY'
import json, math
from pathlib import Path
import mlx.core as mx
from mlx_lm import load as mlx_load
from modelcypher.backends.mlx_backend import MLXBackend
from modelcypher.adapters.model_backbone import resolve_model_backbone
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.riemannian_validation import derive_k_neighbors

model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
model, tokenizer = mlx_load(model_path)
mx.eval(model.parameters())
embed_tokens, layers, _ = resolve_model_backbone(model)
n_layers = len(layers)
layer_ids = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

source_prompts = [
    "The derivative of x squared is two x.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The mitochondria is the powerhouse of the cell.",
    "Newton's second law states F equals m times a.",
    "DNA is a double helix structure discovered by Watson and Crick.",
    "The speed of light in vacuum is approximately 3e8 meters per second.",
    "Euler's identity states e to the i pi plus one equals zero.",
    "The area of a circle is pi times the radius squared.",
    "Water freezes at zero degrees Celsius at standard pressure.",
    "The Pythagorean theorem states a squared plus b squared equals c squared.",
    "Gravity accelerates objects at 9.8 meters per second squared.",
    "The chemical formula for water is H2O.",
    "The boiling point of water is 100 degrees Celsius.",
    "Ohm's law states voltage equals current times resistance.",
    "The speed of sound in air is approximately 343 meters per second.",
]
target_prompts = [
    "Once upon a time, there was a brave knight who fought dragons.",
    "The sunset painted the sky in shades of orange and purple.",
    "She whispered softly, hoping no one else would hear her secret.",
    "The old house on the hill had been abandoned for decades.",
    "He picked up the guitar and played a melody from his childhood.",
    "The rain fell steadily, drumming against the window pane.",
    "They walked hand in hand along the empty beach at midnight.",
    "The forest was silent except for the rustling of leaves.",
    "A single candle flickered in the darkness of the ancient library.",
    "The cat sat on the windowsill, watching the birds outside.",
    "Morning dew glistened on the grass like scattered diamonds.",
    "The train pulled away from the station with a long whistle.",
    "Children laughed as they chased each other through the garden.",
    "The smell of fresh bread filled the kitchen every morning.",
    "Stars appeared one by one as twilight faded into night.",
]

b = MLXBackend()
rg = RiemannianGeometry(b)
sqrt_eps = math.sqrt(math.ldexp(1.0, -23))

def extract(prompts, layer_idx):
    acts = []
    for text in prompts:
        ids = mx.array([tokenizer.encode(text)])
        h = embed_tokens(ids)
        for i in range(layer_idx + 1):
            layer = layers[i]
            if hasattr(layer, "is_attention_layer") and layer.is_attention_layer:
                h = layer(h, mask="causal")
            else:
                h = layer(h, mask=None)
        pooled = mx.mean(h[0], axis=0)
        mx.eval(pooled)
        acts.append(pooled)
    out = mx.stack(acts, axis=0)
    mx.eval(out)
    return out

def spearman(a, c):
    idx_a = sorted(range(len(a)), key=lambda i: a[i])
    idx_c = sorted(range(len(c)), key=lambda i: c[i])
    rank_a = [0] * len(a)
    rank_c = [0] * len(c)
    for r, i in enumerate(idx_a):
        rank_a[i] = r
    for r, i in enumerate(idx_c):
        rank_c[i] = r
    n = len(a)
    d2 = sum((rank_a[i] - rank_c[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))

def density(dist, k):
    s = b.sort(dist, axis=1)
    kd = s[:, 1:k + 1]
    md = b.mean(kd, axis=1)
    b.eval(md)
    out = []
    for i in range(int(md.shape[0])):
        d = float(b.to_scalar(md[i]))
        out.append(1.0 / max(d, 1e-12))
    return out

results = {}
for layer_idx in layer_ids:
    src = extract(source_prompts, layer_idx)
    tgt = extract(target_prompts, layer_idx)

    src_chord = rg._chord_distance_matrix(src, use_cache=False)
    tgt_chord = rg._chord_distance_matrix(tgt, use_cache=False)
    src_geo = rg.geodesic_distances(src).distances
    tgt_geo = rg.geodesic_distances(tgt).distances
    b.eval(src_chord, tgt_chord, src_geo, tgt_geo)

    distortions = []
    n = int(src_chord.shape[0])
    for mat_c, mat_g in [(src_chord, src_geo), (tgt_chord, tgt_geo)]:
        for i in range(n):
            for j in range(i + 1, n):
                c = float(b.to_scalar(mat_c[i][j]))
                g = float(b.to_scalar(mat_g[i][j]))
                if c > 0:
                    distortions.append(abs(g - c) / c)

    k_src = derive_k_neighbors(src, b)
    k_tgt = derive_k_neighbors(tgt, b)
    k = max(1, min(max(k_src, k_tgt), n - 1))

    src_dc = density(src_chord, k)
    src_dg = density(src_geo, k)
    tgt_dc = density(tgt_chord, k)
    tgt_dg = density(tgt_geo, k)

    sign_changes = 0
    for i in range(len(src_dc)):
        wc = src_dc[i] / max(src_dc[i] + tgt_dc[i], 1e-12)
        wg = src_dg[i] / max(src_dg[i] + tgt_dg[i], 1e-12)
        if (wc >= 0.5) != (wg >= 0.5):
            sign_changes += 1

    results[str(layer_idx)] = {
        "mean_distortion": sum(distortions) / len(distortions),
        "max_distortion": max(distortions),
        "sqrt_eps": sqrt_eps,
        "geodesic_needed": max(distortions) > sqrt_eps,
        "spearman_src": spearman(src_dc, src_dg),
        "spearman_tgt": spearman(tgt_dc, tgt_dg),
        "sign_changes": sign_changes,
        "n_compare": len(src_dc),
        "k": k,
    }

Path("results/geodesic_vs_euclidean").mkdir(parents=True, exist_ok=True)
Path("results/geodesic_vs_euclidean/LFM2-350M_density_comparison.json").write_text(json.dumps(results, indent=2))
print("wrote results/geodesic_vs_euclidean/LFM2-350M_density_comparison.json")
PY
```

```bash
poetry run python - <<'PY'
import json, random, math
from pathlib import Path
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

initialize_default_backend()
b = get_default_backend()
loader = ModelLoader(b)
weights = loader.load_weights('/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16')
rg = RiemannianGeometry(b)

layer_keys = [
    'model.layers.2.self_attn.q_proj.weight',
    'model.layers.8.self_attn.q_proj.weight',
    'model.layers.14.self_attn.q_proj.weight',
]
n = 128
reps = 5
k_values = [8, 16, 32, 64]

def summarize(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) * (x - m) for x in vals) / len(vals)
    return {"mean": m, "std": math.sqrt(v), "min": min(vals), "max": max(vals)}

def distortion_stats(X, k):
    chord = rg._chord_distance_matrix(X, use_cache=False)
    geo_res = rg.geodesic_distances(X, k_neighbors=k, use_cache=False)
    geo = geo_res.distances
    b.eval(chord, geo)
    N = int(X.shape[0])
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            c = float(b.to_scalar(chord[i][j]))
            g = float(b.to_scalar(geo[i][j]))
            if c > 0:
                vals.append((g - c) / c)
    vals.sort()
    return {
        "k_used": geo_res.k_neighbors,
        "mean_ratio_minus_1": sum(vals) / len(vals),
        "p95_ratio_minus_1": vals[int(0.95 * len(vals))],
        "max_ratio_minus_1": vals[-1],
    }

result = {
    "model": '/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16',
    "sample_rows": n,
    "replicates": reps,
    "k_values": k_values,
    "layers": {},
}

for layer_key in layer_keys:
    W = b.astype(b.array(weights[layer_key]), 'float32')
    out_dim = int(W.shape[0])
    in_dim = int(W.shape[1])
    out = {
        "shape": [out_dim, in_dim],
        "weight": {"auto": [], "fixed": {str(k): [] for k in k_values}},
        "gaussian_control": {"auto": [], "fixed": {str(k): [] for k in k_values}},
    }

    for rep in range(reps):
        random.seed(1000 + rep)
        idx = b.array(random.sample(range(out_dim), n))
        Xw = b.take(W, idx, axis=0)
        out["weight"]["auto"].append(distortion_stats(Xw, None))
        for k in k_values:
            out["weight"]["fixed"][str(k)].append(distortion_stats(Xw, k))

        b.random_seed(2000 + rep)
        Xg = b.astype(b.random_normal((n, in_dim)), 'float32')
        out["gaussian_control"]["auto"].append(distortion_stats(Xg, None))
        for k in k_values:
            out["gaussian_control"]["fixed"][str(k)].append(distortion_stats(Xg, k))

    def aggregate(block):
        agg = {"auto": {}, "fixed": {}}
        agg["auto"] = {
            "k_used": summarize([x["k_used"] for x in block["auto"]]),
            "mean_ratio_minus_1": summarize([x["mean_ratio_minus_1"] for x in block["auto"]]),
            "p95_ratio_minus_1": summarize([x["p95_ratio_minus_1"] for x in block["auto"]]),
            "max_ratio_minus_1": summarize([x["max_ratio_minus_1"] for x in block["auto"]]),
        }
        for k in k_values:
            vals = block["fixed"][str(k)]
            agg["fixed"][str(k)] = {
                "mean_ratio_minus_1": summarize([x["mean_ratio_minus_1"] for x in vals]),
                "p95_ratio_minus_1": summarize([x["p95_ratio_minus_1"] for x in vals]),
                "max_ratio_minus_1": summarize([x["max_ratio_minus_1"] for x in vals]),
            }
        return agg

    out["weight_summary"] = aggregate(out["weight"])
    out["gaussian_summary"] = aggregate(out["gaussian_control"])
    result["layers"][layer_key] = out

Path('results/weight_geometry').mkdir(parents=True, exist_ok=True)
Path('results/weight_geometry/weight_geodesic_control.json').write_text(json.dumps(result, indent=2))
print('wrote results/weight_geometry/weight_geodesic_control.json')
PY
```

```bash
poetry run python - <<'PY'
import json, random
from pathlib import Path
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.adapters.model_loader import ModelLoader
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

initialize_default_backend()
b = get_default_backend()
rg = RiemannianGeometry(b)
W = b.astype(b.array(ModelLoader(b).load_weights('/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16')['model.layers.8.self_attn.q_proj.weight']), 'float32')
n = 128
random.seed(123)
idx = b.array(random.sample(range(int(W.shape[0])), n))
Xw = b.take(W, idx, axis=0)
b.random_seed(999)
Xg = b.astype(b.random_normal((n, int(W.shape[1]))), 'float32')

def stats(X, k):
    chord = rg._chord_distance_matrix(X, use_cache=False)
    geo = rg.geodesic_distances(X, k_neighbors=k, use_cache=False).distances
    b.eval(chord, geo)
    N = int(X.shape[0])
    vals = []
    for i in range(N):
        for j in range(i + 1, N):
            c = float(b.to_scalar(chord[i][j]))
            g = float(b.to_scalar(geo[i][j]))
            if c > 0:
                vals.append((g - c) / c)
    vals.sort()
    return {
        "mean_ratio_minus_1": sum(vals) / len(vals),
        "p95_ratio_minus_1": vals[int(0.95 * len(vals))],
        "max_ratio_minus_1": vals[-1],
    }

ks = [4, 8, 16, 32, 64, 96, 127]
out = {"layer": "model.layers.8.self_attn.q_proj.weight", "sample_rows": n, "k_results": {}}
for k in ks:
    out["k_results"][str(k)] = {"weight": stats(Xw, k), "gaussian": stats(Xg, k)}

Path('results/weight_geometry/weight_geodesic_k_sensitivity_single_sample.json').write_text(json.dumps(out, indent=2))
print('wrote results/weight_geometry/weight_geodesic_k_sensitivity_single_sample.json')
PY
```

```bash
poetry run python - <<'PY'
import json
from pathlib import Path
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_curvature import SectionalCurvatureEstimator
from modelcypher.adapters.model_loader import ModelLoader

initialize_default_backend()
b = get_default_backend()
est = SectionalCurvatureEstimator()
n = 80

b.random_seed(0)
xy = b.random_uniform(low=-1.0, high=1.0, shape=(n, 2))
flat = b.concatenate([xy, b.zeros((n, 1))], axis=1)

b.random_seed(1)
raw = b.random_normal((n, 3))
norms = b.sqrt(b.sum(raw * raw, axis=1, keepdims=True))
sphere = raw / norms

W = b.astype(b.array(ModelLoader(b).load_weights('/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16')['model.layers.8.self_attn.q_proj.weight']), 'float32')
rows = W[:n]

out = {}
for name, pts in [('flat_plane_r3', flat), ('unit_sphere_r3', sphere), ('weight_rows_r1024', rows)]:
    prof = est.estimate_manifold_profile(pts)
    out[name] = {
        "global_mean": prof.global_mean,
        "global_variance": prof.global_variance,
        "dominant_sign": prof.dominant_sign.value,
        "sign_distribution": {k.value: v for k, v in prof.sign_distribution.items()},
    }

Path('results/weight_geometry/sectional_estimator_sanity.json').write_text(json.dumps(out, indent=2))
print('wrote results/weight_geometry/sectional_estimator_sanity.json')
PY
```

```bash
poetry run python - <<'PY'
import json
from pathlib import Path
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.training.hessian_estimator import Config, hutchinson_trace_estimate, top_eigenvalue

initialize_default_backend()
b = get_default_backend()
A_vals = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
A = b.array(A_vals)
params = {'w': b.array([0.1] * 10)}

def loss_grad(p):
    x = p['w']
    grad = A * x
    loss = 0.5 * b.sum(x * grad)
    b.eval(loss, grad)
    return loss, {'w': grad}

cfg = Config(hutchinson_vectors=64, power_iterations=40)
trace_est = hutchinson_trace_estimate(loss_grad, params, cfg)
top_est = top_eigenvalue(loss_grad, params, cfg)
out = {
    "quadratic_control": {
        "true_trace": sum(A_vals),
        "estimated_trace": trace_est,
        "true_top_eigenvalue": max(A_vals),
        "estimated_top_eigenvalue": top_est,
    }
}
Path('results/weight_geometry/hessian_estimator_quadratic_control.json').write_text(json.dumps(out, indent=2))
print('wrote results/weight_geometry/hessian_estimator_quadratic_control.json')
PY
```

```bash
poetry run python - <<'PY'
import json
from pathlib import Path
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.mode_connectivity import analyze_mode_connectivity, InterpolationMethod

initialize_default_backend()
b = get_default_backend()
b.random_seed(7)
W0 = b.random_normal((64, 64))
W1 = b.random_normal((64, 64))
T = b.random_normal((64, 64))

def loss_fn(W):
    D = W - T
    s = b.sum(D * D)
    b.eval(s)
    return float(b.to_scalar(s))

lin = analyze_mode_connectivity(W0, W1, loss_fn, n_steps=21, method=InterpolationMethod.LINEAR, backend=b)
geo = analyze_mode_connectivity(W0, W1, loss_fn, n_steps=21, method=InterpolationMethod.GEODESIC, backend=b)
out = {
    "max_abs_path_loss_diff": max(abs(a - b) for a, b in zip(lin.path_losses, geo.path_losses)),
    "linear_barrier_height": lin.barrier_height,
    "geodesic_barrier_height": geo.barrier_height,
}
Path('results/weight_geometry/mode_connectivity_linear_vs_geodesic_control.json').write_text(json.dumps(out, indent=2))
print('wrote results/weight_geometry/mode_connectivity_linear_vs_geodesic_control.json')
PY
```

```bash
poetry run python - <<'PY'
import json
from pathlib import Path
from statistics import mean

root = Path('/Users/jasonkempf/ModelCypher')
act = json.loads((root / 'results/geodesic_vs_euclidean/LFM2-350M_density_comparison.json').read_text())
cap = json.loads((root / 'results/weight_geometry/lfm2_350m_capacity.json').read_text())
wk = json.loads((root / 'results/weight_geometry/weight_geodesic_control.json').read_text())
ks = json.loads((root / 'results/weight_geometry/weight_geodesic_k_sensitivity_single_sample.json').read_text())
sec = json.loads((root / 'results/weight_geometry/sectional_estimator_sanity.json').read_text())
hess = json.loads((root / 'results/weight_geometry/hessian_estimator_quadratic_control.json').read_text())['quadratic_control']
mode = json.loads((root / 'results/weight_geometry/mode_connectivity_linear_vs_geodesic_control.json').read_text())

layers = {}
for layer in ['4', '8', '12']:
    x = act[layer]
    layers[layer] = {
        'mean_distortion': x['mean_distortion'],
        'max_distortion': x['max_distortion'],
        'spearman_src': x['spearman_src'],
        'spearman_tgt': x['spearman_tgt'],
        'sign_changes': x['sign_changes'],
        'k': x['k'],
    }
sqrt_eps = act['4']['sqrt_eps']

paired = {}
for layer_name, obj in wk['layers'].items():
    paired[layer_name] = {}
    for k in ['8', '16', '32', '64']:
        w = [x['mean_ratio_minus_1'] for x in obj['weight']['fixed'][k]]
        g = [x['mean_ratio_minus_1'] for x in obj['gaussian_control']['fixed'][k]]
        paired[layer_name][k] = {
            'mean_weight': mean(w),
            'mean_gaussian': mean(g),
            'mean_weight_minus_gaussian': mean([a - b for a, b in zip(w, g)]),
        }

summary = {
    'date': '2026-02-23',
    'model': '/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16',
    'results': {
        'activation_geodesic_vs_euclidean': {
            'sqrt_eps_f32': sqrt_eps,
            'layers': layers,
            'geodesic_required': any(layers[l]['max_distortion'] > sqrt_eps for l in layers),
        },
        'weight_geodesic_vs_gaussian_control': {
            'fixed_k_paired_differences': paired,
            'single_sample_k_sensitivity': ks['k_results'],
        },
        'mode_connectivity_linear_vs_geodesic': mode,
        'sectional_estimator_sanity': sec,
        'hessian_estimator_quadratic_control': hess,
        'weight_spectral_capacity': {
            'analyzed_layers': cap['summary']['analyzedLayers'],
            'mean_effective_rank': cap['summary']['meanEffectiveRank'],
            'mean_capacity_utilization': cap['summary']['meanCapacityUtilization'],
            'reference_rank_dimension': cap['summary']['referenceRankDimension'],
        },
    },
}

out = root / 'results/weight_geometry/benchmark_summary_2026-02-23.json'
out.write_text(json.dumps(summary, indent=2))
print(f'wrote {out}')
PY
```

## Decision Rules

### Rule A: Activation metric

Use geodesic metric for activation density/comparison if:

- any layer has `max_distortion > sqrt(machine_epsilon_float32)`, or
- any layer shows ranking/decision mismatch (`spearman_src < 1` or sign changes > 0).

### Rule W: Weight-space manifold curvature claim

Support `H_weight_curved` only if both hold:

- At fixed matched `k`, weight distortion exceeds Gaussian control distortion.
- Distortion does not collapse toward 0 as `k -> n-1`.

If fixed-`k` weight distortion is consistently below Gaussian control and decays to 0 with increasing `k`, reject `H_weight_curved`.

### Rule I: Interpolation geometry

If linear vs geodesic path losses are identical (`max_abs_path_loss_diff == 0`), treat weight interpolation geometry as Euclidean for this pipeline.

### Rule S: Sectional estimator validity gate

Sectional estimator is valid for this claim only if it distinguishes unit sphere from flat plane in control (`unit_sphere_r3` not flat/zero while flat plane remains near zero).

If it fails this gate, do not use it as evidence for or against weight-space manifold curvature.

### Rule H: Hessian estimator gate

Hessian estimator passes if quadratic control recovers known trace/top-eigenvalue with small numerical error. After gate passes, Hessian spectral outputs are admissible objective-curvature evidence.

## Current Outcome (2026-02-23)

Based on recorded artifacts:

- Activation space: geodesic required.
- Weight space: no support for curved-manifold metric requirement.
- Objective around weights: curvature captured by Hessian spectral quantities.
- Sectional estimator path: currently fails sphere-vs-plane sanity gate for this claim.

## Source Implementation Links

- Weight Euclidean note: `src/modelcypher/core/domain/geometry/transplant.py`
- Weight geodesic equals linear interpolation: `src/modelcypher/core/domain/geometry/mode_connectivity.py`
- Sectional estimator derivative path: `src/modelcypher/core/domain/geometry/manifold_curvature.py`
- High-d curvature call path: `src/modelcypher/core/domain/geometry/riemannian_core_curvature.py`
- Weight spectral capacity analyzer: `src/modelcypher/core/domain/geometry/spectral_capacity.py`
- Hessian estimator: `src/modelcypher/core/domain/training/hessian_estimator.py`
