# Geometric Capacity Paper-to-Experiment Matrix

**Date:** 2026-02-19  
**Purpose:** Map external papers to ModelCypher observables, then define
experiments that can support or falsify geometric capacity claims.

This is a research steering document, not a publication checklist.
Use with:
- `docs/research/GEOMETRIC-CONJECTURES-FALSIFICATION-PROTOCOL.md`
- `docs/research/evidence.json`
- `docs/research/evidence_real_models.json`

Status labels:
- `OPEN`
- `SUPPORTED`
- `FALSIFIED`

## 1) Reading-to-Metric Map

| Theme | Paper signal (in repo) | Claim to test | ModelCypher observable(s) | Run surface |
|---|---|---|---|---|
| Capacity utilization | `Aghajanyan_2021_Intrinsic_Dimensionality_Fine_Tuning.pdf` | Task learning uses low-dimensional subspace | `intrinsic_dimension`, `effective_rank`, `support_ratio`, `null_rank` | `poetry run mc analyze dimension-profile --model <model> --prompt "test"` |
| Capacity utilization | `Denti_2022_GRIDE_Generalized_Ratios_Intrinsic_Dimension.pdf` | Multiscale ID is measurable with uncertainty | TwoNN ID trajectory, per-layer rank/ID gap | `poetry run mc analyze dimension-profile --model <model> --prompt "test"` |
| Capacity utilization | `Ruppik_2025_Local_Intrinsic_Dimensions_Contextual_LMs.pdf` | Local ID shifts predict gains | local/trajectory ID vs benchmark delta | `poetry run mc analyze dimension-profile --model <model> --prompt "test"` + benchmark run |
| Small-model parity | `Huh_2024_Platonic_Representation.pdf` | Relational structure aligns across models after coordinate alignment | `train_cka`, `holdout_cka`, `alignment_gain`, `coverage_ratio`, `gram_condition_number` | `poetry run mc analyze reasoning-geometry-validation --models <A> <B> --output <dir>` |
| Small-model parity | `Cheng_2025_HighDimensional_Abstraction_Phase_LMs.pdf` | Abstraction phase links to cross-model similarity | mid-layer ID profile + held-out CKA | `poetry run mc analyze dimension-profile ...` + `poetry run mc analyze reasoning-geometry-validation ...` |
| Training signal geometry (format) | `Li_2023_Inference_Time_Intervention.pdf` | Trajectory steering can alter behavior without base retraining | intervention vs control on same base geometry | `poetry run python scripts/gradient_projection_experiment.py --arm all --output <dir>` |
| Training signal geometry (format) | `Zhang_2024_Activation_Patching.pdf` | Causal intervention should isolate mechanism | causal arm/reinjection arm deltas | `poetry run python scripts/gradient_projection_experiment.py --arm intervention` and `--arm reinjection` |
| Spectral optimization | `Hu_2022_LoRA_Low_Rank_Adaptation.pdf` | Low-rank adapters can target useful directions | per-layer `sigma_k`, rank capacity, spectral budget | `poetry run mc model capacity <model> --json` |
| Spectral optimization | `TSV_2025_Task_Singular_Vectors.pdf` | Task signal concentrates in singular directions | singular spectrum vs task gain curves | `poetry run mc model capacity <model> --json` + task eval |
| Spectral optimization | `Liu_2024_DoRA_WeightDecomposed_LowRank_Adaptation.pdf` | Magnitude/direction decomposition improves transfer control | direction-preserving updates vs behavioral transfer | `poetry run mc train run ...` + behavioral eval |
| Manifold-aware training | `Martens_2015_Optimizing_Neural_Networks_Kroneckerfactored_Approximate_Curvature.pdf` | Curvature-aware optimization improves stability | Hessian/Lipschitz-derived LR, budget ratio, stop certificate signals | `poetry run mc train run ... --adaptive-lr --dim-monitor` |
| Manifold-aware training | `DiSipio_2024_Information_Geometry_LLM.pdf` | Information geometry terms track learning dynamics | curvature, trajectory geometry, condition number | `poetry run mc analyze geodesic-profile --model <model> --prompt "test"` |
| Null-space opportunity | `Fang_2025_AlphaEdit.pdf` | Null-space edits preserve unrelated behavior | `preserved_fraction`, `projection_loss`, boundary invariance | `poetry run mc merge run -s <source> -t <target> -o <out>` |
| Null-space opportunity | `NUFILT_2025_Null_Space_Projection.pdf` | Projection can isolate safe change directions | `null_rank`, `transfer_strength`, causal intervention report | `poetry run mc merge run ...` + causal report artifacts |
| Null-space opportunity | `Ilharco_2023_Task_Arithmetic.pdf`, `Yadav_2023_TIES_Merging.pdf` | Task vectors/merge directions are composable but interference-prone | alignment + boundary/core shift split | `poetry run mc merge run ...` + reasoning validation |

## 2) Experiment Matrix (What To Dig Into Next)

| ID | Priority | Hypothesis under test | Papers linked | Existing metric(s) to use | New measurement to add | Pass condition | Falsify condition | Initial status |
|---|---|---|---|---|---|---|---|---|
| E1 | P0 | **Format is a causal geometric actuator** (not just surface prompting) | Li 2023, Zhang 2024 | `format_fraction`, `alpha_crit`, MT accuracy, trajectory/ID deltas | Add a format group-action suite: same semantics, permuted headers/layout/syntax; store geometry delta per permutation | Intervention arm (project-out) improves MT while reinjection degrades MT, with consistent sign across seeds | Behavior changes but geometry terms do not move, or geometry moves with no behavior effect | `OPEN` |
| E2 | P0 | **Probe-fit alignment generalizes to held-out manifold regions** | Huh 2024, Cheng 2025 | `train_cka`, `holdout_cka`, `alignment_gain`, `coverage_ratio`, `gram_condition_number` | Domain-stratified held-out atlas splits with bootstrap CI per domain | Positive held-out gain across domains with stable conditioning | Probe-perfect train CKA but held-out CKA collapse in majority of domains | `OPEN` |
| E3 | P0 | **Sub-2B capability onset is a capacity frontier, not a size myth** | Aghajanyan 2021, Ruppik 2025 | layer `effective_rank`, `intrinsic_dimension`, `null_rank`, `spectral_gap`; capability score | Frontier dataset: capability delta vs capacity tuple `(ID, support_ratio, null_rank, spectral_gap)` across model scales | Frontier relation reproduces sign/magnitude across model families | No reproducible relation between capacity tuple and capability onset | `OPEN` |
| E4 | P1 | **Spectral budget predicts safe adapter scale and failure** | Hu 2022, TSV 2025, DoRA 2024 | `sigma_k`, spectral ratio/budget, post-train behavior and coherence | Per-layer perturbation attribution: which singular bands drive gain vs degradation | Budget-respecting runs preserve base behavior while improving target tasks | Budget-respecting runs fail unpredictably or budget-violating runs remain stable | `OPEN` |
| E5 | P1 | **Manifold-aware optimization yields better stability-per-compute** | Martens 2015, DiSipio 2024 | LR from measured curvature, stop/budget traces, condition numbers | Matched-compute optimizer comparison with identical data/model and fixed reporting schema | Geometric optimizer reaches equal/better quality with lower instability and less wasted compute | No improvement in stability or compute efficiency vs baseline optimizer | `OPEN` |
| E6 | P0 | **Null-space transfer has a measurable yield curve** | AlphaEdit 2025, NUFILT 2025, Task Arithmetic 2023 | `preserved_fraction` (behavioral), `projection_loss`, `null_rank`, boundary/core shift | Yield-curve sweep: transfer gain vs preservation vs `null_rank/hidden_dim` and conditioning | Predictable gain-preservation tradeoff by capacity + conditioning | Transfer behavior inconsistent at matched capacity/conditioning | `OPEN` |

## 3) Immediate Run Recipes

1. Format causality:
```bash
poetry run python scripts/gradient_projection_experiment.py --arm all --output results/gradient_projection
```

2. Alignment generalization on held-out domains:
```bash
poetry run mc analyze reasoning-geometry-validation \
  --models LFM2-350M LFM2-700M LFM2-1.2B \
  --benchmark gsm8k arithmetic \
  --samples 500 \
  --output results/reasoning_geometry_validation
```

3. Capacity profile per model:
```bash
poetry run mc analyze dimension-profile --model <model_path> --prompt "test"
poetry run mc model capacity <model_path> --json
```

4. Null-space transfer and preservation:
```bash
poetry run mc merge run -s <source_model> -t <target_model> -o results/merge_out
```

5. Format-bias projection in training (service API only — not exposed in CLI):
```python
from modelcypher.cli.composition import get_dataset_training_service
svc = get_dataset_training_service()
svc.train_from_dataset(
    model_path="<model_path>",
    dataset_path="data/training/ce_reasoning_traces_train.jsonl",
    eval_dataset_path="data/training/ce_reasoning_traces_val.jsonl",
    output_path="results/adapter_format_projection",
    format_projection=True,
    narrow_dataset_path="data/training/ce_reasoning_traces_train.jsonl",
    augmented_dataset_path="data/training/format_augmented_train.jsonl",
)
```

## 4) Decision Discipline

For each experiment `E*`, persist:
- `config.json`
- `full_results.json`
- `analysis.json`
- `decision.json`

Promotion rule:
- Move `OPEN -> SUPPORTED` only when pre-registered primary metrics pass across
  registered model families and seeds.
- Move to `FALSIFIED` immediately when the registered rejection condition is met.

