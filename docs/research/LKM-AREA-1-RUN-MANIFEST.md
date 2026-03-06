# LKM Area 1: Run Manifest and Artifact Schema

**Status:** Draft
**Date:** 2026-03-05
**Parent:** `docs/research/LORA-KNOWLEDGE-MEMORY-CAPACITY-VALIDATION-PROTOCOL.md`

This document specifies the exact runs, datasets, configurations, and artifact
schemas for Area 1 (reproduction infrastructure) and Area 2 (supporting
measurements). It does not contain scripts; it defines what scripts must produce.

---

## 1. Model Selection

### Decision Point: Model Family

The paper uses Qwen3-8B and Llama-3.1-8B. Neither is currently on the volume.

Available on `/Volumes/CodeCypher/models/mlx-community/`:

| Model | Family | Params | Notes |
|-------|--------|--------|-------|
| Qwen3.5-0.8B-bf16 | Qwen3.5 | 0.8B | Hybrid linear_attention + full_attention |
| Qwen3.5-2B-bf16 | Qwen3.5 | 2B | Hybrid |
| Qwen3.5-4B-bf16 | Qwen3.5 | 4B | Hybrid |
| Qwen3.5-9B-bf16 | Qwen3.5 | 9B | Hybrid |
| LFM2-350M-MLX-bf16 | LFM2 | 350M | Hybrid attention-convolution |
| LFM2-700M-bf16 | LFM2 | 700M | Hybrid attention-convolution |

**Option A: Download Qwen3-8B-bf16.**
Exact paper match. Commensurable comparison. Requires download.

**Option B: Run on Qwen3.5 family.**
Available now. Cross-family vs paper (Qwen3.5 != Qwen3). Conclusions are
exploratory until same sign confirmed on paper-matched family. Has the
advantage of testing our predictions on a hybrid-attention architecture the
paper did not study.

**Option C: Both.**
Qwen3.5-0.8B for harness validation (fast). Qwen3-8B for commensurable
reproduction after download. Cross-family comparison becomes a bonus axis.

**Recommended:** Option C. Harness validation requires no download. Commensurable
reproduction requires one model download decision.

### Model Size Policy Compliance

- Smoke test / harness validation: Qwen3.5-0.8B-bf16 (smallest available)
- Primary capacity sweep: Qwen3.5-0.8B-bf16 first, then scale to 2B/4B
- Commensurable paper reproduction: Qwen3-8B-bf16 (requires download)
- Smoke test runs do NOT count for claim promotion

---

## 2. Dataset Generation

### 2.1 PhoneBook Benchmark (PB)

Synthetic key-value dataset matching paper specification.

**Construction:**
- Source: programmatically generated fictional name-phone number pairs
- Names: combine common first names x common last names (no real-world entities)
- Phone numbers: North American format `XXX-XXX-XXXX`, random digits
- Format: QA pairs

```text
Question: What is the phone number of <FirstName> <LastName>? Answer: <XXX-XXX-XXXX>
```

**Slicing:**
- Token counts measured using the target model's tokenizer
- Sizes: 1K, 2K, 4K, 8K, 12K, 16K, 20K tokens
- Deterministic ordering: smaller slices are strict prefixes of larger slices
- Source pool must be large enough that 20K tokens is achievable
  (estimate: ~30 tokens per QA pair -> ~700 pairs for 20K)

**Evaluation:**
- Strict Exact Match (EM) on phone number extraction
- Prompt: `Question: What is the phone number of <Name>?`
- Extract: model generation after `Answer:`
- Match: normalized phone number string comparison
- Evaluate on ALL items in the training slice (memorization test, not generalization)

**Output files:**
- `data/lkm/phonebook_source.csv` — master name-phone pairs
- `data/lkm/phonebook_{size}tok.jsonl` — per-size training slices
- `data/lkm/phonebook_eval.jsonl` — evaluation prompts (one per source pair)
- `data/lkm/phonebook_meta.json` — generation metadata

**`phonebook_meta.json` schema:**
```json
{
  "total_pairs": <int>,
  "tokenizer": "<model_id>",
  "slices": {
    "1000": {"n_pairs": <int>, "actual_tokens": <int>},
    "2000": {"n_pairs": <int>, "actual_tokens": <int>},
    ...
  },
  "generation_seed": <int>,
  "timestamp": "<ISO 8601>"
}
```

### 2.2 CounterFact (CF)

Use the published CounterFact dataset (Meng et al. 2022). No generation needed.
Download from source or construct from existing assets.

**Evaluation:**
- Efficacy score (probability of target completion > probability of original)
- Requires logit access, not just generation

**Defer:** CF is secondary to PB for the capacity sweep. Implement PB first.
CF adds the "fact revision vs. arbitrary association" axis, which is useful
but not required for P-LKM-1 through P-LKM-4.

---

## 3. B0 Run Matrix

### 3.1 Configuration

B0 reproduces the paper's setup as faithfully as possible.

| Parameter | Value | Source |
|-----------|-------|--------|
| Parameterization | Standard LoRA (B=0, A=Gaussian) | Paper default |
| Target modules | All linear layers | Paper default (E.1 confirms) |
| alpha | r (alpha equals rank) | Appendix D, PB/CF setting |
| Steps | 1,500 | Appendix D, PB/CF setting |
| Batch size | 8 | Appendix D, PB/CF setting |
| Learning rate | Paper does not specify for PB/CF | See note below |
| Optimizer | AdamW (assumed, paper standard) | Paper does not specify |
| Precision | bf16 | Match our model weights |

**Learning rate note:** Appendix D specifies lr=5e-5 for PaperQA and lr=5e-4
for NQA/QuALITY, but does NOT specify lr for PB/CF. For the Qwen3 scale study
(Q6), they sweep lr in {1e-5, 5e-5, 1e-4, 5e-4}. For B0 reproduction on PB,
we must either:
1. Use the Q6 per-model-size selected lr (Qwen3-0.6B: 5e-4, Qwen3-8B: 5e-4)
2. Run a small lr sweep ourselves on the B0 baseline

**Decision:** run B0 with lr=5e-4 as the primary setting (matches paper's
selected value for both their 0.6B and 8B Qwen3 models). If results diverge
substantially from paper curves, add lr={1e-4, 5e-5} as secondary B0 arms
to diagnose.

### 3.2 Run Grid

**Axis 1: Rank sweep**
```
r_cap in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
```

**Axis 2: Knowledge load**
```
tokens in {1K, 2K, 4K, 8K, 12K, 16K, 20K}
```

**Total B0 runs:** 10 ranks x 7 loads = 70 runs per model.

For smoke test on Qwen3.5-0.8B: subset to {4, 16, 64, 256} x {1K, 4K, 8K, 16K}
= 16 runs (validates harness before full sweep).

### 3.3 Implementation Path

B0 must use standard LoRA (HuggingFace PEFT or equivalent), NOT our NB-LoRA
pipeline. The point of B0 is paper-matched reproduction. Our pipeline's
geometric step sizes, Cayley parameterization, and stopping criteria would
contaminate the baseline.

**Required:** a thin training harness that wraps PEFT + a standard AdamW
optimizer + fixed step count. This is separate from `mc train run`.

---

## 4. G1-G3 Run Grid

Each geometric arm adds exactly ONE intervention to the previous arm.

### G1: Spectral-Scale Arm

- Same training setup as B0 (same LR, steps, optimizer, parameterization)
- After training, compute per-layer `scale_ratio_i = ||B_i @ A_i||_2 / sigma_k_i`
- If any `scale_ratio_i > 1`, apply geometric scaling before evaluation
- Evaluate with geometric scale applied

**Purpose:** Isolate whether scale safety alone shifts apparent capacity.

**Runs:** Same grid as B0 (70 per model), plus geometry table + scale table emission.

### G2: Tail-Capacity Arm

- Same as G1
- Per-layer rank: `r_i = min(r_cap, tail_dims_i)` instead of uniform `r_cap`
- Total trainable parameters will differ from B0 at matched `r_cap`
- Record actual `sum_i utilized_tail_dims_i` and actual `n_trainable_params`

**Purpose:** Test whether utilized tail capacity predicts saturation better
than raw rank.

**Implementation note:** requires per-layer rank assignment in the PEFT config.
HuggingFace PEFT supports per-module rank via `rank_pattern` dict. Alternatively,
train through our pipeline with Cayley disabled.

### G3: NB-LoRA Arm

- Same as G2 (tail-capacity rank, geometric scale)
- Replace standard LoRA with NB-LoRA (Cayley parameterization)
- This uses our `mc train run-research` pipeline
- Fixed steps matching B0 (override geometric stopping for commensurability)

**Purpose:** Test whether parameterization changes capacity-per-parameter.

---

## 5. Area 2: Measurement Emissions

Every run in every arm must emit the following tables.

### 5.1 Pre-Training Geometry Table

Emitted ONCE per model, before any training run.

**`geometry_table.json` schema:**
```json
{
  "model_id": "<string>",
  "model_family": "<string>",
  "dtype": "<string>",
  "timestamp": "<ISO 8601>",
  "layers": [
    {
      "layer_key": "<string>",
      "shape": [<int>, <int>],
      "full_rank": <int>,
      "effective_rank": <int>,
      "shannon_effective_rank": <float>,
      "tail_dims": <int>,
      "sigma_max": <float>,
      "sigma_k": <float>,
      "spectral_gap": <float>,
      "condition_number": <float>
    }
  ],
  "summary": {
    "total_layers": <int>,
    "total_tail_dims": <int>,
    "mean_tail_dims": <float>,
    "layers_with_capacity": <int>,
    "layers_without_capacity": <int>
  }
}
```

**Source:** `geometric_lora.py` `analyze_weight_geometries()` already produces
most of this. Wrap in a JSON emitter.

### 5.2 Post-Training Scale Ratio Table

Emitted ONCE per trained adapter.

**`scale_ratio_table.json` schema:**
```json
{
  "run_id": "<string>",
  "arm": "<B0|G1|G2|G3>",
  "r_cap": <int>,
  "knowledge_tokens": <int>,
  "layers": [
    {
      "layer_key": "<string>",
      "delta_spectral_norm": <float>,
      "sigma_k": <float>,
      "scale_ratio": <float>,
      "safe": <bool>
    }
  ],
  "summary": {
    "max_scale_ratio": <float>,
    "mean_scale_ratio": <float>,
    "unsafe_layer_count": <int>,
    "all_safe": <bool>
  }
}
```

**Source:** `lora_safety_service.py` `compute_geometric_scale()` provides the
per-layer analysis. Adapted B/A weight extraction depends on whether standard
PEFT or NB-LoRA is used.

### 5.3 Per-Run Evaluation Record

**`raw_scores.jsonl` schema (one line per eval item):**
```json
{"name": "<string>", "phone_true": "<string>", "phone_predicted": "<string>", "exact_match": <bool>}
```

### 5.4 Capacity Curve

Aggregated from raw scores across the rank x load grid.

**`capacity_curve.json` schema:**
```json
{
  "arm": "<B0|G1|G2|G3>",
  "model_id": "<string>",
  "metric": "exact_match",
  "points": [
    {
      "r_cap": <int>,
      "knowledge_tokens": <int>,
      "n_eval_items": <int>,
      "exact_match_rate": <float>,
      "utilized_tail_dims_total": <int or null>,
      "n_trainable_params": <int>,
      "max_scale_ratio": <float or null>
    }
  ]
}
```

### 5.5 Saturation Point Estimates

Derived from capacity curves.

**`saturation_points.json` schema:**
```json
{
  "arm": "<B0|G1|G2|G3>",
  "model_id": "<string>",
  "threshold_tau": <float>,
  "points": [
    {
      "r_cap": <int>,
      "T_sat_tokens": <int or null>,
      "utilized_tail_dims_total": <int or null>,
      "n_trainable_params": <int>
    }
  ]
}
```

Where `T_sat(tau)` = largest token load at which `exact_match_rate >= tau`.
Threshold `tau` values: {0.8, 0.9, 0.95}.

### 5.6 Efficiency Curve

**`efficiency_curve.json` schema:**
```json
{
  "arm": "<B0|G1|G2|G3>",
  "model_id": "<string>",
  "threshold_tau": <float>,
  "points": [
    {
      "r_cap": <int>,
      "eta_mem": <float or null>,
      "T_sat_tokens": <int or null>,
      "n_trainable_params": <int>
    }
  ]
}
```

Where `eta_mem = T_sat / n_trainable_params`.

### 5.7 Falsifier Outcome

**`falsifier_outcome.json` schema:**
```json
{
  "prediction_id": "<P-LKM-1|P-LKM-2|...>",
  "arm_pair": ["<arm_a>", "<arm_b>"],
  "model_id": "<string>",
  "test": "<string>",
  "statistic": <float>,
  "p_value": <float or null>,
  "direction": "<string>",
  "falsifier_triggered": <bool>,
  "notes": "<string>"
}
```

---

## 6. Script Inventory

Scripts to be created in `scripts/lkm/`. None exist yet.

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `generate_phonebook.py` | Generate PB dataset at multiple token sizes | model tokenizer path, seed | `data/lkm/phonebook_*.{csv,jsonl,json}` |
| `emit_geometry_table.py` | Pre-training geometry analysis | model path | `geometry_table.json` |
| `run_b0.py` | Standard LoRA training (PEFT wrapper) | model, data, r_cap, config | adapter dir + `scale_ratio_table.json` |
| `evaluate_phonebook.py` | Exact-match evaluation | model, adapter, eval data | `raw_scores.jsonl` |
| `run_arm.py` | Unified runner for G1/G2/G3 with arm-specific config | arm type, model, data, r_cap | adapter dir + tables |
| `build_capacity_curves.py` | Aggregate raw scores into curves | results dir | `capacity_curve.json`, `saturation_points.json`, `efficiency_curve.json` |
| `evaluate_falsifiers.py` | Test registered predictions | results dir, arm pair | `falsifier_outcome.json` |
| `run_sweep.py` | Orchestrate full rank x load grid | arm, model, config | all artifacts for one arm |

### Dependency Graph

```
generate_phonebook.py
    |
emit_geometry_table.py (once per model, parallel with dataset gen)
    |
    v
run_sweep.py --arm B0
    |
    v
run_sweep.py --arm G1 (uses B0 adapters + geometry table)
    |
    v
run_sweep.py --arm G2 (uses geometry table for per-layer rank)
    |
    v
run_sweep.py --arm G3 (uses NB-LoRA pipeline)
    |
    v
build_capacity_curves.py (all arms)
    |
    v
evaluate_falsifiers.py
```

---

## 7. Run Directory Layout

```
results/lora_memory_capacity_validation/
  <run_id>/
    config.json
    geometry_table.json          (symlink to model-level table)
    scale_ratio_table.json
    raw_scores.jsonl
    adapter/                     (trained adapter weights)
      adapter_config.json
      adapter_model.safetensors
```

Aggregate outputs:
```
results/lora_memory_capacity_validation/
  <model_id>/
    geometry_table.json
    B0/
      capacity_curve.json
      saturation_points.json
      efficiency_curve.json
    G1/
      capacity_curve.json
      ...
    G2/
      ...
    G3/
      ...
    falsifier_outcomes/
      P-LKM-1.json
      P-LKM-2.json
      P-LKM-4.json
```

---

## 8. Execution Order

1. Generate PhoneBook dataset for Qwen3.5-0.8B tokenizer
2. Emit geometry table for Qwen3.5-0.8B-bf16
3. Smoke test: B0 on Qwen3.5-0.8B, subset grid (16 runs)
4. Verify harness produces well-formed artifacts
5. Full B0 sweep on Qwen3.5-0.8B (70 runs)
6. Emit scale ratio tables for all B0 adapters
7. Build B0 capacity/efficiency curves
8. Run G1 sweep (apply scale enforcement, re-evaluate)
9. Build G1 curves, evaluate P-LKM-2
10. Run G2 sweep (per-layer rank from tail_dims)
11. Build G2 curves, evaluate P-LKM-1
12. Run G3 sweep (NB-LoRA)
13. Build G3 curves, evaluate P-LKM-4
14. Cross-arm falsifier evaluation

**Decision gate after step 7:** If B0 on Qwen3.5-0.8B shows capacity saturation
matching the paper's qualitative shape (capacity increases with rank, saturates
at load), proceed. If no saturation observed, investigate whether 0.8B is too
small or the harness has a bug before scaling up.

**Decision gate after step 9:** If P-LKM-2 shows no signal (scale ratios all
safe OR no coherence shift), the scale-safety mechanism may not be active at
paper-style alpha=r settings. Record and proceed to G2.

---

## 9. Commensurability Notes

- Qwen3.5-0.8B results are NOT commensurable with the paper's Qwen3-8B results.
  Different family, different scale, different architecture (hybrid vs standard).
- Qwen3.5 results test our geometric predictions on a NEW architecture family
  that the paper did not study. This is valuable but does not reproduce the paper.
- For commensurable reproduction, Qwen3-8B-bf16 must be downloaded.
- All claims from Qwen3.5 runs are classified `[EXPLORATORY]` with explicit
  cross-family caveat until confirmed on a paper-matched model.

---

## 10. Open Questions for Jason

1. **Model download:** Should we download Qwen3-8B-bf16 for commensurable
   reproduction, or run the full protocol on Qwen3.5 first?

2. **PEFT dependency:** B0 requires standard HuggingFace PEFT for paper-matched
   LoRA. Is adding `peft` as a dev dependency acceptable, or should we implement
   a minimal standard-LoRA training loop ourselves?

3. **Compute budget:** Full B0 sweep is 70 runs at 1,500 steps each on 0.8B.
   Estimated: ~2 hours on Metal. Full protocol (B0+G1+G2+G3) is ~280 runs.
   Run all, or start with the smoke-test subset?

4. **CounterFact:** Defer CF entirely until PB capacity sweep is complete, or
   generate CF in parallel as a secondary benchmark?
