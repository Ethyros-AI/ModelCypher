# Qwen-Anchored Feasibility Map on M4 Max (128 GB)

## Objective
Build a measurement-first envelope for local large-model experimentation:
- hard memory limits by stage (`load`, `tokenize`, `forward`, `decode`)
- bounded decode behavior (no unbounded generation)
- 70B/120B projections derived from measured overhead, not guesswork

## Baseline
- Human brain neuron estimate for the opening question: ~86 billion neurons (Azevedo 2009; von Bartheld 2025).
- Local runtime: Apple M4 Max, 128 GB unified memory, MLX backend.

## Implemented Instrumentation
1. `mc infer run --max-tokens` to force bounded decode.
2. `mc system memory-profile` to emit real backend memory telemetry:
   - `active_gb`, `peak_gb`, per-stage timestamps
   - `decode_slope.gb_per_token` from bounded decode windows
   - optional `train_probe` payload:
     - primary mode: `nblora_step` (streaming geometry + NB-LoRA inject + 1 train iteration)
     - fallback mode: `forward_surrogate` when backend/probe path is unavailable
3. Script: `scripts/qwen_feasibility_map.py`:
   - profiles multiple local models
   - writes raw per-model JSON and aggregated feasibility map JSON
   - computes projections for 70B/120B by precision tier

## Runbook
```bash
poetry run python scripts/qwen_feasibility_map.py \
  --model /path/to/qwen3-8b \
  --model /path/to/qwen-7b-quant4 \
  --model /path/to/gemma-3-27b \
  --decode-tokens 32 \
  --train-probe \
  --auto-quantize-8bit
```

Output:
- `results/feasibility_map/<timestamp>/feasibility_map.json`
- one JSON profile per model in the same directory

## Current Artifact Snapshot (2026-02-25)
Run directory:
- `results/feasibility_map/20260225T160732Z`

Measured points:
- Qwen3-1.7B bf16: load `3.20 GiB`, forward peak `3.54 GiB`
- Qwen3-8B bf16: load `15.26 GiB`, forward peak `15.53 GiB`
- Gemma-3-27B bf16: load `52.93 GiB`, forward peak `53.27 GiB`
- Mistral-7B 4-bit: load `3.80 GiB`, forward peak `3.84 GiB`
- Auto-generated Qwen3-1.7B 8-bit and Qwen3-8B 8-bit variants were profiled in the same run.

Generated projection table (`decode_tokens=8`):
- 70B @ 4-bit: `35.87 GiB` projected decode active (fits 128 GiB)
- 70B @ 8-bit: `65.77 GiB` projected decode active (fits 128 GiB)
- 70B @ 16-bit: `130.39 GiB` projected decode active (does not fit 128 GiB)
- 120B @ 4-bit: `59.15 GiB` projected decode active (fits 128 GiB)
- 120B @ 8-bit: `112.34 GiB` projected decode active (fits 128 GiB)
- 120B @ 16-bit: `223.52 GiB` projected decode active (does not fit 128 GiB)

## Projection Math
- Static weight memory: `params * bits / 8`.
- Runtime overhead terms are measured from profiled points:
  - `load_overhead_gib = load_active_gib - static_weight_gib`
  - `forward_delta_gib = forward_active_gib - load_active_gib`
  - `decode_slope_gib_per_token`
- 70B/120B projections apply the per-tier empirical means:
  - `load_active_gib = static_weight_gib + mean(load_overhead_gib)`
  - `forward_active_gib = load_active_gib + mean(forward_delta_gib)`
  - `decode_active_gib = forward_active_gib + mean(decode_slope_gib_per_token) * decode_tokens`

## Confidence and Caveats
- Confidence is highest for interpolation within measured model families and precision tiers.
- Extrapolation to 70B/120B assumes overhead structure remains in-family.
- Checkpoints with model-name parameter hints that differ by order-of-magnitude from measured parameters are excluded from projection fitting.
