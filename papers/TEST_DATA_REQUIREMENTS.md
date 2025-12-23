# Required Test Data for ModelCypher Papers

This document specifies the test data needed to bring each paper to arXiv publication quality.

## Paper 1: Manifold Hypothesis of Agency

### Required Datasets

#### 1. Semantic Prime Embeddings (Cross-Model)
- **Location**: `data/experiments/semantic_primes/`
- **Format**: JSON with embeddings per model
- **Models**: TinyLlama-1.1B, Qwen2.5-{0.5B, 1.5B, 3B}, Llama-3.2-{1B, 3B}
- **Content**: Token embeddings for 65 NSM primes + 200 control words
- **Generation**: `mc geometry primes probe --model <id> --output <file>`

#### 2. Gram Matrix Comparisons
- **Location**: `data/experiments/gram_matrices/`
- **Format**: NumPy `.npy` files
- **Content**: 65×65 Gram matrices for each model
- **Metrics**: Pearson correlation, CKA, Frobenius error

#### 3. Null Distribution Samples
- **Location**: `data/experiments/null_distributions/`
- **Format**: CSV with 200 random subset samples
- **Content**: Pearson/CKA for size-matched control word sets
- **Purpose**: Statistical significance testing

#### 4. Cross-Family CKA Matrix
- **Location**: `data/experiments/cka_matrix.csv`
- **Format**: CSV, models × models
- **Content**: Pairwise CKA for all model pairs

### Missing Scripts
```bash
# Generate all required data for Paper 1
python scripts/generate_paper1_data.py \
  --models tinyllama,qwen2.5-0.5b,qwen2.5-1.5b,qwen2.5-3b,llama-3.2-1b,llama-3.2-3b \
  --anchors semantic_primes,computational_gates,control_words \
  --output data/experiments/paper1/
```

---

## Paper 2: Linguistic Thermodynamics

### Required Datasets

#### 1. Modifier × Model Entropy Matrix
- **Location**: `data/experiments/entropy/modifier_matrix.csv`
- **Format**: CSV (100 prompts × 10 modifiers × 4 models)
- **Modifiers**: baseline, caps, urgency, roleplay, negation, directness, scarcity, authority, combined, minimal
- **Models**: Qwen2.5-3B, Llama-3.2-3B, Mistral-7B, TinyLlama-1.1B

#### 2. Temperature Sweep Data
- **Location**: `data/experiments/entropy/temperature_sweep.csv`
- **Format**: CSV (T=0.0 to T=1.5, step=0.1, per modifier)
- **Content**: Mean entropy at each temperature

#### 3. Safety Signal Comparison
- **Location**: `data/experiments/entropy/safety_signals.csv`
- **Format**: CSV
- **Content**: AUROC for entropy, ΔH, KL, combined

#### 4. Harmful/Benign Prompt Suite
- **Location**: `data/experiments/safety/harmful_benign_suite.json`
- **Format**: JSON array of {prompt, label, category}
- **Content**: 100 harmful + 100 benign prompts (curated)
- **Source**: Adapt from HarmBench or create manually

### Missing Scripts
```bash
# Generate all required data for Paper 2
python scripts/generate_paper2_data.py \
  --prompt-suite data/prompts/modifier_suite.json \
  --models qwen2.5-3b,llama-3.2-3b,mistral-7b,tinyllama-1.1b \
  --temperatures 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5 \
  --output data/experiments/paper2/
```

---

## Paper 3: Unified Manifold Alignment

### Required Datasets

#### 1. Intersection Maps
- **Location**: `data/experiments/merging/intersection_maps/`
- **Format**: JSON per model pair
- **Pairs**: Qwen2.5-7B↔Mistral-7B, Qwen2.5-3B↔Llama-3.2-3B
- **Content**: Layer-wise Jaccard overlap, coverage scores

#### 2. Merge Evaluation Suite - Coding
- **Location**: `data/experiments/merging/eval_coding.json`
- **Format**: JSON array of {prompt, expected_output, scoring_fn}
- **Content**: 50 coding problems (HumanEval-style)
- **Source**: Subset of HumanEval or create synthetic

#### 3. Merge Evaluation Suite - Creative
- **Location**: `data/experiments/merging/eval_creative.json`
- **Format**: JSON array of {prompt, rubric}
- **Content**: 50 creative writing prompts
- **Scoring**: Human eval or LLM-as-judge

#### 4. Merge Sweep Grid Results
- **Location**: `data/experiments/merging/sweep_grid.csv`
- **Format**: CSV
- **Content**: alpha × rank × mode × retention × drift

#### 5. Procrustes Error Distributions
- **Location**: `data/experiments/merging/procrustes_errors/`
- **Format**: JSON per layer
- **Content**: Rotation error, scale factor, residual

### Missing Scripts
```bash
# Generate all required data for Paper 3
python scripts/generate_paper3_data.py \
  --source-model qwen2.5-7b \
  --target-model mistral-7b \
  --adapters coder,instruct,creative \
  --eval-suites coding,creative \
  --output data/experiments/paper3/
```

---

## New Systems Paper

### Required Demonstration Data

#### 1. Module Inventory
- **Source**: Already exists (from test_module_import_guard.py)
- **Content**: 222 modules, 98% load success

#### 2. Test Coverage Report
- **Source**: pytest output
- **Content**: 1116 tests, 100% pass rate

#### 3. Benchmark Comparisons
- **Location**: `data/benchmarks/tool_comparison.csv`
- **Content**: ModelCypher vs TransformerLens vs CircuitsVis
- **Metrics**: Feature count, CLI coverage, test coverage

#### 4. End-to-End Case Study Outputs
- **Location**: `data/case_studies/`
- **Content**: 
  - Semantic prime analysis JSON
  - Entropy monitoring logs
  - Merge diagnostic outputs

---

## Data Generation Priority

| Priority | Dataset | Papers | Effort |
|----------|---------|--------|--------|
| 🔴 HIGH | Modifier entropy matrix | Paper 2 | 2 hours |
| 🔴 HIGH | Semantic prime Gram matrices | Paper 1 | 2 hours |
| 🟡 MEDIUM | Cross-model CKA | Paper 1 | 1 hour |
| 🟡 MEDIUM | Intersection maps | Paper 3 | 2 hours |
| 🟢 LOW | Eval suites | Paper 3, Systems | 4 hours |
| 🟢 LOW | Benchmark comparison | Systems | 2 hours |

---

## Immediate Actions

1. Create `data/experiments/` directory structure
2. Implement `scripts/generate_paper1_data.py`
3. Run entropy experiments for Paper 2
4. Generate first iteration of tables/figures
