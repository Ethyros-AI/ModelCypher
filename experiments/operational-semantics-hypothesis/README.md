# Operational Semantics Hypothesis

**Hypothesis**: Mathematical concepts are encoded not as tokens but as *structural relationships*
in the latent space. The Pythagorean formula a² + b² = c² should be literally computable from
the geometry of number embeddings.

## Quick Start

```bash
cd experiments/operational-semantics-hypothesis

# Run cross-model invariance analysis
poetry run python analysis/cross_model_invariance.py \
  --model /path/to/model

# Run fast Pythagorean probe
poetry run python analysis/fast_pythagorean.py \
  --model /path/to/model
```

## Experiments

### 1. Clustering (Weak Form)
Do valid Pythagorean triples cluster separately from invalid ones?
- **File**: `run_experiment.py --experiment pythagorean`
- **Measure**: Silhouette score, centroid separation

### 2. Cross-Modal Invariance
Do different presentations of the same theorem converge?
- **File**: `run_experiment.py --experiment cross_modal`
- **Measure**: CKA between formula, word problem, geometric description

### 3. Transformation Recovery
Is there a learnable mapping from (a,b) to c?
- **File**: `run_experiment.py --experiment transformation`
- **Measure**: R² of linear transformation, generalization to novel triples

### 4. Arithmetic Geometry
Do arithmetic operations manifest as consistent directions?
- **File**: `run_experiment.py --experiment arithmetic`
- **Measure**: Direction consistency for adding, doubling

### 5. Formula as Geometry (Strong Form)
Is the formula *literally* encoded in latent positions?
- **File**: `analysis/formula_as_geometry.py`
- **Tests**:
  - Does ||embed(n)||² encode n²?
  - Is there a consistent "squaring direction"?
  - Do valid triples lie on a constraint surface?
  - Can we recover a² + b² - c² from embeddings?
  - Is the position of 5 constrained by its relationship to 3 and 4?

## Falsification

The hypothesis is **rejected** if:
- Valid and invalid triples don't separate
- Novel triples fail to follow the pattern (memorization, not abstraction)
- No consistent transformation exists
- Cross-modal presentations diverge
- The formula cannot be recovered from geometry

## Implications

If confirmed, this suggests:
1. LLMs encode *operational* semantics, not just distributional
2. Mathematical structure emerges from natural language training
3. The "shape of knowledge" includes formal relationships
4. Physical laws may be projections from conceptual geometry

## Files

```
operational-semantics-hypothesis/
├── HYPOTHESIS.md                  # Detailed hypothesis statement
├── README.md                      # This file
├── analysis/
│   ├── cross_model_invariance.py  # Procrustes alignment across models
│   ├── fast_pythagorean.py        # Direct MLX embedding extraction
│   ├── formula_as_geometry.py     # Strong form tests
│   ├── pythagorean_probe.py       # LocalInferenceEngine-based probes
│   ├── arithmetic_directions.py   # Arithmetic direction analysis
│   └── riemannian_pythagorean.py  # Manifold geometry tests
└── results/
    ├── SUMMARY.md                 # Summary of all findings
    ├── cross_model.json           # Cross-model invariance (88.5%)
    ├── fast_*.json                # Per-model fast probe results
    └── *.json                     # Individual model results
```

## Status

**Validated** - Results support the Operational Semantics Hypothesis.
Integrated into Paper 0, Claim 4: "Formulas Are Constraint Surfaces".
