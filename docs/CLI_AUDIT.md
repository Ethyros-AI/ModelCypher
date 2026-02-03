# CLI Audit: Geometry Analysis Tools

## Current State

### Existing CLI Commands (mc safety)

| Command | What It Does | Method |
|---------|--------------|--------|
| `spectral-trajectory` | Layer-wise spectral entropy | SVD singular values |
| `dimension-profile` | Per-layer intrinsic dimension | TwoNN estimator |
| `comp-phi` | Per-prompt comp/φ | TwoNN peak/final ratio |
| `entropy-trajectory` | Layer-wise entropy | Token distributions |

### Our Exploration Scripts (scripts/)

| Script | What It Does | Method |
|--------|--------------|--------|
| `explore_phi_trajectories.py` | Layer-wise norm + comp/φ | L2 norm tracking |
| `hidden_state_analysis.py` | EffDim, kurtosis, sparsity | Participation ratio |
| `layer_contribution_analysis.py` | Layer role classification | Expansion/compression counts |
| `final_layer_weight_analysis.py` | Weight matrix properties | SVD effective rank |
| `measure_phi_distribution.py` | Batch comp/φ by task type | Peak/final norm |
| `validate_phi_proxy.py` | Phi proxy validation | Differentiable approximation |
| `attention_entropy_trajectory.py` | (incomplete) | Attention weight entropy |

---

## Overlap Analysis

### Redundant (similar to existing)

| Our Script | CLI Equivalent | Keep Which? |
|------------|----------------|-------------|
| `measure_phi_distribution.py` | `mc safety comp-phi` | CLI (more complete) |

### Complementary (different method, same goal)

| Our Script | CLI Equivalent | Difference |
|------------|----------------|------------|
| `explore_phi_trajectories.py` | `spectral-trajectory` | Norms vs spectral entropy |
| `hidden_state_analysis.py` | `dimension-profile` | Participation ratio vs TwoNN |

**Recommendation**: Keep both methods. Norm-based is simpler/faster, TwoNN is more principled.

### Unique (no CLI equivalent)

| Our Script | What's Unique |
|------------|---------------|
| `layer_contribution_analysis.py` | **Compression gate detection** |
| `final_layer_weight_analysis.py` | **Weight-space signature** |

**Recommendation**: Promote these to CLI commands.

---

## New Discoveries Not in CLI

### 1. Geometric Fingerprint

**What**: expansion_ratio variance across task types identifies model specialization.
- Specialist = 0 variance (constant ~1.0, flat trajectory)
- Base = high variance (1.0-2.0+)

**Current state**: `comp-phi` measures single prompts, doesn't compute variance. Note: φ normalization deprecated per PHI_FINDINGS.md.

**Needed**: `mc model fingerprint` that runs multiple task types and reports expansion_ratio variance.

### 2. Compression Gate

**What**: Identify which layers consistently compress vs expand.
- Base models: L15-16 always compress
- Specialists: No compression layers

**Current state**: Not in CLI.

**Needed**: Integrate `layer_contribution_analysis.py` as `mc geometry compression-gate`.

### 3. Dimension Recovery

**What**: Final layer EffDim recovery indicates generalization capability.
- Base: Recover to 3-8 dims
- Specialist: Stay at 1-2 dims

**Current state**: `dimension-profile` shows trajectory but doesn't quantify recovery.

**Needed**: Add recovery metric to dimension-profile output.

### 4. Weight-Space Signature

**What**: Final layer sparsity and effective rank differ by model type.
- Specialist: 42% sparse, low rank
- Base: 13% sparse, high rank

**Current state**: Not in CLI.

**Needed**: `mc model weight-analysis` or integrate into `mc model probe`.

---

## Proposed CLI Structure

```
mc model
├── probe              # Existing - architecture details
├── fingerprint        # NEW - geometric signature (comp/φ variance)
├── weight-analysis    # NEW - weight matrix properties
└── ...

mc geometry
├── compression-gate   # NEW - layer role classification
└── ...

mc safety
├── spectral-trajectory    # Existing
├── dimension-profile      # Existing - ADD recovery metric
├── comp-phi               # Existing
└── ...
```

---

## Action Items

### Immediate (integrate existing work)

1. [ ] Add `--recovery` flag to `dimension-profile` showing final vs mid EffDim
2. [ ] Add `--variance` flag to `comp-phi` for multi-prompt analysis
3. [ ] Create `mc geometry compression-gate` from `layer_contribution_analysis.py`

### Short-term (new commands)

4. [ ] Create `mc model fingerprint` - one-command model classification
5. [ ] Create `mc model weight-analysis` - weight matrix properties

### Cleanup

6. [ ] Delete `measure_phi_distribution.py` (redundant with `comp-phi`)
7. [ ] Move `validate_phi_proxy.py` to tests/
8. [ ] Archive `attention_entropy_trajectory.py` (incomplete)

---

## Scripts to Keep vs Archive

| Script | Decision | Reason |
|--------|----------|--------|
| `explore_phi_trajectories.py` | **Keep** | Faster norm-based alternative |
| `hidden_state_analysis.py` | **Keep** | Richer metrics than TwoNN alone |
| `layer_contribution_analysis.py` | **Promote to CLI** | Unique, valuable |
| `final_layer_weight_analysis.py` | **Promote to CLI** | Unique, valuable |
| `measure_phi_distribution.py` | Archive | Redundant with `comp-phi` |
| `validate_phi_proxy.py` | Move to tests/ | Validation only |
| `attention_entropy_trajectory.py` | Archive | Incomplete |
