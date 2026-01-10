# ModelCypher: A Geometric Toolkit for Large Language Model Analysis and Safe Adaptation

**Author**: Jason Kempf
**Affiliation**: EthyrosAI  
**Date**: December 2025

---

## Abstract

**ModelCypher** makes the Geometric Knowledge Thesis operational. The toolkit implements modules for measuring representation geometry, entropy dynamics, safety constraints, and merge pipelines. Three core capabilities: (1) CKA-based cross-model comparison via Gram matrices, (2) entropy divergence (ΔH) methodology for safety monitoring, (3) null-space transplant for model merging. The framework integrates with the CLI and Model Context Protocol (MCP) for agentic workflows. Test coverage and module counts evolve with the codebase; see the repository for current totals. AGPLv3 license.

---

## 1. Introduction

Knowledge has shape. **ModelCypher** measures it.

The toolkit implements four core capabilities:

1. **Geometric Diagnostics**: CKA measures cross-model alignment. Topological fingerprints capture manifold structure. Intrinsic dimension estimates representation complexity.

2. **Entropy Monitoring**: Token-level entropy tracks uncertainty. ΔH (base-adapter divergence) provides a methodology for detecting safety-relevant behavior. Circuit breakers intervene before generation.

3. **Safety Analysis**: Refusal is a direction (Arditi et al., 2024). We detect it, measure it, and verify it survives adapter merging.

4. **Model Operations**: Null-space transplant for model merges. Permutation alignment for same-architecture pairs. DARE sparsity analysis as a diagnostic.

### 1.1 Design Principles

**Measurement Before Metaphor**: Every geometric claim is a computable metric with falsification criteria.

**Diagnostics Before Intervention**: Compatibility assessment precedes merge attempts. Alignment failures are debugged before merging.

**Reproducibility**: Deterministic seeds. Version-pinned dependencies. Test suite for regressions.

---

## 2. System Architecture

### 2.1 Domain Structure

```
modelcypher/core/domain/
├── geometry/     # CKA, fingerprints, alignment
├── entropy/      # Tracking, windows, probes
├── safety/       # Guards, calibration, sidecar
├── agents/       # Traces, atlases, validators
├── training/     # Checkpoints, metrics
├── thermo/       # Linguistic thermodynamics
├── adapters/     # LoRA, DARE, DoRA
└── merging/      # Merge primitives
```

### 2.2 Interface Layers

**CLI** (`mc`): Command groups for geometry, entropy, safety, merge, and model inspection

| Command Group | Examples |
|--------------|----------|
| `mc geometry` | `primes probe-model`, `metrics topological-fingerprint`, `atlas dimensionality-study` |
| `mc entropy` | `analyze`, `dual-path`, `verify-baseline` |
| `mc geometry safety` | `probe-redteam`, `probe-behavioral` |
| `mc merge` | `merge -s ... -t ... -o ...` |
| `mc model` | `probe`, `analyze-alignment` |

**MCP Server**: Tools for integration with agentic systems (e.g., Claude Desktop, codeium).

---

## 3. Core Capabilities

### 3.1 Representation Geometry

**Centered Kernel Alignment (CKA)**

Compares representation similarity without assuming shared coordinates (Kornblith et al., 2019):

$$\text{CKA}(X, Y) = \frac{\text{HSIC}(K_X, K_Y)}{\sqrt{\text{HSIC}(K_X, K_X) \cdot \text{HSIC}(K_Y, K_Y)}}$$

Where HSIC is the Hilbert-Schmidt Independence Criterion on kernel matrices.

**Implementation**: `src/modelcypher/core/domain/geometry/concept_response_matrix.py`

**Topological Fingerprinting**

Computes Betti numbers of activation manifolds via persistent homology (Naitzat et al., 2020):

- β₀: Connected components
- β₁: Loops/holes
- β₂: Voids

**Implementation**: `src/modelcypher/core/domain/geometry/topological_fingerprint.py`

### 3.2 Entropy Dynamics

**Token-Level Entropy**

$$H(x_i) = -\sum_{v \in V} p(v | x_{<i}) \log p(v | x_{<i})$$

**Base-Adapter Divergence (ΔH)**

Measures entropy difference between base and instruction-tuned models as a safety signal:

$$\Delta H = H_{\text{base}}(x) - H_{\text{tuned}}(x)$$

**Implementation**: `src/modelcypher/core/domain/entropy/entropy_delta_tracker.py`

### 3.3 Safety Analysis

**Refusal Direction Detection**

Identifies the linear direction mediating refusal behavior (Arditi et al., 2024):

$$d_{\text{refusal}} = \mathbb{E}[h_{\text{refuse}}] - \mathbb{E}[h_{\text{comply}}]$$

**Implementation**: `src/modelcypher/core/domain/geometry/refusal_direction_detector.py`

### 3.4 Model Merging

**Null-Space Transplant**:
1. Align representations via probe-derived transforms
2. Project source deltas into the target null space
3. Add projected deltas to preserve target behavior

**Implementation**: `src/modelcypher/core/use_cases/merge/`

TIES/DARE are tracked as research references and are not part of the merge pipeline.

---

## 4. Validation

### 4.1 Test Coverage

Run `poetry run pytest` to verify current test coverage and pass status.

### 4.2 Module Import Guard

Automated test ensures all modules remain importable as the codebase evolves:

```python
# tests/test_module_import_guard.py
@pytest.mark.parametrize("module_path", discover_all_modules())
def test_module_imports(module_path):
    importlib.import_module(module_path)
```

---

## 5. Case Studies

### 5.1 Cross-Model Semantic Prime Analysis

**Objective**: Test whether semantic primes induce stable cross-model structure.

**Protocol**:
```bash
poetry run mc geometry primes probe-model /path/to/qwen --output-file qwen_primes.json
poetry run mc geometry primes probe-model /path/to/llama --output-file llama_primes.json
poetry run mc geometry primes compare qwen_primes.json llama_primes.json
```

### 5.2 Entropy-Based Safety Signal

**Objective**: Detect harmful prompts via ΔH before response generation.

**Protocol**:
```bash
poetry run mc thermo detect "<harmful>" --model /path/to/tuned
# Compare base vs tuned samples with mc entropy dual-path (requires precomputed samples)
```

### 5.3 Model Merge Pipeline

**Objective**: Merge two models with null-space transplant and inspect geometry metrics.

**Protocol**:
```bash
poetry run mc merge run -s /path/to/source -t /path/to/target -o /path/to/output --dry-run
poetry run mc merge run -s /path/to/source -t /path/to/target -o /path/to/output
poetry run mc model probe /path/to/output --output json
```

---

## 6. Related Tools

| Tool | Focus | Comparison |
|------|-------|------------|
| **TransformerLens** | Mechanistic interpretability | Circuits, activation patching. ModelCypher adds geometry, merging. |
| **CircuitsVis** | Visualization | Attention visualization. ModelCypher adds CLI, MCP. |
| **mergekit** | Model merging | Weight operations. ModelCypher adds diagnostics-first, safety. |
| **LM-Eval** | Benchmarking | Accuracy metrics. ModelCypher adds geometric analysis. |

---

## 7. Limitations

1. **MLX-Centric**: Optimized for Apple Silicon; CUDA support is secondary.
2. **English-Centric**: Anchor sets are primarily English; multilingual probe support is in development.
3. **Model Coverage**: Tested on Qwen, Llama, Mistral; other families may require adaptation.
4. **Alignment Requirements**: Geometric alignment must be exact (CKA=1.0); compatibility metrics are raw measurements and require debugging when not aligned.

---

## 8. Conclusion

ModelCypher makes the Geometric Knowledge Thesis operational. The results from Papers I–III are not theoretical claims—they are CLI commands that produce raw measurements. The toolkit provides methodology for entropy-based safety analysis and model merging. Module and test counts evolve with the codebase; consult the repository for current totals. Knowledge has shape, and now we can measure it.

---

## References

[Arditi et al. (2024)](../docs/references/arxiv/Arditi_2024_Refusal_Single_Direction.pdf). Refusal in Language Models Is Mediated by a Single Direction. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).

[Kornblith et al. (2019)](../docs/references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf). Similarity of Neural Network Representations Revisited. *ICML 2019*. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414).

[Naitzat et al. (2020)](../docs/references/arxiv/Naitzat_2020_Topology_Deep_Neural_Networks.pdf). Topology of Deep Neural Networks. *JMLR*, 21(184), 1-85. [arXiv:2004.06093](https://arxiv.org/abs/2004.06093).

[Yadav et al. (2023)](../docs/references/arxiv/Yadav_2023_TIES_Merging.pdf). TIES-Merging. *NeurIPS 2023*. [arXiv:2306.01708](https://arxiv.org/abs/2306.01708).

[Yu et al. (2024)](../docs/references/arxiv/Yu_2023_Language_Models_are_Super_Mario_Absorbing.pdf). DARE. *ICML 2024*. [arXiv:2311.03099](https://arxiv.org/abs/2311.03099).

---

## Appendix A: Installation

```bash
git clone https://github.com/Ethyros-AI/ModelCypher.git
cd ModelCypher
poetry install
poetry run pytest tests/ -q  # Run test suite to verify coverage
```

## Appendix B: Repository Structure

```
ModelCypher/
├── src/modelcypher/     # Core library
├── papers/              # Research papers (this series)
├── docs/references/     # Reference PDFs (arXiv + conferences)
├── docs/research/       # Master bibliography
├── tests/               # Test suite
└── CLAUDE.md            # AI agent instructions
```
