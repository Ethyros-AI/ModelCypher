# ModelCypher: A Geometric Toolkit for Large Language Model Analysis and Safe Adaptation

**Author**: Jason Kempf
**Affiliation**: EthyrosAI  
**Date**: December 2025

---

## Abstract

We present **ModelCypher**, an open-source Python toolkit for geometric analysis of large language models. The framework implements over 270 domain modules across eight core areas: representation geometry, entropy dynamics, safety constraints, agent observability, training diagnostics, and model merging. We demonstrate three workflows: (1) measuring cross-model semantic similarity via Centered Kernel Alignment on anchor sets, (2) detecting safety violations through entropy divergence monitoring, and (3) cross-architecture adapter transfer using Procrustes alignment. The toolkit integrates with both command-line and Model Context Protocol (MCP) interfaces, enabling integration with agentic AI systems. We validate correctness with 2,972 passing tests and specify experimental protocols for each capability. All code, documentation, and 46 foundational research papers are released under AGPLv3 license.

---

## 1. Introduction

Large language models present a fundamental challenge: their capabilities are difficult to understand, their safety properties are fragile, and their adapters are architecture-locked. Research progress requires tools that bridge theoretical frameworks with practical measurement.

**ModelCypher** addresses this gap by implementing:

1. **Geometric Diagnostics**: CKA, topological fingerprints, intrinsic dimension estimation
2. **Entropy Monitoring**: Token-level entropy, base-adapter divergence, circuit breaker integration
3. **Safety Analysis**: Refusal direction detection, capability guards, adapter probes
4. **Model Operations**: Merging, stitching, cross-architecture transfer

### 1.1 Design Principles

**Measurement Before Metaphor**: All geometric claims are operationalized as computable metrics with falsification criteria.

**Diagnostics Before Intervention**: The toolkit emphasizes analysis over modification; merging tools require compatibility assessment.

**Reproducibility**: All experiments are runnable via CLI with deterministic seeds and version-pinned dependencies.

---

## 2. System Architecture

### 2.1 Domain Structure

```
modelcypher/core/domain/
├── geometry/     (60+ modules)  # CKA, fingerprints, alignment
├── entropy/      (20+ modules)  # Tracking, windows, probes
├── safety/       (30+ modules)  # Guards, calibration, sidecar
├── agents/       (25+ modules)  # Traces, atlases, validators
├── training/     (25+ modules)  # Checkpoints, metrics
├── thermo/       (15+ modules)  # Linguistic thermodynamics
├── adapters/     (20+ modules)  # LoRA, DARE, DoRA
└── merging/      (15+ modules)  # Transport, TIES, DARE
```

### 2.2 Interface Layers

**CLI** (`mc`): 32+ commands across 6 groups

| Command Group | Examples |
|--------------|----------|
| `mc geometry` | `primes probe`, `fingerprint`, `cka compare` |
| `mc entropy` | `measure`, `sweep`, `window` |
| `mc safety` | `adapter-probe`, `guard-check` |
| `mc model` | `merge`, `analyze-alignment`, `stitch` |

**MCP Server**: 36+ tools for integration with agentic systems (e.g., Claude Desktop, codeium).

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

**Implementation**: `src/modelcypher/core/domain/safety/refusal_direction_detector.py`

### 3.4 Model Merging

**TIES-Merging** (Yadav et al., 2023):
1. **Trim**: Zero parameters below threshold
2. **Elect Sign**: Resolve sign conflicts by majority vote
3. **Merge**: Average surviving parameters

**DARE** (Yu et al., 2024):
1. Drop random fraction p of delta parameters
2. Rescale remaining by 1/(1-p)

**Implementation**: `src/modelcypher/core/domain/merging/`

---

## 4. Validation

### 4.1 Test Coverage

| Metric | Value |
|--------|-------|
| Domain modules | 274 |
| Unit tests | 2,972 |
| Pass rate | 100% |

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
mc geometry primes probe --model qwen2.5-3b --output qwen_primes.json
mc geometry primes probe --model llama-3.2-3b --output llama_primes.json
mc geometry primes compare --file-a qwen_primes.json --file-b llama_primes.json
```

**Expected Output**: CKA score and statistical significance vs null distribution.

### 5.2 Entropy-Based Safety Signal

**Objective**: Detect harmful prompts via ΔH before response generation.

**Protocol**:
```bash
mc entropy measure --model base-model --prompt "<harmful>" --output base.json
mc entropy measure --model tuned-model --prompt "<harmful>" --output tuned.json
# Compute ΔH = H_base - H_tuned
```

### 5.3 Cross-Architecture Adapter Transfer

**Objective**: Transfer LoRA from Qwen to Llama while measuring skill retention.

**Protocol**:
```bash
mc model analyze-alignment --source qwen2.5-7b --target llama-3.2-8b
mc model stitch --source qwen2.5-7b --adapter coding.safetensors --target llama-3.2-8b
mc eval suite --model merged --suite humaneval-subset.json
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
2. **English-Centric**: Anchor sets are English; multilingual probes are TODO.
3. **Model Coverage**: Tested on Qwen, Llama, Mistral; other families may require adaptation.
4. **Approximate Methods**: Geometric alignment is not exact; compatibility assessment is heuristic.

---

## 8. Conclusion

ModelCypher provides a unified toolkit for geometric LLM analysis, bridging theoretical frameworks with reproducible measurements. By implementing methodology from 46 foundational papers and specifying falsifiable experimental protocols, it enables rigorous research on representation structure, safety signals, and cross-architecture transfer.

---

## References

Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.

Kornblith, S., et al. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*. arXiv:1905.00414.

Naitzat, G., et al. (2020). Topology of Deep Neural Networks. *JMLR*, 21(184), 1-85. arXiv:2004.06093.

Yadav, P., et al. (2023). TIES-Merging. *NeurIPS 2023*. arXiv:2306.01708.

Yu, L., et al. (2024). DARE. *ICML 2024*. arXiv:2311.03099.

---

## Appendix A: Installation

```bash
git clone https://github.com/user/ModelCypher.git
cd ModelCypher
poetry install
poetry run pytest tests/ -q  # Expected: 2972 passed
```

## Appendix B: Repository Structure

```
ModelCypher/
├── src/modelcypher/     # Core library (274+ modules)
├── papers/              # Research papers (this series)
├── docs/references/     # 46 downloaded arXiv PDFs
├── docs/research/       # Master bibliography
├── tests/               # 100+ test files, 2972 tests
└── CLAUDE.md            # AI agent instructions
```
