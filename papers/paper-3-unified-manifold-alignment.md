# Paper III Draft: Unified Manifold Alignment (Engineering)

## Abstract (Draft)

We present a geometric alignment framework for cross-architecture adapter transfer and model merging. The approach decomposes alignment into three spaces: weight space (low-rank updates), representation space (anchor geometry), and probability space (distributional drift). We extend Cross-LoRA with anchor-locked k-space rotation, introduce intersection maps via Sliced Gromov-Wasserstein topology matching, and implement Verb/Noun asymmetric fusion to preserve specialized skills.

## 1. Introduction (Draft)

Adapter lock-in and brittle merges block reliable deployment of bounded agents across model families. A LoRA trained on one base model rarely transfers to another without retraining, and naive weight averaging can destroy behavior. The industry needs a measurable alignment protocol that can say when transfer is likely to succeed and when it should be rejected.

We treat alignment as a geometry problem. If shared concepts exist as stable invariants (Paper I), then adapters and merges should be expressible as transport operations across those invariants. Our framework decomposes alignment into three spaces: weight space (LoRA transport), representation space (anchors and rotation), and probability space (drift and smooth embedding). Each space provides a diagnostic and a guardrail.

Methodologically, we build on Cross-LoRA transport through truncated SVD bases \cite{xia2025crosslora}, then orient those bases using anchor-locked orthogonal Procrustes rotations. We introduce intersection maps to quantify layerwise correspondence via anchor activation sets and use permutation alignment (Git Re-Basin style) with soft-gated fusion to merge aligned subspaces \cite{ainsworth2023git,yadav2023ties}. These components are implemented on-device in TrainingCypher and exposed through a CLI pipeline designed for reproducibility.

The evidence is mixed by design. Some components are supported by existing experiments; others are prototype-level and require benchmarking. This paper makes that boundary explicit and provides a roadmap for the missing evaluations. The goal is not to claim a solved problem, but to provide a credible, testable, engineering framework for alignment.

## 2. Related Work (Draft)

Low-rank adaptation and PEFT methods make adaptation efficient but tightly coupled to the base model \cite{hu2022lora}. Cross-LoRA proposes data-free adapter transfer by projecting updates through truncated SVD bases \cite{xia2025crosslora}. Model merging methods such as TIES-Merging and Model Soups show that weight-space composition can work within a basin but do not address cross-architecture alignment \cite{yadav2023ties,wortsman2022soups}. Git Re-Basin demonstrates permutation-based alignment within a model family and motivates symmetry-aware merging \cite{ainsworth2023git}.

On the representation side, convergence hypotheses and similarity metrics motivate the existence of shared invariants but do not provide transport maps \cite{huh2024platonic,kornblith2019cka}. Our contribution is to combine these threads into a three-space alignment stack with concrete diagnostics and a portable implementation.

## 3. Methods (Draft)

### 3.1 Three-Space Alignment Stack

We separate alignment into three coupled spaces:

1. Weight space: transport low-rank adapter operators across model parameterizations.
2. Representation space: use anchor geometry to orient and validate transport.
3. Probability space: measure distributional drift to assess smooth embedding.

Each space supplies a measurable signal: subspace projection error, anchor alignment error, and KL drift, respectively.

### 3.2 Cross-LoRA Transport (Weight Space)

Let Delta W_s be a LoRA update on source model S and W_t a target base matrix. Cross-LoRA approximates a transport via truncated SVD bases:

W ~= U_k Sigma_k V_k^T

Delta W_t ~= U_t (U_s^T Delta W_s V_s) V_t^T

TrainingCypher performs factor-level projection without materializing Delta W and enforces alignmentRank >= LoRA rank (see `docs/research/CROSS_LORA_ADAPTER_PROJECTION.md`).

### 3.3 Anchor-Locked k-Space Rotation (Representation Space)

Truncated SVD bases are only defined up to rotation, sign, and permutation. We orient k-space using anchors (semantic primes and computational gates). For each module, anchors are projected into the source and target SVD bases, then an orthogonal Procrustes rotation Omega is solved to align them. The rotation is applied directly to LoRA factors during transport:

B_t = U_t (Omega_out (U_s^T B_s))
A_t = ((A_s V_s) Omega_in) V_t^T

A compatibility metric e = ||Z_s Omega - Z_t||_F / ||Z_t||_F is reported as a guardrail. TrainingCypher warns when mean e > 0.3 and aborts when max e > 0.8.

### 3.4 Intersection Maps and Sliced Gromov-Wasserstein

Standard alignment assumes rigid rotation (Procrustes). We relax this by computing Sliced Gromov-Wasserstein (SGW) distance between layer manifolds. SGW projects high-dimensional manifolds into 1D slices and computes optimal transport cost, providing a robust metric for topological similarity that survives non-rigid deformations. This yields a "confidence map" for layer-wise merging.

### 3.5 Verb/Noun Decomposition (Asymmetric Fusion)

We introduce a semantic basis for fusion. Dimensions are classified as "Nouns" (Knowledge/Stability) or "Verbs" (Skills/Variance) based on their activation variance across Computational Gates versus Semantic Primes.
- **Noun Dimensions:** High stability across primes. Trust the Target model (Base Knowledge).
- **Verb Dimensions:** High variance across gates. Trust the Source model (New Skill).
This allows for asymmetric merging where a coding adapter transfers "Verb" dimensions without overwriting the "Noun" facts of the base model.

### 3.6 Permutation Alignment and Soft-Gated Fusion

Permutation symmetries can hide aligned subspaces. We apply permutation alignment (Git Re-Basin style) to MLP blocks, then fuse weights with a confidence-weighted gate. This limits destructive interference and preserves residual stream continuity \cite{ainsworth2023git,yadav2023ties}.

### 3.6 Smooth Embedding (Probability Space)

Transported adapters can still introduce nonlinear mismatches. We use probe prompts and KL drift to measure probability-space deviation and apply smoothing when drift exceeds a threshold (see `docs/research/GEOMETRIC_ADAPTER_TRANSFER_VIA_OPERATIONAL_INVARIANTS.md`). This step is diagnostic today and not yet fully benchmarked.

## 4. Experiments (Draft)

### 4.1 Anchor Stability (Prerequisite)

We rely on anchor stability evidence from Paper I to justify anchor-locked rotation: primes and gates show higher centered alignment than control words across models (`docs/research/SEMANTIC_PRIME_SKELETON_EXPERIMENT.md`, `docs/research/cross-cultural-geometry-experiment.md`).

### 4.2 Cross-LoRA Transport with Anchor Lock

We apply Cross-LoRA projection with anchor-locked rotations and record per-module alignment error e. This provides a transport feasibility signal and a guardrail for aborting incompatible transfers (`docs/research/CROSS_LORA_ADAPTER_PROJECTION.md`).

### 4.3 Intersection Map Diagnostics

We generate activation fingerprints for anchors and compute layerwise Jaccard matches with confidence scores. This is currently implemented as tooling; cross-family quantitative results are still pending (`docs/research/INTERSECTION_MAP_THEORY.md`).

### 4.4 Cross-Family Merge Prototype

We ran the 4KD pipeline on Qwen3-8B <-> Mistral-7B, aligning 128 layers with mean alignment error around 8.03. The run uses MLP-only permutation alignment and confidence-weighted fusion (`docs/research/GEOMETRIC_MERGING_MILESTONE.md`). This is a prototype result without task-suite benchmarks.

## 5. Results (Draft)

- A1 (Cross-LoRA transport) is supported at the level of implementation and alignment diagnostics, but lacks task-level transfer benchmarks.
- A2 (anchor-locked rotation) is implemented with a measurable compatibility metric; no controlled ablations yet.
- A3 (intersection maps) provide a layer confidence signal, but cross-family evaluations remain pending.
- A4 (permutation alignment plus soft-gated fusion) shows a successful cross-family merge run with mean alignment error ~8.03, but no downstream evaluation.

These results motivate the framework while making clear that external benchmarks are still required.

## 6. Discussion (Draft)

Alignment is diagnostic before it is transformative. High anchor alignment is a necessary condition for safe transfer, not a sufficient one. The three-space stack exposes failure modes: weight-space transport can succeed while probability-space drift remains high, and representation-space alignment can be strong while permutation mismatches still damage behavior.

The framework is also compatible with safety boundaries from Paper II. Entropy and conflict signals can be incorporated into probability-space smoothing to reject transfers that destabilize model behavior. This connects geometric alignment to operational safety.

## 7. Limitations (Draft)

- Tokenizer mismatch and non-bijective layer correspondence remain unresolved.
- Cross-family benchmarks are missing; current results are diagnostic and prototype-level.
- Intersection maps use top-k sets and Jaccard overlap, which can be brittle.
- Compatibility thresholds are heuristic and need calibration across families.

## 8. Conclusion (Draft)

We present a unified alignment framework that decomposes transfer into weight, representation, and probability spaces. The pipeline is implemented and produces measurable diagnostics, including a cross-family merge prototype, while explicitly marking missing evaluations. This positions alignment as a geometric engineering problem grounded in verifiable invariants and provides the practical bridge between the geometry of Paper I and the stability signals of Paper II.

## Appendix A. Benchmark Plan (Draft)

This plan targets the open benchmarks needed to move A1-A4 from diagnostic to validated.

### A.1 Intersection Map Runs

- Models: Qwen2.5-7B, Qwen2.5-Coder-7B, Mistral-7B, Llama-3.2-3B (all local MLX or HF IDs).
- Anchors: semantic primes (multilingual), computational gates (ensemble).
- Output artifacts: intersection map JSON, layer confidence summaries, fingerprint dumps.
- Metrics: mean Jaccard overlap per layer, coverage (matched dims / total), confidence-weighted score.

### A.2 Merge Evaluation Suite

- Suites: `docs/research/eval_suites/coding_suite.v1.json` and `docs/research/eval_suites/creativity_suite.v1.json`.
- Baselines: source model, target model, TIES-Merging, model soup (weight average), 4KD pipeline.
- Retention metrics: score(merged, suite) / score(parent, suite) for each domain.
- Scoring: apply suite constraints to `tc infer suite` outputs (manual/custom scorer).

### A.3 Merge Sweep and Diagnostics

- Sweep grid: alpha in {0.2, 0.35, 0.5, 0.65, 0.8}, rank in {16, 32, 64}, anchor mode in {semantic-primes, intersection, rebasin}.
- Diagnostics: mean/max Procrustes error, rotation roughness, anchor coverage, permutation alignment stats.
- Report: merged model score vs diagnostics to identify safe operating regions.

### A.4 Publication Table Targets

- Table 1: Cross-family adapter transfer (accuracy delta vs direct LoRA).
- Table 2: Merge retention tradeoff curve (coding vs creativity).
- Table 3: Intersection map confidence vs merge stability (correlation summary).

## References (Draft)

\\bibliographystyle{plain}
\\bibliography{references}
