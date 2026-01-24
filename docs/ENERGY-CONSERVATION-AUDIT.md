# Energy-Information Conservation Audit (Draft)

Purpose
- Treat information as energy; audit where algorithms transform, discard, or normalize energy and whether those steps are intended and measured.
- Focus on code paths that modify representations, not just measurements.

Definitions (operational)
- Energy = information content; measure using existing raw metrics (entropy, behavioral norms, spectral energy).
- Conservation check = compare pre/post values in the same space and units; record deltas.

Audit map: candidate non-conservation points

1) Blending or averaging (explicit mixing instead of additive transfer)
- `src/modelcypher/core/domain/geometry/transplant.py` (delta_scale < 0.5 triggers active/spectral/uniform blending)
- `src/modelcypher/core/domain/geometry/active_subspace_blend.py`
- `src/modelcypher/core/domain/geometry/spectral_blend.py`
- `src/modelcypher/core/domain/merging/lora_adapter_merger.py` (mean across aligned matrices)
- `src/modelcypher/core/domain/geometry/concept_response_matrix.py` (layer alignment interpolation)
- `src/modelcypher/core/domain/geometry/cross_cultural_geometry.py` (averaged grams)

Audit questions
- For each blend, what energy metric is being preserved (if any)?
- Is blending ever used in production paths, or only for diagnostics/experiments?
- If blending is intentional, do we log pre/post energy and preserved_fraction (behavioral or spectral)?

2) Null-space projection and filtering (intentional component removal)
- `src/modelcypher/core/domain/geometry/transplant.py` (null-space projection and geodesic filtering)
- `src/modelcypher/core/domain/geometry/geodesic_null_space.py` (projection_loss, preserved_fraction)
- `src/modelcypher/core/domain/geometry/channel_projector.py` (per-channel projection_loss)
- `src/modelcypher/core/domain/geometry/constrained_transplant.py`

Audit questions
- Are projection_loss and preserved_fraction logged for every merge path?
- Are behavioral norms computed on the same activation distribution before/after projection?
- Is any projection applied without a corresponding preserved_fraction measurement?

3) Rank truncation, compression, or denoising (explicit loss)
- `src/modelcypher/core/domain/compression/rmt_compressor.py` (rank-truncated pinv)
- `src/modelcypher/core/domain/geometry/rmt_signal_separation.py`
- `src/modelcypher/core/domain/geometry/intrinsic_compression.py`
- `src/modelcypher/core/domain/geometry/rank_selection.py`

Audit questions
- Is the removed energy explicitly accounted for (signal_variance_fraction, reconstruction_error)?
- Do compression paths compare behavioral impact, not just Frobenius error?

4) Dimension mismatch and truncation/padding
- `src/modelcypher/core/domain/vocabulary/embedding_projector.py` (_resize_features truncates or pads)
- `src/modelcypher/core/domain/vocabulary/cross_vocab_merger.py` (interpolated token mappings)
- `src/modelcypher/core/domain/geometry/shared_subspace_projector.py` (CCA shared_dimension)

Audit questions
- When truncating dimensions, is the discarded energy measured?
- When padding with zeros, is downstream normalization hiding energy drops?
- Do shared subspace projections report shared_variance_ratio alongside energy changes?

5) Normalization, clipping, and epsilon guards (unit changes)
- `src/modelcypher/core/domain/geometry/numerical_stability.py` (safe_log_epsilon, regularization_epsilon)
- `src/modelcypher/core/domain/entropy/logit_entropy_calculator.py` (entropy normalization to [0, 1])
- `src/modelcypher/core/domain/entropy/layer_entropy_projector.py` (eps in log)
- `src/modelcypher/core/domain/geometry/birkhoff_projector.py` (Sinkhorn normalization, spectral clipping)

Audit questions
- Are these strictly numeric guards (no semantic loss) or do they mask real energy deltas?
- Is a consistent unit tracked before and after normalization/clipping?

6) Proxy entropy measurements or simulated values
- `src/modelcypher/core/domain/entropy/logit_entropy_calculator.py` (proxy for semantic entropy)
- `src/modelcypher/core/domain/entropy/sep_probe.py` (predicted entropy)
- `src/modelcypher/core/use_cases/geometry_safety_service.py` (simulated prompt entropy)
- `src/modelcypher/core/use_cases/consolidation_service.py` (entropy from mean_density proxy)

Audit questions
- Where do proxies replace direct entropy? Is the replacement labeled in outputs?
- Do proxy paths propagate uncertainty about the proxy error?

7) Sampling, coreset selection, and top-k filtering
- `src/modelcypher/core/domain/geometry/acquisition_coreset.py`
- `src/modelcypher/core/domain/geometry/orthogonal_probe_generator.py`
- `src/modelcypher/core/domain/geometry/riemannian_sampling.py`
- `src/modelcypher/core/domain/semantics/vector_space.py` (threshold + argpartition top-k)

Audit questions
- What fraction of the manifold is sampled, and how does that affect energy estimates?
- Are energy metrics computed on full distributions or sampled subsets?

8) Precision and dtype limits
- `src/modelcypher/core/domain/geometry/numerical_stability.py` (precision caps/floors)
- `src/modelcypher/core/domain/geometry/precision_utils.py`

Audit questions
- Are any merges or projections performed at lower precision than the stored weights?
- Is energy compared across different dtypes without rescaling?

9) Temperature scaling and thermodynamic transforms
- `src/modelcypher/core/domain/thermo/phase_transition_theory.py` (temperature-scaled softmax)
- `src/modelcypher/core/domain/thermo/linguistic_calorimeter.py` (entropy trajectory and derived temperature)

Audit questions
- Are temperature-derived entropies treated as energy or as a transformed variable?
- Is energy conservation evaluated in the pre-temperature or post-temperature space?

Measurements to capture (existing signals)
- `projection_loss`, `preserved_fraction`, `transfer_strength` from null-space filtering and transplant.
- `reconstruction_error`, `signal_variance_fraction`, `signal_rank` from compression paths.
- `mean_entropy`, `entropy_variance`, `entropy_ratio` from entropy validators and profiles.
- `behavioral_norm` and `delta_norm` from transplant outputs.
- `cka` / `alignment_error` / `shared_variance_ratio` for alignment and shared subspace.

Cross-cutting checks (no thresholds)
- For every operation that changes representation: record pre/post energy metrics in the same space.
- For every proxy: tag outputs so later analyses can separate measured vs predicted energy.
- For every normalization/clipping step: log the scale factor or guard value used.
- For every sampling-based estimate: record sample count, coverage_ratio, and any selection criteria.

Search cues for future passes
- Keywords: blend, interpolate, project, truncate, clip, normalize, rank, coreset, argpartition, top-k, epsilon, precision.

Notes
- This document is a mapping of risk surfaces, not a verdict. Each item needs local measurement to decide whether energy is conserved or intentionally transformed.
