# Quantization Frontier Bedrock Review (2026-03-05)

**Selected item:** `Quantization crossing frontier vs CKA floor`

Source questions:
- `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`
- `docs/RESEARCH-ROADMAP.md`

## Scope

This note does not run new experiments. It reduces the question to the operator
that actually matters, reviews the current literature, and audits the MLX and
ModelCypher code paths that implement the relevant quantities.

## Thesis

The current candidate closure observable,

`max(||E_q||_2 / (gap_k / 2))`

is not the right causal quantity for the 4-bit CKA floor.

It measures whether a scalar perturbation budget exceeds a local singular-value
ordering boundary at the Shannon structural-rank cutoff of the **weight**
matrix. But the quantity we are trying to explain, `min_cka`, is a property of
the **centered output Gram geometry** induced by quantized inference on actual
activations.

The bedrock object is therefore not `||E_q||_2` alone, but the
activation-weighted perturbation

`E_q Sigma_x^(1/2)`

and the induced centered-Gram perturbation

`Delta K_c`.

## Literature Review

The state of the art has already moved toward activation-aware or
subspace-aware objects, even when papers do not phrase them in exactly these
terms.

| Thread | Representative source | What it preserves |
|---|---|---|
| Hessian-aware PTQ | GPTQ | Input/Hessian-weighted reconstruction, not raw Frobenius error |
| Activation-aware equivalent transform | AWQ, SmoothQuant | Salient activation directions / outlier migration |
| Rotation-based invariance | QuaRot, SpinQuant | Function-preserving orthogonal reparameterizations that reshape quantization difficulty |
| Low-rank error reconstruction | LQER, QERA, Preserve-Then-Quantize | Output/activation error or dominant subspace, not arbitrary tail ordering |
| Post-hoc recovery | Recover-LoRA, RILQ | Functional recovery after degradation, often exposing high-rank or activation-localized damage |

### Reading of the field

1. **GPTQ is already geometry-aware.**
   The recent geometric treatment shows GPTQ is equivalent to Babai's nearest
   plane algorithm on a lattice induced by the layer Hessian. This is not a
   "quantize by raw distance" story; it is a weighted nearest-vector problem.

2. **AWQ and SmoothQuant act on activation geometry.**
   Both methods reduce quantization difficulty by changing the basis in which
   activation outliers are seen. AWQ uses activation-aware channel scaling.
   SmoothQuant uses a mathematically equivalent migration of difficulty from
   activations to weights.

3. **QuaRot and SpinQuant make the key invariance explicit.**
   Orthogonal transforms can leave the full-precision transformer function
   unchanged while dramatically changing quantization difficulty. That is
   direct evidence that low-bit degradation is governed by subspace geometry and
   outlier orientation, not by individual singular-value ordering alone.

4. **QERA, LQER, and Preserve-Then-Quantize are even closer to our internal
   direction.**
   These methods reconstruct quantization error in activation-weighted or
   dominant-subspace terms. Their objective is output error, not weight error.

5. **Recover-LoRA and RILQ reinforce the same point from the recovery side.**
   When degradation is severe, low-rank recovery succeeds or fails based on the
   functional structure of the error. RILQ is especially important because it
   shows that at very low precision, some quantization error is effectively
   high-rank relative to simple SVD-based assumptions.

## Bedrock Math

### 1. MLX affine group quantization

MLX affine quantization is row-wise grouped quantization over contiguous chunks
of the last dimension. For each group:

`alpha = max_i w_i`

`beta = min_i w_i`

`s = (alpha - beta) / (2^b - 1)`

`w_hat_i = round((w_i - beta) / s)`

and dequantization returns

`w_i ~= s w_hat_i + beta`

with packed integers stored in `uint32`.

So the quantized weight is exactly

`W_q = W + E_q`

with deterministic, block-structured `E_q`.

### 2. The layer-output perturbation

For a linearized layer with input activations `X` and weight `W`,

`Y = X W^T`

`Y_q = X (W + E_q)^T = Y + Delta Y`

where

`Delta Y = X E_q^T`

This already shows the core mistake in the raw Weyl frontier observable:
output drift is not determined by `E_q` alone, but by `E_q` composed with the
input activation distribution.

### 3. Functional error is activation-weighted

Let `Sigma_x = E[x^T x]` be the input covariance for the layer's input
distribution. Then

`E ||Delta W x||^2 = tr(Delta W Sigma_x Delta W^T)`

`= ||Delta W Sigma_x^(1/2)||_F^2`

This is the exact right-side activation weighting already used in our own
`rmt_quantization_error.py`.

So the natural perturbation operator is

`A = E_q Sigma_x^(1/2)`

not `E_q`.

### 4. CKA is a centered-Gram observable

ModelCypher's linear CKA computes centered dot-product Gram similarity:

`K = H Y Y^T H`

`K_q = H Y_q Y_q^T H`

with

`CKA(Y, Y_q) = <K, K_q>_F / (||K||_F ||K_q||_F)`

Expanding `K_q` with `Y_q = Y + Delta Y`:

`Delta K = H (Y Delta Y^T + Delta Y Y^T + Delta Y Delta Y^T) H`

So the thing that moves CKA is the centered Gram perturbation induced by
`Delta Y`, not a singular-value swap at an arbitrary tail boundary of `W`.

### 5. Why raw Weyl crossing is insufficient

Weyl answers a specific question:

`|sigma_i(W + E) - sigma_i(W)| <= ||E||_2`

and the no-crossing corollary says local ordering at a boundary is preserved if

`||E||_2 < gap / 2`

But CKA does **not** care about individual singular values the way this test
does.

Two reasons:

1. **CKA is invariant to orthogonal feature rotations and isotropic scaling.**
   If quantization mostly rotates representations inside a subspace, CKA can
   remain near 1 even though individual singular vectors or singular values are
   reordered.

2. **Subspace stability is governed by a different theorem family.**
   Davis-Kahan and Wedin control eigenspace/singular-subspace motion by
   `||E|| / gap_subspace`, where the relevant gap is the separation of the
   subspace of interest from the rest, not the local gap at a Shannon
   structural-rank cutoff in weight space.

Therefore:

- A large `||E_q||_2 / (gap_k / 2)` at a near-degenerate tail boundary is **not
  sufficient** to predict low CKA.
- A model can have many raw crossings and still preserve output Gram geometry.
- The right gap is the gap of the **activation-relevant output covariance**
  or the centered Gram spectrum, not the weight-tail boundary by itself.

## What the Local Data Already Says

The repo's own results already point in this direction.

1. The 8-bit Weyl validation reports `0/448` layers safe under the raw
   no-crossing condition, while `sigma_max`, `sigma_k`, and `tail_dims` remain
   almost unchanged.

2. The deep-dive note already states the paradox correctly: many raw tail
   crossings occur, yet function is largely preserved.

3. The activation-weighted RMT analysis shows the model experiences error in
   the directions it uses through `E_q Sigma_x^(1/2)`, which is a closer causal
   object than raw `E_q`.

4. The Tikhonov correction path already operates in the activation covariance
   eigenbasis, not on raw weight-space gaps.

So the internal codebase already contains the bridge to the right mechanism.
The open question is not waiting for a new heuristic. It is waiting for the
correct operator to be promoted.

## MLX Handling

### What MLX actually does

In MLX:

- `mx.quantize(...)` produces packed quantized weights plus scales and
  optional biases.
- `nn.QuantizedLinear.__call__()` uses `mx.quantized_matmul(...)` directly.
- The quantized forward path is fused around the packed representation rather
  than materializing a dequantized weight matrix on every call.

This matters because MLX is not "adding random noise." It is executing a fixed,
group-structured operator with shared per-group affine parameters.

### Supported modes

The current MLX docs and installed package expose:

- `affine`
- `mxfp4`
- `mxfp8`
- `nvfp4`

ModelCypher's current quantization service is primarily written around affine
group quantization, with shape-based inference and a bias-aware fallback for
`mxfp4`.

## ModelCypher Handling

### What the current precheck measures

`run_quantization_weyl_precheck()` does the following:

1. dequantize or load matching FP and quantized weights
2. compute `LayerGeometry` on the FP weight
3. take `spectral_gap` at the Shannon effective-rank boundary
4. compute `||W_fp - W_q||_2`
5. flag a crossing when `||E_q||_2 >= gap / 2`

That is mathematically coherent for the question

"did the perturbation exceed the raw Weyl no-crossing budget at this chosen
weight-space boundary?"

But it is not yet the right question for `min_cka`.

### Where the mismatch enters

`compute_layer_geometry()` defines the tracked boundary by Shannon effective
rank. That is a structural weight-space boundary, useful for LoRA capacity and
null-space reasoning. It is not automatically the dominant output subspace for
the layer's actual activation distribution.

So the current frontier proxy mixes two different objects:

- a weight-space boundary from spectral entropy
- an activation-space similarity observable from centered Grams

### The strongest code-level tension

`dataset_training_service.py` originally treated any detected raw crossing as a
hard block unless the legacy crossing override was enabled. The current
frontier gate instead blocks only when the activation-aware operator cannot be
measured, unless `research_allow_quantization_frontier_invalid=True`.

That is stronger than the measured local evidence supports. The repo already
contains an 8-bit result where all layers violate the raw frontier but major
spectral structure and much of the function are retained.

## The Corrected Frontier Observable

If the goal is to explain `min_cka`, the frontier should be defined in one of
the following activation-aware ways.

### Option A: activation-weighted subspace ratio

Per layer, define

`A_l = E_l Sigma_x,l^(1/2)`

and let `C_y,l = W_l Sigma_x,l W_l^T`.

Then use

`rho_act,l = ||A_l||_2 / gap_eff,l`

where `gap_eff,l` separates the output covariance subspace we actually care
about. In practice this gap must be conditioned on the effective dimensions of
the activation distribution, not the full hidden dimension: `D_eff` should be
measured from the activation-relevant output spectrum, then `gap_eff,l` taken at
that `D_eff`-conditioned boundary.

This matches perturbation theory on the output-producing operator rather than
the raw weight matrix.

### Option B: direct Gram perturbation ratio

For probe activations,

`eps_K,l = ||Delta K_c,l||_F / ||K_c,l||_F`

with

`Delta K_c,l = K_c,l^q - K_c,l`

This is even cleaner because it measures perturbation in the same centered-Gram
space that CKA itself uses.

ModelCypher already has the exact lower bound:

`CKA >= (1 - eps_K) / (1 + eps_K)`

So if the mission question is "what explains the CKA floor?", this bound is
closer to bedrock than `max(error/(gap/2))`.

### What this precheck can and cannot predict

An activation-aware frontier precheck predicts the **base** perturbation between
the quantized model and its full-precision reference on measured probes. It does
not by itself predict corrective reach. The deep-dive correction experiments
show that recovery is compensatory rather than restorative: the correction acts
in the observed activation subspace and need not reduce `E_q` in weight space.
So the frontier observable should gate measurability of base divergence first,
not pretend to upper-bound post-correction CKA without an additional model of
the correction operator.

## Consequences For The Open Question

The question should be tightened from

"Are 4-bit CKA limits explained by measured Weyl crossing severity?"

to

"Are 4-bit CKA limits explained by activation-weighted subspace perturbation or
centered-Gram perturbation, and does raw Weyl crossing add any residual causal
information once those are measured?"

That version matches:

- the observable we care about (`min_cka`)
- the actual inference operator
- the structure of MLX quantization
- the direction of the current literature
- the activation-weighted and Tikhonov code already present in this repo

## No-Guess Next Step

Before any new experiment:

1. Derive the layerwise bound from
   `Y_q = Y + X E_q^T`
   to
   `Delta K_c`
   and the existing CKA lower bound.
2. Define `gap_eff` on the `D_eff`-conditioned effective dimensions of the
   output covariance or centered Gram spectra, not the full ambient space.
3. Implement an activation-weighted precheck alongside the raw Weyl precheck.
4. Only then run the 4-bit confirmation pass.

The current raw Weyl metric should be retained as telemetry, but not promoted
as the sole causal frontier until it beats the activation-aware observable on
the same models and bitwidths.

## Sources

### Local

- `docs/research/OPEN-MATHEMATICAL-QUESTIONS.md`
- `docs/RESEARCH-ROADMAP.md`
- `docs/research/quantization_geometry_deep_dive.md`
- `src/modelcypher/core/domain/training/quantization_weyl_precheck.py`
- `src/modelcypher/core/domain/training/geometric_lora.py`
- `src/modelcypher/core/domain/geometry/cka.py`
- `src/modelcypher/core/use_cases/quantization_utils.py`
- `src/modelcypher/core/use_cases/quantization_correction_service.py`
- `scripts/rmt_quantization_error.py`

### External

- Citation check (2026-03-05): the previously questioned arXiv entries
  `2507.18553` and `2602.02001` were verified against arXiv before this note
  was retained.
- GPTQ geometry: https://arxiv.org/abs/2507.18553
- AWQ: https://arxiv.org/abs/2306.00978
- SmoothQuant: https://arxiv.org/abs/2211.10438
- QuaRot: https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2405.16406
- LQER: https://arxiv.org/abs/2402.02446
- QERA: https://arxiv.org/abs/2410.06040
- Preserve-Then-Quantize: https://arxiv.org/abs/2602.02001
- Recover-LoRA: https://aclanthology.org/2025.emnlp-industry.164/
- Wedin perturbation note: https://gauss.uc3m.es/fdopico/papers/bit2000.pdf
- Davis-Kahan variant note: https://personal.lse.ac.uk/wangt60/talks/DKvariant.pdf
- MLX quantized matmul docs: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- MLX dequantize docs: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html
