# 15-Hyperparameter Research Program

**Status:** Research program with mixed runtime evidence
**Updated:** 2026-07-04

This document tracks the downstream training derivation program separately from
the shipped measurement workbench. The program goal is still to derive the
traditional controls from spectral structure, activation geometry, IEEE 754
precision, or direct measurements. The public claim is derivation status, not
optimizer superiority.

The current shipped default is not "all 15 controls replaced." `mc train run`
derives the adapter surface, rank, batch sizing, scale budget, and stopping
certificate. Its canonical optimizer path uses calibrated AdamW with `lr=2e-4`,
cosine decay, and AdamW betas `0.9/0.999`. MASS step sizing remains wired for
research optimizer modes.

## Evidence States

| Evidence state | Meaning |
|---|---|
| `derived+shipped-default` | The derivation is wired into the canonical `mc train run` path. |
| `derived+research-mode-only` | A derivation exists and is wired only behind non-default research modes. |
| `formula-exists-unwired` | A formula or helper exists, but the shipped runtime does not consume it as the public claim says. |
| `dead-code` | Code exists without a shipped runtime consumer. |
| `removed` | The control was intentionally removed rather than replaced by an active formula. |

## Per-Control Status

| # | Control | Evidence state | Current runtime truth | Falsifier before promotion |
|---|---|---|---|---|
| 1 | Learning rate | `derived+research-mode-only` | Default path is calibrated AdamW `2e-4` with cosine; MASS is research-mode only. | Same-model same-data closure benchmark shows MASS path beats calibrated AdamW outside the pre-registered tie band. |
| 2 | Adam epsilon | `formula-exists-unwired` | `compute_geometric_epsilon` has weight-singular-value units; shipped AdamW does not consume it. | Units are matched to Adam second-moment state and wired, or the claim stays removed from shipped docs. |
| 3 | Momentum | `derived+research-mode-only` | Default path uses AdamW betas `0.9/0.999`; Fisher/MASS moment logic is research-only. | Research-mode moment law beats AdamW betas under the frozen benchmark without task losses outside the tie band. |
| 4 | Weight decay | `formula-exists-unwired` | Condition-ratio formula exists; default runtime passes `weight_decay=0.0`. | Formula is wired and improves or preserves the closure benchmark against `0.0` and recipe decay. |
| 5 | Gradient clipping | `derived+research-mode-only` | No canonical geometric clipper; MASS research modes bound updates through controller terms. | Controller-bound updates replace clipping on the default path and logged bounds predict avoided failures. |
| 6 | Warmup | `derived+research-mode-only` | Canonical path starts calibrated cosine immediately; research modes rely on MASS ceilings. | No-warmup MASS path survives the frozen benchmark and its early-step displacement stays within measured bounds. |
| 7 | LR schedule | `derived+research-mode-only` | Default path uses cosine over six data-epochs; MASS no-schedule behavior is research-only. | No-schedule MASS path beats or ties cosine under the closure benchmark. |
| 8 | Batch size | `derived+shipped-default` | Logical batch size derives from gradient-noise scale, with memory-safe micro-batching as an implementation constraint. | Derived batch size fails to match measured gradient-noise scaling or harms the closure benchmark. |
| 9 | Early stopping | `derived+shipped-default` | Geometric certificate and measured validation-loss windows are wired. | Stop decisions fail to predict held-out degradation or stop too late under retained telemetry. |
| 10 | LoRA scale | `derived+shipped-default` | Adapter scale budget and saturation telemetry are enforced during training. | Per-layer effective delta violates the stated spectral bound or the bound fails to predict behavior. |
| 11 | LoRA rank | `derived+shipped-default` | Per-module ranks derive from tail dimensions and data capacity samples. | Derived ranks underperform fixed-rank baselines outside the tie band. |
| 12 | Target modules | `derived+shipped-default` | Target surface derives from layer spectral geometry. | Spectral target selection loses to fixed `q+v` or measured behavioral damage is not predicted. |
| 13 | Dropout | `formula-exists-unwired` | `compute_geometric_dropout` can appear in config payloads; the training adapter does not apply it as runtime dropout. | Runtime dropout is wired and validated, or the row remains unwired. |
| 14 | Weight init | `formula-exists-unwired` | Default init is PiSSA; the older spectral-normalized-to-`sigma_k` init is not the shipped default. | The documented init and the runtime init are unified, with retained benchmark evidence. |
| 15 | Residual scaling | `dead-code` | `residual_scaling.py` has no shipped training-path consumer. | A shipped consumer appears with a derivation and tests, or the code is deleted. |

## Canonical Links

- Runtime matrix generator: `scripts/generate_knob_matrix.py`
- README drift test: `tests/test_knob_matrix.py`
- Formula history: `docs/research/geometric_hyperparameter_rosetta_stone.md`
- Closure ladder: `docs/RESEARCH-ROADMAP.md`
