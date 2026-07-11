# Four-Bit Extension Report

Updated: 2026-03-08

## Question

Does the geometric RMT gate still pass after extending the Qwen3-1.7B bf16
model to 4-bit group-64 affine quantization?

## Retained Evidence

- Measurement JSON:
  `results/four_bit_extension/20260226T023950Z/four_bit_extension.json`
- Family summary JSON: `results/four_bit_extension/summary.json`
- Historical quantized model path now recorded only as provenance:
  `results/four_bit_extension/20260226T023950Z/derived_models/Qwen3-1.7B-MLX-bf16-4bit-g64-affine`

## Observed Values

- `n_layers`: `196`
- `n_layers_with_signal`: `196`
- `mean_signal_rank`: `425.2755102040816`
- `mean_signal_variance_fraction`: `0.5366134927340533`
- `mean_frobenius_norm`: `8.135853826999664`
- `mean_spectral_norm`: `0.44805739204190215`
- `rank_ratio_4b_over_8b`: `0.9999160278787442`
- `frob_ratio_4b_over_8b`: `16.93933705139871`
- `gate_pass`: `true`

## Verdict

4-bit gate passes. All 196 measured layers retain signal, the error structure is
nearly unchanged relative to 8-bit in rank terms, and the Frobenius-scale
damage is much larger. This preserves the geometric justification for
corrective LoRA, but the repo no longer keeps the 4-bit model dump itself in
the worktree.

## Cleanup Performed

- Downgraded the family to `summary_only`: keep the measurement JSON and family
  summary, delete the repo-local derived 4-bit model directory.
- Deleted duplicate run `20260226T023857Z` after verifying that its
  `model.safetensors` file matched the retained run byte-for-byte.
- Verified duplicate model SHA256:
  `1282ae3fdf8c9e6ce1e8c680e621181013c5d1bb5ea1b43940092271b46cf685`
- Deleted the retained run's derived-model directory after recording the
  historical `model.safetensors` SHA256:
  `1282ae3fdf8c9e6ce1e8c680e621181013c5d1bb5ea1b43940092271b46cf685`
- Deleted `results/four_bit_extension/latest.log`; the family summary now
  carries the useful provenance.
- Deleted raw payload in this pass: about `1040.95 MB`.

## Next Falsifier

Rerun `scripts/four_bit_extension.py` to materialize a fresh 4-bit model path,
then pass that path explicitly with `--quantized-model` to the downstream
correction and QK-correlation scripts. The historical path above is retained as
provenance only, not as a live model directory.
