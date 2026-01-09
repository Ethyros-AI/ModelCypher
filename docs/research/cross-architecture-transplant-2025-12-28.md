# Cross-Architecture Transplant Validation (2025-12-28)

This note records the successful cross-architecture transplant validation using
null-space constrained functional replacement.

## Summary

- Source to target: Qwen2 -> SmolLM
- Method: null-space constrained transplant with GRAM_TRANSPORT projection
- Boundary preservation: relative diff reported as 0.00
- Core alignment improvement: +27.4%
- Output coherence: correct Fibonacci definition (no gibberish)

## Evidence

Results file:
`/path/to/experiments/cross-arch-transplant-2025-12-28/qwen2-to-smolm/transplant_result.json`

This confirms transplant viability for cross-architecture pairs without
changing the pipeline code, and validates the shift from weight blending to
functional replacement.
