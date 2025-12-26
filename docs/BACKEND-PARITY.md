# Backend Parity Checklist

Goal: MLX, JAX, and CUDA users should get the same capabilities, with backend-appropriate
performance defaults and no MLX-only blockers in shared paths.

## Current Snapshot

- Backend protocol coverage is complete (MLX/JAX/CUDA).
- Training engines exist for MLX/JAX/CUDA.
- Dual-path inference exists for MLX/JAX/CUDA.

## Parity Checklist

- [x] Backend selection honors MC_BACKEND (alias MODELCYPHER_BACKEND) and detects CUDA/JAX.
- [x] System health reports MLX/JAX/CUDA versions and availability.
- [x] Inference package exports platform-selected DualPathGenerator classes.
- [x] Training package exports platform-selected TrainingEngine/CheckpointManager.
- [ ] CLI geometry commands avoid MLX-only loaders where possible (add platform loaders).
- [ ] Activation probing for merge pipeline supports CUDA/JAX.
- [ ] Evaluation service supports CUDA/JAX inference (not just mlx_lm).
- [ ] Entropy calibration supports CUDA/JAX inference.
- [ ] Adapter wrapping has CUDA/JAX equivalents to wrap-mlx.
- [ ] MCP system status includes CUDA/JAX performance flags (Flash/SDP, JAX device info).
- [ ] Tests cover JAX/CUDA parity for core services (system, inference, training).

## Backlog (Prioritized)

1) Platform loaders for CLI geometry probes
   - Provide CUDA/JAX model loaders and tokenizers alongside MLX.
   - Avoid MLX-only helpers in shared CLI command paths.

2) Merge pipeline activation collection
   - Implement collect_layer_activations_cuda/jax with HF models or backend-native hooks.
   - Preserve current MLX behavior and keep probe mode consistent.

3) Evaluation + calibration
   - Add CUDA/JAX implementations in use_cases/evaluation_service.py
   - Add CUDA/JAX entropy calibration path in use_cases/entropy_calibration_service.py

4) Adapter tooling parity
   - Add wrap-cuda/wrap-jax or a backend-agnostic wrapper with explicit layout metadata.

5) Parity tests
   - Add tests for system service and MCP status fields across CUDA/JAX.
   - Add focused tests for inference platform selection.

