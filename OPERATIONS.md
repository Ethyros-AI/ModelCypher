# ModelCypher Operations

This file keeps machine-local operating notes out of the doctrine guide.
`AGENTS.md` is the single source for principles, architecture, and guardrails.

## Model Work

Before training, inference, evaluation, or any script that loads a model, run:

```bash
pgrep -af 'python|mlx' | grep -v grep
```

If any Python or MLX process may be using the GPU, ask the owner before
starting model work. Do not run tests while training is using the GPU.

Owner-side models live under `/Volumes/CodeCypher/models/`, for example:

| Path | Use |
|---|---|
| `mlx-community/LFM2-350M-MLX-bf16` | Smallest default validation model |
| `mlx-community/LFM2-700M-bf16` | Small scale check |
| `mlx-community/Qwen3.5-0.8B-bf16` | Hybrid architecture check |
| `mlx-community/LFM2.5-1.2B-Base-bf16` | Mid-scale validation |
| `mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16` | Final confidence only |

Use 350M to 700M models for debugging and math validation. Use 1B to 2B after
small models pass. Use 3B to 8B only for final validation.

## Common Commands

```bash
poetry install
poetry run pytest
poetry run mc train run -m MODEL -d DATA -o OUTPUT
poetry run mc train evaluate -m MODEL -a ADAPTER -d VAL
poetry run mc train compare -m MODEL --adapter-a A --adapter-b B -d VAL
poetry run mc analyze family --model MODEL --manifest data/probes/prompt_family_minimal_pairs.json
poetry run mc merge run -s SOURCE -t TARGET -o OUTPUT
poetry run mc model info MODEL
poetry run mc infer run --model MODEL --prompt "Hello"
```

On Linux or any machine without MLX/Metal, use:

```bash
MC_DISABLE_MLX=1 poetry run pytest -m "not real_model and not slow" -q
```

## Troubleshooting

- If a model path fails, confirm the external volume is mounted.
- If MLX tests fail on a non-Apple machine, use the JAX CPU fallback command.
- If CKA after closed-form alignment is below `1.0` on training probes, inspect
  the alignment path; held-out failures indicate probe coverage failure.
- If merge output damages target behavior, measure behavioral norm
  `||X delta_W^T||`, not Frobenius norm.

## Archived Material

Owner-side archives are under `/Volumes/CodeCypher/archive/` and are not part of
the portable repository state.
