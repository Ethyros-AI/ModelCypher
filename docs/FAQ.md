# Frequently Asked Questions

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options must come before the command path (example: `mc --output text model probe ./model`).

## Skepticism

### "This is just PCA with marketing."

ModelCypher uses a lot of linear algebra, and PCA is a useful baseline. The difference is that many of the tools here assume the data is not globally linear, and measure quantities (e.g., curvature, geodesic distances on a k-NN graph) that PCA does not model directly.

**The math:** PCA finds linear projections that maximize variance under an Euclidean inner product. If your representation geometry is locally curved or varies by layer/dataset slice, linear projections can miss structure that shows up under manifold-aware measurements.

**Quick check:** Run a curvature profile. If the reported curvature is near 0 across layers (and stable under resampling), a linear/Euclidean approximation may be reasonable for that dataset. If curvature varies by layer or slice, manifold-aware tools can be a better fit.

```bash
poetry run mc --output text geometry research curvature-profile ./your-model
```

### "Where's the peer review?"

Many parts of this repo are preprints, research notes, and reproducible experiments. Not everything is peer-reviewed yet.

**What we do have:**
- A large test suite (run `poetry run pytest`)
- Reproducible CLI commands for analyses
- A living bibliography: [docs/references/BIBLIOGRAPHY.md](references/BIBLIOGRAPHY.md)

If a doc claim doesn’t match the output you see, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues) with the command you ran and the output.

### "Why geometry instead of benchmarks?"

Benchmarks measure outputs on labeled tasks. Geometry measures representation structure (similarity, curvature, intrinsic dimension, drift). They answer different questions and work best together.

| Approach | What it helps with | What it misses |
|----------|---------------------|---------------|
| Benchmarks (MMLU, etc.) | Task performance tracking | Structural drift/alignment issues that don’t show up in accuracy |
| Geometry (CKA, curvature, etc.) | Structural comparison and change detection | Doesn’t replace task evals or safety review |

**Example:** Two models can score similarly on a benchmark while differing in representation geometry. Geometry tools can surface those differences so you know where to dig deeper.

### "How do I try it on my model?"

Start with small, fast checks:

```bash
poetry run mc --output text model probe /path/to/your/model
poetry run mc --output text geometry research curvature-profile /path/to/your/model
```

Then branch into the sub-area you care about (merge readiness, safety drift, domain transfer, etc.).

If a command crashes, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues) with your model path and the error message.

### "Isn't 'knowledge as geometry' just a metaphor?"

It’s an operational framing: define probe sets, measure representation structure, and report the numbers. If you prefer, call it “representation structure analysis.”

Examples of measurable quantities used across the repo include:
- CKA similarity (see [Kornblith et al., 2019](references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf))
- Topology/geometry analyses of deep nets (see [Naitzat et al., 2020](references/arxiv/Naitzat_2020_Topology_Deep_Neural_Networks.pdf))

We define concepts operationally (probe sets, response directions), then measure similarities/distances and geometric properties of the resulting point clouds.

---

## Technical

### "What backends are supported?"

| Backend | Platform | Install |
|---------|----------|---------|
| MLX | macOS (Apple Silicon) | Default (`poetry install`) |
| CUDA | Linux (NVIDIA GPU) | `poetry install -E cuda` |
| JAX | Linux/TPU/GPU | `poetry install -E jax` |

Set explicitly: `MC_BACKEND=cuda poetry run mc ...` or `MC_BACKEND=jax poetry run mc ...`

### "How long do probes take?"

It depends on model size, backend, and probe corpus size. If you’re unsure, start with a single command (like `model probe` or a small `geometry` subcommand) and scale up.

### "Can I use this with vLLM / Ollama / llama.cpp?"

Currently requires direct weight access (safetensors/PyTorch format). Integration with inference servers is on the roadmap.

### "What's the minimum Python version?"

Python 3.11+

---

## Philosophy

### "Do you claim to solve alignment?"

No. We provide measurement tools. Alignment is a goal; measurement is a prerequisite.

**Analogy:** A thermometer doesn't cure fever. But you can't treat fever without measuring temperature.

### "What can these metrics NOT tell me?"

- Whether a model is "conscious" or "understands" (undefined terms)
- Whether outputs will be harmful (we measure structure, not content)
- Whether the model will generalize to novel domains (we measure current state)

See [AI-ASSISTANT-GUIDE.md](AI-ASSISTANT-GUIDE.md) for detailed limitations.

### "Why AGPL license?"

Knowledge should be free. If you use this as a service, you share your improvements. If you use it internally, no obligations.
