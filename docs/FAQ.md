# Frequently Asked Questions

## Skepticism

### "This is just PCA with marketing."

No. PCA assumes Euclidean geometry. LLM activations live on curved manifolds.

**The math:** PCA finds linear projections that maximize variance. But in high-dimensional curved space, the shortest path between two points isn't a straight line—it's a geodesic. Euclidean distance *underestimates* true distance in positively curved regions and *overestimates* in negatively curved regions.

**The proof:** Run `mc geometry manifold analyze` on any model. If sectional curvature ≈ 0 everywhere, it's flat (PCA works). If curvature varies by layer (it does), you need Riemannian tools.

```bash
mc geometry manifold analyze ./your-model --output text
# Typical output: Layer 12 curvature = 0.23, Layer 18 = -0.15
# Non-zero curvature = curved manifold = PCA insufficient
```

### "Where's the peer review?"

Honest answer: these are preprints. Publication is pending data insertion.

**What we have instead:**
- 2400+ passing tests with deterministic seeds
- Falsification-first experimental design (hypotheses stated before results)
- Reproducible CLI commands for every claim
- 46 cited papers from NeurIPS, ICML, ICLR, Nature

**The code is the proof.** Run the commands. Check the numbers. If they don't match, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues).

### "Why geometry instead of benchmarks?"

Benchmarks measure *what the model says*. Geometry measures *how knowledge is organized*.

| Approach | Measures | Can Be Gamed? |
|----------|----------|---------------|
| Benchmarks (MMLU, etc.) | Output accuracy | Yes (train on test set) |
| Geometry (CKA, curvature) | Internal structure | No (topology is invariant) |

**Example:** Two models can score identically on MMLU but have completely different internal organization. One might be robust to adversarial prompts; the other might collapse. Geometry tells you which is which.

You can't fake topology. Betti numbers don't lie.

### "Prove it works on my model."

Run this:

```bash
mc geometry spatial probe-model /path/to/your/model --output text
```

**Expected output:**
```
World Model Score: 0.30-0.60 (typical for 1-7B models)
Axis Orthogonality: 85-95%
Physics Engine: DETECTED or NOT DETECTED
```

If you get wildly different numbers, that's real data about your model's geometry. If the command crashes, [file an issue](https://github.com/Ethyros-AI/ModelCypher/issues) with your model path and error message.

### "Isn't 'knowledge as geometry' just a metaphor?"

No. It's a measurable hypothesis.

**Metaphor:** "The model understands math."
**Measurement:** "CKA similarity between 'addition' and 'subtraction' probes = 0.73."

We define concepts operationally: a "concept" is a direction in activation space that responds consistently to a probe set. We measure distances between these directions. We compute curvature of the manifold they span.

If you prefer, call it "representation structure analysis." The math is the same.

---

## Technical

### "What backends are supported?"

| Backend | Platform | Install |
|---------|----------|---------|
| MLX | macOS (Apple Silicon) | Default, no extra install |
| JAX | Linux, TPU, CUDA | `poetry install -E jax` |

Set explicitly: `MC_BACKEND=jax mc geometry ...`

### "How long do probes take?"

| Model Size | Geometry Probe | Full Analysis |
|------------|----------------|---------------|
| 0.5B | ~30 seconds | ~2 minutes |
| 3B | ~2 minutes | ~10 minutes |
| 7B | ~5 minutes | ~30 minutes |

Times on M2 Max. Your mileage may vary.

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
