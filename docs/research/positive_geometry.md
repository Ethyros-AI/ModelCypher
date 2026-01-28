# Positive Geometry (Amplituhedron Probe)

This note documents how we probe for **positive-geometry signatures** inside
LLM representation spaces. It is a measurement-only scaffold, not a claim.

## Conceptual placement in ModelCypher

- **Invariant manifold geometry** is measured on activations before sampling.
  This aligns with the pre-collapse regime in `docs/research/dimensional_hierarchy.md`.
- **Collapse (0D → 1D)** is treated as **sampling**, not softmax, and is downstream
  of the manifold geometry. The amplituhedron probe is therefore **pre-collapse**.
- **Fractional intrinsic dimension** and expansion/compression dynamics are already
  tracked in `docs/MANIFOLD-LEARNING-SYNTHESIS.md`. The positive-geometry signature
  can be compared against those measurements to test correlations.

## What we measure

We treat the column space of probe activations (ordered by atlas probe order)
 as a point on the Grassmannian. We then compute **ordered minors** of the
orthonormal basis matrix (Plücker coordinates) and report:

- fraction of **positive**, **negative**, and **near-zero** minors
- **sign entropy** over these three buckets
- raw summary stats for minors (min/max/mean, mean |minor|)
- Plücker norm and max absolute minor

No thresholds, no interpretation strings.

## CLI

```bash
poetry run mc geometry research positive-geometry /path/to/model \
  --layer 0 \
  --probe-count 128 \
  --max-minors 256
```

The probe order is the atlas order. Positivity here is *ordered* positivity; if
we change the ordering, we change the measurement. That is expected and is part
of the test.

## How to use the measurement

- Compare **sign entropy** and **positive fraction** across layers.
- Compare these values against **intrinsic dimension** and **entropy trajectories**
  for the same layers to test whether positive-geometry signatures correlate with
  expansion/compression regimes.

This is a hypothesis probe. The output is raw geometry only.
