# Math Primer (for explaining ModelCypher)

ModelCypher uses “high-dimensional geometry” to turn training artifacts (weights, gradients, response trajectories) into *summaries* that humans can reason about.

This is not a full math textbook. It’s a translation layer: the smallest set of ideas you need to explain what the tools measure and why it’s useful.

If you want the academic citations behind the “knowledge as geometry” framing, see [research/KnowledgeasHighDimensionalGeometryInLLMs.md](research/KnowledgeasHighDimensionalGeometryInLLMs.md).

Notes:
- In this repo, run commands as `poetry run mc ...`.
- Global CLI options can appear anywhere on the command line (example: `mc model probe ./model --output text`).

## The core idea: everything is a vector

Most objects we care about can be treated as a long list of numbers:

- a model’s weights → one huge vector
- a gradient update → one huge vector (the “step” training wants to take)
- a layer’s activations → a vector per token
- a response trajectory → a path through some representation space

Once you accept “it’s a vector”, the rest is distance + direction + shape.

## Distance vs direction (what changed vs how it changed)

### Distance (magnitude)

Distance answers: **"How much changed?"**

- Many ModelCypher geometry commands report **geodesic distance** on a k-NN graph (shortest path distance), rather than raw Euclidean distance.
- Euclidean distance is still used for the bootstrap step (building k-NN edges) and can be useful for sanity checks.
- Bigger distance usually means bigger updates or bigger drift.

How to explain to a human:
"Distance is the size of the change, measured along the shape of the data—like road distance between cities, not straight-line distance through the earth."

> **Why geodesic?** Euclidean distance can become less informative in high dimensions (distances concentrate). Graph-geodesic distances track neighborhood structure implied by the point cloud and are often more informative under curved/nonlinear geometry.

### Direction (angle)

Direction answers: **“Is the change of the same kind as some known direction?”**

- The dot product tells you whether two vectors point in similar directions.
- Cosine similarity is the dot product after normalization; it acts like an “angle score”.

How to explain to a human:
“Angle is about *what kind* of change it is, not just how large. Two changes can be big but in unrelated directions.”

## Counter-Intuitive Properties of High Dimensions

In very high dimensions:

- Random vectors are *almost orthogonal* (angles cluster near 90°).
- Distances often *concentrate* (many points look similarly far apart).

Why this matters here:
When a direction stops looking random (e.g., updates repeatedly align with a “refusal direction”), that’s a *stronger signal* than it would be in low dimensions.

## Aligning spaces (when two models use different coordinates)

Two models can represent the “same concept” but with rotated/scaled coordinates. Comparing raw vectors can be misleading unless you align them.

### Procrustes alignment

Procrustes finds the least-squares rotation (and sometimes scaling) to align one set of vectors to another.

How to explain to a human:
“It’s like rotating one map so north lines up before comparing routes.”

### Generalized Procrustes (GPA)

GPA aligns *multiple* spaces to a shared consensus, not just two.

How to explain to a human:
“It’s the group version: find a common coordinate system everyone agrees on.”

## Comparing shapes instead of coordinates

Sometimes you care less about exact alignment and more about whether two spaces have the same *structure*.

### Gromov–Wasserstein distance (GW)

GW compares two point clouds by matching their internal pairwise distances, not their coordinates.

How to explain to a human:
“Instead of matching points by name, it matches by neighborhood structure — like comparing two constellations by the distances between stars.”

## “Flatness” and curvature (stability heuristics)

Training is often described as optimizing a landscape:

- **Flat regions**: small perturbations don’t change loss much → generally more stable.
- **Sharp regions**: small perturbations change loss a lot → can be brittle.

ModelCypher uses proxies (not full Hessians) to estimate whether the current region looks flat or sharp.

How to explain to a human:
“Flatness is a stability hint under the measured setup. Treat it as a diagnostic signal, not a guarantee about generalization or safety.”

## Adapter math (LoRA/DARE/DoRA)

Adapters represent a weight change without editing the full base model.

### DARE sparsity

DARE-style sparsity analysis asks: **“How many adapter deltas are near-zero?”**

How to explain to a human:
“Sparsity tells you whether the adapter is a small, focused change (easy to merge/prune) or a dense rewrite.”

### DoRA decomposition

DoRA decomposes changes into:

- **magnitude** (scaling existing directions)
- **direction** (rotating to new features)

How to explain to a human:
“It distinguishes ‘turning up the volume on existing features’ vs ‘learning new directions’.”

## Paths / trajectories (responses as sequences)

Some tools treat a response as a path through detected “gates” or motifs (coarse computational steps).

How to explain to a human:
“It’s a fingerprint of *how* the model arrived at an answer, not just what it answered.”

## What these concepts are *not*

- They are not a replacement for evaluation suites.
- They do not certify safety.
- They do not remove the need for policy review, red teaming, or human judgment.

Use them to report: “something changed — here’s where, how much, and the confidence of that change.”

## Selected references

- CKA similarity: Kornblith et al. (2019) ([PDF](references/arxiv/Kornblith_2019_CKA_Neural_Similarity.pdf), [arXiv:1905.00414](https://arxiv.org/abs/1905.00414))
- k-NN graph geodesics (Isomap): Tenenbaum et al. (2000) ([DOI:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319))
- Procrustes alignment: Gower (1975) ([DOI:10.1007/BF02291478](https://doi.org/10.1007/BF02291478))
- Optimal transport background (incl. GW): Peyré & Cuturi (2018) ([PDF](references/arxiv/Peyre_2018_Computational_Optimal_Transport.pdf), [arXiv:1803.00567](https://arxiv.org/abs/1803.00567))
