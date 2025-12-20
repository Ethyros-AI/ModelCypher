# Math Primer (for explaining ModelCypher)

ModelCypher uses “high-dimensional geometry” to turn training artifacts (weights, gradients, response trajectories) into *summaries* that humans can reason about.

This is not a full math textbook. It’s a translation layer: the smallest set of ideas you need to explain what the tools measure and why it’s useful.

If you want the academic citations behind the “knowledge as geometry” framing, see `KnowledgeasHighDimensionalGeometryInLLMs.md`.

## The core idea: everything is a vector

Most objects we care about can be treated as a long list of numbers:

- a model’s weights → one huge vector
- a gradient update → one huge vector (the “step” training wants to take)
- a layer’s activations → a vector per token
- a response trajectory → a path through some representation space

Once you accept “it’s a vector”, the rest is distance + direction + shape.

## Distance vs direction (what changed vs how it changed)

### Distance (magnitude)

Distance answers: **“How much changed?”**

- In practice, this is often an L2 norm (Euclidean distance) or a normalized variant.
- Bigger distance usually means bigger updates or bigger drift.

How to explain to a human:
“Distance is the size of the change. It’s like ‘how far the weights moved’.”

### Direction (angle)

Direction answers: **“Is the change of the same kind as some known direction?”**

- The dot product tells you whether two vectors point in similar directions.
- Cosine similarity is the dot product after normalization; it acts like an “angle score”.

How to explain to a human:
“Angle is about *what kind* of change it is, not just how large. Two changes can be big but in unrelated directions.”

## Why high-dimensional spaces feel weird (and why that helps)

In very high dimensions:

- Random vectors are *almost orthogonal* (angles cluster near 90°).
- Distances often *concentrate* (many points look similarly far apart).

Why this matters here:
When a direction stops looking random (e.g., updates repeatedly align with a “refusal direction”), that’s a *stronger signal* than it would be in low dimensions.

## Aligning spaces (when two models use different coordinates)

Two models can represent the “same concept” but with rotated/scaled coordinates. Comparing raw vectors can be misleading unless you align them.

### Procrustes alignment

Procrustes finds the best rotation (and sometimes scaling) to align one set of vectors to another.

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
“Flatness is a stability hint. Sharpness can mean the model is sensitive and might generalize worse or become unstable.”

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

They are best used as: “something changed — here’s where, how, and how worried we should be.”
