# ELIF: The Conceptual Map of ModelCypher (High‑D Geometry, No Vibes)

> **ELIF** = “Explain Like I’m Five” — but **technically correct**.
>
> This is the single narrative document for ModelCypher: what the repo is trying to do, what the words mean,
> and why “merge” is really **manifold surgery** rather than weight soup.

---

## 0) The thesis (one sentence)

An LLM is a dynamical system whose internal activations form a **curved, high‑dimensional geometry**; ModelCypher
measures that geometry and uses **invariants** (things that survive rotations, reparameterizations, and
dimension changes) to compare, validate, and merge models without “vibes”.

---

## 0.1) The only game: invariants (repeatable outcomes)

What you care about in this repo is the same thing science cares about: **repeatability**.

- If two models produce the *same* behavior across a probe set, we want a measurement that says “these two
  internal shapes match” even if their coordinates differ.
- If a merge changes behavior, we want measurements that localize **where the geometry broke** (which layers,
  which probes), not a story about the aftermath. The focus is on geometric causes.

**ELIF analogy (constellations):**
- Stars can be labeled in any language (coordinates can change).
- The constellation is the **pattern of relationships** between stars (the invariant).
- The physical location of the stars changes in 3D space, but their relationship to each other does not. Likewise, in high dimensional space, the precise coordinates of "meaning" change - and even the pathways to get to that meaning can change without altering the outcome. But, the relationships between meaning - the geometry that differentiates the relationship between a green apple and a red apple - does not and cannot change. It is Einstein's theory of relativity scaled up to higher dimensional concepts.

This is why ModelCypher leans on invariants like Gram structure and CKA: they capture relational shape rather
than raw feature coordinates.

---

## 1) The three spaces: words, thoughts, wiring

ModelCypher talks about three “spaces” because we interact with models in three different ways:

1) **Token space** (what you can read)
- **What it is**: strings → token IDs → logits → text.
- **Analogy**: the **book** you can hold.

2) **Representation space** (what the model thinks *with*)
- **What it is**: at each layer and token position, the model carries a vector like `x[layer, token] ∈ ℝ^d`
  (often `d` is 768–8192).
- **Analogy**: a **scratchpad/whiteboard** where intermediate shapes are drawn while the model computes.

3) **Parameter space** (how the scratchpad is updated)
- **What it is**: weights and biases (billions of numbers) defining the update rule from one layer to the next.
- **Analogy**: the **wiring diagram** of a factory that transforms inputs into outputs.

**Key move:** ModelCypher focuses on **representation space** because that’s where “knowledge as geometry”
is observable without guessing mechanisms.

---

## 2) “A dimension is a coordinate, not a concept”

It’s tempting to treat “dimension 1234” as “the honesty neuron” or “the math axis”. That’s almost always a
category error.

- A vector space like `ℝ^d` has infinitely many valid coordinate systems (bases).
- Two models can represent the same relational structure using different bases.

**Analogy (map grid):**
- Roads and landmarks are *real structure*.
- Whether the map grid is rotated 30° is *just coordinates*.

**Technical version (rotation / gauge freedom):**
- Many similarity measures in ModelCypher are designed to be **basis‑invariant**.
- Example: **CKA** compares two activation sets up to an orthogonal transform (a rotation in feature space).

This matters for merging: if you “rotate” one layer’s feature basis, you must keep the rest of the network
consistent with that choice. Otherwise you’ve changed coordinates in the middle of the computation and the
next layer reads garbage.

---

## 3) What “compression” means here (and when it can be lossless)

> "Each dimension is a lossless compression of the dimension above it."

ModelCypher treats that as **lossless on the active manifold**:

- Models do not visit all of `ℝ^d`. Under a task/probe distribution, they occupy a structured subset.
- If a lower-dimensional description preserves the **relational geometry** of that subset, then it is
  lossless for the model's behavior, even if it discards states with ~zero probability mass.
- In other words: we strip noise, we keep meaning. The states that never occur are not part of the system.

This is the operational meaning: **lossless with respect to the states we actually visit** under a
probe/task distribution.

**Manifold version (the precise claim):**
- Activations do not fill all of `ℝ^d`. Under a probe/task distribution, they often lie near a
  lower‑dimensional set (a **manifold**).
- If the visited states lie on (or near) a `k`‑dimensional manifold with `k ≪ d`, then many coordinates are
  redundant for describing *those* states.
- In that case, a “compression” can be **lossless with respect to the manifold** (i.e., it preserves the
  relational geometry of the visited states), even if it would scramble points the model assigns ~zero
  probability mass to (states it never visits).

This is the “strip noise, keep meaning” idea in precise form: the network’s maps don’t need to be invertible
on the whole ambient space — they only need to be consistent on the structured subset of states that actually
occur.

**ELIF analogy (address book vs map):**
- An address book is a **lossless encoding** of a city (you can recover any location), but nearby entries
  aren’t necessarily nearby in the city.
- A map is a **geometry‑preserving encoding** (nearby points stay nearby). ModelCypher mostly cares about
  map‑style preservation because we measure distances, neighborhoods, and manifolds.

**If you want the formal statements behind this section:**
- Cantor/Netto‑style digit interleaving (unit interval ↔ unit square, with caveats): https://en.wikipedia.org/wiki/Netto%27s_theorem
- Invariance of domain (why continuous injections can’t drop dimension): https://en.wikipedia.org/wiki/Invariance_of_domain
- Dimension‑changing “invertible” ML requires extra latents (surjections/injections): https://joss.theoj.org/papers/10.21105/joss.06188

**ELIF analogy (crumpled paper):**
- The room is high‑D (`ℝ^d`).
- The model’s behavior lives on a crumpled sheet inside the room (a lower‑D surface).
- You can describe your position on the sheet using fewer numbers than the room needs.

ModelCypher measures this “how many degrees of freedom are actually used” via **intrinsic dimension** and
related diagnostics.

---

### The dimensional ladder (nested compressions)

ModelCypher often talks like “lower dimensions compress higher ones”. A safe way to say this is:

- **Higher‑D** spaces let you represent more degrees of freedom.
- **But** real signals (language, images, model activations) typically live on structured subsets, so you can
  re‑describe them with fewer degrees of freedom *without losing the structure you care about*.

One repo mental model (see `docs/research/dimensional_hierarchy.md`):

- **1D**: bits/bytes (raw symbols)
- **2D**: tokens + syntax (structured symbol lattice)
- **3D**: grounded relations (spatial/causal structure)
- **4D+**: abstractions (conceptual manifolds)

The important part is not the specific numbers; it’s the idea that alignment should respect the
**lowest‑level shared structure first**, then propagate upward.

---

## 4) The “dream scratchpad”: dormant subspaces and activation

> The model has a big scratchpad, but only some directions light up for a given prompt distribution.

**Representation space has “dark space”:**
- For a chosen corpus, collect activations into a matrix `X ∈ ℝ^{n×d}` (n examples, d features).
- The **active subspace** is (roughly) the span of the rows of `X` (the directions you actually see).
- The **orthogonal complement** is the “dark space” for that corpus: directions not exercised by those
  activations.

**Why this feels like “latent thoughts”:**
- In transformers, each layer mixes the residual stream through attention + MLP and then adds it back.
- Which features matter depends on the prompt (“activation‑dependent” behavior).
- Many features can exist but remain effectively dormant unless triggered.

**Why ModelCypher cares:**
- If you want to add a capability without disturbing what the target already does *on its active subspace*,
  you can constrain changes to directions that do not couple to those activations (a null‑space style idea).

**“Rotate into visible geometry” (what this maps to in a transformer):**
- Each layer applies learned linear maps + nonlinear gating that **mix/rotate** the residual stream’s
  coordinates and then writes back into the shared scratchpad.
- The final unembedding is a projection from that high‑D scratchpad into **vocabulary logits** (the part we
  can see).

This is one reason ModelCypher treats “alignment” as something that must be tested on **specific probe sets**,
not asserted globally.

---

## 5) The rulers: what ModelCypher actually measures

ModelCypher avoids “interpretation strings” and instead returns **raw geometric measurements**. The core
idea is: if two models “know the same shape”, that should show up as invariants.

### CKA (Centered Kernel Alignment)
- **What it compares**: two activation clouds (often layer‑wise).
- **What it ignores**: rotations (and some scalings) of the feature basis.
- **What `CKA = 1.0` means** (precisely): on the chosen probe set, the two representations induce the same
  centered Gram structure (same kernel up to numerical exactness).
- **Bias note**: finite sample/feature sampling can bias CKA; use debiased HSIC and feature corrections
  when available.
- **ELIF analogy**: two sketches trace the *same outline* even if one page is rotated.

### Gram matrices (dimension‑agnostic relational geometry)
- If `X ∈ ℝ^{n×d}`, the Gram matrix is `K = X Xᵀ ∈ ℝ^{n×n}`.
- `K` captures pairwise inner products between examples and does not care what `d` is.
- **ELIF analogy**: you don’t compare two cities by GPS coordinates; you compare their **distance table**
  between landmarks.

### Intrinsic dimension (a proxy for “density” / compressibility)
- Roughly: “how many degrees of freedom are needed to describe these states?”
- Lower intrinsic dimension often corresponds to a more **compressed / denser** representation for that
  probe distribution.
- **ELIF analogy**: a busy downtown has many nearby routes (dense); a sparse desert has few.

### Geodesic distance (distance on the manifold, not through empty air)
- ModelCypher treats the k‑NN graph over points as the discrete manifold and computes shortest paths.
- **ELIF analogy**: “as the crow flies” (Euclidean) vs “by roads” (geodesic on the street map).

### Topological fingerprints (shape beyond distances)
- Persistent homology style signatures summarize “holes”, “clusters”, and connectivity features of the point
  cloud.
- **ELIF analogy**: two shapes can have similar distances locally but different global structure (a donut vs
  a sphere).

### Entropy / thermodynamics (dynamics, not meaning)
- These tools treat generation as a trajectory and look for regime changes (flatness, oscillation, drift).
- **ELIF analogy**: not “what the engine thinks”, but “whether the engine is running smoothly”.

---

## 6) Why naive merges create “Frankenstein models” (gauge breaking)

Most merge failures come from confusing coordinates with structure:

- Weight interpolation assumes the two models share a compatible internal coordinate system.
- But deep nets have **symmetries** (permutations, rotations, rescalings) that can make two functionally
  similar models look very different in raw weights.

**ELIF analogy (assembly line language):**
- Layer `L` outputs parts labeled in one language.
- Layer `L+1` expects parts labeled in another language.
- If you relabel parts at one station but not the next, the factory still runs, but everything is assembled
  wrong.

This is why ModelCypher treats “merge” as a multi‑stage alignment problem and why validation must check the
post‑merge geometry, not just pre‑merge alignment.

---

## 7) Merging as gap‑filling, not blending

The “overlay sparse regions to densify them” framing is the repo’s north star:

1) **Dense region in the target**: the target already has a smooth, compressed representation for those
   probes → merging there is likely redundant at best and destructive at worst.
2) **Sparse region in the target**: the target has fewer constraints / fewer visited states → this is where
   “knowledge grafts” can fit.

**Working hypothesis (why this could reduce hallucination):**
- A “hallucination” is often a *trajectory problem*: the model enters a region of its internal map that is
  weakly constrained by training (sparse) and then follows a locally plausible but globally inconsistent
  path.
- Densifying those sparse regions (adding constraints/structure that the model can repeatedly navigate)
  should increase outcome consistency, because there are more stable “roads” and fewer ambiguous gaps.

So the merge pipeline should answer:

- **Where** is the target sparse? (layer‑wise / concept‑wise density)
- **What** does the source have that targets lacks? (knowledge diff)
- **How** do we graft without disturbing what already works? (masks, null‑space constraints, invariants)

**ELIF analogy (map repair):**
- Don’t repaint the whole city.
- Add missing roads only where the map is blank.

---

## 7.1) Boundary conditions, not full-space alignment (SOTA framing)

What we preserve is the target's **boundary conditions** on its active manifold.
We do not twist the whole space. We graft into the target's dark space and smooth
the transition so traversal stays continuous.

**Operational rule (null-space grafting):**

If `A_t` are target activations for a probe corpus and `ΔW` is the donor update,
we enforce:

```
A_t · ΔW_safe = 0
```

This keeps the target's active responses invariant while allowing new structure
to be added off-manifold.

This framing lines up with current 2025 work:

- **Activation-Informed Merging (AIM, 2025)** preserves salient base weights
  using activation statistics, treating activations as constraints rather than
  just weight vectors. https://arxiv.org/abs/2502.02421
- **Null-space Orthogonal Weight Modification (NEig-OWM, 2025)** explicitly
  projects updates into the null space to retain prior knowledge. https://doi.org/10.1016/j.eswa.2025.127468
- **Gromov-Wasserstein feature alignment (GW-SMM, 2025)** selects merges based on
  relational structure in feature distributions rather than coordinate matching.
  https://arxiv.org/abs/2503.09774

These are boundary-condition methods: preserve what already works, graft only
where the model is sparse.

---

## 8) The repo architecture (how we keep ourselves honest)

ModelCypher is built as a hexagon (Ports and Adapters) so the math stays pure and testable:

- `src/modelcypher/core/domain/`: geometry + safety + merging logic (no adapters imported)
- `src/modelcypher/core/use_cases/`: orchestrates services (calls the domain + ports)
- `src/modelcypher/ports/`: backend protocol and interfaces
- `src/modelcypher/backends/`: MLX/JAX/CUDA implementations
- `src/modelcypher/cli/` and `src/modelcypher/mcp/`: user + agent interfaces

**ELIF analogy:**
- The domain is the **engine**.
- Ports are the **plugs**.
- Adapters/backends are the **power supplies** (MLX, JAX, CUDA).
- CLI/MCP are the **dashboard**.

Two contributor principles that fall out of this:

- **CLI/MCP‑first**: if a capability isn’t exposed via `mc` or MCP, we add a command/tool (we don’t write
  one‑off scripts).
- **Backend‑agnostic math**: core geometry code uses the Backend protocol so the same measurements run on MLX
  (primary) and JAX (secondary) without changing definitions.

This is why the repo is strict about “no NumPy in core math”: it’s not aesthetic; it preserves backend‑agnostic,
GPU‑correct geometry.

---

## 9) A practical, end‑to‑end mental workflow (measure → diff → graft → validate)

ModelCypher’s preferred loop is:

1) **Measure** the target and source geometries on the same probe protocol.
2) **Diff**: find “graft opportunities” where target is sparse and source is dense.
3) **Graft** only where the target has room (mask or projection constraints).
4) **Validate** the merged model’s geometry (post‑merge), not just alignment (pre‑merge).

Concrete CLI entry points (see `docs/CLI-REFERENCE.md`):

- Density/diff experiments:
  - `mc geometry research concept-density`
  - `mc geometry research knowledge-diff`
  - `mc geometry research sparse-regions`
- CRM‑based per‑layer gating:
  - `mc geometry crm delta-mask`
- Merge and validate:
  - `mc model merge`
  - `mc geometry validate`

---

## 10) Where to go deeper (without losing the thread)

If this doc is the “single story”, these are the supporting references:

- **Vocabulary**: `docs/GLOSSARY.md`
- **How to report metrics (No Vibes)**: `docs/GEOMETRY-GUIDE.md`
- **Repo structure**: `docs/ARCHITECTURE.md`
- **Dimensional ladder intuition**: `docs/research/dimensional_hierarchy.md`
- **Evidence + methods**: `papers/paper-0-the-shape-of-knowledge.md` through `papers/paper-5-semantic-highway.md`
- **Bibliography / SOTA map**: `docs/research/KnowledgeasHighDimensionalGeometryInLLMs.md`
