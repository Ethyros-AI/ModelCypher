# ArXiv Style Guide: The Shape of a 2025 ML Paper

> **Status**: Internal Guide for ModelCypher Publications.
> **Last Updated**: December 2025.
> **Source**: Synthesis of NeurIPS 2024 winners, DeepMind GDL papers, and arXiv CS.LG guidelines.

## 1. The "Shape" of a Modern ML Paper

The days of dry, purely academic papers are over. Influential papers in 2025 (e.g., *Mixtral*, *AlphaGeometry*, *Llama 2*) follow a specific narrative arc that balances **Scientific Rigor** with **Engineering Transparency**.

### The "Hourglass" Structure
1.  **Wide Introduction**: Contextualize the problem broadly (Safety, Agency, Geometry).
2.  **Narrow Methodology**: Zoom in to the specific, reproducible mechanism (The Procrustes Anchor, The Entropy Monitor).
3.  **Wide Implication**: Zoom back out to the impact, limitations, and what would falsify the claim.

### Required Sections (The Template)

1.  **Abstract**: 
    -   **If validated**: A key empirical result (“metric drop”) with enough context to interpret.
    -   **If not yet validated**: A clear scope label (prototype/position paper) and an explicit evaluation plan.
    -   **Must-Have**: The "Noun" (Name the architecture/method, e.g., "We introduce *Geometric Anchors*").
    -   **Must-Have**: Claim discipline (what is measured vs hypothesized).
    -   **Style**: Active voice. "We show," not "It is shown."

2.  **Introduction**:
    -   **The Hook**: Start with the "Crisis" (e.g., "LLMs are black boxes," "Safety is fragile").
    -   **The Turn**: Introduce the geometric perspective as the solution.
    -   **Contributions**: A bulleted list of *exactly* what this paper adds. (e.g., "1. A reproducible metric... 2. An open-source toolkit...")

3.  **Related Work (The moat)**:
    -   **Crucial for 2025**: Acknowledge the "Generative AI Boom."
    -   **Positioning**: Show how we are *distinct* from "Vibes-based" safety (RLHF) and aligned with "Mechanistic" safety (feature steering).

4.  **Methodology (The "Recipes")**:
    -   **Transparency**: Must include exact algorithms. No "magic boxes."
    -   **Diagrams**: DeepMind papers always have a high-level "System Diagram" (Figure 1). We need this for the 4-Paper series.
    -   **Formalism**: Use standard mathematical notation (e.g., manifolds $\mathcal{M}$, tangent spaces $T_p\mathcal{M}$).

5.  **Experiments (The "Proof")**:
    -   **Baselines**: Compare against standard methods (e.g., "Random Controls" for anchors, "Standard RLHF" for safety).
    -   **Ablations**: Show *why* it works by breaking it. (e.g., "What happens if we remove the anchors?").
    -   **Reproducibility**: Explicitly link to code, configs, and artifacts needed to rerun.
    -   **Negative results**: If a hypothesis fails, say so (this is often what earns trust).

6.  **Safety & Ethics Statement (New Standard)**:
    -   **Mandatory**: Almost all top-tier papers now include a dedicated section on the broader impact, misuse potential, and mitigation.

7.  **The "Plain Geometry" Rule (No Anthropomorphism)**:
    -   **Banned**: "Thinking," "Reasoning" (as a verb), "Living," "Understanding," "Mind."
    -   **Allowed**: "Computing," "Navigating," "Instantiated in," "Encoding," "Manifold."
    -   **Why**: We must not imply consciousness. The model is a physical system processing a trajectory.

## 2. ArXiv Submission Logistics (2025)

### Category Selection
*   **Primary**: `cs.LG` (Machine Learning).
*   **Secondary**: `cs.AI` (Artificial Intelligence), `cs.CL` (Computation and Language).
*   **Note**: `stat.ML` is often cross-listed automatically.

### The "Review Article" Trap
*   **New Rule (Oct 2025)**: *Review* articles (surveys) now require peer review *before* arXiv submission to `cs.*`.
*   **Action**: Paper 0 ("The Shape of Knowledge") must NOT be framed purely as a survey. It must be a **Position Paper** proposing a *novel framework* (Hypothesis), supported by literature, rather than just summarizing it.

### Formatting
*   **LaTeX**: Mandatory. We should use the official `neurips_2024.sty` or `iclr2025.sty` templates for a professional look.
*   **Citations**: Use BibTeX. The `KnowledgeasHighDimensionalGeometryInLLMs.md` file needs to be converted to `references.bib`.

## 3. Action Plan for ModelCypher Papers

1.  **Paper 0 (Shape)**: Reframe from "Survey" to "Framework Proposal" to avoid the new review-article ban. Focus on the *novelty* of the synthesis.
2.  **Visuals**: We need "Figure 1" for each paper.
    -   *Paper 1*: A diagram of the manifold with "Anchor Points".
    -   *Paper 2*: A "Phase Transition" diagram (Entropy vs Temperature).
    -   *Paper 3*: A "Stitching" diagram (Model A + Adapter -> Model B).
3.  **Benchmarks**: Prefer reproducible numbers over impressive numbers. If numbers are preliminary, label them and provide the harness needed to reproduce or falsify them.
