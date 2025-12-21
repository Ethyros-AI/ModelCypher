# Research Disclaimer

> **Summary**: This software is an experimental research platform for investigating the high-dimensional geometry of Large Language Models. It is provided "as is" under the MIT License.

## 1. Experimental Nature
This codebase implements novel theoretical frameworks ("Linguistic Thermodynamics", "Geometric Adaptation", "Entropy Differential Safety"). While these concepts are grounded in academic literature (see `docs/research/`), they are **experimental**. 
-   **Not Medical/Legal/Financial Advice**: The outputs of models trained or analyzed with this tool should not be relied upon for critical decision-making.
-   **Falsifiability**: The experiments defined in `docs/research/falsification_experiments.md` are designed to *test* these theories, not just demonstrate them. Users are encouraged to report negative results.

## 2. Safety Mechanisms
The `CircuitBreaker` and `EntropyDifferential` modules are designed to prevent model collapse and mitigate harmful outputs during training. However, no safety system is 100% foolproof. 
-   **Human-in-the-Loop**: Always maintain human oversight when deploying models trained with these tools.
-   **Defense-in-Depth**: These geometric safety measures should be used *alongside*, not *instead of*, traditional content filtering and RLHF.

## 3. Acknowledgements
This project bridges the gap between high-level conceptual safety (Law/Policy) and low-level mechanical implementation (Code/Math). The "Logic" and "Architecture" are as critical as the "Arithmetic". Key contributions include:
-   **Legal/Policy Rigor**: Defining *why* we measure entropy.
-   **Geometric Rigor**: Defining *how* we measure entropy (Procrustes, Manifold Topology).
