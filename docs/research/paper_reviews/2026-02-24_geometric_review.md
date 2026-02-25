# ModelCypher Paper Review — 2026-02-24

**Generated**: 2026-02-25 19:58
**Papers scanned**: 25
**Papers with geometric relevance**: 2

---

## 1. ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation

**arXiv**: [2602.20093](https://arxiv.org/abs/2602.20093)
**Authors**: Kun Yang, Yuxuan Zhu, Yazhe Chen, Siyao Zheng, Bangyang Hong et al. (10 total)
**HF Upvotes**: 23
**Geometric Relevance Score**: 6.0
**Code**: [https://github.com/FuCongResearchSquad/ManCAR](https://github.com/FuCongResearchSquad/ManCAR)

### Summary
Sequential recommendation increasingly employs latent multi-step reasoning to enhance test-time computation. Despite empirical gains, existing approaches largely drive intermediate reasoning states via target-dominant objectives without imposing explicit feasibility constraints. This results in latent drift, where reasoning trajectories deviate into implausible regions. We argue that effective recommendation reasoning should instead be viewed as navigation on a collaborative manifold rather than free-form latent refinement. To this end, we propose ManCAR (Manifold-Constrained Adaptive Reasoning), a principled framework that grounds reasoning within the topology of a global interaction graph. ManCAR constructs a local intent prior from the collaborative neighborhood of a user's recent actions, represented as a distribution over the item simplex. During training, the model progressively aligns its latent predictive distribution with this prior, forcing the reasoning trajectory to remain within the valid manifold. At test time, reasoning proceeds adaptively until the predictive distribution stabilizes, avoiding over-refinement. We provide a variational interpretation of ManCAR to theoretically validate its drift-prevention and adaptive test-time stopping mechanisms. Experiments on seven benchmarks demonstrate that ManCAR consistently outperforms state-of-the-art baselines, achieving up to a 46.88% relative improvement w.r.t. NDCG@10. Our code is available at https://github.com/FuCongResearchSquad/ManCAR.

### Keyword Matches
- **Core geometric** (2): \bmanifold\b, \btopolog

### ModelCypher Integration Notes
<!-- FILL: Claude deep-dive pass populates this section -->
_Pending deep analysis — run with --deep flag or review manually._

---

## 2. tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

**arXiv**: [2602.20160](https://arxiv.org/abs/2602.20160)
**Authors**: Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu et al. (9 total)
**HF Upvotes**: 4
**Geometric Relevance Score**: 3.0
**Code**: [https://github.com/cwchenwang/tttLRM](https://github.com/cwchenwang/tttLRM)

### Summary
We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

### Keyword Matches
- **Mechanistic/structural** (2): \brepresentation\b.*\b(geometry|structure|space|manifold|alignment), \brepresentation\b.*\b(similar|compari|align|converg)

### ModelCypher Integration Notes
<!-- FILL: Claude deep-dive pass populates this section -->
_Pending deep analysis — run with --deep flag or review manually._

---

## Quick Reference

| # | Score | Paper | arXiv | Code |
|---|-------|-------|-------|------|
| 1 | 6.0 | ManCAR: Manifold-Constrained Latent Reasoning with Adaptive ... | [2602.20093](https://arxiv.org/abs/2602.20093) | [repo](https://github.com/FuCongResearchSquad/ManCAR) |
| 2 | 3.0 | tttLRM: Test-Time Training for Long Context and Autoregressi... | [2602.20160](https://arxiv.org/abs/2602.20160) | [repo](https://github.com/cwchenwang/tttLRM) |
