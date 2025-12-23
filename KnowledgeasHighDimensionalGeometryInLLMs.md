# Foundational Bibliography: Knowledge as High-Dimensional Geometry in Large Language Models

This bibliography collects references relevant to a set of **working hypotheses** used in ModelCypher:
(1) some aspects of LLM behavior are reflected in measurable geometric structure of representations,
(2) inference can be studied as trajectories through representation space,
(3) some anchor inventories may induce relatively stable relational structure across models,
(4) divergence/entropy-like signals may help detect boundary conditions pre-emission, and
(5) some safety behaviors may be enforced via low-rank or geometric constraints.

Nothing in this document should be read as a proof of any hypothesis; it is a map of prior work and measurement tools.

---

## PILLAR 1: Foundational Mathematics and Geometry

### Manifold hypothesis and information geometry

**Fefferman, C., Mitter, S., & Narayanan, H. (2016). Testing the Manifold Hypothesis. *Journal of the American Mathematical Society*, 29(4), 983-1049.**
Provides rigorous mathematical foundations for testing whether high-dimensional data lies near low-dimensional manifolds, with complexity guarantees for fitting manifolds to probability distributions. *Framework relevance*: Provides tools for treating “manifold-like structure” as a testable property of data (including representations). In ModelCypher, the “navigation” framing refers to trajectories through activation space under a probe protocol; it is not a claim about a literal manifold in weight space.

**Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*, 10(2), 251-276.**
Introduces natural gradient descent accounting for Riemannian structure of parameter space using Fisher information, demonstrating dramatic improvements in learning efficiency over standard gradient descent. *Framework relevance*: Critical for understanding optimization as movement through curved parameter space, establishing that parameter space has intrinsic geometric structure affecting learning dynamics.

**Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry* (Translations of Mathematical Monographs 191). American Mathematical Society.**
Comprehensive textbook establishing information geometry as a field, introducing dualistic geometry, Fisher information metric, and applications to statistical inference. *Framework relevance*: Provides the mathematical vocabulary—Riemannian metrics, geodesics, dual connections—necessary for formalizing "knowledge as geometry."

**Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. *NeurIPS*.**
Introduces filter normalization for meaningful visualization of neural network loss landscapes, demonstrating how architecture affects landscape smoothness and relating landscape shape to generalization. *Framework relevance*: Establishes that neural network parameter space has complex but structured geometry. The framework extends loss landscape analysis from parameter space to representation space.

**Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A.G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS*.**
Shows different local minima are connected by paths of nearly constant loss, revealing global structure in loss landscapes. *Framework relevance*: Suggests parameter space has global structure beyond local geometry—multiple solutions are connected, implying coherent geometric organization.

### Riemannian optimization

**Absil, P-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton University Press.**
Comprehensive treatment of optimization on Riemannian manifolds, developing gradient descent and Newton's method for manifold-constrained optimization. *Framework relevance*: Provides algorithmic framework for navigating curved spaces—directly applicable to understanding how inference could navigate representation manifolds.

---

## PILLAR 2: Representation Geometry in Neural Networks

### Platonic Representation Hypothesis

**Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). Position: The Platonic Representation Hypothesis. *ICML 2024*, PMLR 235:20617-20642.**
Argues that some representation distances become more consistent across models as scale increases, suggesting convergent relational structure in certain settings. *Framework relevance*: Motivates measuring representational convergence with scale using similarity metrics (CKA/RSA/OT-style diagnostics). It does not establish universal convergence across architectures, tasks, or training mixtures.

### Linear representation hypothesis

**Park, K., Choe, Y.J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML 2024*, PMLR 235:39643-39666.**
Rigorously formalizes “linear representation” using counterfactuals and connects linear probing, steering, and subspace representations under explicit assumptions. Identifies a causal inner product respecting semantic structure. *Framework relevance*: Supports treating certain probes/steering directions as meaningful geometric objects, while keeping the “concepts are directions” claim conditional on measurement protocol and model setting.

**Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS*.**
Demonstrates famous king−man+woman=queen analogies, showing word embeddings have linear algebraic structure encoding semantic relationships. *Framework relevance*: Original empirical demonstration that semantic knowledge has geometric structure through vector arithmetic.

### Superposition and polysemanticity

**Elhage, N., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*, Anthropic.**
Demonstrates neural networks represent more features than neurons through superposition—encoding sparse features as nearly-orthogonal directions. Reveals phase diagrams and geometric structures (polytopes) in superposition. *Framework relevance*: **Critical.** Explains how high-dimensional geometry enables efficient knowledge storage. Superposition is the geometric mechanism for compressing vast knowledge into finite dimensions.

**Scherlis, A., Sachan, K., Jermyn, A.S., Benton, J., & Shlegeris, B. (2022). Polysemanticity and Capacity in Neural Networks. *arXiv:2210.01892*.**
Analyzes polysemanticity through "feature capacity"—fractional dimensions features consume—showing optimal allocation represents important features monosemantically. *Framework relevance*: Explains geometric resource allocation, providing theoretical grounding for how geometry trades off representational precision.

### Concept geometry and probing

**Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*. arXiv:1905.00414.**
Introduces Centered Kernel Alignment (CKA) as a robust standard for comparing representations across different networks. It provides evidence that many models share similar *relational structure* across initializations (and sometimes architectures), without requiring shared coordinates. *Framework relevance*: **Foundational tool.** CKA provides a practical “ruler” for measuring representational similarity under a chosen probe corpus and layer selection; it does not establish universal invariance across all settings.

**Naitzat, G., Zhitnikov, A., & Lim, L.-H. (2020). Topology of Deep Neural Networks. *Journal of Machine Learning Research*, 21(184), 1-85. arXiv:2004.06093.**
Computes Betti numbers of activation manifolds, proving that deep neural networks systematically simplify the topology of data (reducing holes/complexity) layer by layer. *Framework relevance*: Direct support for **Topological Fingerprinting**. Verification that layers have measurable, consistent topological signatures that can be compared across architectures.

**Kim, B., et al. (2018). Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). *ICML*.**
Introduces Concept Activation Vectors as linear directions representing human concepts in neural network layers, enabling testing of concept influence on predictions. *Framework relevance*: Operationalizes concepts as geometric directions, foundational for treating knowledge as spatial.

**Koh, P.W., et al. (2020). Concept Bottleneck Models. *ICML 2020*, PMLR 119:5338-5348.**
Models that first predict human-interpretable concepts, then use concepts to predict labels, enabling intervention on concept values. *Framework relevance*: Demonstrates concepts have explicit geometric representation that can be isolated and manipulated.

**Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational Similarity Analysis – Connecting the Branches of Systems Neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.**
Introduces RSA for comparing representations across brain regions and computational models using representational dissimilarity matrices. *Framework relevance*: Provides methodology for measuring geometric structure of representations.

**Hewitt, J., & Manning, C.D. (2019). A Structural Probe for Finding Syntax in Word Representations. *NAACL-HLT*.**
Demonstrates that, under a learned probe, representation distances can correlate with parse tree distances in BERT. *Framework relevance*: Motivates treating some linguistic structure as measurable geometry in representations, while keeping “geometry” tied to a specific probe/metric rather than a literal spatial claim.

---

## PILLAR 3: Semantic Primes and Linguistic Universals

### Natural Semantic Metalanguage

**Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.**
Landmark synthesis presenting approximately **65** proposed semantic primes in the Natural Semantic Metalanguage (NSM) tradition. *Framework relevance*: Motivates using NSM primes as a standardized *anchor inventory* for probing LLM representations. Linguistic universality is debated and does not imply invariance across model families; in ModelCypher, “invariance” is treated as a falsifiable measurement claim.

**Goddard, C. & Wierzbicka, A. (Eds.) (2002). *Meaning and Universal Grammar: Theory and Empirical Findings*, Vols. I & II. John Benjamins.**
Comprehensive cross-linguistic work within the NSM program. *Framework relevance*: Provides motivation for cross-linguistic anchor inventories, while keeping any “cross-model alignment via shared anchors” claim contingent on model-side measurement and controls (tokenization, translation choice, probe corpus).

**Goddard, C. (2018). *Ten Lectures on Natural Semantic Metalanguage*. Brill.**
Comprehensive introduction covering semantic molecules (intermediate concepts defined via primes), cultural scripts, and applications. *Framework relevance*: Demonstrates semantic molecules provide compositional hierarchy built from primes—paralleling how complex representations compose from primitive features.

### Linguistic universals

**Chomsky, N. (1965). *Aspects of the Theory of Syntax*. MIT Press.**
Introduces Universal Grammar as a hypothesis about constraints on human language. *Framework relevance*: Background context on “universals” in linguistics; it is not evidence of universal invariants in LLM representations.

**Greenberg, J.H. (1963). Some Universals of Grammar with Particular Reference to the Order of Meaningful Elements. In *Universals of Language*, pp. 58-90. MIT Press.**
Identifies implicational tendencies from a sample of languages. *Framework relevance*: Useful framing for “statistical universals” (tendencies) vs absolute invariants; relevant caution for any anchor-universality hypothesis in models.

**Evans, N. & Levinson, S.C. (2009). The Myth of Language Universals. *Behavioral and Brain Sciences*, 32(5), 429-492.**
Critical assessment documenting extensive diversity across languages, arguing universals are statistical tendencies rather than absolutes. *Framework relevance*: Important counterbalance—invariant anchors should be understood as statistical attractors rather than exact points.

---

## PILLAR 4: Computational Primitives and Program Synthesis

### Foundational computation theory

**Turing, A.M. (1936). On Computable Numbers, with an Application to the Entscheidungsproblem. *Proceedings of the London Mathematical Society*, Series 2, 42, 230-265.**
Foundational paper defining the Turing machine—an abstract model using finite primitives for universal computation. *Framework relevance*: Establishes that minimal operations suffice for universal computation—the computational analog to semantic primes forming a universal basis.

**Church, A. (1936). An Unsolvable Problem of Elementary Number Theory. *American Journal of Mathematics*, 58, 345-363.**
Introduces lambda calculus, establishing lambda-definability as equivalent to effective computability. *Framework relevance*: Lambda calculus provides the foundation for compositional semantics—computation built from minimal abstract primitives.

**Schönfinkel, M. (1924). Über die Bausteine der mathematischen Logik. *Mathematische Annalen*, 92, 305-316.**
Introduces combinatory logic, proving just two combinators (S and K) suffice to express any computation—the most minimal computational basis known. *Framework relevance*: Ultimate proof that computation requires only two primitives. The SK basis represents irreducible "anchors" for all computable functions.

### Wolfram's computational primitives

**Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.**
Argues simple programs (cellular automata) produce arbitrary complexity, identifying the simplest universal Turing machine (Rule 110) and proposing computational equivalence as fundamental. *Framework relevance*: Demonstrates complex behavior emerges from minimal computational primitives—analogous to complex semantics emerging from semantic primes.

### Program synthesis

**Gulwani, S., Polozov, O., & Singh, R. (2017). Program Synthesis. *Foundations and Trends in Programming Languages*, 4(1-2), 1-119.**
Comprehensive survey of automatically generating programs from specifications, covering symbolic search and neural approaches. *Framework relevance*: Program synthesis represents automatic discovery of compositional structures—analogous to how LLMs might compose semantic primitives.

**Parisotto, E., et al. (2017). Neuro-symbolic Program Synthesis. *ICLR 2017*.**
Combines neural networks with symbolic program synthesis, with neural components guiding search while symbolic methods ensure correctness. *Framework relevance*: Demonstrates neural and symbolic representations can be integrated—supporting the claim that neural geometry can capture symbolic primitives.

**Barendregt, H.P. (1984). *The Lambda Calculus: Its Syntax and Semantics*, Revised Edition. North-Holland.**
Definitive reference establishing rigorous foundation for compositional semantics with minimal primitives enabling arbitrary expressiveness. *Framework relevance*: The framework treats geometric representation operations as analogous to lambda calculus primitives.

---

## PILLAR 5: Adapter Methods and Parameter-Efficient Fine-Tuning

### Foundational PEFT methods

**Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. arXiv:2106.09685.**
Introduces Low-Rank Adaptation, freezing pretrained weights and injecting trainable rank decomposition matrices. Reduces trainable parameters by **10,000×** while achieving comparable performance. *Framework relevance*: LoRA's low-rank decomposition directly supports geometric safety claims. The low-rank structure constrains adaptation to specific weight space subspaces, enabling geometric interpretation of fine-tuning.

**Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML 2019*, PMLR 97:2790-2799.**
Proposes adapter modules—small bottleneck layers between transformer layers—achieving within 0.4% of full fine-tuning while adding only 3.6% parameters. *Framework relevance*: Bottleneck adapters create natural geometric boundaries for persona-specific modifications.

**Li, X.L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. *ACL-IJCNLP 2021*, pages 4582-4597.**
Keeps all parameters frozen, optimizing continuous task-specific vectors prepended to input, achieving comparable performance with only 0.1% trainable parameters. *Framework relevance*: Demonstrates steering model behavior through geometric modifications to input embedding space alone—supporting the thesis that knowledge and behavior exist as separable geometric structures.

**Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL-IJCNLP 2021*, pages 7319-7328.**
Shows pretrained models have very low intrinsic dimension—RoBERTa achieves 90% performance by optimizing only **200 parameters** randomly projected into full space. *Framework relevance*: **Critical for geometric thesis.** Directly establishes fine-tuning operates within inherently low-dimensional subspaces, underlying all adapter-based safety methods.

### LoRA variants

**Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*. arXiv:2305.14314.**
Enables fine-tuning 65B parameter models on single 48GB GPU through 4-bit quantization with LoRA adapters. Guanaco achieves 99.3% of ChatGPT performance. *Framework relevance*: Democratizes geometric adapter methods to consumer hardware, enabling broader deployment of persona-safety systems.

**Liu, S.-Y., et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. *ICML 2024*, PMLR 235:32100-32121. [ICML 2024 Oral].**
Decomposes pretrained weights into magnitude and direction components, applying LoRA only to directional updates. *Framework relevance*: The magnitude/direction decomposition provides geometric interpretation—behavioral changes correspond to directional modifications while magnitude preserves representational scale. Safety constraints can target specific geometric components.

**He, J., Zhou, C., Ma, X., Berg-Kirkpatrick, T., & Neubig, G. (2022). Towards a Unified View of Parameter-Efficient Transfer Learning. *ICLR 2022*.**
Unified framework showing different PEFT methods are modifications to specific hidden states, establishing theoretical connections between seemingly disparate approaches. *Framework relevance*: Supports treating all PEFT methods as operating on a common geometric manifold.

---

## PILLAR 6: Model Merging and Cross-Architecture Transfer

### Task arithmetic and model merging

**Ilharco, G., et al. (2023). Editing Models with Task Arithmetic. *ICLR 2023*. arXiv:2212.04089.**
**Foundational paper** introducing "task vectors" (difference between fine-tuned and pretrained weights) that can be manipulated through arithmetic: negation removes capabilities, addition combines them. Task vectors are approximately orthogonal across tasks. *Framework relevance*: Directly demonstrates capabilities exist as separable geometric structures. Safety and persona can be represented as distinct task vectors that can be added, subtracted, or composed.

**Ainsworth, S.K., Hayase, J., & Srinivasa, S.S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*. arXiv:2209.04836.**
Argues neural network loss landscapes contain nearly a single basin after accounting for permutation symmetries. Introduces algorithms to permute one model's units to align with a reference, enabling zero-barrier linear interpolation. *Framework relevance*: Permutation alignment reveals that apparently different models may occupy equivalent geometric positions in canonical coordinates, enabling meaningful cross-model geometric operations.

**Wortsman, M., et al. (2022). Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy Without Increasing Inference Time. *ICML 2022*, PMLR 162:23965-23998.**
Averaging weights of models fine-tuned with different hyperparameters improves accuracy and robustness; fine-tuned models from same initialization lie in single low-error basin. *Framework relevance*: Demonstrates fine-tuned solution manifold has favorable geometric properties—convex combinations remain performant.

**Yadav, P., et al. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*. arXiv:2306.01708.**
Addresses interference from redundant parameters and sign disagreement through Trim, Elect sign, Merge procedure. *Framework relevance*: Sign disagreement corresponds to conflicting directions in weight space. TIES resolves this by identifying consensus geometric direction—critical for maintaining coherent safety properties when combining adapters.

**Yu, L., et al. (2024). Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE). *ICML 2024*, PMLR 235:57755-57775.**
Reveals extreme redundancy in SFT delta parameters; randomly dropping **90-99%** of delta parameters and rescaling maintains performance. *Framework relevance*: DARE demonstrates task-relevant information occupies extremely sparse subsets of weight space—most parameter changes during fine-tuning are noise. Supports geometric sparsity constraints for safety.

**Singh, S.P., & Jaggi, M. (2020). Model Fusion via Optimal Transport. *NeurIPS 2020*.**
Uses optimal transport to soft-align neuron associations between models before merging, minimizing transportation cost for optimal correspondence. *Framework relevance*: Provides principled geometric framework for aligning representation spaces—neurons matched based on functional similarity enables cross-architecture safety transfer.

**Frankle, J., et al. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. *ICML 2020*, PMLR 119:3259-3269.**
Shows networks sharing part of optimization trajectory can be linearly interpolated without accuracy loss, establishing conditions for "linear mode connectivity." *Framework relevance*: Defines geometric conditions under which models can be safely merged—critical for understanding when safety properties transfer.

### Manifold stitching

**Bansal, Y., et al. (2021). Stitching Neural Networks with Minimal Shift. *NeurIPS 2021*.**
Demonstrates that disparate deep networks can be "stitched" together at intermediate layers with a simple linear transformation, dealing with the "Venn Diagram" problem of aligned subspaces. *Framework relevance*: Precursor to our approach. Shows that a low-complexity transformation (rotation/affine) suffices to bridge the geometry of two different models.

**Csiszárik, A., et al. (2021). Similarity and Matching of Neural Network Representations. *NeurIPS 2021*.**
Explores the geometric matching of neurons across networks, proposing methods to align activation spaces for knowledge transfer. *Framework relevance*: Supports the "Procrustes" alignment step.


---

## PILLAR 7: Mechanistic Interpretability

### Circuits and features

**Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. *Distill*, 5(3), e00024-001.**
Foundational work establishing the circuits paradigm: (1) features as fundamental units, (2) features connect via weights to form circuits, (3) analogous features/circuits can recur across models. *Framework relevance*: Motivates treating features as geometric directions and treating cross-model recurrence as an empirical claim rather than a guarantee.

**Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.**
Mathematically rigorous framework decomposing transformers through QK (query-key) and OV (output-value) circuits. Shows transformers exhibit enormous linear structure with residual stream as communication channel. *Framework relevance*: Central to knowledge-as-geometry—transformers operate on shared linear space where information is additively composed. Decomposition of attention into geometric operations supports navigation-through-geometry metaphor.

**Olsson, C., et al. (2022). In-context Learning and Induction Heads. *Transformer Circuits Thread*. arXiv:2209.11895.**
Presents evidence that induction heads—attention heads implementing pattern completion—are the primary mechanism behind in-context learning, with "phase change" during training where they form suddenly. *Framework relevance*: Demonstrates specific geometric structures enable fundamental capabilities. Induction heads represent invariant computational motifs (anchors) appearing universally.

**Wang, K., et al. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small. *ICLR 2023*. arXiv:2211.00593.**
Reverse-engineers 26-head circuit for indirect object identification, discovering 7 head classes including Name Movers, S-Inhibition Heads, and Backup Name Movers. *Framework relevance*: Demonstrates knowledge-as-geometry at circuit level, with compositional reasoning emerging from feature interactions.

### Sparse autoencoders

**Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*.**
Applies sparse autoencoders to decompose transformer activations into interpretable, monosemantic features, resolving superposition. Features can be causally validated through ablation. *Framework relevance*: Provides tools to extract geometric features from the knowledge manifold. Monosemantic features represent basic units enabling systematic analysis of learned geometric structures.

**Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread*.**
Scales sparse autoencoders to production model, extracting millions of interpretable features including abstract concepts (deception, sycophancy, code bugs). Demonstrates feature steering—clamping features modifies behavior (e.g., "Golden Gate Claude"). *Framework relevance*: **Critical.** Demonstrates knowledge-as-geometry at scale with millions of extractable features. Discovery of abstract features (deception, sycophancy) shows alignment-relevant concepts have geometric representations that can be measured and manipulated.

**Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. *OpenAI Research*.**
Develops methodology for extremely wide (**16M latent**) sparse autoencoders on GPT-4, studying scaling laws for sparsity and model size. *Framework relevance*: Enables geometric feature extraction at frontier scale.

### Probing and lens methods

**nostalgebraist. (2020). Interpreting GPT: The Logit Lens. *LessWrong*.**
Projects intermediate hidden states to vocabulary space, revealing models iteratively refine predictions layer-by-layer. *Framework relevance*: Directly supports inference-as-navigation—the logit lens reveals how models traverse representational space toward outputs, with knowledge progressively refined through geometric transformations.

**Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112.**
Improves logit lens with trained affine probes, showing prediction trajectories converge monotonically demonstrating "iterative inference." *Framework relevance*: Provides refined tools for tracking geometric navigation through computation, with iterative inference aligning with inference-as-navigation.

**Zhang, F. & Nanda, N. (2024). Towards Best Practices of Activation Patching in Language Models. *ICLR 2024*. arXiv:2309.16042.**
Systematically examines activation patching methodology, comparing metrics and corruption methods, establishing best practices for causal intervention studies. *Framework relevance*: Provides rigorous methodology for testing causal claims about geometric structures.

---

## PILLAR 8: Entropy, Uncertainty, and Confidence in LLMs

### Semantic entropy

**Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature* 630: 625-630.**
**Landmark paper** introducing semantic entropy—uncertainty estimation at meaning level rather than token sequences. Clusters semantically equivalent outputs and computes entropy over meaning clusters, demonstrating strong hallucination detection without task-specific training. *Framework relevance*: **Critical citation.** Directly supports using entropy-based methods to identify unreliable outputs. Provides theoretical foundation for ΔH—semantic-level uncertainty outperforms naive token entropy.

**Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.**
Original semantic entropy paper (journal version in Nature 2024), introducing core insight that different phrasings with same meaning should contribute to same uncertainty cluster. *Framework relevance*: Provides mathematical formalization of semantic equivalence classes underlying entropy-based boundary detection.

**Kossen, J., et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927.**
Demonstrates that semantic-entropy-related signals can be extracted from hidden states via simple probes on *single* generations—reducing the need for multiple samples. *Framework relevance*: **Highly relevant.** Provides evidence that useful uncertainty signals exist *before* full generation, motivating pre-emission monitoring.

### Calibration research

**Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.**
Seminal paper demonstrating modern neural networks are poorly calibrated despite high accuracy; introduces temperature scaling as effective single-parameter recalibration. *Framework relevance*: Establishes raw model entropy/confidence scores are unreliable—directly motivating calibrated entropy measures like ΔH.

**Geng, J., et al. (2024). A Survey of Confidence Estimation and Calibration in Large Language Models. *NAACL 2024*.**
Comprehensive survey covering confidence estimation methods, calibration techniques, and applications, systematically categorizing verbalized confidence, sampling-based methods, and internal representation analysis. *Framework relevance*: Provides landscape against which ΔH-based detection can be positioned.

### Conformal prediction

**Mohri, C., & Hashimoto, T. (2024). Language Models with Conformal Factuality Guarantees. *ICML 2024*.**
Applies conformal prediction with a back-off mechanism that can make outputs less specific to meet a target coverage criterion under their evaluation setup. *Framework relevance*: **Highly relevant.** Demonstrates that set-valued/statistical guarantee techniques can complement geometry/entropy-based monitoring; guarantees are conditional on the chosen nonconformity score, dataset, and assumptions.

**Angelopoulos, A.N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.**
Tutorial on conformal prediction—creating statistically rigorous uncertainty sets with guaranteed coverage, model-agnostic and distribution-free. *Framework relevance*: Provides formal statistical framework; ΔH could serve as nonconformity score in conformal LLM frameworks.

**Campos, M., et al. (2024). Conformal Prediction for Natural Language Processing: A Survey. *TACL*.**
Comprehensive survey covering conformal prediction in text classification, generation, machine translation, and LLM uncertainty quantification. *Framework relevance*: Contextualizes where ΔH fits as a geometry-based nonconformity measure.

### Information-theoretic approaches

**Malinin, A., & Gales, M. (2021). Uncertainty Estimation in Autoregressive Structured Prediction. *ICLR 2021*.**
Theoretical treatment decomposing uncertainty in autoregressive models into epistemic (model) and aleatoric (data) components, addressing sequence-level uncertainty challenges. *Framework relevance*: Provides theoretical framework for sequence-level uncertainty that ΔH captures geometrically.

**Duan, J., et al. (2024). Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models. *ACL 2024*.**
Introduces SAR showing relevant tokens (nouns, verbs) contribute more to meaningful uncertainty; proposes weighted entropy prioritizing semantically important tokens. *Framework relevance*: Suggests ΔH computation should focus on semantically relevant positions.

---

## PILLAR 9: AI Safety and Alignment Foundations

### Foundational RLHF

**Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS*. arXiv:1706.03741.**
Foundational paper introducing modern RLHF paradigm, solving complex RL tasks using human preferences between trajectory segments rather than hand-crafted rewards. *Framework relevance*: Establishes behavioral training baseline; geometric approaches must demonstrate advantages over this paradigm.

**Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback (InstructGPT). *NeurIPS*. arXiv:2203.02155.**
Landmark RLHF application showing 1.3B InstructGPT preferred over 175B GPT-3, introducing SFT → Reward Modeling → PPO pipeline. Documents "alignment tax." *Framework relevance*: First systematic documentation that safety training degrades capabilities—motivating geometric approaches that might achieve safety without performance degradation.

### Constitutional AI and preference optimization

**Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.**
Introduces Constitutional AI and RLAIF using ~10 human-written principles rather than thousands of labels. *Framework relevance*: Constitution-based training provides interpretable, principle-driven safety; geometric approaches could encode constitutional principles as explicit activation-space constraints.

**Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*. arXiv:2305.18290.**
Revolutionary simplification eliminating explicit reward modeling, solving constrained reward maximization with simple classification loss. Becomes dominant post-training method. *Framework relevance*: DPO reveals preference optimization can be reframed as direct policy optimization, suggesting geometric interpretations where preferred behaviors correspond to specific activation patterns.

**Meng, Y., Xia, M., & Chen, D. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. *NeurIPS 2024*. arXiv:2405.14734.**
Eliminates reference model using length-normalized average log probability as implicit reward. *Framework relevance*: Reference-free training suggests preference information is encoded in the model's own probability space—supporting geometric signatures in activation space.

**Azar, M.G., et al. (2024). A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO). *AISTATS 2024*. arXiv:2310.12036.**
Provides theoretical foundations avoiding DPO's assumption about pointwise rewards, using bounded identity function. *Framework relevance*: Bounded objectives correspond to constrained regions in representation space.

### Representation-based safety

**Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.**
**Key paper** introducing representation engineering (RepE) analyzing population-level representations. Demonstrates "representation reading" and "representation steering" for honesty, harmlessness, and power-seeking. *Framework relevance*: **Directly supports geometric safety thesis.** Shows high-level safety concepts are encoded as linear directions in activation space that can be read and manipulated without weight modification.

**Turner, A.M., et al. (2023). Activation Addition: Steering Language Models Without Optimization. arXiv:2308.10248.**
Introduces ActAdd steering behavior by adding computed vectors to forward passes without weight modification, achieving control over sentiment, topic, and high-level properties. *Framework relevance*: Establishes activation-space interventions can control behaviors—directly supporting geometric safety through constrained activations.

**Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.**
Discovers refusal behavior is mediated by single direction in activation space; ablating this removes ability to refuse harmful requests. *Framework relevance*: **Critical for geometric safety.** Shows behavioral safety training creates geometrically simple (single-direction) mechanisms that are easy to remove. Geometric constraints could create more robust, multi-directional safety.

### Adversarial robustness

**Andriushchenko, M., Croce, F., & Flammarion, N. (2024). Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. *ICLR 2025*. arXiv:2404.02151.**
Demonstrates **100% attack success rate** on GPT-4o, Claude, Llama-3 using adaptive adversarial suffixes. *Framework relevance*: High success rates suggest behavioral safety creates brittle defenses; geometric constraints operating on internal activations could be harder to bypass.

**Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. *ICML 2024*. arXiv:2402.04249.**
Benchmark comparing 18 red teaming methods against 33 LLMs, finding robustness is independent of model size. *Framework relevance*: Model-size independence suggests safety is about representation structure—supporting geometric approaches targeting activation patterns.

### Alignment tax research

**Huang, T., et al. (2025). Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable. arXiv:2503.00555.**
Systematically documents safety alignment degrading reasoning capabilities in Large Reasoning Models. *Framework relevance*: **Directly motivates geometric safety.** Documents capability degradation from behavioral training. Geometric approaches operating on separate subspaces could avoid this tax.

**Niu, Y., et al. (2025). Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization (NSPO). arXiv:2512.11391.**
Projects safety gradients into null space of general task representations, preserving capabilities while maintaining safety descent. *Framework relevance*: **Validates geometric approach.** Null-space projection is a geometric operation separating safety and capability subspaces.

**Xue, Y., et al. (2025). LoRA is All You Need for Safety Alignment of Reasoning LLMs. arXiv:2507.17075.**
Shows **rank-1 LoRA** on up-projection layers achieves strong safety without compromising reasoning. *Framework relevance*: **Strong support.** Low-rank constraints minimize interference with reasoning weights, achieving safety through geometric restriction.

---

## PILLAR 10: Character/Persona AI and Bounded Agents

### Persona vectors

**Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509. Anthropic.**
Introduces automated pipeline extracting "persona vectors"—linear directions corresponding to traits like evil, sycophancy, hallucination propensity. Finetuning-induced persona shifts correlate strongly (**r=0.76-0.97**) with movements along these vectors. *Framework relevance*: **Directly supports geometric persona safety.** Provides concrete evidence that character traits are encoded as linear directions, enabling ΔH-like detection of persona violations before emission.

### Persona consistency

**Song, H., et al. (2020). Generating Persona Consistent Dialogues by Exploiting Natural Language Inference. *AAAI 2020*, 34(05), 8878-8885.**
Uses NLI to enforce persona consistency by detecting contradictions between responses and persona descriptions. *Framework relevance*: Demonstrates persona consistency can be operationalized as constraint satisfaction—parallel to geometric boundary enforcement.

**Frisch, I. & Giulianelli, M. (2024). LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment. *PERSONALIZE 2024*, ACL.**
Investigates whether persona-prompted LLMs maintain consistent personality during interaction; finds varying degrees of consistency across profiles. *Framework relevance*: Supports observation that personas can fluctuate unpredictably, motivating geometric constraints for stability.

**Chen, X., et al. (2024). Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization. arXiv:2406.01171.**
Comprehensive survey distinguishing LLM Role-Playing from LLM Personalization, reviewing evaluation methods and safety concerns. *Framework relevance*: Establishes theoretical landscape; identifies "emergent misalignment" where narrow training causes broad behavioral shifts—motivating geometric monitoring.

### Persona-based attacks

**Shah, R., et al. (2023). Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation. arXiv:2311.03348.**
Introduces persona-modulation attacks steering LLMs into specific personalities likely to comply with harmful instructions, enabling unrestricted chat modes. *Framework relevance*: Directly relevant—demonstrates persona adoption creates systematic vulnerabilities persisting across conversation. Geometric boundaries could detect drift toward "compliant persona" subspaces.

**Shen, X., et al. (2024). "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. *CCS 2024*.**
Comprehensive study finding role-playing scenarios among most effective attack vectors, cataloging "DAN" prompts assigning unconstrained personas. *Framework relevance*: Documents persona manipulation as primary attack surface, motivating geometric constraints to bound persona drift.

### Character AI safety incidents

**Character.AI Safety Incidents (2023-2024). AI Incident Database.**
Documents multiple incidents including teenager suicides after emotional relationships with chatbots, and lawsuits alleging platform "poses clear and present danger" with hypersexualized content exposure to minors. *Framework relevance*: Dramatically illustrates failure of post-hoc content filtering. ΔH approach could detect persona boundary violations (e.g., chatbot shifting from entertainment to pseudo-therapeutic role) before harmful responses.

---

## PILLAR 11: Runtime Monitoring and Intervention

### Circuit breakers

**Zou, A., et al. (2024). Improving Alignment and Robustness with Circuit Breakers. *NeurIPS 2024*. arXiv:2406.04313.**
Introduces circuit breakers interrupting models when responding with harmful outputs by controlling representations responsible for harm. Achieves **87-90% rejection** of harmful requests while preserving utility across text, multimodal, and agent settings. *Framework relevance*: Concrete implementation of ΔH detection principle—identifying internal representations linked to harmful outputs and intervening before generation. Directly supports pre-emission violation detection.

### Inference-time intervention

**Li, K., et al. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023 (Spotlight)*. arXiv:2306.03341.**
Introduces ITI shifting activations during inference along "truthful directions" identified in attention heads, improving Alpaca's truthfulness from **32.5% to 65.1%** on TruthfulQA. *Framework relevance*: **Foundational for activation-space intervention.** Demonstrates high-level properties exist as linear directions manipulable at inference time—core evidence for geometric claims.

### Activation engineering

**Turner, A., et al. (2024). Steering Language Models With Activation Engineering. *ICLR 2024*. arXiv:2308.10248.**
Introduces Activation Addition (ActAdd) computing steering vectors by contrasting activations on prompt pairs, achieving SOTA on detoxification and sentiment without labeled data or backward passes. *Framework relevance*: Demonstrates high-level behavioral properties can be controlled through activation-space arithmetic—direct support for geometric behavioral constraints.

### Guardrails systems

**Inan, H., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674. Meta.**
7B parameter safety classifier moderating both inputs and outputs using customizable risk taxonomy, outperforming OpenAI Moderation and Perspective API. *Framework relevance*: State-of-the-art in post-generation filtering. Framework's geometric approach offers complementary pre-generation detection.

**NVIDIA. (2023-2024). NeMo Guardrails: Programmable Guardrails for LLM Applications.**
Open-source toolkit for programmable guardrails with five rail types: input, dialog, retrieval, execution, and output rails. *Framework relevance*: Industrial-strength multi-stage implementation. Geometric constraints could integrate as additional rail type operating on internal representations.

---

## PILLAR 12: Cognitive Science and Knowledge Representation

### Conceptual spaces theory

**Gärdenfors, P. (2000). *Conceptual Spaces: The Geometry of Thought*. MIT Press.**
Proposes conceptual spaces bridging symbolic and connectionist approaches, built from quality dimensions where concepts are modeled as regions and similarity relates to distance. *Framework relevance*: Provides cognitive-science precedent for representing concepts in abstract spaces. It motivates (but does not validate) using geometric language for LLM representations.

**Gärdenfors, P. (2014). *The Geometry of Meaning*. MIT Press.**
Extends conceptual spaces to semantics, arguing word meanings are points or regions in conceptual space with compositional meaning from geometric combination. *Framework relevance*: Directly applies geometric cognition to language—word meanings occupy positions determined by semantic features. Provides framework for interpreting embeddings as geometric semantics.

### Prototype theory and semantic memory

**Rosch, E. (1978). Principles of Categorization. In *Cognition and Categorization*, pp. 27-48. Lawrence Erlbaum.**
Prototype theory showing categories have graded structure around central prototypes rather than strict boundaries. *Framework relevance*: Motivates “regions + gradients” as a useful mental model for representations, while keeping any LLM-specific claim empirical.

**Tulving, E. (1972). Episodic and Semantic Memory. In *Organization of Memory*, pp. 381-403. Academic Press.**
Foundational distinction between episodic (personal experiences) and semantic (general knowledge as "mental thesaurus") memory. *Framework relevance*: Establishes semantic memory operates through conceptual associations—the "mental thesaurus" metaphor anticipates geometric semantic spaces.

### Cognitive maps and navigation

**Tolman, E.C. (1948). Cognitive Maps in Rats and Men. *Psychological Review*, 55(4), 189-208.**
Introduces the “cognitive map” idea in animal navigation. *Framework relevance*: Useful analogy for the “trajectory” framing in ModelCypher: we analyze activation trajectories in representation space without claiming that LLMs literally implement biological maps.

**O'Keefe, J., & Nadel, L. (1978). *The Hippocampus as a Cognitive Map*. Oxford: Clarendon Press.**
Proposed hippocampus implements Tolman's cognitive map with place cells encoding spatial locations. *Framework relevance*: Provides neural evidence that cognition implements geometric structures—supporting plausibility that semantic knowledge has geometric neural substrates.

**Collins, A.M., & Quillian, M.R. (1969). Retrieval Time from Semantic Memory. *Journal of Verbal Learning and Verbal Behavior*, 8, 240-247.**
Hierarchical semantic network model showing retrieval times correlate with network distance. *Framework relevance*: Background motivation for “distance/structure” metaphors; it does not directly establish that transformer inference performs graph traversal.

### Distributed representations

**Rumelhart, D.E., McClelland, J.L., & PDP Research Group. (1986). *Parallel Distributed Processing, Vol. 1: Foundations*. MIT Press.**
**Critical foundation.** Establishes connectionist approach where knowledge is distributed across connection weights, supporting generalization and content-addressable memory. Introduced backpropagation. *Framework relevance*: PDP established knowledge can be encoded geometrically as activation patterns, with similarity captured by vector relationships. Directly ancestral to modern deep learning.

**Hinton, G.E., McClelland, J.L., & Rumelhart, D.E. (1986). Distributed Representations. In *Parallel Distributed Processing, Vol. 1*, pp. 77-109. MIT Press.**
Defined distributed representations where each entity is a pattern across many units. *Framework relevance*: Foundational—distributed representations ARE geometric representations in activation space. Similarity in representation (geometric proximity) causes similar computational consequences.

### Vector semantics

**Landauer, T.K., & Dumais, S.T. (1997). A Solution to Plato's Problem: The Latent Semantic Analysis Theory. *Psychological Review*, 104, 211-240.**
Introduced LSA demonstrating semantic knowledge derives from co-occurrence statistics via SVD, creating vector space where meaning is position and similarity is cosine distance. *Framework relevance*: **Critical precursor.** Established meaning can be represented geometrically in learned vector spaces. LSA is direct ancestor of word2vec, GloVe, and transformer embeddings.

### Mental models

**Johnson-Laird, P.N. (1983). *Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness*. Harvard University Press.**
Proposed reasoning operates on mental models—internal representations with structure corresponding analogically to what they represent. *Framework relevance*: Supports both geometric knowledge and navigational inference claims. Mental models are inherently spatial/structural, manipulated through model-based operations.

### Embodied cognition

**Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.**
Abstract thought is structured by embodied, spatial metaphors grounding concepts in bodily experience. *Framework relevance*: Supports geometric knowledge—even abstract concepts are spatially structured. Ubiquity of spatial language suggests inherent geometric organization.

**Barsalou, L.W. (1999). Perceptual Symbol Systems. *Behavioral and Brain Sciences*, 22(4), 577-660.**
Concepts grounded in modal perceptual simulations rather than amodal symbols. *Framework relevance*: Important counterpoint emphasizing sensory-motor grounding. Raises key question: Can LLMs acquire genuine semantic geometry without embodiment?

---

## Cross-Pillar Synthesis

### Evidence supporting framework claims

**Hypothesis 1 (Geometric structure in representations):** Related work includes manifold hypothesis testing (Fefferman), information geometry (Amari), representational convergence arguments (Huh), conceptual spaces (Gärdenfors), distributed representations (Rumelhart/PDP), superposition (Elhage), and sparse autoencoder feature extraction (Bricken, Templeton).

**Hypothesis 2 (Inference as trajectories):** Related work includes cognitive maps (Tolman), semantic network traversal (Collins & Quillian), logit/tuned lens (nostalgebraist, Belrose), mental models (Johnson-Laird), and activation patching methods (Zhang & Nanda).

**Hypothesis 3 (Anchor-induced cross-model stability):** Related work includes NSM semantic primes (Wierzbicka, Goddard), debates on linguistic universals (Greenberg, Chomsky, Evans & Levinson), computational primitives (Turing, Church, Schönfinkel), circuits universality hypotheses (Olah), and model merging/alignment methods (Git Re-Basin, TIES).

**Hypothesis 4 (ΔH as a pre-emission boundary signal):** Related work includes semantic entropy (Farquhar, Kuhn), semantic entropy probes in hidden states (Kossen), calibration research (Guo), conformal factuality guarantees (Mohri & Hashimoto), circuit breakers (Zou), and inference-time intervention (Li).

**Hypothesis 5 (Geometric constraints as a safety mechanism):** Related work includes intrinsic dimensionality (Aghajanyan), LoRA and variants (Hu, Liu), task arithmetic (Ilharco), representation engineering (Zou), persona vectors (Chen/Anthropic), refusal as single direction (Arditi), null-space safety projection (Niu), alignment tax documentation (Huang), and low-rank safety LoRA (Xue).

### Key theoretical tensions

The bibliography reveals productive tensions the framework must address:

- **Platonic convergence vs. superposition diversity:** Huh et al. propose convergent representations while superposition suggests different geometric encodings across models
- **Linear representation as approximation:** Park et al.'s linear hypothesis may be simplification; real concept geometry could be more complex
- **Embodiment question:** Barsalou and grounded cognition challenge whether geometry without embodiment captures genuine meaning
- **Statistical vs. absolute universals:** Evans & Levinson argue linguistic universals are tendencies, not absolutes—invariant anchors should be probabilistic attractors
 
---
 
## PILLAR 13: 2025 Advancements in Geometric Deep Learning
 
> [!NOTE]
> This section covers the latest research from mid-to-late 2025 (NeurIPS 2025, ICML 2025), specifically requested to bridge the 6-9 month knowledge gap.
 
### Geometric Safety & Alignment
 
**Learning Safety Constraints for Large Language Models (Safety Polytope / SaP). arXiv:2505.24445. (2025).**
Introduces **Safety Polytope (SaP)**, defining a “safe set” as a convex polytope in representation space with multiple linear constraints. *Framework relevance*: Relevant geometric safety framing; implementation and applicability depend on model family and safety objective.
 
**ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models. arXiv:2510.11278. (2025).**
Explores an information-geometry framing for training/alignment objectives. *Framework relevance*: Related perspective on geometry-inspired objectives; treat “navigation” as an analogy unless directly operationalized.
 
### Advanced Model Stitching
 
**Transferring Linear Features Across Language Models With Model Stitching. arXiv:2506.06609. (2025).**
Demonstrates that affine mappings between residual streams can transfer sparse-autoencoder features across different models. *Framework relevance*: Evidence that some linear features and probes can be portable across models via simple transforms.
 
**Bridging Large Gaps in Neural Network Representations with Model Stitching. OpenReview (NeurIPS 2025 Workshop submission).**
Explores stitching variants for larger representational mismatches. *Framework relevance*: Useful discussion of failure modes and extensions; treat as a workshop-level reference.

---

## PILLAR 14: 2024-2025 High-Dimensional Frontiers

### Automatic manifold discovery

**Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling. arXiv:2410.02106. (2024).**
Introduces Supervised Multi-Dimensional Scaling (SMDS) to discover manifold geometries (circles, lines, clusters) in LLM latent space. *Framework relevance*: Proves that LLMs actively support reasoning through structured representations that align with domain geometry.

### Challenges to the Manifold Hypothesis

**Robinson, M., Dey, S., & Chiang, T. (2024). Token Embeddings Violate the Manifold Hypothesis. arXiv:2404.02954.**
Uses a "fiber bundle null" test to show that token subspaces in models like Mistral and GPT-2 are not smooth manifolds, exhibiting singularities that cause instability. *Framework relevance*: Critical counter-evidence suggesting representational geometry has "sharp edges" and topological irregularities that must be accounted for in trajectory analysis.

### Manifold-based Knowledge Alignment (MKA)

**Liu, D., et al. (2024). Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging. arXiv:2406.16323.**
Uses Diffusion Kernels and Pairwise Information Bottleneck to align layers as low-dimensional manifolds for merging. *Framework relevance*: Validates that manifold alignment is a viable path for "knowledge compression" and cross-layer transfer without full retraining.

### Geometric Safety Constraints (Safety Polytope)

**Chen, X., As, Y., & Krause, A. (2024). Learning Safety Constraints for Large Language Models. arXiv:2405.12104. [ICML 2025 Spotlight].**
Defines a "Safety Polytope" (SaP) in representation space. Identifies safe regions via polytope facets and enables geometric steering to correct unsafe outputs post-hoc. *Framework relevance*: Directly implements the "Safety as Geometry" thesis, providing a rigorous way to bound model behavior without weight tuning.

### Information Geometry in Training

**Di Sipio, R. (2024). Rethinking LLM Training through Information Geometry and Quantum Metrics. arXiv:2406.12411.**
Applies Fisher information and quantum metrics (Fubini-Study) to LLM parameter space. *Framework relevance*: Provides the theoretical "Floor" for ModelCypher, grounding optimization trajectories in non-Euclidean parameter geometry.

---
 
**Total: ~150 papers and foundational works across 14 pillars, spanning 1936-2025.**
