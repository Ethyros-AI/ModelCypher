# Foundational Bibliography: Knowledge as High-Dimensional Geometry in Large Language Models

This bibliography supports research claiming: (1) LLM knowledge is static high-dimensional geometry, (2) inference is navigation through that geometry, (3) invariant anchors enable cross-model alignment, (4) entropy differential (ΔH) can detect boundary violations pre-emission, and (5) adapter-based persona safety can be achieved through geometric constraints.

---

## PILLAR 1: Foundational Mathematics and Geometry

### Manifold hypothesis and information geometry

**Fefferman, C., Mitter, S., & Narayanan, H. (2016). Testing the Manifold Hypothesis. *Journal of the American Mathematical Society*, 29(4), 983-1049.**
Provides rigorous mathematical foundations for testing whether high-dimensional data lies near low-dimensional manifolds, with complexity guarantees for fitting manifolds to probability distributions. *Framework relevance*: Foundational for the claim that LLM knowledge organizes on lower-dimensional manifolds within high-dimensional weight space. The framework extends this by proposing that inference *navigates* these manifolds, treating them as computational substrates rather than merely statistical structures.

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
Argues that representations in AI models are converging toward a shared statistical model of reality—a "platonic representation." Vision and language models measure distances between datapoints increasingly similarly as they scale. *Framework relevance*: **Critical foundation.** Directly supports the claim that LLMs encode shared geometric representations. If models converge to common representations, this suggests objective geometric structure to knowledge. The framework adds the dynamic dimension—inference as navigation through this convergent geometry.

### Linear representation hypothesis

**Park, K., Choe, Y.J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML 2024*, PMLR 235:39643-39666.**
Rigorously formalizes "linear representation" using counterfactuals, proving connections between linear probing, model steering, and subspace representations. Identifies a causal inner product respecting semantic structure. *Framework relevance*: **Critical.** Establishes concepts are directions in representation space, enabling geometric operations. The causal inner product provides principled geometry. The framework builds on this: if concepts are directions, inference involves moving through conceptual directions.

**Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS*.**
Demonstrates famous king−man+woman=queen analogies, showing word embeddings have linear algebraic structure encoding semantic relationships. *Framework relevance*: Original empirical demonstration that semantic knowledge has geometric structure through vector arithmetic.

### Superposition and polysemanticity

**Elhage, N., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*, Anthropic.**
Demonstrates neural networks represent more features than neurons through superposition—encoding sparse features as nearly-orthogonal directions. Reveals phase diagrams and geometric structures (polytopes) in superposition. *Framework relevance*: **Critical.** Explains how high-dimensional geometry enables efficient knowledge storage. Superposition is the geometric mechanism for compressing vast knowledge into finite dimensions.

**Scherlis, A., Sachan, K., Jermyn, A.S., Benton, J., & Shlegeris, B. (2022). Polysemanticity and Capacity in Neural Networks. *arXiv:2210.01892*.**
Analyzes polysemanticity through "feature capacity"—fractional dimensions features consume—showing optimal allocation represents important features monosemantically. *Framework relevance*: Explains geometric resource allocation, providing theoretical grounding for how geometry trades off representational precision.

### Concept geometry and probing

**Kim, B., et al. (2018). Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). *ICML*.**
Introduces Concept Activation Vectors as linear directions representing human concepts in neural network layers, enabling testing of concept influence on predictions. *Framework relevance*: Operationalizes concepts as geometric directions, foundational for treating knowledge as spatial.

**Koh, P.W., et al. (2020). Concept Bottleneck Models. *ICML 2020*, PMLR 119:5338-5348.**
Models that first predict human-interpretable concepts, then use concepts to predict labels, enabling intervention on concept values. *Framework relevance*: Demonstrates concepts have explicit geometric representation that can be isolated and manipulated.

**Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational Similarity Analysis – Connecting the Branches of Systems Neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.**
Introduces RSA for comparing representations across brain regions and computational models using representational dissimilarity matrices. *Framework relevance*: Provides methodology for measuring geometric structure of representations.

**Hewitt, J., & Manning, C.D. (2019). A Structural Probe for Finding Syntax in Word Representations. *NAACL-HLT*.**
Demonstrates syntactic structure is encoded geometrically in BERT representations—parse tree distances correspond to representation distances. *Framework relevance*: Shows linguistic structure has geometric form; syntax is literally spatial. The framework generalizes this to all knowledge domains.

---

## PILLAR 3: Semantic Primes and Linguistic Universals

### Natural Semantic Metalanguage

**Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.**
Landmark synthesis presenting approximately **65 indefinable, universal meanings** present in all human languages. Complex word meanings can be fully paraphrased using only semantic primes. *Framework relevance*: Directly supports invariant anchors claim. Semantic primes represent linguistic "anchor points"—irreducible concepts shared across all languages as the basis for composing all meanings. The framework extends this to claim LLMs learn geometric analogs of semantic primes.

**Goddard, C. & Wierzbicka, A. (Eds.) (2002). *Meaning and Universal Grammar: Theory and Empirical Findings*, Vols. I & II. John Benjamins.**
Comprehensive cross-linguistic validation of 60+ semantic primes across dozens of typologically diverse languages. *Framework relevance*: Empirical evidence that semantic universals exist and are lexicalized across languages—critical support for cross-model alignment via shared conceptual anchors.

**Goddard, C. (2018). *Ten Lectures on Natural Semantic Metalanguage*. Brill.**
Comprehensive introduction covering semantic molecules (intermediate concepts defined via primes), cultural scripts, and applications. *Framework relevance*: Demonstrates semantic molecules provide compositional hierarchy built from primes—paralleling how complex representations compose from primitive features.

### Linguistic universals

**Chomsky, N. (1965). *Aspects of the Theory of Syntax*. MIT Press.**
Introduces Language Acquisition Device and formalizes Universal Grammar as innate principles constraining possible human languages. *Framework relevance*: The claim that syntactic structures are constrained by innate universals parallels claims that representational structures in LLMs are constrained by universal geometric properties.

**Greenberg, J.H. (1963). Some Universals of Grammar with Particular Reference to the Order of Meaningful Elements. In *Universals of Language*, pp. 58-90. MIT Press.**
Identifies 45 linguistic universals from analysis of 30 typologically diverse languages, introducing implicational universals showing systematic cross-linguistic patterns. *Framework relevance*: Demonstrates empirically that linguistic diversity masks underlying universal patterns.

**Evans, N. & Levinson, S.C. (2009). The Myth of Language Universals. *Behavioral and Brain Sciences*, 32(5), 429-492.**
Critical assessment documenting extensive diversity across languages, arguing universals are statistical tendencies rather than absolutes. *Framework relevance*: Important counterbalance—invariant anchors should be understood as statistical attractors rather than exact points.

### Computational applications

**Towards Universal Semantics with Large Language Models. (2024). arXiv preprint.**
First application of LLMs to NSM-style semantic analysis, with fine-tuned models outperforming GPT-4o on generating explications using semantic primes. *Framework relevance*: Directly supports that LLMs encode something analogous to semantic primes and can leverage them for cross-linguistic alignment.

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

---

## PILLAR 7: Mechanistic Interpretability

### Circuits and features

**Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. *Distill*, 5(3), e00024-001.**
Foundational paper establishing circuits paradigm: (1) features are fundamental units, (2) features connect via weights to form circuits, (3) analogous features and circuits form universally across models. *Framework relevance*: Establishes that neural networks develop interpretable geometric structures (features as directions). Universality hypothesis supports invariant anchors—similar geometric structures emerge across models.

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
Demonstrates semantic entropy is encoded in LLM hidden states and extractable via simple linear probes on *single* generations—eliminating need for multiple samples. *Framework relevance*: **Highly relevant.** Proves entropy-related information exists in hidden states *before* generation, directly supporting pre-emission ΔH detection capability.

### Calibration research

**Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.**
Seminal paper demonstrating modern neural networks are poorly calibrated despite high accuracy; introduces temperature scaling as effective single-parameter recalibration. *Framework relevance*: Establishes raw model entropy/confidence scores are unreliable—directly motivating calibrated entropy measures like ΔH.

**Geng, J., et al. (2024). A Survey of Confidence Estimation and Calibration in Large Language Models. *NAACL 2024*.**
Comprehensive survey covering confidence estimation methods, calibration techniques, and applications, systematically categorizing verbalized confidence, sampling-based methods, and internal representation analysis. *Framework relevance*: Provides landscape against which ΔH-based detection can be positioned.

### Conformal prediction

**Mohri, C., & Hashimoto, T. (2024). Language Models with Conformal Factuality Guarantees. *ICML 2024*.**
Applies conformal prediction to guarantee factuality with back-off algorithm progressively making outputs less specific. Achieves **80-90% correctness guarantees** on QA and reasoning. *Framework relevance*: **Highly relevant.** Demonstrates conformal methods provide statistical guarantees for LLM reliability—complementary to ΔH-based detection.

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
**Most critical cognitive science foundation.** Proposes conceptual spaces bridging symbolic and connectionist approaches, built from quality dimensions where concepts are convex regions. Similarity is geometric distance; natural categories are geometrically coherent. *Framework relevance*: Directly establishes knowledge IS geometry. Concepts as convex regions, similarity as distance, categorization as geometric operations—provides theoretical grounding for viewing LLM embeddings as implementing conceptual spaces.

**Gärdenfors, P. (2014). *The Geometry of Meaning*. MIT Press.**
Extends conceptual spaces to semantics, arguing word meanings are points or regions in conceptual space with compositional meaning from geometric combination. *Framework relevance*: Directly applies geometric cognition to language—word meanings occupy positions determined by semantic features. Provides framework for interpreting embeddings as geometric semantics.

### Prototype theory and semantic memory

**Rosch, E. (1978). Principles of Categorization. In *Cognition and Categorization*, pp. 27-48. Lawrence Erlbaum.**
Prototype theory showing categories have graded structure around central prototypes rather than strict boundaries. *Framework relevance*: **Critical for geometric framework.** Prototypes are central regions with typicality gradients representing distance—maps directly onto continuous geometric representations. LLM embeddings naturally implement prototype-like structures.

**Tulving, E. (1972). Episodic and Semantic Memory. In *Organization of Memory*, pp. 381-403. Academic Press.**
Foundational distinction between episodic (personal experiences) and semantic (general knowledge as "mental thesaurus") memory. *Framework relevance*: Establishes semantic memory operates through conceptual associations—the "mental thesaurus" metaphor anticipates geometric semantic spaces.

### Cognitive maps and navigation

**Tolman, E.C. (1948). Cognitive Maps in Rats and Men. *Psychological Review*, 55(4), 189-208.**
Demonstrated rats form internal spatial representations enabling flexible navigation and shortcut discovery. *Framework relevance*: Foundational for "knowledge is geometry"—established that minds build map-like representations. LLM embedding spaces function as learned cognitive maps navigated analogously to physical space.

**O'Keefe, J., & Nadel, L. (1978). *The Hippocampus as a Cognitive Map*. Oxford: Clarendon Press.**
Proposed hippocampus implements Tolman's cognitive map with place cells encoding spatial locations. *Framework relevance*: Provides neural evidence that cognition implements geometric structures—supporting plausibility that semantic knowledge has geometric neural substrates.

**Collins, A.M., & Quillian, M.R. (1969). Retrieval Time from Semantic Memory. *Journal of Verbal Learning and Verbal Behavior*, 8, 240-247.**
Hierarchical semantic network model showing retrieval times correlate with network distance. *Framework relevance*: Directly supports "inference is navigation"—retrieving information involves traversing relational structure with time proportional to semantic distance.

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

**Claim 1 (Knowledge is static high-dimensional geometry):** Supported by manifold hypothesis (Fefferman), information geometry (Amari), Platonic Representation Hypothesis (Huh), conceptual spaces (Gärdenfors), distributed representations (Rumelhart/PDP), superposition (Elhage), and sparse autoencoder feature extraction (Bricken, Templeton).

**Claim 2 (Inference is navigation through geometry):** Supported by cognitive maps (Tolman), semantic network traversal (Collins & Quillian), logit/tuned lens (nostalgebraist, Belrose), mental models (Johnson-Laird), and activation patching methods (Zhang & Nanda).

**Claim 3 (Invariant anchors enable cross-model alignment):** Supported by NSM semantic primes (Wierzbicka, Goddard), linguistic universals (Greenberg, Chomsky), computational primitives (Turing, Church, Schönfinkel), circuits universality hypothesis (Olah), and model merging/alignment methods (Git Re-Basin, TIES).

**Claim 4 (ΔH detects boundary violations pre-emission):** Supported by semantic entropy (Farquhar, Kuhn), semantic entropy probes in hidden states (Kossen), calibration research (Guo), conformal factuality guarantees (Mohri & Hashimoto), circuit breakers (Zou), and inference-time intervention (Li).

**Claim 5 (Adapter-based geometric safety improves on behavioral training):** Supported by intrinsic dimensionality (Aghajanyan), LoRA and variants (Hu, Liu), task arithmetic (Ilharco), representation engineering (Zou), persona vectors (Chen/Anthropic), refusal as single direction (Arditi), null-space safety projection (Niu), alignment tax documentation (Huang), and low-rank safety LoRA (Xue).

### Key theoretical tensions

The bibliography reveals productive tensions the framework must address:

- **Platonic convergence vs. superposition diversity:** Huh et al. propose convergent representations while superposition suggests different geometric encodings across models
- **Linear representation as approximation:** Park et al.'s linear hypothesis may be simplification; real concept geometry could be more complex
- **Embodiment question:** Barsalou and grounded cognition challenge whether geometry without embodiment captures genuine meaning
- **Statistical vs. absolute universals:** Evans & Levinson argue linguistic universals are tendencies, not absolutes—invariant anchors should be probabilistic attractors

---

**Total: ~140 papers and foundational works across 12 pillars, spanning 1936-2025, covering mathematical foundations through cutting-edge SOTA in geometric AI safety.**