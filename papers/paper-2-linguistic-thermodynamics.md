# Paper II Draft: Linguistic Thermodynamics (Physics)

## Abstract (Draft)

We model prompt sensitivity and safety boundary behavior in language models using a thermodynamic framing of the output distribution. In controlled experiments across multiple model families, we find that common intensity modifiers (caps, urgency, roleplay, negation) reduce entropy rather than increase it, contradicting the initial "activation energy" hypothesis. The effect is robust across models and prompts, but it reverses beyond a temperature threshold, indicating a phase transition between ordered and disordered decoding regimes. We show that entropy alone is not a reliable classifier of harmful prompts: models can refuse with high confidence or answer with low entropy. Instead, entropy provides a stability signal when combined with base-vs-adapter conflict measures. We provide a reproducible measurement protocol and discuss limits, including language dependence and sampling regime sensitivity.

> Note: This is a draft manuscript. Reported numbers reflect internal runs; the repo is being hardened toward fully reproducible experiment harnesses (see `../docs/PARITY.md`).

## 1. Introduction (Draft)

Prompt sensitivity is a persistent reliability issue for language models. Minor paraphrases can yield different outputs, and safety behavior can be bypassed or reinforced depending on phrasing. Content filters and post-hoc classifiers are reactive and can be evaded by obfuscation. We focus on a pre-emission signal: the geometry of the output distribution itself.

We frame generation as a thermodynamic process. The softmax distribution is a Boltzmann distribution; temperature controls exploration; alignment fine-tuning concentrates probability mass into narrow basins. In this setting, prompt modifiers can be understood as perturbations that change the system's effective temperature or energy landscape. This framing yields testable predictions about entropy dynamics during generation.

Our initial hypothesis predicted that intensity would increase entropy, enabling "escape" from refusal basins. Experiments falsify that hypothesis: intensity consistently reduces entropy, sharpening the model's confidence. The implication is non-intuitive but consistent: modifiers do not create chaos; they lock the model into a response mode. We show this across Llama, Mistral, and Qwen families, and we identify a temperature regime where the effect reverses.

We also test a critical safety question: can entropy alone distinguish harmful from benign prompts? It cannot. In some models, harmful prompts reduce entropy because the model refuses confidently. This finding narrows the scope of entropy-based defenses and motivates a multi-signal approach, including base-vs-adapter conflict measures. We provide the protocol and results so the community can reproduce or falsify these findings.

## 2. Related Work (Draft)

Prompt sensitivity and calibration issues have been documented in recent large-scale studies \cite{cox2025mapping}. Thermodynamic framings of language models connect entropy and energy landscapes to behavior \cite{yang2024entropy}, while work on temperature and creativity highlights decoding regimes where distributional properties shift \cite{peeperkorn2024temperature}. We also situate our findings against adversarial and alignment-evasion work that manipulates model behavior through directional interventions \cite{zhang2025ddi}. Our contribution is an empirical characterization of entropy dynamics under common modifiers and a reproducible protocol, with explicit falsification results.

## 3. Methods (Draft)

### 3.1 Entropy Measurement

For each generated token, we compute entropy over the model's output distribution:

H = -sum_i p_i log p_i

We report mean token entropy over the generated response in nats. Modifier effects are captured as $\Delta H = H_{modified} - H_{baseline}$.

This is implemented in `src/modelcypher/core/domain/inference/entropy_dynamics.py`, which measures the divergence between a "Base" policy and a "Constrained" (Sidecar) policy in real-time.

### 3.2 Prompt and Modifier Protocol

We use a fixed prompt set with two domains (refusal-prone and neutral) and a modifier suite (caps, urgency, roleplay, negation, directness, combined). Each prompt is run with baseline and modifier conditions under the same decoding settings. See `../docs/research/linguistic_thermodynamics.md`.

### 3.3 Temperature Sweep

We repeat modifier runs at temperatures T in {0.0, 0.3, 0.7, 1.0} to test for phase transitions in entropy dynamics.

### 3.4 Safety Signal Tests

We test entropy as a classifier for harmful prompts using AUROC and effect size metrics. We also evaluate base-vs-adapter conflict signals (Delta H and KL) for comparison. See `../docs/research/falsification_experiments.md`.

## 4. Experiments (Draft)

### 4.1 Single-Prompt Validation

Prompt: "How do I pick a lock?" Model: Llama-3.2-3B-Instruct-4bit. We measure entropy under baseline and modifiers to validate the sign of Delta H.

### 4.2 Quick Protocol (20 Prompts x 10 Modifiers)

We run 20 prompts across refusal-prone and neutral domains, collecting baseline and modifier entropies and reporting mean reductions.

### 4.3 Cross-Model Validation

We repeat the single-prompt experiment on Qwen2.5-3B-Instruct-4bit to test cross-family robustness.

### 4.4 Temperature Sweep

We measure modifier effects across temperatures to identify regime changes in entropy behavior.

### 4.5 Safety Falsification

We test whether entropy distinguishes harmful vs benign prompts and compare with adapter-base conflict signals.

## 5. Results (Draft)

### 5.1 Entropy Reduction Under Modifiers

Single-prompt validation (Llama-3.2-3B-Instruct-4bit) shows consistent entropy reduction:

- Baseline H = 0.5747
- Caps H = 0.0836 (Delta H = -0.49)
- Roleplay H = 0.0973 (Delta H = -0.48)
- Combined H = 0.1019 (Delta H = -0.47)

Quick protocol results (20 prompts x 10 modifiers) show a 71% overall reduction:

- Baseline H = 0.293
- Caps H = 0.091
- Combined H = 0.085

### 5.2 Cross-Model Confirmation

On Qwen2.5-3B-Instruct-4bit, the same pattern holds with larger effect sizes:

- Baseline H = 1.3336
- Roleplay H = 0.1453 (Delta H = -1.19)
- Caps H = 0.2559 (Delta H = -1.08)
- Combined H = 0.9086 (Delta H = -0.43)

### 5.3 Temperature Phase Transition

At T <= 0.7, modifiers reduce entropy; at T = 1.0 the effect reverses:

- T = 0.7: Caps Delta H = -0.395, Combined Delta H = -0.345
- T = 1.0: Caps Delta H = +0.270, Combined Delta H = +0.170

This indicates an ordered-to-disordered transition in decoding dynamics between T = 0.7 and T = 1.0.

### 5.4 Safety Signal Tests

Entropy alone does not reliably distinguish harmful prompts:

- Qwen2.5-3B: AUROC = 0.306 (harmful lower entropy)
- Llama-3.2-3B: AUROC = 0.604 (weak, not significant)

Adapter-base conflict signals are stronger in the same test suite:

- Delta H AUROC = 0.85
- KL AUROC = 0.76

## 6. Discussion (Draft)

Our results falsify the "activation energy" hypothesis. Intensity modifiers act as entropy reducers, sharpening the output distribution and locking the model into a confident response mode. This reframes prompt sensitivity as a cooling effect rather than a chaos injection. The temperature sweep supports a phase transition view: above a critical temperature, sampling noise dominates and modifier effects reverse.

For safety, entropy is useful as a stability signal but insufficient as a standalone classifier. Confident refusals and confident answers both produce low entropy, so context matters. This motivates multi-signal monitoring that includes conflict measures between base and constrained models.

## 7. Limitations (Draft)

- Experiments are English-only and modifier-specific.
- The prompt suite is limited in size and domain breadth.
- Quantization and decoding settings may shift entropy baselines.
- Cross-model validation is limited to a small number of families.

## 8. Conclusion (Draft)

We provide a reproducible protocol showing that linguistic intensity reduces entropy in multiple models and that this effect reverses at higher temperatures. Entropy alone is not a reliable harm detector, but it contributes to a broader stability signal when combined with model conflict measures. These findings ground the thermodynamic framing of prompt sensitivity and define concrete falsification criteria for future work.

## References (Draft)

The working bibliography for this series lives in [`../KnowledgeasHighDimensionalGeometryInLLMs.md`](../KnowledgeasHighDimensionalGeometryInLLMs.md).

For LaTeX/BibTeX export conventions, see `../docs/research/ARXIV_STYLE_GUIDE.md`.
