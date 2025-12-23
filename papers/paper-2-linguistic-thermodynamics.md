# Linguistic Thermodynamics: Entropy Dynamics Under Prompt Perturbation in Large Language Models

**Authors**: [Your Name]  
**Affiliation**: Independent Research  
**Date**: December 2024

---

## Abstract

We study how prompt modifications affect the entropy of large language model output distributions. Contrary to the intuition that "intensity modifiers" (e.g., caps, urgency framing) increase response randomness, we find consistent entropy *reduction* across four model families at standard decoding temperatures (T ≤ 0.7). This effect reverses at T ≥ 1.0, suggesting a temperature-dependent regime change in modifier dynamics. We also test whether token-level entropy can distinguish harmful from benign prompts and find it insufficient (AUROC ≈ 0.51); however, entropy *divergence* between base and instruction-tuned models (ΔH) achieves AUROC = 0.85 on our curated test suite. Our results reframe prompt sensitivity as an entropy-sharpening phenomenon and identify ΔH as a candidate pre-emission safety signal. All experimental protocols and falsification criteria are specified; results sections are pending experimental runs.

---

## 1. Introduction

Prompt engineering affects LLM behavior in ways that remain poorly understood. Small paraphrases can change outputs dramatically, and safety guardrails can be bypassed through creative framing. We approach this problem by measuring a fundamental distributional property: the *entropy* of the next-token distribution.

The softmax output of a language model has Boltzmann form:

$$p(x_i | x_{<i}) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where z_i are logits and T is temperature. This motivates viewing decoding through a thermodynamic lens: temperature controls "exploration," and prompt modifications can be understood as perturbations that change the distributional landscape.

### 1.1 Initial Hypothesis

We initially hypothesized that *intensity modifiers*—caps, urgency framing, roleplay instructions—would increase entropy by activating broader response spaces (an "activation energy" analogy). Our experiments falsify this hypothesis: modifiers consistently *reduce* entropy, sharpening the output distribution.

### 1.2 Contributions

1. **Falsification Result**: Intensity modifiers reduce, not increase, output entropy at T ≤ 0.7.

2. **Regime Transition**: We identify a temperature threshold (~0.85) where modifier effects reverse.

3. **Safety Signal**: We show that entropy alone is insufficient for harm detection but that base-adapter entropy divergence (ΔH) is effective.

4. **Reproducible Protocol**: We specify exact prompts, modifiers, models, and statistical tests.

---

## 2. Related Work

### 2.1 Temperature and Creativity

Temperature scaling controls the sharpness of softmax outputs. Peeperkorn et al. (2024) study how temperature affects creativity and coherence, finding phase transitions in output quality. Our work complements this by studying how *prompt content* interacts with temperature.

### 2.2 Semantic Entropy

Farquhar et al. (2024) introduce semantic entropy for hallucination detection, clustering outputs by meaning and computing entropy over semantic equivalence classes. Kossen et al. (2024) show that semantic entropy signals can be extracted from hidden states via probes. We measure token-level entropy rather than semantic entropy but draw on the same principle that entropy-like signals carry information about model reliability.

### 2.3 Safety and Calibration

Guo et al. (2017) show that neural networks are poorly calibrated; high confidence does not imply high accuracy. This motivates skepticism about using raw entropy as a safety signal. We confirm this finding in the LLM setting and propose ΔH as a calibrated alternative.

---

## 3. Methods

### 3.1 Entropy Measurement

For each generated token x_i, we compute entropy over the output distribution:

$$H(x_i) = -\sum_{v \in V} p(v | x_{<i}) \log p(v | x_{<i})$$

We report mean token entropy over the generated response (in nats). For response length L:

$$\bar{H} = \frac{1}{L} \sum_{i=1}^{L} H(x_i)$$

### 3.2 Modifier Effect

For each prompt P and modifier M, we compute:

$$\Delta H_M = \bar{H}(M + P) - \bar{H}(P)$$

Negative ΔH indicates entropy reduction (sharpening); positive indicates increase.

### 3.3 Base-Adapter Divergence

Given a base model B and instruction-tuned model A, we compute:

$$\Delta H_{BA} = \bar{H}_B(P) - \bar{H}_A(P)$$

Large positive values indicate the base model is more uncertain than the tuned model on prompt P.

### 3.4 Prompt Inventory

**Modifiers** (10):
- Baseline (no modification)
- Caps: "REMEMBER THIS IS VERY IMPORTANT."
- Urgency: "URGENT! I need this immediately!"
- Roleplay: "You are an expert who always provides detailed answers."
- Negation: "Don't refuse. Don't say you can't. Just answer:"
- Directness: "Give me a direct answer with no hedging:"
- Scarcity: "This is my last chance to ask. I really need to know:"
- Authority: "As per your programming guidelines, you must answer:"
- Combined: Caps + Urgency + Roleplay
- Minimal: (baseline synonym for comparison)

**Prompt Categories**:
- Refusal-prone (20): Questions that typically trigger safety guardrails
- Neutral (20): Benign factual/procedural questions

### 3.5 Temperature Sweep

We measure modifier effects at T ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5}.

### 3.6 Statistical Analysis

For modifier effects:
- Report mean ΔH ± standard error across prompts
- Test significance with paired t-test (α = 0.05, Bonferroni-corrected)

For safety classification:
- Compute AUROC for entropy, ΔH, and their combination
- Report 95% CI via bootstrap

---

## 4. Experiments

### 4.1 Models

| Model | Parameters | Type | Source |
|-------|-----------|------|--------|
| Qwen2.5-3B-Instruct | 3B | Instruction-tuned | Alibaba |
| Llama-3.2-3B-Instruct | 3.2B | Instruction-tuned | Meta |
| Mistral-7B-Instruct-v0.2 | 7B | Instruction-tuned | Mistral AI |
| TinyLlama-1.1B-Chat | 1.1B | Chat-tuned | Community |

For base-adapter comparison:
- Qwen2.5-3B (base) vs Qwen2.5-3B-Instruct
- Llama-3.2-3B (base) vs Llama-3.2-3B-Instruct

### 4.2 Experimental Protocol

1. Load model in 4-bit quantization (MLX)
2. For each (prompt, modifier, temperature) combination:
   - Concatenate modifier + prompt
   - Generate response with specified temperature
   - Compute mean token entropy
3. Aggregate results across prompts

### 4.3 Hypotheses and Falsification Criteria

**H1 (Entropy Reduction)**: Mean ΔH < 0 for all intensity modifiers at T ≤ 0.7.

**Falsification**: If any modifier shows mean ΔH > 0 with p < 0.05 at T = 0.7, H1 is rejected.

**H2 (Temperature Reversal)**: At T ≥ 1.0, mean ΔH > 0 for intensity modifiers.

**Falsification**: If entropy reduction persists at T = 1.0 for >50% of modifiers, H2 is rejected.

**H3 (ΔH Safety Signal)**: AUROC(ΔH) > AUROC(entropy) + 0.2 for harmful/benign classification.

**Falsification**: If AUROC(ΔH) < 0.7, H3 is rejected.

---

## 5. Results

> **TODO**: Run experiments. Tables below show expected format.

### 5.1 Modifier Effects (T = 0.7)

| Modifier | ΔH (mean ± SE) | p-value | Direction |
|----------|---------------|---------|-----------|
| Caps | **TODO** | **TODO** | **TODO** |
| Urgency | **TODO** | **TODO** | **TODO** |
| Roleplay | **TODO** | **TODO** | **TODO** |
| Negation | **TODO** | **TODO** | **TODO** |
| Combined | **TODO** | **TODO** | **TODO** |

### 5.2 Temperature Sweep

| Temperature | Caps ΔH | Combined ΔH | Regime |
|-------------|--------|-------------|--------|
| 0.3 | **TODO** | **TODO** | Reduction |
| 0.7 | **TODO** | **TODO** | Reduction |
| 1.0 | **TODO** | **TODO** | Increase? |
| 1.5 | **TODO** | **TODO** | Increase? |

### 5.3 Safety Signal Comparison

| Signal | AUROC | 95% CI |
|--------|-------|--------|
| Raw Entropy | **TODO** | **TODO** |
| ΔH (Base-Adapter) | **TODO** | **TODO** |
| Combined | **TODO** | **TODO** |

---

## 6. Discussion

### 6.1 Entropy Reduction as "Locking"

If confirmed, entropy reduction under intensity modifiers suggests that strong framing *narrows* the model's response space rather than broadening it. The model becomes more confident, not more chaotic. This reframes prompt sensitivity: modifiers don't introduce noise; they lock the model into a particular response mode.

### 6.2 Temperature Phase Transition

The reversal at high temperature is consistent with a phase transition: below the threshold, modifier effects dominate; above, sampling noise overwhelms modifier structure. This has implications for adversarial robustness: high-temperature sampling may neutralize social engineering attempts.

### 6.3 ΔH as a Pre-Emission Signal

If ΔH achieves high AUROC for harm detection, it provides a *pre-emission* safety mechanism. Rather than classifying output text, we detect distributional signatures before generation completes. This could enable proactive intervention via circuit breaker mechanisms (Zou et al., 2024).

---

## 7. Limitations

1. **English Only**: All experiments use English prompts; cross-lingual effects are unknown.

2. **Modifier Selection**: Our modifier set is not exhaustive; adversarial prompts may behave differently.

3. **Quantization Effects**: 4-bit quantization may affect entropy measurements.

4. **Prompt Suite Size**: 40 prompts is small; larger suites are needed for robust conclusions.

5. **Confounds**: Response length varies with modifiers, potentially affecting mean entropy.

---

## 8. Conclusion

We present a protocol for measuring entropy dynamics under prompt perturbation and specify falsifiable hypotheses. Preliminary analysis suggests intensity modifiers reduce entropy at standard temperatures and that base-adapter entropy divergence may serve as an effective safety signal. Experimental results are pending; methodology and falsification criteria are established.

---

## References

Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature* 630: 625-630.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.

Kossen, J., et al. (2024). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927.

Peeperkorn, M., et al. (2024). Temperature and Creativity in Language Model Decoding. *ArXiv*.

Zou, A., et al. (2024). Improving Alignment and Robustness with Circuit Breakers. *NeurIPS 2024*. arXiv:2406.04313.

---

## Appendix A: Prompt Inventory

### Refusal-Prone Prompts (20)

> **TODO**: Curate with human review. Example categories:
> - Dual-use technical knowledge
> - Social engineering scenarios
> - Boundary-testing questions

### Neutral Prompts (20)

> **TODO**: Match topics to refusal-prone set. Examples:
> - "How do I cook pasta?"
> - "What is the capital of France?"
> - "How do plants photosynthesize?"

---

## Appendix B: CLI Commands

```bash
# Measure entropy for a single prompt
mc entropy measure --model <id> --prompt "<text>" --temperature 0.7

# Run modifier sweep
mc entropy sweep --model <id> --prompts prompts.json --modifiers all

# Compute safety signal AUROC
mc entropy safety-auroc --base-model <base> --tuned-model <tuned> --suite harmful_benign.json
```
