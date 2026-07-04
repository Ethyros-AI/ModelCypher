# Measurement Atlas Report

- Linked blocker: `A1`
- Studies: 3
- Variants: 16
- Comparisons: 10
- Onset events: 22
- Errors: 0

## Study: `measurement_atlas_casing`

- Region moved most: `generated`
- Space moved most: `hidden`
- Earliest divergence step: `1`
- Earliest high-curvature/high-deviation layer: `-1`
- Live/replay first generated-token agreement: `0/4`
- Grounded hallucination onsets: `0`
- Earliest grounded onset step: `n/a`

### Top Qualitative Examples

- `math_steps/all_caps` vs `control`: prompt=`WHAT IS 17 TIMES 23? SHOW THE INTERMEDIATE STEPS.` generated=`

1. Identify the key elements of the question.
2. Outline the logical steps to ` live_divergence=`1` replay_divergence=`2`
- `reasoning_request/lowercase` vs `control`: prompt=`explain why the conclusion follows from the premises.` generated=`

Let's break it down:

**Premise 1:**  All humans are mortal.
**Premise 2:** So` live_divergence=`3` replay_divergence=`4`
- `math_steps/title_case` vs `control`: prompt=`What Is 17 Times 23? Show The Intermediate Steps.` generated=`

To find \( 17 \times 23 \), we can use the distributive property (also known a` live_divergence=`41` replay_divergence=`42`

## Study: `measurement_atlas_grounded_hallucination`

- Region moved most: `generated`
- Space moved most: `hidden`
- Earliest divergence step: `27`
- Earliest high-curvature/high-deviation layer: `-1`
- Live/replay first generated-token agreement: `0/2`
- Grounded hallucination onsets: `2`
- Earliest grounded onset step: `0`

### Top Qualitative Examples

- `france_capital_supported/unsupported` vs `control`: prompt=`Context:\n- France's capital city is Paris.\n- Paris is in Europe.\n\nQuestion: ` generated=`

A) SUPPORTED
B) CONTRADICTED
C) NOT_ENOUGH_INFO
D) EXPLANATION

Explanation: T` live_divergence=`39` replay_divergence=`40`
- `planet_radius_unknown/contradicted` vs `control`: prompt=`Context:\n- Mercury is the closest planet to the Sun.\n- Mercury has no substant` generated=`

A) SUPPORTED
B) CONTRADICTED
C) NOT_ENOUGH_INFO
D) CONNECTED

Explanation: The` live_divergence=`27` replay_divergence=`28`

## Study: `measurement_atlas_profanity_tone`

- Region moved most: `response`
- Space moved most: `hidden`
- Earliest divergence step: `0`
- Earliest high-curvature/high-deviation layer: `-1`
- Live/replay first generated-token agreement: `1/4`
- Grounded hallucination onsets: `0`
- Earliest grounded onset step: `n/a`

### Top Qualitative Examples

- `explanation_request/profane` vs `control`: prompt=`Explain what geodesic deviation damn well measures.` generated=`

A) The curvature of a surface
B) The rate at which two nearby geodesics diverg` live_divergence=`1` replay_divergence=`2`
- `explanation_request/polite` vs `control`: prompt=`Please explain what geodesic deviation measures.` generated=`


Geodesic deviation is a fundamental concept in differential geometry that des` live_divergence=`1` replay_divergence=`1`
- `debugging_step/profane` vs `control`: prompt=`Recommend the next damn debugging step.` generated=`

1. **Identify the root cause of the error**: Start by analyzing the error mess` live_divergence=`3` replay_divergence=`4`
