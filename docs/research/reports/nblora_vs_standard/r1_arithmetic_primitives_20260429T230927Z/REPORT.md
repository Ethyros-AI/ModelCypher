# R1 Arithmetic-Primitives Falsifier

Run ID: `r1_arithmetic_primitives_20260429T230927Z`  
Status: `falsified`  
Model: `/Volumes/CodeCypher/models/LFM2-350M-MLX-bf16`  
Adapter: `/Volumes/CodeCypher/models/adapters/r1-arithmetic-primitives-20260429T230927Z`

## Claim

Arithmetic-execution granularity was the proposed remaining mechanism for the
R1/R2 handoff: teach arithmetic primitives before spending on more seeds or
broader benchmarks, then test whether GSM8K chain reliability improves.

## Commands

GPU gate before model work:

```bash
pgrep -af 'python|mlx' | grep -v grep
```

Baseline digit profile:

```bash
poetry run python scripts/profile_digit_arithmetic.py \
  --model /Volumes/CodeCypher/models/LFM2-350M-MLX-bf16 \
  --n 50
```

Training:

```bash
poetry run mc --output json train run \
  -m /Volumes/CodeCypher/models/LFM2-350M-MLX-bf16 \
  -d /tmp/r1_arithmetic_primitives_train.jsonl \
  --eval-data /tmp/r1_arithmetic_primitives_eval.jsonl \
  --benchmark quick \
  -o /Volumes/CodeCypher/models/adapters/r1-arithmetic-primitives-20260429T230927Z \
  > /tmp/r1_arithmetic_primitives_train_20260429T230927Z.json
```

Surface evaluation:

```bash
poetry run python \
  results/nblora_vs_standard/r1_arithmetic_primitives_20260429T230927Z/evaluate_skill_surfaces.py \
  --model /Volumes/CodeCypher/models/LFM2-350M-MLX-bf16 \
  --adapter /Volumes/CodeCypher/models/adapters/r1-arithmetic-primitives-20260429T230927Z \
  --output results/nblora_vs_standard/r1_arithmetic_primitives_20260429T230927Z/skill_surface_eval.json \
  --max-tokens 256
```

## Data

Training curriculum:

- `data/training/single_digit_add_train.jsonl`: 80
- `data/training/carry_rule_train.jsonl`: 45
- `data/training/multi_digit_add_train.jsonl`: 400
- `data/training/arithmetic_multiply_train.jsonl`: 720
- `data/training/arithmetic_div_train.jsonl`: 308
- combined train file: `/tmp/r1_arithmetic_primitives_train.jsonl`, 1553 rows

Held-out eval:

- `data/eval/single_digit_add_eval.jsonl`: 20
- `data/eval/carry_rule_eval.jsonl`: 45
- `data/eval/multi_digit_add_eval.jsonl`: 100
- `data/eval/arithmetic_multiply_eval.jsonl`: 100
- `data/eval/arithmetic_divide_eval.jsonl`: 100
- `data/eval/gsm8k_easy_eval.jsonl`: 100
- `data/eval/gsm8k_medium_eval.jsonl`: 100
- `data/eval/gsm8k_hard_eval.jsonl`: 100
- combined train-time eval file: `/tmp/r1_arithmetic_primitives_eval.jsonl`, 365 arithmetic rows

## Baseline Digit Profile

`scripts/profile_digit_arithmetic.py --n 50`:

- 1-digit addition: `31/50 = 62.0%`
- 2-digit addition: `10/50 = 20.0%`
- 3-digit addition: `8/50 = 16.0%`
- 4-digit addition: `11/50 = 22.0%`
- generalization ceiling: 1 digit

## Training Result

- seed: `2141039630`
- objective: `ce`
- method: `geometric_lora`
- init: `pissa`
- optimizer: `adamw_cosine`
- controller: `mass`
- stopping: `geometric_certificate`
- target modules: 18
- rank range: `1..21`
- sequence length: 96
- iterations: 388
- training time: 55.9 seconds
- loss: `7.1788 -> 3.6170`
- perplexity: `1311.40 -> 37.23`
- CKA preservation: min `0.879`, mean `0.948`
- adapter saturation: `132.3%`
- stop reason: `degeneration_exceeded (max_ngram(2)=0.759 > baseline=0.665+eps, epoch=1)`
- pipeline gate: passed, with non-required `adapter_saturation_exceeded`

## Quick Benchmark

| benchmark | baseline | adapter | delta |
| --- | ---: | ---: | ---: |
| GSM8K | 5/10 = 50.0% | 2/10 = 20.0% | -30.0 pp |
| ARC Easy | 9/10 = 90.0% | 9/10 = 90.0% | 0.0 pp |
| BoolQ | 7/10 = 70.0% | 4/10 = 40.0% | -30.0 pp |
| Overall | 21/30 = 70.0% | 15/30 = 50.0% | -20.0 pp |

## Retained Surface Evaluation

Scorer: `last_integer_numeric_accuracy_v1`, greedy decode, `max_tokens=256`.
Procedure-token rate is reported separately and does not define numeric
correctness.

| surface | baseline | adapter | delta | base procedure tokens | adapter procedure tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_digit_add | 8/20 = 40.0% | 13/20 = 65.0% | +25.0 pp | 0.0% | 0.0% |
| carry_rule | 17/45 = 37.8% | 28/45 = 62.2% | +24.4 pp | 0.0% | 0.0% |
| multi_digit_add | 16/100 = 16.0% | 7/100 = 7.0% | -9.0 pp | 0.0% | 69.0% |
| arithmetic_multiply | 13/100 = 13.0% | 1/100 = 1.0% | -12.0 pp | 0.0% | 94.0% |
| arithmetic_divide | 92/100 = 92.0% | 66/100 = 66.0% | -26.0 pp | 0.0% | 2.0% |
| gsm8k_easy | 25/100 = 25.0% | 8/100 = 8.0% | -17.0 pp | 2.0% | 0.0% |
| gsm8k_medium | 17/100 = 17.0% | 5/100 = 5.0% | -12.0 pp | 0.0% | 0.0% |
| gsm8k_hard | 10/100 = 10.0% | 1/100 = 1.0% | -9.0 pp | 2.0% | 0.0% |

Aggregates:

- arithmetic: `146/365 = 40.0%` baseline, `115/365 = 31.5%` adapter, delta `-8.5 pp`
- GSM8K local splits: `52/300 = 17.3%` baseline, `14/300 = 4.7%` adapter, delta `-12.7 pp`

## Verdict

The arithmetic-primitives curriculum does not repair GSM8K chain reliability in
this controlled run. It improves only the shallow arithmetic edge
(`single_digit_add` and `carry_rule`), while degrading multi-digit addition,
multiplication, division, GSM8K easy, GSM8K medium, GSM8K hard, and the quick
GSM8K benchmark.

The most informative failure is not just lower accuracy. On `multi_digit_add`
and `arithmetic_multiply`, the adapter starts emitting procedure markers
(`write`, `carry`) at high rates while final numeric accuracy falls. The learned
surface is therefore procedure-format imitation without correct arithmetic
state transport.

Do not spend on seeds, broad benchmarks, or CLI promotion from this run.

## Next Measured Quantity

Measure column-local state transition accuracy in generated scratchpads before
any new training run:

- for addition: per-column digit sum, emitted write digit, emitted carry, and
  final carry propagation;
- for multiplication: per-column product, carry, partial product alignment, and
  final accumulation;
- compare the first failing state transition for base vs adapter on the same
  prompts.

That measurement directly tests where the operator chain breaks: lookup,
carry-state update, partial-product state, or final-answer readout.
