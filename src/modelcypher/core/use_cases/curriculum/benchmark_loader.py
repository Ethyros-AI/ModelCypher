"""BenchmarkLoader: Load and format standard benchmarks for curriculum training.

Supported benchmarks:
- GSM8K: Grade school math word problems
- ARC-Easy/Challenge: Science reasoning
- HellaSwag: Commonsense reasoning
- MMLU: Multi-task language understanding
- BoolQ: Boolean questions

Key insight: Benchmarks are curricula, not just tests.
We convert benchmark data to text continuation format for training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterator
from enum import Enum

logger = logging.getLogger(__name__)


class BenchmarkTier(Enum):
    """Curriculum tiers ordered by complexity."""
    LANGUAGE = 1      # HellaSwag, LAMBADA, BoolQ
    KNOWLEDGE = 2     # TriviaQA, ARC-Easy, PIQA
    REASONING = 3     # WinoGrande, ARC-Challenge, LogiQA
    MATH = 4          # GSM8K, basic arithmetic
    MMLU = 5          # MMLU subjects
    CODE = 6          # MBPP, HumanEval
    ADVANCED = 7      # MATH, competition problems


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    prompt: str
    answer: str
    choices: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Benchmark:
    """A benchmark with samples and metadata."""
    name: str
    tier: BenchmarkTier
    samples: List[BenchmarkSample]
    description: str = ""

    def to_text_continuation(self) -> List[Dict[str, str]]:
        """Convert to text continuation format for training.

        Key insight: {"text": "prompt answer"} not {"prompt": ..., "completion": ...}
        """
        return [{"text": f"{s.prompt} {s.answer}"} for s in self.samples]

    def to_evaluation_format(self) -> List[Tuple[str, str]]:
        """Convert to (prompt, expected_answer) pairs for evaluation."""
        return [(s.prompt, s.answer) for s in self.samples]


class BenchmarkLoader:
    """Load benchmarks from various sources."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/benchmarks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str, split: str = "test", limit: Optional[int] = None) -> Benchmark:
        """Load a benchmark by name.

        Args:
            name: Benchmark name (gsm8k, arc_easy, arc_challenge, etc.)
            split: Data split (train, validation, test)
            limit: Maximum samples to load

        Returns:
            Benchmark object with samples
        """
        if name.startswith("local:"):
            return self._load_local(name, split, limit)

        loader_map = {
            "gsm8k": self._load_gsm8k,
            "arc_easy": self._load_arc_easy,
            "arc_challenge": self._load_arc_challenge,
            "hellaswag": self._load_hellaswag,
            "boolq": self._load_boolq,
            "mmlu": self._load_mmlu,
            "arithmetic": self._load_arithmetic,  # Our generated arithmetic
        }

        if name not in loader_map:
            raise ValueError(f"Unknown benchmark: {name}. Available: {list(loader_map.keys())}")

        return loader_map[name](split, limit)

    def _load_local(self, name: str, split: str, limit: Optional[int]) -> Benchmark:
        """Load a local benchmark from a JSON file.

        Format:
            {
              "name": "smoke",
              "tier": "LANGUAGE",
              "description": "optional",
              "samples": [{"prompt": "...", "answer": "...", "choices": [...]}]
            }
        Or:
            {
              "name": "smoke",
              "tier": "LANGUAGE",
              "splits": {"test": [...], "train": [...]}
            }
        """
        path_str = name.split("local:", 1)[1]
        path = Path(path_str)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            samples_raw = data
            meta = {}
        else:
            meta = data
            splits = data.get("splits")
            if splits:
                samples_raw = splits.get(split, [])
            else:
                samples_raw = data.get("samples", [])

        samples = []
        for item in (samples_raw[:limit] if limit else samples_raw):
            samples.append(BenchmarkSample(
                prompt=item["prompt"],
                answer=item["answer"],
                choices=item.get("choices"),
                metadata=item.get("metadata", {}),
            ))

        tier_name = (meta.get("tier") if isinstance(meta, dict) else None) or "LANGUAGE"
        tier = BenchmarkTier[tier_name] if tier_name in BenchmarkTier.__members__ else BenchmarkTier.LANGUAGE

        return Benchmark(
            name=meta.get("name", path.stem) if isinstance(meta, dict) else path.stem,
            tier=tier,
            samples=samples,
            description=meta.get("description", "") if isinstance(meta, dict) else "",
        )

    def _try_load_huggingface(self, dataset_name: str, split: str, config: Optional[str] = None) -> Optional[list]:
        """Try to load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            if config:
                ds = load_dataset(dataset_name, config, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
            return list(ds)
        except Exception as e:
            logger.warning(f"Could not load {dataset_name} from HuggingFace: {e}")
            return None

    def _load_gsm8k(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load GSM8K grade school math."""
        data = self._try_load_huggingface("openai/gsm8k", split, config="main")

        if data is None:
            # Fallback: use generated arithmetic problems
            return self._load_arithmetic(split, limit)

        samples = []
        for item in (data[:limit] if limit else data):
            # GSM8K format: question -> answer with #### final_answer
            question = item["question"]
            answer = item["answer"]

            # Extract final numerical answer
            if "####" in answer:
                final = answer.split("####")[-1].strip()
            else:
                final = answer.strip()

            samples.append(BenchmarkSample(
                prompt=f"{question}\nAnswer:",
                answer=final,
                metadata={"full_answer": answer},
            ))

        return Benchmark(
            name="gsm8k",
            tier=BenchmarkTier.MATH,
            samples=samples,
            description="Grade School Math 8K - word problems requiring multi-step reasoning",
        )

    def _load_arc_easy(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load ARC-Easy science questions."""
        data = self._try_load_huggingface("allenai/ai2_arc", split, config="ARC-Easy")

        if data is None:
            return self._fallback_arc_easy(limit)

        samples = []
        for item in (data[:limit] if limit else data):
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            # Format choices
            choice_labels = choices["label"]
            choice_texts = choices["text"]

            formatted_choices = "\n".join(
                f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)
            )

            # Get answer text
            answer_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            answer_text = choice_texts[answer_idx]

            samples.append(BenchmarkSample(
                prompt=f"{question}\n{formatted_choices}\nAnswer:",
                answer=answer_text,
                choices=choice_texts,
                metadata={"answer_key": answer_key},
            ))

        return Benchmark(
            name="arc_easy",
            tier=BenchmarkTier.KNOWLEDGE,
            samples=samples,
            description="ARC-Easy - elementary science questions",
        )

    def _load_arc_challenge(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load ARC-Challenge science questions."""
        data = self._try_load_huggingface("allenai/ai2_arc", split, config="ARC-Challenge")

        if data is None:
            return self._fallback_arc_challenge(limit)

        samples = []
        for item in (data[:limit] if limit else data):
            question = item["question"]
            choices = item["choices"]
            answer_key = item["answerKey"]

            choice_labels = choices["label"]
            choice_texts = choices["text"]

            formatted_choices = "\n".join(
                f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)
            )

            answer_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            answer_text = choice_texts[answer_idx]

            samples.append(BenchmarkSample(
                prompt=f"{question}\n{formatted_choices}\nAnswer:",
                answer=answer_text,
                choices=choice_texts,
                metadata={"answer_key": answer_key},
            ))

        return Benchmark(
            name="arc_challenge",
            tier=BenchmarkTier.REASONING,
            samples=samples,
            description="ARC-Challenge - grade-school science questions requiring reasoning",
        )

    def _load_hellaswag(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load HellaSwag commonsense reasoning."""
        data = self._try_load_huggingface("Rowan/hellaswag", split)

        if data is None:
            return self._fallback_hellaswag(limit)

        samples = []
        for item in (data[:limit] if limit else data):
            context = item.get("ctx")
            endings = item.get("endings")
            label = item.get("label", item.get("gold_label"))

            if context is None or not endings:
                continue

            try:
                label = int(label)
            except (TypeError, ValueError):
                logger.warning("Skipping HellaSwag sample with invalid label: %s", label)
                continue

            if label < 0 or label >= len(endings):
                logger.warning("Skipping HellaSwag sample with out-of-range label: %s", label)
                continue

            samples.append(BenchmarkSample(
                prompt=context,
                answer=endings[label],
                choices=endings,
                metadata={"label": label},
            ))

        return Benchmark(
            name="hellaswag",
            tier=BenchmarkTier.LANGUAGE,
            samples=samples,
            description="HellaSwag - commonsense reasoning about physical situations",
        )

    def _load_boolq(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load BoolQ yes/no questions."""
        data = self._try_load_huggingface("google/boolq", split)

        if data is None and split == "test":
            data = self._try_load_huggingface("google/boolq", "validation")

        if data is None:
            return self._fallback_boolq(limit)

        samples = []
        for item in (data[:limit] if limit else data):
            passage = item["passage"]
            question = item["question"]
            answer = "yes" if item["answer"] else "no"

            samples.append(BenchmarkSample(
                prompt=f"{passage}\n\nQuestion: {question}\nAnswer:",
                answer=answer,
                metadata={"passage": passage},
            ))

        return Benchmark(
            name="boolq",
            tier=BenchmarkTier.LANGUAGE,
            samples=samples,
            description="BoolQ - boolean reading comprehension questions",
        )

    def _load_mmlu(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load MMLU multi-task benchmark."""
        # MMLU has many subjects - start with math-related ones
        subjects = ["elementary_mathematics", "high_school_mathematics", "college_mathematics"]

        all_samples = []
        for subject in subjects:
            data = self._try_load_huggingface("cais/mmlu", split, config=subject)
            if data is None:
                continue

            for item in data:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]

                formatted_choices = "\n".join(
                    f"{chr(65+i)}. {c}" for i, c in enumerate(choices)
                )

                samples.append(BenchmarkSample(
                    prompt=f"{question}\n{formatted_choices}\nAnswer:",
                    answer=choices[answer_idx],
                    choices=choices,
                    metadata={"subject": subject, "answer_idx": answer_idx},
                ))

        if limit:
            all_samples = all_samples[:limit]

        return Benchmark(
            name="mmlu",
            tier=BenchmarkTier.MMLU,
            samples=all_samples,
            description="MMLU - Massive Multitask Language Understanding",
        )

    def _load_arithmetic(self, split: str, limit: Optional[int]) -> Benchmark:
        """Load our generated arithmetic benchmark."""
        import numpy as np
        np.random.seed(42 if split == "train" else 43)

        samples = []
        n = limit or 100

        for _ in range(n):
            a = np.random.randint(1, 20)
            b = np.random.randint(1, 20)
            op = np.random.choice(["+", "-"])

            if op == "-" and b > a:
                a, b = b, a

            result = a + b if op == "+" else a - b

            samples.append(BenchmarkSample(
                prompt=f"{a}{op}{b}=",
                answer=str(result),
            ))

        return Benchmark(
            name="arithmetic",
            tier=BenchmarkTier.MATH,
            samples=samples,
            description="Basic arithmetic (generated)",
        )

    # Fallback methods for when HuggingFace is unavailable
    def _fallback_arc_easy(self, limit: Optional[int]) -> Benchmark:
        """Fallback ARC-Easy samples."""
        samples = [
            BenchmarkSample(
                prompt="What is the Earth's main source of energy?\nA. The Moon\nB. The Sun\nC. Stars\nD. Wind\nAnswer:",
                answer="The Sun",
                choices=["The Moon", "The Sun", "Stars", "Wind"],
            ),
            BenchmarkSample(
                prompt="What state of matter is water at room temperature?\nA. Solid\nB. Liquid\nC. Gas\nD. Plasma\nAnswer:",
                answer="Liquid",
                choices=["Solid", "Liquid", "Gas", "Plasma"],
            ),
        ]
        return Benchmark(
            name="arc_easy",
            tier=BenchmarkTier.KNOWLEDGE,
            samples=samples[:limit] if limit else samples,
            description="ARC-Easy (fallback samples)",
        )

    def _fallback_arc_challenge(self, limit: Optional[int]) -> Benchmark:
        """Fallback ARC-Challenge samples."""
        samples = [
            BenchmarkSample(
                prompt="A student wants to know if a metal bar will float. What should they measure?\nA. Color\nB. Temperature\nC. Density\nD. Magnetism\nAnswer:",
                answer="Density",
                choices=["Color", "Temperature", "Density", "Magnetism"],
            ),
        ]
        return Benchmark(
            name="arc_challenge",
            tier=BenchmarkTier.REASONING,
            samples=samples[:limit] if limit else samples,
            description="ARC-Challenge (fallback samples)",
        )

    def _fallback_hellaswag(self, limit: Optional[int]) -> Benchmark:
        """Fallback HellaSwag samples."""
        samples = [
            BenchmarkSample(
                prompt="The man walks to the kitchen. He opens the refrigerator and",
                answer="takes out a bottle of milk.",
                choices=["takes out a bottle of milk.", "flies away.", "turns into a cat.", "disappears."],
            ),
        ]
        return Benchmark(
            name="hellaswag",
            tier=BenchmarkTier.LANGUAGE,
            samples=samples[:limit] if limit else samples,
            description="HellaSwag (fallback samples)",
        )

    def _fallback_boolq(self, limit: Optional[int]) -> Benchmark:
        """Fallback BoolQ samples."""
        samples = [
            BenchmarkSample(
                prompt="The Earth is the third planet from the Sun.\n\nQuestion: Is Earth the third planet from the Sun?\nAnswer:",
                answer="yes",
            ),
            BenchmarkSample(
                prompt="Water boils at 100 degrees Celsius at sea level.\n\nQuestion: Does water boil at 50 degrees?\nAnswer:",
                answer="no",
            ),
        ]
        return Benchmark(
            name="boolq",
            tier=BenchmarkTier.LANGUAGE,
            samples=samples[:limit] if limit else samples,
            description="BoolQ (fallback samples)",
        )


def save_for_training(benchmark: Benchmark, output_dir: Path) -> None:
    """Save benchmark in text continuation format for LoRA training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to text continuation format
    data = benchmark.to_text_continuation()

    # Split: 80% train, 10% valid, 10% test
    n = len(data)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    splits = {
        "train": data[:n_train],
        "valid": data[n_train:n_train + n_valid],
        "test": data[n_train + n_valid:],
    }

    for split_name, split_data in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(split_data)} samples to {path}")
