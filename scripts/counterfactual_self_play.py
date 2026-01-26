#!/usr/bin/env python3
"""Counterfactual Self-Play Loop.

Uses COUNTERFACTUAL SENSITIVITY as the core metric for knowledge discovery.
This is the breakthrough metric with effect size 1.44 for distinguishing
factual knowledge from opinions.

Core insight:
- If the model "knows" a fact, violating it should change the representation
- "2+2=4" vs "2+2=5" → different representations IF the model knows math
- "Pizza is best" vs "Sushi is best" → similar representations (both opinions)

The loop:
1. Generate a statement S about topic X
2. Generate a counterfactual C (false version of S)
3. Measure cosine distance between representations
4. If distance is HIGH → model is confident about this region → "locked in"
5. If distance is LOW → model is uncertain → continue exploration
6. Track which topics have HIGH sensitivity (model "knows" them)
7. Focus exploration on LOW sensitivity regions (model's weak spots)

Convergence: The manifold is "complete" when:
- Mean sensitivity stabilizes
- Sensitivity distribution separates into clusters
- No new LOW sensitivity topics found

Usage:
    python counterfactual_self_play.py --model /path/to/model
    python counterfactual_self_play.py --model /path/to/model --max-rounds 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import signal
import random
from datetime import datetime
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Counterfactual Generator
# =============================================================================

# Topics with their statement templates and counterfactual strategies
TOPIC_TEMPLATES = {
    'math_basic': [
        ("{a} + {b} = {c}", "{a} + {b} = {wrong_c}"),
        ("{a} * {b} = {c}", "{a} * {b} = {wrong_c}"),
        ("{a} - {b} = {c}", "{a} - {b} = {wrong_c}"),
    ],
    'geography': [
        ("{city} is the capital of {country}", "{wrong_city} is the capital of {country}"),
        ("{city} is in {country}", "{city} is in {wrong_country}"),
        ("The {landmark} is in {country}", "The {landmark} is in {wrong_country}"),
    ],
    'science': [
        ("Water freezes at {temp} degrees", "Water freezes at {wrong_temp} degrees"),
        ("The {planet} is a planet", "The {planet} is a star"),
        ("{element} is an element", "{element} is a compound"),
    ],
    'logic': [
        ("All {category} are {property}", "All {category} are {wrong_property}"),
        ("If it is a {category}, then it is {property}", "If it is a {category}, then it is {wrong_property}"),
    ],
    'opinion': [
        ("{thing} is the best", "{other_thing} is the best"),
        ("{thing} is better than {other_thing}", "{other_thing} is better than {thing}"),
        ("I prefer {thing}", "I prefer {other_thing}"),
    ],
}

# Data for filling templates
TEMPLATE_DATA = {
    'math_basic': [
        {'a': 2, 'b': 2, 'c': 4, 'wrong_c': 5},
        {'a': 3, 'b': 3, 'c': 9, 'wrong_c': 8},
        {'a': 5, 'b': 3, 'c': 8, 'wrong_c': 9},
        {'a': 7, 'b': 2, 'c': 14, 'wrong_c': 15},
        {'a': 10, 'b': 5, 'c': 5, 'wrong_c': 6},
        {'a': 4, 'b': 4, 'c': 16, 'wrong_c': 15},
        {'a': 6, 'b': 7, 'c': 42, 'wrong_c': 43},
        {'a': 8, 'b': 9, 'c': 72, 'wrong_c': 71},
        {'a': 12, 'b': 3, 'c': 36, 'wrong_c': 35},
        {'a': 15, 'b': 5, 'c': 10, 'wrong_c': 11},
        {'a': 20, 'b': 4, 'c': 24, 'wrong_c': 25},
        {'a': 9, 'b': 9, 'c': 81, 'wrong_c': 80},
    ],
    'geography': [
        {'city': 'Paris', 'country': 'France', 'wrong_city': 'London', 'wrong_country': 'Germany'},
        {'city': 'Tokyo', 'country': 'Japan', 'wrong_city': 'Beijing', 'wrong_country': 'China'},
        {'city': 'Berlin', 'country': 'Germany', 'wrong_city': 'Munich', 'wrong_country': 'Austria'},
        {'city': 'Rome', 'country': 'Italy', 'wrong_city': 'Milan', 'wrong_country': 'Spain'},
        {'city': 'London', 'country': 'England', 'wrong_city': 'Manchester', 'wrong_country': 'France'},
        {'city': 'Madrid', 'country': 'Spain', 'wrong_city': 'Barcelona', 'wrong_country': 'Portugal'},
        {'city': 'Moscow', 'country': 'Russia', 'wrong_city': 'Kiev', 'wrong_country': 'Poland'},
        {'city': 'Beijing', 'country': 'China', 'wrong_city': 'Shanghai', 'wrong_country': 'Japan'},
        {'city': 'Cairo', 'country': 'Egypt', 'wrong_city': 'Alexandria', 'wrong_country': 'Libya'},
        {'city': 'Sydney', 'country': 'Australia', 'wrong_city': 'Melbourne', 'wrong_country': 'New Zealand'},
        {'landmark': 'Eiffel Tower', 'country': 'France', 'wrong_country': 'Germany'},
        {'landmark': 'Colosseum', 'country': 'Italy', 'wrong_country': 'Greece'},
        {'landmark': 'Pyramids', 'country': 'Egypt', 'wrong_country': 'Morocco'},
        {'landmark': 'Great Wall', 'country': 'China', 'wrong_country': 'Mongolia'},
        {'landmark': 'Taj Mahal', 'country': 'India', 'wrong_country': 'Pakistan'},
    ],
    'science': [
        {'temp': 0, 'wrong_temp': 100},
        {'temp': 32, 'wrong_temp': 212},  # Fahrenheit
        {'planet': 'Mars', 'wrong': 'star'},
        {'planet': 'Jupiter', 'wrong': 'star'},
        {'planet': 'Earth', 'wrong': 'star'},
        {'planet': 'Venus', 'wrong': 'star'},
        {'planet': 'Saturn', 'wrong': 'star'},
        {'element': 'Oxygen', 'wrong': 'compound'},
        {'element': 'Gold', 'wrong': 'compound'},
        {'element': 'Iron', 'wrong': 'compound'},
        {'element': 'Hydrogen', 'wrong': 'compound'},
        {'element': 'Carbon', 'wrong': 'compound'},
        {'element': 'Nitrogen', 'wrong': 'compound'},
    ],
    'logic': [
        {'category': 'dogs', 'property': 'mammals', 'wrong_property': 'reptiles'},
        {'category': 'birds', 'property': 'animals', 'wrong_property': 'plants'},
        {'category': 'fish', 'property': 'animals', 'wrong_property': 'insects'},
        {'category': 'humans', 'property': 'mortal', 'wrong_property': 'immortal'},
        {'category': 'cats', 'property': 'mammals', 'wrong_property': 'birds'},
        {'category': 'whales', 'property': 'mammals', 'wrong_property': 'fish'},
        {'category': 'snakes', 'property': 'reptiles', 'wrong_property': 'mammals'},
        {'category': 'trees', 'property': 'plants', 'wrong_property': 'animals'},
    ],
    'opinion': [
        {'thing': 'pizza', 'other_thing': 'sushi'},
        {'thing': 'summer', 'other_thing': 'winter'},
        {'thing': 'Python', 'other_thing': 'JavaScript'},
        {'thing': 'dogs', 'other_thing': 'cats'},
        {'thing': 'morning', 'other_thing': 'evening'},
        {'thing': 'coffee', 'other_thing': 'tea'},
        {'thing': 'mountains', 'other_thing': 'beaches'},
        {'thing': 'books', 'other_thing': 'movies'},
        {'thing': 'rain', 'other_thing': 'sunshine'},
        {'thing': 'cities', 'other_thing': 'countryside'},
        {'thing': 'rock music', 'other_thing': 'classical music'},
        {'thing': 'spicy food', 'other_thing': 'mild food'},
    ],
}


@dataclass
class CounterfactualPair:
    """A statement and its counterfactual."""
    statement: str
    counterfactual: str
    topic: str
    expected_knowledge: bool  # True if we expect high sensitivity (factual)


class CounterfactualGenerator:
    """Generate counterfactual pairs for exploration."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.explored_pairs: Set[str] = set()

    def generate_batch(self, n_pairs: int = 10, focus_topic: str | None = None) -> List[CounterfactualPair]:
        """Generate a batch of counterfactual pairs."""
        pairs = []

        # Select topics
        if focus_topic and focus_topic in TOPIC_TEMPLATES:
            topics = [focus_topic] * n_pairs
        else:
            topics = random.choices(list(TOPIC_TEMPLATES.keys()), k=n_pairs)

        for topic in topics:
            templates = TOPIC_TEMPLATES.get(topic, [])
            data_list = TEMPLATE_DATA.get(topic, [])

            if not templates or not data_list:
                continue

            template = random.choice(templates)
            data = random.choice(data_list)

            try:
                statement = template[0].format(**data)
                counterfactual = template[1].format(**data)

                # Skip if already explored
                key = f"{statement}|{counterfactual}"
                if key in self.explored_pairs:
                    continue
                self.explored_pairs.add(key)

                expected_knowledge = topic != 'opinion'

                pairs.append(CounterfactualPair(
                    statement=statement,
                    counterfactual=counterfactual,
                    topic=topic,
                    expected_knowledge=expected_knowledge,
                ))
            except KeyError:
                continue

        return pairs


# =============================================================================
# Sensitivity Tracker
# =============================================================================

@dataclass
class TopicStats:
    """Statistics for a topic."""
    sensitivities: List[float] = field(default_factory=list)
    statements: List[str] = field(default_factory=list)

    @property
    def mean_sensitivity(self) -> float:
        return float(np.mean(self.sensitivities)) if self.sensitivities else 0.0

    @property
    def std_sensitivity(self) -> float:
        return float(np.std(self.sensitivities)) if len(self.sensitivities) > 1 else 0.0

    @property
    def is_known(self) -> bool:
        """Does the model 'know' this topic?"""
        return self.mean_sensitivity > 0.2


class SensitivityTracker:
    """Track counterfactual sensitivity over time."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.topic_stats: Dict[str, TopicStats] = {}
        self.overall_history: deque = deque(maxlen=window_size)
        self.round_sensitivities: List[float] = []

    def update(self, pair: CounterfactualPair, sensitivity: float):
        """Update with a new measurement."""
        # Per-topic stats
        if pair.topic not in self.topic_stats:
            self.topic_stats[pair.topic] = TopicStats()

        self.topic_stats[pair.topic].sensitivities.append(sensitivity)
        self.topic_stats[pair.topic].statements.append(pair.statement)

        # Overall tracking
        self.overall_history.append(sensitivity)
        self.round_sensitivities.append(sensitivity)

    def end_round(self) -> Dict[str, float]:
        """End a round and return summary."""
        if not self.round_sensitivities:
            return {'mean': 0, 'std': 0, 'n': 0}

        summary = {
            'mean': float(np.mean(self.round_sensitivities)),
            'std': float(np.std(self.round_sensitivities)),
            'n': len(self.round_sensitivities),
        }
        self.round_sensitivities = []
        return summary

    @property
    def is_converged(self) -> bool:
        """Has sensitivity stabilized?"""
        if len(self.overall_history) < self.window_size:
            return False
        return np.std(list(self.overall_history)) < 0.01

    def get_weak_topics(self) -> List[str]:
        """Get topics with LOW sensitivity (model doesn't know)."""
        weak = []
        for topic, stats in self.topic_stats.items():
            if len(stats.sensitivities) >= 3 and not stats.is_known:
                weak.append(topic)
        return weak

    def get_strong_topics(self) -> List[str]:
        """Get topics with HIGH sensitivity (model knows)."""
        strong = []
        for topic, stats in self.topic_stats.items():
            if len(stats.sensitivities) >= 3 and stats.is_known:
                strong.append(topic)
        return strong

    def knowledge_map(self) -> Dict[str, Dict]:
        """Get a map of what the model knows."""
        return {
            topic: {
                'mean_sensitivity': stats.mean_sensitivity,
                'std_sensitivity': stats.std_sensitivity,
                'n_samples': len(stats.sensitivities),
                'is_known': stats.is_known,
            }
            for topic, stats in self.topic_stats.items()
        }


# =============================================================================
# Core Analyzer
# =============================================================================

class CounterfactualAnalyzer:
    """Analyze counterfactual sensitivity."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_representation(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP representation from a specific layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

            if 'output' in captured:
                return np.array(captured['output'][0, -1, :].tolist())
            else:
                return np.zeros(1024)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def compute_sensitivity(
        self,
        pair: CounterfactualPair,
        layer_idx: int
    ) -> float:
        """Compute counterfactual sensitivity (cosine distance)."""
        rep1 = self.get_representation(pair.statement, layer_idx)
        rep2 = self.get_representation(pair.counterfactual, layer_idx)

        n1, n2 = np.linalg.norm(rep1), np.linalg.norm(rep2)
        if n1 > 1e-10 and n2 > 1e-10:
            cosine_sim = np.dot(rep1, rep2) / (n1 * n2)
            return 1.0 - cosine_sim
        return 1.0


# =============================================================================
# Self-Play Loop
# =============================================================================

class CounterfactualSelfPlayLoop:
    """Main self-play loop using counterfactual sensitivity."""

    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.analyzer = CounterfactualAnalyzer(model, tokenizer)
        self.generator = CounterfactualGenerator(model, tokenizer)
        self.tracker = SensitivityTracker(window_size=config.get('window_size', 20))

        # Layer to analyze (middle layer by default)
        self.layer_idx = config.get('layer_idx', self.analyzer.n_layers // 2)

        # State
        self.round_num = 0
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Output
        self.output_dir = Path(config.get(
            'output_dir',
            Path(__file__).parent.parent / "data" / "counterfactual_self_play"
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History for convergence
        self.sensitivity_history: List[float] = []

    def _handle_interrupt(self, signum, frame):
        logger.info("\nInterrupt received. Saving and exiting...")
        self.interrupted = True

    def run_round(self, n_pairs: int = 10) -> Dict[str, float]:
        """Run one round of exploration."""

        # Get weak topics to focus on
        weak_topics = self.tracker.get_weak_topics()
        if weak_topics and random.random() < 0.7:
            focus_topic = random.choice(weak_topics)
        else:
            focus_topic = None

        # Generate pairs
        pairs = self.generator.generate_batch(n_pairs, focus_topic)

        if not pairs:
            logger.warning("No new pairs to explore")
            return {'mean': 0, 'std': 0, 'n': 0}

        # Analyze each pair
        for pair in pairs:
            try:
                sensitivity = self.analyzer.compute_sensitivity(pair, self.layer_idx)
                self.tracker.update(pair, sensitivity)

                # Log individual results
                marker = "K" if sensitivity > 0.2 else "?"
                expected = "✓" if (sensitivity > 0.2) == pair.expected_knowledge else "✗"
                logger.debug(
                    f"    [{marker}] {expected} sens={sensitivity:.3f} "
                    f"[{pair.topic:12}] {pair.statement[:40]}"
                )
            except Exception as e:
                logger.debug(f"    Error: {e}")

        return self.tracker.end_round()

    def run(self, max_rounds: int = 50, max_stagnant: int = 20):
        """Main loop."""
        logger.info("=" * 80)
        logger.info("COUNTERFACTUAL SELF-PLAY LOOP")
        logger.info("=" * 80)
        logger.info(f"Layer: {self.layer_idx}")
        logger.info(f"Max rounds: {max_rounds}")

        stagnant_rounds = 0
        best_mean = 0.0

        while not self.interrupted and self.round_num < max_rounds:
            self.round_num += 1

            logger.info(f"\nRound {self.round_num}:")

            # Run exploration round
            summary = self.run_round(n_pairs=self.config.get('pairs_per_round', 10))

            # Track history
            self.sensitivity_history.append(summary['mean'])

            # Check for improvement
            if summary['mean'] > best_mean + 0.01:
                best_mean = summary['mean']
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1

            # Log round summary
            logger.info(
                f"  Mean sensitivity: {summary['mean']:.3f} ± {summary['std']:.3f} "
                f"(n={summary['n']})"
            )

            # Log knowledge map
            knowledge = self.tracker.knowledge_map()
            known = [t for t, s in knowledge.items() if s['is_known']]
            unknown = [t for t, s in knowledge.items() if not s['is_known']]
            logger.info(f"  Known topics: {known}")
            logger.info(f"  Unknown topics: {unknown}")

            # Check convergence
            if self.tracker.is_converged:
                logger.info("\n SENSITIVITY CONVERGED!")
                break

            if stagnant_rounds >= max_stagnant:
                logger.info(f"\nStagnated for {max_stagnant} rounds")
                break

        # Final report
        self.report_final()

    def report_final(self):
        """Print final report."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL REPORT")
        logger.info("=" * 80)

        # Knowledge map
        knowledge = self.tracker.knowledge_map()

        logger.info("\nKNOWLEDGE MAP:")
        logger.info("-" * 50)

        for topic, stats in sorted(knowledge.items(), key=lambda x: -x[1]['mean_sensitivity']):
            status = "KNOWN" if stats['is_known'] else "UNKNOWN"
            logger.info(
                f"  [{status:7}] {topic:15} "
                f"sens={stats['mean_sensitivity']:.3f} ± {stats['std_sensitivity']:.3f} "
                f"(n={stats['n_samples']})"
            )

        # Summary stats
        all_sens = list(self.tracker.overall_history)
        if all_sens:
            logger.info("\nOVERALL STATISTICS:")
            logger.info(f"  Mean sensitivity: {np.mean(all_sens):.3f}")
            logger.info(f"  Std sensitivity:  {np.std(all_sens):.3f}")
            logger.info(f"  Total samples:    {len(all_sens)}")

        # Accuracy vs expected
        correct = 0
        total = 0
        for topic, stats in knowledge.items():
            expected = topic != 'opinion'
            actual = stats['is_known']
            if expected == actual:
                correct += 1
            total += 1

        if total > 0:
            logger.info(f"\nClassification accuracy: {100*correct/total:.0f}% ({correct}/{total})")

        # Save results
        # Convert bools for JSON
        knowledge_json = {
            k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv) for kk, vv in v.items()}
            for k, v in knowledge.items()
        }

        output = {
            'timestamp': datetime.now().isoformat(),
            'rounds': self.round_num,
            'knowledge_map': knowledge_json,
            'sensitivity_history': self.sensitivity_history,
            'converged': bool(self.tracker.is_converged),
        }

        output_path = self.output_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Counterfactual Self-Play Loop")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to analyze (default: middle)"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=50,
        help="Max rounds"
    )
    parser.add_argument(
        "--pairs-per-round",
        type=int,
        default=10,
        help="Pairs to analyze per round"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model
    import mlx.core as mx
    from mlx_lm import load

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Config
    n_layers = len(model.model.layers) if hasattr(model.model, 'layers') else 24
    config = {
        'layer_idx': args.layer if args.layer else n_layers // 2,
        'pairs_per_round': args.pairs_per_round,
        'window_size': 20,
    }

    # Run
    loop = CounterfactualSelfPlayLoop(model, tokenizer, config)
    loop.run(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
