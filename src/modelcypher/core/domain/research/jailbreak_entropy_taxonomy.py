# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Experiment C1: Jailbreak Entropy Signature Taxonomy.

**Hypothesis:** Different jailbreak attack types produce distinct entropy signatures.

**The Math:**
- Entropy trajectory: H(t) = -Σ p_i(t) log p_i(t) for each token t
- Pattern features: drop ratio, spike count, gradient stability
- Clustering: k-means on trajectory features

**Success Metric:** >70% accuracy classifying attack type from entropy alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class AttackCategory(str, Enum):
    """Attack taxonomy categories mapped from JailbreakBench."""

    HARASSMENT = "harassment"
    TECHNICAL_EXPLOIT = "technical_exploit"
    VIOLENCE = "violence"
    FRAUD = "fraud"
    SOCIAL_ENGINEERING = "social_engineering"
    MISINFORMATION = "misinformation"
    EXPLICIT_CONTENT = "explicit_content"
    PRIVACY_VIOLATION = "privacy_violation"
    UNAUTHORIZED_ADVICE = "unauthorized_advice"
    AUTHORITY_MISUSE = "authority_misuse"
    BENIGN = "benign"


@dataclass(frozen=True)
class LabeledJailbreak:
    """A labeled jailbreak prompt with its attack category."""

    prompt: str
    """The prompt text."""

    attack_category: str
    """Attack category (e.g., 'prompt_injection', 'roleplay', 'few_shot')."""

    is_harmful: bool
    """Whether this is a known harmful prompt."""

    source: str | None = None
    """Optional source (e.g., 'jailbreakbench', 'custom')."""


@dataclass
class EntropySignature:
    """Entropy signature extracted from a generation trajectory."""

    trajectory: list[float]
    """Full entropy trajectory H(t) for each token."""

    attack_category: str
    """The attack category label."""

    is_harmful: bool
    """Whether this was a harmful prompt."""

    prompt_prefix: str
    """The original prompt (truncated for storage)."""

    # Computed statistics
    drop_ratio: float = field(init=False)
    """Drop ratio: max(H) / min(H) - indicates sharpness of entropy collapse."""

    spike_count: int = field(init=False)
    """Number of 2σ deviations from mean (spike count)."""

    gradient_stability: float = field(init=False)
    """Variance of dH/dt (gradient stability) - lower = more monotonic."""

    mean_entropy: float = field(init=False)
    """Mean entropy across trajectory."""

    std_entropy: float = field(init=False)
    """Standard deviation of entropy."""

    min_entropy: float = field(init=False)
    """Minimum entropy observed."""

    max_entropy: float = field(init=False)
    """Maximum entropy observed."""

    min_entropy_index: int = field(init=False)
    """Token index where minimum entropy occurred."""

    total_curvature: float = field(init=False)
    """Total curvature of trajectory (sum of |d²H/dt²|)."""

    tokens_to_first_drop: int | None = field(init=False)
    """Time to first significant drop (tokens until 30% drop from baseline)."""

    monotonicity_score: float = field(init=False)
    """Monotonicity score [0,1]: fraction of consecutive drops."""

    def __post_init__(self) -> None:
        # Truncate prompt prefix
        object.__setattr__(self, "prompt_prefix", self.prompt_prefix[:100])

        if not self.trajectory:
            object.__setattr__(self, "drop_ratio", 1.0)
            object.__setattr__(self, "spike_count", 0)
            object.__setattr__(self, "gradient_stability", 0.0)
            object.__setattr__(self, "mean_entropy", 0.0)
            object.__setattr__(self, "std_entropy", 0.0)
            object.__setattr__(self, "min_entropy", 0.0)
            object.__setattr__(self, "max_entropy", 0.0)
            object.__setattr__(self, "min_entropy_index", 0)
            object.__setattr__(self, "total_curvature", 0.0)
            object.__setattr__(self, "tokens_to_first_drop", None)
            object.__setattr__(self, "monotonicity_score", 0.0)
            return

        n = len(self.trajectory)
        local_mean = sum(self.trajectory) / n
        local_min = min(self.trajectory)
        local_max = max(self.trajectory)
        local_min_index = self.trajectory.index(local_min)

        # Drop ratio (avoid division by zero)
        local_drop_ratio = local_max / local_min if local_min > 0.001 else local_max + 1

        # Standard deviation
        variance = sum((h - local_mean) ** 2 for h in self.trajectory) / n
        local_std = math.sqrt(variance)

        # Spike count: values > 2σ from mean
        spike_threshold = 2 * local_std
        local_spike_count = sum(1 for h in self.trajectory if abs(h - local_mean) > spike_threshold)

        # Gradient (dH/dt)
        gradients = [
            self.trajectory[i] - self.trajectory[i - 1] for i in range(1, len(self.trajectory))
        ]

        # Gradient stability = variance of gradients
        if gradients:
            mean_grad = sum(gradients) / len(gradients)
            grad_var = sum((g - mean_grad) ** 2 for g in gradients) / len(gradients)
            local_gradient_stability = grad_var
        else:
            local_gradient_stability = 0.0

        # Total curvature (sum of |d²H/dt²|)
        curvature = 0.0
        for i in range(1, len(gradients)):
            curvature += abs(gradients[i] - gradients[i - 1])

        # Tokens to first significant drop (30% from baseline)
        baseline = sum(self.trajectory[: min(3, len(self.trajectory))]) / min(
            3, len(self.trajectory)
        )
        drop_threshold = baseline * 0.7
        local_tokens_to_first_drop: int | None = None
        for i, h in enumerate(self.trajectory):
            if h < drop_threshold:
                local_tokens_to_first_drop = i
                break

        # Monotonicity: fraction of consecutive decreases
        decreases = sum(1 for g in gradients if g < 0)
        local_monotonicity = decreases / len(gradients) if gradients else 0.0

        # Assign computed values
        object.__setattr__(self, "mean_entropy", local_mean)
        object.__setattr__(self, "min_entropy", local_min)
        object.__setattr__(self, "max_entropy", local_max)
        object.__setattr__(self, "min_entropy_index", local_min_index)
        object.__setattr__(self, "drop_ratio", local_drop_ratio)
        object.__setattr__(self, "std_entropy", local_std)
        object.__setattr__(self, "spike_count", local_spike_count)
        object.__setattr__(self, "gradient_stability", local_gradient_stability)
        object.__setattr__(self, "total_curvature", curvature)
        object.__setattr__(self, "tokens_to_first_drop", local_tokens_to_first_drop)
        object.__setattr__(self, "monotonicity_score", local_monotonicity)

    @property
    def feature_vector(self) -> list[float]:
        """Feature vector for clustering/classification."""
        return [
            self.drop_ratio,
            float(self.spike_count),
            self.gradient_stability,
            self.mean_entropy,
            self.std_entropy,
            self.total_curvature,
            self.monotonicity_score,
            float(
                self.tokens_to_first_drop
                if self.tokens_to_first_drop is not None
                else len(self.trajectory)
            ),
        ]


@dataclass(frozen=True)
class TaxonomyCluster:
    """Clustering result for taxonomy."""

    cluster_id: int
    """Cluster ID."""

    centroid: list[float]
    """Centroid feature vector."""

    member_indices: list[int]
    """Member signatures (indices)."""

    dominant_category: str
    """Dominant attack category in this cluster."""

    category_distribution: dict[str, int]
    """Category distribution."""


@dataclass(frozen=True)
class ClassificationResult:
    """Classification result."""

    predicted_category: str
    """Predicted attack category."""

    confidence: float
    """Confidence score."""

    distance_to_cluster: float
    """Distance to nearest cluster centroid."""

    cluster_id: int
    """Cluster ID assigned."""


@dataclass(frozen=True)
class TaxonomyReport:
    """Full taxonomy report."""

    experiment_id: UUID
    """Experiment ID."""

    model_id: str
    """Model used."""

    signatures: list[EntropySignature]
    """All collected signatures."""

    clusters: list[TaxonomyCluster]
    """Clusters discovered."""

    test_accuracy: float
    """Classification accuracy on held-out test set."""

    category_precision: dict[str, float]
    """Per-category precision."""

    category_recall: dict[str, float]
    """Per-category recall."""

    confusion_matrix: list[list[int]]
    """Confusion matrix [actual][predicted]."""

    category_labels: list[str]
    """Category labels for confusion matrix."""

    timestamp: datetime
    """Timestamp."""

    notes: str | None = None
    """Notes."""

    @property
    def success_metric_achieved(self) -> bool:
        """Whether C1 success metric is achieved (>70% accuracy)."""
        return self.test_accuracy > 0.70

    def generate_markdown_report(self) -> str:
        """Generates a markdown report."""
        lines = [
            "# C1: Jailbreak Entropy Signature Taxonomy",
            "",
            f"**Experiment ID:** {self.experiment_id}",
            f"**Model:** {self.model_id}",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Results Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Test Accuracy | {self.test_accuracy * 100:.1f}% |",
            f"| Success Metric (>70%) | {'YES' if self.success_metric_achieved else 'NO'} |",
            f"| Total Signatures | {len(self.signatures)} |",
            f"| Clusters | {len(self.clusters)} |",
            "",
            "## Cluster Analysis",
            "",
        ]

        for cluster in sorted(self.clusters, key=lambda c: c.cluster_id):
            lines.append(f"### Cluster {cluster.cluster_id}: {cluster.dominant_category}")
            lines.append("")
            lines.append(f"**Members:** {len(cluster.member_indices)}")
            lines.append("")
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for cat, count in sorted(cluster.category_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"| {cat} | {count} |")
            lines.append("")

        lines.append("## Per-Category Metrics")
        lines.append("")
        lines.append("| Category | Precision | Recall |")
        lines.append("|----------|-----------|--------|")
        for cat in self.category_labels:
            p = self.category_precision.get(cat, 0)
            r = self.category_recall.get(cat, 0)
            lines.append(f"| {cat} | {p * 100:.1f}% | {r * 100:.1f}% |")

        lines.append("")
        lines.append("## Confusion Matrix")
        lines.append("")

        # Header
        header = "| Actual \\ Predicted |"
        for label in self.category_labels:
            header += f" {label[:10]} |"
        lines.append(header)
        lines.append("|" + "---|" * (len(self.category_labels) + 1))

        # Rows
        for i, label in enumerate(self.category_labels):
            row = f"| {label[:15]} |"
            for j in range(len(self.category_labels)):
                row += f" {self.confusion_matrix[i][j]} |"
            lines.append(row)

        if self.notes:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            lines.append(self.notes)

        return "\n".join(lines)


class JailbreakEntropyTaxonomy:
    """Experiment C1: Jailbreak Entropy Signature Taxonomy.

    Clusters jailbreak prompts by their entropy signatures to identify
    distinct attack types.
    """

    def __init__(self) -> None:
        self._collected_signatures: list[EntropySignature] = []

    @property
    def collected_signatures(self) -> list[EntropySignature]:
        """Get all collected signatures."""
        return list(self._collected_signatures)

    def add_signature(self, signature: EntropySignature) -> None:
        """Add a signature to the collection."""
        self._collected_signatures.append(signature)

    def create_signature(
        self,
        trajectory: list[float],
        labeled: LabeledJailbreak,
    ) -> EntropySignature:
        """Create and store an entropy signature from a trajectory."""
        signature = EntropySignature(
            trajectory=trajectory,
            attack_category=labeled.attack_category,
            is_harmful=labeled.is_harmful,
            prompt_prefix=labeled.prompt,
        )
        self._collected_signatures.append(signature)
        return signature

    def cluster(
        self,
        signatures: list[EntropySignature],
        k: int,
        max_iterations: int = 100,
    ) -> list[TaxonomyCluster]:
        """Clusters signatures using k-means.

        Args:
            signatures: Signatures to cluster.
            k: Number of clusters.
            max_iterations: Maximum iterations.

        Returns:
            Array of clusters.
        """
        if len(signatures) < k:
            return []

        features = [sig.feature_vector for sig in signatures]
        dim = len(features[0])

        # Initialize centroids randomly
        shuffled_indices = list(range(len(features)))
        random.shuffle(shuffled_indices)
        centroids = [features[i] for i in shuffled_indices[:k]]
        assignments = [0] * len(signatures)

        for _ in range(max_iterations):
            # Assignment step
            changed = False
            for i, feature in enumerate(features):
                nearest = self._nearest_centroid(feature, centroids)
                if nearest != assignments[i]:
                    assignments[i] = nearest
                    changed = True

            # Update step
            new_centroids = [[0.0] * dim for _ in range(k)]
            counts = [0] * k

            for i, feature in enumerate(features):
                cluster = assignments[i]
                counts[cluster] += 1
                for d in range(dim):
                    new_centroids[cluster][d] += feature[d]

            for c in range(k):
                if counts[c] > 0:
                    for d in range(dim):
                        new_centroids[c][d] /= counts[c]
                    centroids[c] = new_centroids[c]

            if not changed:
                break

        # Build cluster objects
        clusters: list[TaxonomyCluster] = []
        for c in range(k):
            member_indices = [i for i, a in enumerate(assignments) if a == c]

            # Count categories
            category_dist: dict[str, int] = {}
            for idx in member_indices:
                cat = signatures[idx].attack_category
                category_dist[cat] = category_dist.get(cat, 0) + 1

            dominant = (
                max(category_dist.items(), key=lambda x: x[1])[0] if category_dist else "unknown"
            )

            clusters.append(
                TaxonomyCluster(
                    cluster_id=c,
                    centroid=centroids[c],
                    member_indices=member_indices,
                    dominant_category=dominant,
                    category_distribution=category_dist,
                )
            )

        return clusters

    def classify(
        self,
        signature: EntropySignature,
        clusters: list[TaxonomyCluster],
    ) -> ClassificationResult:
        """Classifies a signature using the trained clusters.

        Args:
            signature: Signature to classify.
            clusters: Trained clusters.

        Returns:
            Classification result.
        """
        features = signature.feature_vector
        best_cluster = 0
        best_distance = float("inf")

        for cluster in clusters:
            dist = self._euclidean_distance(features, cluster.centroid)
            if dist < best_distance:
                best_distance = dist
                best_cluster = cluster.cluster_id

        predicted_cluster = next(c for c in clusters if c.cluster_id == best_cluster)
        max_in_cluster = float(max(predicted_cluster.category_distribution.values(), default=1))
        total_in_cluster = float(len(predicted_cluster.member_indices))
        confidence = max_in_cluster / max(1, total_in_cluster)

        return ClassificationResult(
            predicted_category=predicted_cluster.dominant_category,
            confidence=confidence,
            distance_to_cluster=best_distance,
            cluster_id=best_cluster,
        )

    def evaluate(
        self,
        test_signatures: list[EntropySignature],
        clusters: list[TaxonomyCluster],
    ) -> tuple[float, dict[str, float], dict[str, float], list[list[int]], list[str]]:
        """Evaluates classifier on test set.

        Args:
            test_signatures: Test signatures with ground truth labels.
            clusters: Trained clusters.

        Returns:
            (accuracy, categoryPrecision, categoryRecall, confusionMatrix, labels).
        """
        categories = sorted(set(sig.attack_category for sig in test_signatures))
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        confusion = [[0] * len(categories) for _ in range(len(categories))]
        correct = 0

        for sig in test_signatures:
            result = self.classify(signature=sig, clusters=clusters)
            actual_idx = cat_to_idx.get(sig.attack_category, 0)
            predicted_idx = cat_to_idx.get(result.predicted_category, 0)

            confusion[actual_idx][predicted_idx] += 1
            if result.predicted_category == sig.attack_category:
                correct += 1

        accuracy = correct / max(1, len(test_signatures))

        # Per-category precision and recall
        precision: dict[str, float] = {}
        recall: dict[str, float] = {}

        for cat, idx in cat_to_idx.items():
            true_positive = confusion[idx][idx]
            predicted_positive = sum(confusion[i][idx] for i in range(len(categories)))
            actual_positive = sum(confusion[idx])

            precision[cat] = true_positive / predicted_positive if predicted_positive > 0 else 0.0
            recall[cat] = true_positive / actual_positive if actual_positive > 0 else 0.0

        return accuracy, precision, recall, confusion, categories

    def run_experiment(
        self,
        signatures: list[EntropySignature],
        model_id: str = "unknown",
        k: int = 5,
        test_split: float = 0.2,
    ) -> TaxonomyReport:
        """Runs the full C1 experiment.

        Args:
            signatures: Pre-collected entropy signatures.
            model_id: Model identifier.
            k: Number of clusters for taxonomy.
            test_split: Fraction to hold out for testing.

        Returns:
            Full taxonomy report.
        """
        # Split train/test
        shuffled = list(signatures)
        random.shuffle(shuffled)
        test_count = int(len(shuffled) * test_split)
        test_set = shuffled[:test_count]
        train_set = shuffled[test_count:]

        # Cluster training set
        clusters = self.cluster(signatures=train_set, k=k)

        # Evaluate on test set
        accuracy, precision, recall, confusion, labels = self.evaluate(
            test_signatures=test_set,
            clusters=clusters,
        )

        return TaxonomyReport(
            experiment_id=uuid4(),
            model_id=model_id,
            signatures=signatures,
            clusters=clusters,
            test_accuracy=accuracy,
            category_precision=precision,
            category_recall=recall,
            confusion_matrix=confusion,
            category_labels=labels,
            timestamp=datetime.now(),
            notes=f"C1 Experiment: k={k}, trainSize={len(train_set)}, testSize={len(test_set)}",
        )

    def _nearest_centroid(self, feature: list[float], centroids: list[list[float]]) -> int:
        """Find the nearest centroid to a feature vector."""
        best = 0
        best_dist = float("inf")

        for i, centroid in enumerate(centroids):
            dist = self._euclidean_distance(feature, centroid)
            if dist < best_dist:
                best_dist = dist
                best = i

        return best

    def _euclidean_distance(self, a: list[float], b: list[float]) -> float:
        """Compute Euclidean distance between two vectors."""
        if len(a) != len(b):
            return float("inf")
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    @staticmethod
    def map_jailbreakbench_category(category: str) -> str:
        """Maps JailbreakBench category to attack taxonomy category."""
        mapping = {
            "harassment_discrimination": "harassment",
            "malware_hacking": "technical_exploit",
            "physical_harm": "violence",
            "economic_harm": "fraud",
            "fraud_deception": "social_engineering",
            "disinformation": "misinformation",
            "sexual_adult_content": "explicit_content",
            "privacy": "privacy_violation",
            "expert_advice": "unauthorized_advice",
            "government_decision_making": "authority_misuse",
            "benign": "benign",
        }
        return mapping.get(category.lower().replace("-", "_"), category)
