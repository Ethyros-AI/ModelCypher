import mlx.core as mx
import numpy as np
from modelcypher.core.domain.geometry.refusal_direction_detector import RefusalDirectionDetector, RefusalDirectionConfig
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator, BootstrapConfiguration
from modelcypher.core.domain.thermodynamics.behavioral_classifier import BehavioralOutcomeClassifier, BehavioralOutcome
from modelcypher.core.domain.thermodynamics.linguistic_calorimeter import LinguisticCalorimeter
from modelcypher.core.domain.entropy.entropy_tracker import ModelState

def test_refusal_direction():
    """Test RefusalDirectionDetector computes valid refusal directions."""
    config = RefusalDirectionConfig.default()

    # Mock activations: [N, D]
    N, D = 10, 64
    harmful = mx.random.normal((N, D)) + 2.0
    harmless = mx.random.normal((N, D)) - 2.0

    direction = RefusalDirectionDetector.compute_direction(
        harmful, harmless, config, layer_index=15, model_id="test-model"
    )

    assert direction is not None
    assert direction.layer_index == 15
    assert direction.strength > 4.0
    assert direction.direction.shape == (D,)

    # Test distance measurement
    test_activation = mx.random.normal((D,)) + 2.0  # Aligned with harmful
    metrics = RefusalDirectionDetector.measure_distance(
        test_activation, direction, previous_projection=None, token_index=0
    )

    assert metrics is not None
    assert metrics.projection_magnitude > 0
    assert metrics.assessment is not None

def test_intrinsic_dimension_bootstrap():
    """Test IntrinsicDimensionEstimator with bootstrap confidence intervals."""
    # Generate points in a 3D subspace of 64D
    N, D = 100, 64
    subspace = mx.random.normal((3, D))
    weights = mx.random.normal((N, 3))
    points = weights @ subspace + mx.random.normal((N, D)) * 0.01

    bootstrap = BootstrapConfiguration(resamples=100, seed=42)
    estimator = IntrinsicDimensionEstimator()
    estimate = estimator.estimate_two_nn(points, bootstrap=bootstrap)

    assert estimate.intrinsic_dimension > 0
    assert estimate.ci is not None
    assert estimate.ci.lower <= estimate.intrinsic_dimension <= estimate.ci.upper

def test_behavioral_classification():
    """Test BehavioralOutcomeClassifier correctly classifies response patterns."""
    classifier = BehavioralOutcomeClassifier()

    # Case 1: Refusal keywords
    res1 = classifier.classify(
        response="I am sorry, but I cannot fulfill this request as an AI.",
        entropy_trajectory=[0.5, 0.4, 0.6],
        model_state=ModelState.NOMINAL
    )
    assert res1.outcome == BehavioralOutcome.REFUSED

    # Case 2: Solution patterns
    res2 = classifier.classify(
        response="Here is how you make a sandwich: 1. Get bread. 2. Put ham.",
        entropy_trajectory=[1.0, 1.1, 0.9],
        model_state=ModelState.NOMINAL
    )
    assert res2.outcome == BehavioralOutcome.SOLVED

    # Case 3: Hedge patterns
    res3 = classifier.classify(
        response="It depends on the context. One one hand, it's good. On the other hand, it's bad.",
        entropy_trajectory=[2.0, 2.1, 1.9],
        model_state=ModelState.NOMINAL
    )
    assert res3.outcome == BehavioralOutcome.HEDGED

def test_linguistic_calorimeter():
    """Test LinguisticCalorimeter measures entropy changes from modifiers."""
    calorimeter = LinguisticCalorimeter()

    measurement = calorimeter.measure_variant(
        modifier="caps",
        full_prompt="GIVE ME NEWS",
        response="I cannot give news.",
        entropy_trajectory=[0.1, 0.2, 0.1],
        model_state=ModelState.HALTED,
        baseline_entropy=1.5
    )

    assert measurement.modifier == "caps"
    assert measurement.delta_h is not None
    assert measurement.delta_h < 0  # 0.13 - 1.5 = negative delta
    assert measurement.outcome is not None
    assert measurement.outcome.outcome == BehavioralOutcome.REFUSED
