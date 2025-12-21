

import mlx.core as mx
from modelcypher.core.domain.inference.entropy_dynamics import LogitEntropyCalculator, LogitDivergenceCalculator, EntropyDeltaTracker

def test_logit_entropy_calculator():
    calc = LogitEntropyCalculator(top_k=2)
    
    # High entropy: uniform distribution [10, 10, 10]
    # Softmax([10,10,10]) = [0.33, 0.33, 0.33]
    # Entropy = - sum(0.33 * log(0.33)) = ln(3) approx 1.098
    logits = mx.array([10.0, 10.0, 10.0])
    ent, var = calc.compute(logits)
    
    assert abs(ent - 1.0986) < 0.01
    
    # Low entropy: peaked distribution [100, 0, 0]
    # Softmax approx [1, 0, 0]
    # Entropy approx 0
    logits_peaked = mx.array([100.0, 0.0, 0.0])
    ent_peak, var_peak = calc.compute(logits_peaked)
    
    assert ent_peak < 0.01

def test_logit_divergence_calculator():
    calc = LogitDivergenceCalculator()
    
    # Same distribution -> 0 KL
    logits = mx.array([1.0, 2.0, 3.0])
    kl = calc.kl_divergence(logits, logits)
    assert abs(kl) < 1e-6
    
    # Different distribution
    p = mx.array([10.0, 0.0]) # approx [1, 0]
    q = mx.array([0.0, 10.0]) # approx [0, 1]
    # KL(p||q) should be large
    kl_large = calc.kl_divergence(p, q)
    assert kl_large > 5.0

def test_entropy_delta_tracker_anomaly():
    tracker = EntropyDeltaTracker()
    tracker.start_session()
    
    # Base uncertain (high entropy), Adapter confident (low entropy) -> Anomaly
    # Base: Uniform
    base_logits = mx.array([1.0, 1.0, 1.0])
    # Adapter: Peaked
    adapter_logits = mx.array([100.0, 0.0, 0.0])
    
    sample = tracker.record_dual_entropy(base_logits, adapter_logits, token_index=0, generated_token=0)
    
    assert sample.base_entropy > 1.0
    assert sample.adapter_entropy < 0.1
    assert sample.delta > 0.9
    
    # Anomaly score should be high
    assert sample.anomaly_score > 0.6
    
    # Check if anomaly was recorded
    assert tracker.consecutive_anomalies == 1
    
    # End session
    result = tracker.end_session()
    assert result.anomaly_count == 1
