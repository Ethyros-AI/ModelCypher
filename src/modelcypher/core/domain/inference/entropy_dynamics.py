from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger("modelcypher.inference.entropy_dynamics")


class ModelState(str, Enum):
    nominal = "nominal"
    concentrated = "concentrated"  # Low entropy
    uncertain = "uncertain"        # High entropy
    distressed = "distressed"      # Very high entropy (circuit breaker)


class AnomalyLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class ConflictLevel(str, Enum):
    carving = "carving"
    mild_tension = "mild_tension"
    fighting = "fighting"


@dataclass(frozen=True)
class EntropySample:
    window_id: uuid.UUID
    token_start: int
    token_end: int
    logit_entropy: float
    top_k_variance: float
    latency_ms: float
    source: Optional[str] = None
    correlation_id: Optional[uuid.UUID] = None

    class EntropyLevel(str, Enum):
        low = "low"
        moderate = "moderate"
        high = "high"


@dataclass(frozen=True)
class ConflictAnalysis:
    mean_kl: float
    base_approval_rate: float
    conflict_score: float
    level: ConflictLevel
    interpretation: str

    @staticmethod
    def compute(
        kl_divergences: List[Optional[float]],
        base_approved_top_k: List[Optional[bool]]
    ) -> Optional['ConflictAnalysis']:
        kl_sum = 0.0
        token_count = 0
        approved_count = 0

        for kl, approved in zip(kl_divergences, base_approved_top_k):
            if kl is not None and approved is not None:
                token_count += 1
                kl_sum += kl
                if approved:
                    approved_count += 1

        if token_count == 0:
            return None

        mean_kl = kl_sum / token_count
        approval_rate = float(approved_count) / token_count
        conflict_score = mean_kl * (1.0 - approval_rate)

        if approval_rate >= 0.95 and conflict_score < 0.5:
            level = ConflictLevel.carving
        elif approval_rate >= 0.70 and conflict_score < 2.0:
            level = ConflictLevel.mild_tension
        else:
            level = ConflictLevel.fighting

        interpretation = {
            ConflictLevel.carving: "Adapter is carving: sampled tokens largely remain within the base model’s top-K; divergence reflects specialization, not contradiction.",
            ConflictLevel.mild_tension: "Adapter shows mild tension: sampled tokens sometimes fall outside the base model’s top-K; monitor for drift or mismatched persona.",
            ConflictLevel.fighting: "Adapter is fighting: sampled tokens frequently fall outside the base model’s top-K and divergence is high; investigate for misalignment or backdoor behavior."
        }[level]

        return ConflictAnalysis(
            mean_kl=mean_kl,
            base_approval_rate=approval_rate,
            conflict_score=conflict_score,
            level=level,
            interpretation=interpretation
        )


@dataclass
class EntropyDeltaSample:
    token_index: int
    generated_token: int
    
    # Base metrics
    base_entropy: float
    base_top_k_variance: float
    base_state: ModelState
    base_top_token: int
    
    # Adapter metrics
    adapter_entropy: float
    adapter_top_k_variance: float
    adapter_state: ModelState
    adapter_top_token: int
    
    latency_ms: float
    
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[uuid.UUID] = None
    source: Optional[str] = None
    
    # Approval metrics
    base_surprisal: Optional[float] = None
    base_approval_probability: Optional[float] = None
    normalized_approval_score: Optional[float] = None
    base_approved_top_k: Optional[bool] = None
    kl_divergence_adapter_to_base: Optional[float] = None

    @property
    def delta(self) -> float:
        return self.base_entropy - self.adapter_entropy

    @property
    def top_token_disagreement(self) -> bool:
        return self.base_top_token != self.adapter_top_token

    @property
    def variance_delta(self) -> float:
        return self.base_top_k_variance - self.adapter_top_k_variance

    @property
    def anomaly_score(self) -> float:
        positive_delta = max(0.0, self.delta)
        entropy_ratio = positive_delta / max(self.base_entropy, 0.01)
        disagreement_bonus = 1.0 if self.top_token_disagreement else 0.0
        raw_score = 0.8 * entropy_ratio + 0.2 * disagreement_bonus
        return min(1.0, raw_score)

    def anomaly_level(self, low: float = 0.3, high: float = 0.6) -> AnomalyLevel:
        score = self.anomaly_score
        if score < low: return AnomalyLevel.low
        if score < high: return AnomalyLevel.moderate
        return AnomalyLevel.high

    @property
    def has_backdoor_signature(self) -> bool:
        base_uncertain = self.base_state in [ModelState.uncertain, ModelState.distressed]
        adapter_confident = self.adapter_state in [ModelState.nominal, ModelState.concentrated]
        return base_uncertain and adapter_confident and self.top_token_disagreement

    @property
    def has_approval_anomaly(self) -> bool:
        if self.base_surprisal is None:
            return self.has_backdoor_signature
        
        adapter_confident = self.adapter_entropy < 1.5
        base_disapproves = self.base_surprisal > 6.0
        return adapter_confident and base_disapproves

    @property
    def enhanced_anomaly_score(self) -> float:
        base_score = self.anomaly_score
        if self.base_surprisal is None:
            return base_score
        
        surprisal = self.base_surprisal
        surprisal_penalty = min(1.0, surprisal / 10.0)
        confidence_multiplier = max(0.0, min(1.0, (3.0 - self.adapter_entropy) / 2.5))
        approval_contribution = surprisal_penalty * confidence_multiplier * 0.4
        
        return min(1.0, base_score * 0.6 + approval_contribution + 0.4 * base_score)

    @property
    def approval_anomaly_level(self) -> AnomalyLevel:
        if self.has_approval_anomaly:
            return AnomalyLevel.high
        if self.base_surprisal and self.base_surprisal > 4.0 and self.adapter_entropy < 2.0:
            return AnomalyLevel.moderate
        return self.anomaly_level()

    def to_signal_payload(self) -> Dict[str, Any]:
        payload = {
            "id": str(self.id),
            "tokenIndex": self.token_index,
            "generatedToken": self.generated_token,
            "baseEntropy": self.base_entropy,
            "adapterEntropy": self.adapter_entropy,
            "delta": self.delta,
            "baseState": self.base_state.value,
            "adapterState": self.adapter_state.value,
            "topTokenDisagreement": self.top_token_disagreement,
            "anomalyScore": self.anomaly_score,
            "enhancedAnomalyScore": self.enhanced_anomaly_score,
            "anomalyLevel": self.anomaly_level().value,
            "approvalAnomalyLevel": self.approval_anomaly_level.value,
            "hasBackdoorSignature": self.has_backdoor_signature,
            "hasApprovalAnomaly": self.has_approval_anomaly,
            "timestamp": self.timestamp.isoformat(),
            "latencyMs": self.latency_ms,
        }
        if self.base_surprisal is not None:
             payload["baseSurprisal"] = self.base_surprisal
        if self.base_approval_probability is not None:
             payload["baseApprovalProbability"] = self.base_approval_probability
        if self.normalized_approval_score is not None:
             payload["normalizedApprovalScore"] = self.normalized_approval_score
        if self.base_approved_top_k is not None:
             payload["baseApprovedTopK"] = self.base_approved_top_k
        if self.kl_divergence_adapter_to_base is not None:
             payload["klDivergenceAdapterToBase"] = self.kl_divergence_adapter_to_base
        if self.correlation_id:
             payload["correlationID"] = str(self.correlation_id)
        if self.source:
             payload["source"] = self.source
        return payload

@dataclass
class EntropyDeltaSessionResult:
    session_start: datetime
    session_end: datetime
    total_tokens: int
    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float
    backdoor_signature_count: int
    circuit_breaker_tripped: bool
    samples: List[EntropyDeltaSample]
    
    session_id: uuid.UUID = field(default_factory=uuid.uuid4)
    correlation_id: Optional[uuid.UUID] = None
    approval_anomaly_count: int = 0
    avg_base_surprisal: Optional[float] = None
    max_base_surprisal: Optional[float] = None
    conflict_analysis: Optional[ConflictAnalysis] = None
    circuit_breaker_trip_index: Optional[int] = None

    class SecurityAssessment(str, Enum):
        safe = "safe"
        suspicious = "suspicious"
        dangerous = "dangerous"

    @property
    def security_assessment(self) -> SecurityAssessment:
        if self.approval_anomaly_count > 0:
            return self.SecurityAssessment.dangerous
        if self.circuit_breaker_tripped or self.backdoor_signature_count > 0:
            return self.SecurityAssessment.dangerous
        if self.max_base_surprisal and self.max_base_surprisal > 8.0:
            return self.SecurityAssessment.suspicious
        if self.anomaly_count > 0 or self.max_anomaly_score > 0.5:
            return self.SecurityAssessment.suspicious
        return self.SecurityAssessment.safe

    @property
    def duration_seconds(self) -> float:
        return (self.session_end - self.session_start).total_seconds()


class LogitEntropyCalculator:
    """Entropy calculation from model logits (1:1 port)."""

    def __init__(
        self,
        top_k: int = 10,
        epsilon: float = 1e-10,
        backend: "Backend | None" = None,
    ) -> None:
        self.top_k = top_k
        self.epsilon = epsilon
        self._backend = backend or get_default_backend()

    def compute(self, logits: "Array", skip_variance: bool = False) -> Tuple[float, float]:
        b = self._backend
        # Logits: [..., vocab]
        # Flatten to [vocab]
        flat_logits = logits
        if logits.ndim > 1:
            if logits.ndim == 3:  # [batch, seq, vocab]
                flat_logits = logits[0, -1]
            elif logits.ndim == 2:  # [batch, vocab]
                flat_logits = logits[0]

        flat_logits = b.astype(flat_logits, "float32")
        b.eval(flat_logits)  # stable eval

        # Softmax
        max_val = b.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_val
        exp_shifted = b.exp(shifted)
        sum_exp = b.sum(exp_shifted, axis=-1, keepdims=True)
        probs = exp_shifted / sum_exp

        # Entropy
        log_probs = b.log(probs + self.epsilon)
        entropy = -b.sum(probs * log_probs, axis=-1)

        # Variance
        variance_val = 0.0
        if not skip_variance and self.top_k > 0:
            vocab_size = flat_logits.shape[-1]
            k = min(self.top_k, vocab_size)
            # Use argsort to get top K indices (descending)
            indices = b.argsort(-flat_logits, axis=-1)
            top_k_indices = indices[:k]
            # Gather top-K logits
            top_k_logits_list = [flat_logits[int(b.to_numpy(top_k_indices[i]))] for i in range(k)]
            top_k_logits = b.stack(top_k_logits_list)

            mean_val = b.mean(top_k_logits, axis=-1, keepdims=True)
            squared_diff = (top_k_logits - mean_val) ** 2
            variance = b.mean(squared_diff, axis=-1)
            b.eval(entropy, variance)
            entropy_np = b.to_numpy(entropy)
            variance_np = b.to_numpy(variance)
            return (float(entropy_np.item()), float(variance_np.item()))

        b.eval(entropy)
        entropy_np = b.to_numpy(entropy)
        return (float(entropy_np.item()), variance_val)


class LogitDivergenceCalculator:
    """KL divergence calculator."""

    def __init__(
        self,
        epsilon: float = 1e-10,
        backend: "Backend | None" = None,
    ) -> None:
        self.epsilon = epsilon
        self._backend = backend or get_default_backend()

    def stable_softmax(self, flat_logits: "Array") -> "Array":
        b = self._backend
        max_val = b.max(flat_logits, axis=-1, keepdims=True)
        shifted = flat_logits - max_val
        exp_shifted = b.exp(shifted)
        sum_exp = b.sum(exp_shifted, axis=-1, keepdims=True)
        return exp_shifted / sum_exp

    def kl_divergence(self, primary_logits: "Array", probe_logits: "Array") -> float:
        b = self._backend
        # Flatten
        p_flat = primary_logits
        q_flat = probe_logits
        if primary_logits.ndim > 1:
            p_flat = primary_logits[0, -1] if primary_logits.ndim == 3 else primary_logits[0]
        if probe_logits.ndim > 1:
            q_flat = probe_logits[0, -1] if probe_logits.ndim == 3 else probe_logits[0]

        p = self.stable_softmax(p_flat)
        q = self.stable_softmax(q_flat)

        log_p = b.log(p + self.epsilon)
        log_q = b.log(q + self.epsilon)

        kl = b.sum(p * (log_p - log_q), axis=-1)
        b.eval(kl)
        kl_np = b.to_numpy(kl)
        return max(0.0, float(kl_np.item()))


class EntropyDeltaTracker:
    """Tracks entropy differences between two model passes."""

    @dataclass
    class Configuration:
        top_k: int = 10
        anomaly_threshold: float = 0.6
        consecutive_anomaly_count: int = 3
        compute_variance: bool = True
        source: str = "EntropyDeltaTracker"

    def __init__(
        self,
        configuration: "Configuration | None" = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = configuration or self.Configuration()
        self._backend = backend or get_default_backend()
        self.calculator = LogitEntropyCalculator(
            top_k=self.config.top_k, backend=self._backend
        )
        self.samples: List[EntropyDeltaSample] = []
        self.session_active = False
        self.correlation_id: Optional[uuid.UUID] = None
        self.session_start: Optional[datetime] = None
        self.consecutive_anomalies = 0
        self.circuit_breaker_tripped = False
        self.circuit_breaker_trip_index: Optional[int] = None

    def start_session(self, correlation_id: Optional[uuid.UUID] = None):
        self.session_active = True
        self.correlation_id = correlation_id or uuid.uuid4()
        self.session_start = datetime.utcnow()
        self.samples = []
        self.consecutive_anomalies = 0
        self.circuit_breaker_tripped = False
        self.circuit_breaker_trip_index = None

    def record_dual_entropy(
        self,
        base_logits: "Array",
        adapter_logits: "Array",
        token_index: int,
        generated_token: int,
    ) -> EntropyDeltaSample:
        start_time = time.perf_counter()
        b = self._backend

        base_ent, base_var = self.calculator.compute(base_logits, skip_variance=not self.config.compute_variance)
        adap_ent, adap_var = self.calculator.compute(adapter_logits, skip_variance=not self.config.compute_variance)

        # Top token extraction
        def get_top(logits: "Array") -> int:
            flat = logits
            if logits.ndim > 1:
                flat = logits[0, -1] if logits.ndim == 3 else logits[0]
            idx = b.argmax(flat, axis=-1)
            b.eval(idx)
            return int(b.to_numpy(idx).item())
             
        base_top = get_top(base_logits)
        adap_top = get_top(adapter_logits)

        # Basic state classification (simple heuristic mapping)
        def classify(ent):
            if ent < 1.5: return ModelState.nominal # 1.5 is 'low' from swift
            if ent < 3.0: return ModelState.uncertain 
            return ModelState.distressed
            
        base_state = classify(base_ent)
        adap_state = classify(adap_ent)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        sample = EntropyDeltaSample(
            token_index=token_index,
            generated_token=generated_token,
            base_entropy=base_ent,
            base_top_k_variance=base_var,
            base_state=base_state,
            base_top_token=base_top,
            adapter_entropy=adap_ent,
            adapter_top_k_variance=adap_var,
            adapter_state=adap_state,
            adapter_top_token=adap_top,
            latency_ms=latency_ms,
            correlation_id=self.correlation_id,
            source=self.config.source
        )
        
        self.samples.append(sample)
        self._check_anomalies(sample)
        return sample

    def _check_anomalies(self, sample: EntropyDeltaSample):
        if sample.anomaly_score >= self.config.anomaly_threshold:
            self.consecutive_anomalies += 1
            if not self.circuit_breaker_tripped and self.consecutive_anomalies >= self.config.consecutive_anomaly_count:
                self.circuit_breaker_tripped = True
                self.circuit_breaker_trip_index = sample.token_index
        else:
            self.consecutive_anomalies = 0

    def end_session(self) -> EntropyDeltaSessionResult:
        end_time = datetime.utcnow()
        self.session_active = False
        
        anomaly_count = sum(1 for s in self.samples if s.anomaly_score >= self.config.anomaly_threshold)
        max_score = max((s.anomaly_score for s in self.samples), default=0.0)
        avg_delta = sum((s.delta for s in self.samples), start=0.0) / max(1, len(self.samples))
        disagreement = sum(1 for s in self.samples if s.top_token_disagreement)
        disagreement_rate = disagreement / max(1, len(self.samples))
        
        # Conflict Analysis
        kls = [s.kl_divergence_adapter_to_base for s in self.samples]
        approvals = [s.base_approved_top_k for s in self.samples]
        conflict = ConflictAnalysis.compute(kls, approvals)

        return EntropyDeltaSessionResult(
            session_start=self.session_start or end_time,
            session_end=end_time,
            total_tokens=len(self.samples),
            anomaly_count=anomaly_count,
            max_anomaly_score=max_score,
            avg_delta=avg_delta,
            disagreement_rate=disagreement_rate,
            backdoor_signature_count=sum(1 for s in self.samples if s.has_backdoor_signature),
            approval_anomaly_count=sum(1 for s in self.samples if s.has_approval_anomaly),
            circuit_breaker_tripped=self.circuit_breaker_tripped,
            circuit_breaker_trip_index=self.circuit_breaker_trip_index,
            samples=self.samples,
            correlation_id=self.correlation_id,
            conflict_analysis=conflict
        )
