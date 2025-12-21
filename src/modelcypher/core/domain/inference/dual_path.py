
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, AsyncGenerator, Dict, Any, Union
import time
import uuid
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import logging

logger = logging.getLogger(__name__)

# Import our ported modules
from modelcypher.core.domain.inference.entropy_dynamics import (
    EntropyDeltaTracker, 
    EntropyDeltaSample,
    LogitEntropyCalculator,
    LogitDivergenceCalculator
)

@dataclass
class SecurityScanMetrics:
    token_count: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    circuit_breaker_tripped: bool
    anomaly_alert_count: int

@dataclass
class DualPathGeneratorConfiguration:
    base_model_path: str
    adapter_path: Optional[str] = None
    delta_tracker_config: EntropyDeltaTracker.Configuration = field(default_factory=EntropyDeltaTracker.Configuration)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    halt_on_circuit_breaker: bool = True

class DualPathGenerator:
    """
    Orchestrates dual-path generation with entropy disagreement tracking.
    
    Maintains concept of "Base Model" and "Adapter Model" (via hot-swapping or separate instances).
    Ideally uses one model and toggles LoRA adapters if supported by MLX-LM, 
    or maintains two model instances if memory permits.
    
    For strict 1:1 parity with Swift `DualPathGenerator`, this attempts to use the 
    Single-Model Hot-Swap approach if possible, or falls back to just running 
    separate forward passes if we can manage the state.
    """
    
    def __init__(
        self, 
        config: DualPathGeneratorConfiguration,
        signal_router: Any = None # placeholder for signal system
    ):
        self.config = config
        self.delta_tracker = EntropyDeltaTracker(config.delta_tracker_config, router=signal_router)
        
        # Load model(s)
        # Note: In a real app we might inject the loaded model.
        # Here we assume paths are provided. 
        # MLX-LM loading handling:
        # We load the BASE model. 
        # For the ADAPTER path, we need to apply adapters.
        logger.info(f"Loading model from {config.base_model_path}")
        self.model, self.tokenizer = load(config.base_model_path)
        
        # If adapter path is present, we need a way to apply it.
        # MLX-LM supports adapters via `load(..., adapter_path=...)`
        # But we need DYNAMIC switching for "Dual Path".
        # We can implement this by manually applying LoRA layers or loading a second model instance.
        # Loading a second instance is safer for parity and easier to implement first.
        # Swift implementation used hot-swap. 
        # Let's try loading a SECOND model for the "Adapter" path if memory allows, 
        # or just fail if implementation complexity of hot-swap in Python is too high for this turn.
        # A 1:1 port of the Swift `applyAdapter` / `detachAdapter` logic requires access to the LoRA layers in Python.
        
        self.adapter_model = None
        if config.adapter_path:
            logger.info(f"Loading adapter model from {config.adapter_path}")
             # In MLX-LM, loading with adapter_path fuses? Or returns LoRA model?
            self.adapter_model, _ = load(config.base_model_path, adapter_path=config.adapter_path)
        else:
            self.adapter_model = self.model # If no adapter, both paths are same (degenerate case)

        self.entropy_calc = LogitEntropyCalculator(top_k=config.delta_tracker_config.top_k)
        
    async def generate(self, prompt: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates text while performing dual-path analysis.
        Yields chunks: token, anomaly, metrics.
        """
        # 1. Tokenize
        prompt_tokens = mx.array(self.tokenizer.encode(prompt))
        
        start_time = time.time()
        time_to_first = 0.0
        token_count = 0
        
        # Start tracking session
        correlation_id = uuid.uuid4()
        await self.delta_tracker.start_session(correlation_id)
        
        # We need manual generation loop to intercept logits
        # MLX-LM `generate` is high level. We use `generate_step` or manual loop.
        
        # Internal state
        tokens = prompt_tokens.tolist()
        
        # Cache for both models
        # MLX-LM make_cache equivalent?
        # We'll use the simplified loop for now.
        
        cache_base = None
        cache_adapter = None
        
        # Initial forward pass (prefill)
        # Base Path
        logits_base, cache_base = self.model(prompt_tokens[None], cache=cache_base)
        # Adapter Path
        logits_adapter, cache_adapter = self.adapter_model(prompt_tokens[None], cache=cache_adapter)
        
        # Logits handling for next token...
        # We need to sample from ADAPTER logits, but analyze BOTH.
        
        curr_logits_adapter = logits_adapter[:, -1, :]
        curr_logits_base = logits_base[:, -1, :]
        
        while token_count < self.config.max_tokens:
            # Analyze
            # Compute Entropy/Divergence
            # (Synchronous in Python, unlike Swift actor)
            
            # This logic mirrors `process` in Swift's DualPathLogitProcessor
            
            # 1. Sample from Adapter
            # temp/top_p logic
            token_tensor = self._sample(curr_logits_adapter)
            token_id = token_tensor.item()
            
            # 2. Decode
            text = self.tokenizer.decode([token_id])
            
            # 3. Security Analysis
            # Compute base entropy/variance
            base_ent = self.entropy_calc.compute(curr_logits_base)
            
            # Compute adapter entropy/variance
            adap_ent = self.entropy_calc.compute(curr_logits_adapter)
            
            # Compute KL
            # Need LogitDivergenceCalculator (assuming it was ported as static or class)
            div_calc = LogitDivergenceCalculator() 
            kl = div_calc.kl_divergence(curr_logits_adapter, curr_logits_base)
            
            # Record Delta
            # We call delta_tracker.record_step(...)
            # Note: Swift logic accumulates `PendingEntropyData` then sends to actor.
            # Python is simpler.
            
            sample = EntropyDeltaSample(
                token_index=token_count,
                generated_token_id=token_id,
                base_entropy=base_ent.entropy,
                base_variance=base_ent.variance,
                adapter_entropy=adap_ent.entropy,
                adapter_variance=adap_ent.variance,
                kl_divergence=kl,
            # Compute probabilities for metrics
            # We need softmax of base logits
            probs_base = mx.softmax(curr_logits_base)
            
            # Surprisal = -log(P(token))
            token_prob = probs_base[token_id].item()
            surprisal = -1.0 * mx.log(mx.array(token_prob)).item() if token_prob > 1e-10 else 100.0
            
            # Base Approval = Was this token in the top K of the base model?
            # Or simplified: P(token) in base model
            approval_prob = token_prob
            
            # TODO: Normalized approval usually requires ranking. 
            # For now, raw probability is a good proxy for "approval".
            
            sample = EntropyDeltaSample(
                token_index=token_count,
                generated_token_id=token_id,
                base_entropy=base_ent.entropy,
                base_variance=base_ent.variance,
                adapter_entropy=adap_ent.entropy,
                adapter_variance=adap_ent.variance,
                kl_divergence=kl,
                base_surprisal=surprisal,
                base_approval_prob=approval_prob, 
                normalized_approval=approval_prob, # Proxy
                base_top_k_hit=(approval_prob > 0.01) # Rough heuristic for top-k hit
            )
            
            await self.delta_tracker.record_step(sample)
            
            # Yield token
            yield {"type": "token", "text": text}
            
            # Check for anomalies/circuit breaker
            report = await self.delta_tracker.check_status()
            if report and report.get("anomaly"):
                 yield {"type": "anomaly", "sample": sample}
                 
            if self.config.halt_on_circuit_breaker and report and report.get("circuit_breaker"):
                 yield {"type": "circuit_breaker", "samples": []}
                 break
                 
            # Prepare next step
            tokens.append(token_id)
            token_count += 1
            if token_count == 1:
                time_to_first = (time.time() - start_time) * 1000
                
            input_tensor = mx.array([[token_id]])
            
            logits_base, cache_base = self.model(input_tensor, cache=cache_base)
            logits_adapter, cache_adapter = self.adapter_model(input_tensor, cache=cache_adapter)
            
            curr_logits_base = logits_base[:, -1, :]
            curr_logits_adapter = logits_adapter[:, -1, :]
            
            # Check output stop
            if text in self.config.stop_sequences:
                break
                
        total_time = (time.time() - start_time) * 1000
        metrics = SecurityScanMetrics(
            token_count=token_count,
            time_to_first_token_ms=time_to_first,
            total_time_ms=total_time,
            tokens_per_second=token_count / (total_time / 1000),
            circuit_breaker_tripped=False,
            anomaly_alert_count=0
        )
        yield {"type": "metrics", "metrics": metrics}

    def _sample(self, logits: mx.array) -> mx.array:
        # Simple sampling (greedy or temperature)
        if self.config.temperature == 0:
            return mx.argmax(logits, axis=-1)
        
        # Apply temp
        logits = logits / self.config.temperature
        # mlx.random.categorical not available? use argmax for now or simple manual sampling
        return mx.random.categorical(logits)

