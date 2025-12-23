"""
Tests for LoRAAdapterMerger (TIES and DARE-TIES strategies).
"""
import pytest
from modelcypher.core.domain.merging.lora_adapter_merger import (
    LoRAAdapterMerger,
    Strategy,
    Config,
    SeededGenerator,
    TIESMergeResult,
)


class TestSeededGenerator:
    """Tests for the deterministic random number generator."""
    
    def test_reproducibility(self):
        """Same seed should produce same sequence."""
        rng1 = SeededGenerator(42)
        rng2 = SeededGenerator(42)
        
        for _ in range(10):
            assert rng1.next_uint64() == rng2.next_uint64()
    
    def test_different_seeds_differ(self):
        """Different seeds should produce different sequences."""
        rng1 = SeededGenerator(42)
        rng2 = SeededGenerator(43)
        
        values1 = [rng1.next_uint64() for _ in range(10)]
        values2 = [rng2.next_uint64() for _ in range(10)]
        
        assert values1 != values2
    
    def test_next_float_range(self):
        """next_float should return values in [0, 1)."""
        rng = SeededGenerator(12345)
        
        for _ in range(100):
            f = rng.next_float()
            assert 0.0 <= f < 1.0


class TestTrimVector:
    """Tests for the TIES trimming operation."""
    
    def test_trim_top_k_20_percent(self):
        """Top-K 20% should keep ~20% of values."""
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        trimmed, kept = LoRAAdapterMerger.trim_vector(values, top_k=0.2)
        
        # Should keep top 2 values (0.8 and 0.9)
        assert kept == 2
        assert trimmed[8] == 0.8
        assert trimmed[9] == 0.9
        # Others should be zeroed
        assert all(trimmed[i] == 0.0 for i in range(8))
    
    def test_trim_top_k_zero(self):
        """Top-K 0 should zero everything."""
        values = [1.0, 2.0, 3.0]
        trimmed, kept = LoRAAdapterMerger.trim_vector(values, top_k=0.0)
        
        assert kept == 0
        assert all(v == 0.0 for v in trimmed)
    
    def test_trim_top_k_one(self):
        """Top-K 1 should keep everything."""
        values = [1.0, 2.0, 3.0]
        trimmed, kept = LoRAAdapterMerger.trim_vector(values, top_k=1.0)
        
        assert kept == 3
        assert trimmed == values
    
    def test_trim_empty(self):
        """Empty input should return empty."""
        trimmed, kept = LoRAAdapterMerger.trim_vector([], top_k=0.5)
        
        assert trimmed == []
        assert kept == 0


class TestTIESMerge:
    """Tests for the TIES merging algorithm."""
    
    def test_ties_identical_vectors(self):
        """Identical vectors should merge to same values."""
        vectors = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
        result = LoRAAdapterMerger.ties_merge(vectors)
        
        assert result.conflict_count == 0
        assert result.merged == [1.0, 2.0, 3.0]
    
    def test_ties_opposite_signs_conflict(self):
        """Opposite signs should create conflicts."""
        vectors = [
            [1.0, -2.0, 3.0],
            [-1.0, 2.0, 3.0],
        ]
        result = LoRAAdapterMerger.ties_merge(vectors)
        
        # First two elements have sign conflicts
        assert result.conflict_count == 2
        # Third element has no conflict
        assert result.merged[2] == 3.0
    
    def test_ties_zeros_ignored(self):
        """Zero values should not contribute to sign consensus."""
        vectors = [
            [1.0, 0.0, 3.0],
            [2.0, 5.0, 0.0],
        ]
        result = LoRAAdapterMerger.ties_merge(vectors)
        
        # No conflicts (zeros don't count)
        assert result.conflict_count == 0
        # First position: only (1.0, 2.0) contribute, avg = 1.5
        assert result.merged[0] == 1.5
        # Second position: only 5.0 contributes
        assert result.merged[1] == 5.0
        # Third position: only 3.0 contributes
        assert result.merged[2] == 3.0
    
    def test_ties_sign_majority_wins(self):
        """Majority sign should determine final direction."""
        vectors = [
            [1.0],
            [2.0],
            [-0.5],
        ]
        result = LoRAAdapterMerger.ties_merge(vectors)
        
        # Positive majority (2 vs 1), so only 1.0 and 2.0 contribute
        # Average = 1.5
        assert result.conflict_count == 1  # Mixed signs
        assert result.merged[0] == 1.5
    
    def test_ties_empty(self):
        """Empty input should return empty result."""
        result = LoRAAdapterMerger.ties_merge([])
        
        assert result.merged == []
        assert result.conflict_count == 0
        assert result.merged_non_zero == 0


class TestDAREDrop:
    """Tests for DARE dropout application."""
    
    def test_dare_drop_zero_rate(self):
        """Drop rate 0 should keep all values unchanged."""
        values = [1.0, 2.0, 3.0]
        rng = SeededGenerator(42)
        result = LoRAAdapterMerger.apply_dare_drop(values, drop_rate=0.0, rng=rng)
        
        assert result == values
    
    def test_dare_drop_one_rate(self):
        """Drop rate 1 should zero everything."""
        values = [1.0, 2.0, 3.0]
        rng = SeededGenerator(42)
        result = LoRAAdapterMerger.apply_dare_drop(values, drop_rate=1.0, rng=rng)
        
        assert all(v == 0.0 for v in result)
    
    def test_dare_drop_scaling(self):
        """Kept values should be scaled by 1/(1-drop_rate)."""
        # With drop_rate=0.5, kept values should be scaled by 2
        values = [1.0] * 100  # Many values to reduce variance
        rng = SeededGenerator(42)
        result = LoRAAdapterMerger.apply_dare_drop(values, drop_rate=0.5, rng=rng)
        
        # Non-zero values should be 2.0 (scaled)
        non_zero = [v for v in result if v != 0.0]
        assert all(abs(v - 2.0) < 0.01 for v in non_zero)
    
    def test_dare_drop_approximately_correct_fraction(self):
        """Approximately (1-drop_rate) fraction should be kept."""
        values = [1.0] * 1000
        rng = SeededGenerator(42)
        result = LoRAAdapterMerger.apply_dare_drop(values, drop_rate=0.3, rng=rng)
        
        kept_fraction = sum(1 for v in result if v != 0.0) / len(result)
        
        # Should be approximately 70% kept (with some variance)
        assert 0.6 < kept_fraction < 0.8


class TestConfig:
    """Tests for merge configuration."""
    
    def test_default_config(self):
        """Default config should have reasonable values."""
        config = Config()
        
        assert config.strategy == Strategy.TIES
        assert 0.0 < config.ties_top_k < 1.0
        assert config.drop_rate is None
        assert config.seed == 0
    
    def test_dare_ties_strategy(self):
        """DARE-TIES config should be valid."""
        config = Config(
            strategy=Strategy.DARE_TIES,
            ties_top_k=0.3,
            drop_rate=0.5,
            seed=42,
        )
        
        assert config.strategy == Strategy.DARE_TIES
        assert config.ties_top_k == 0.3
        assert config.drop_rate == 0.5
        assert config.seed == 42
