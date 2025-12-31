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

"""Tests for cross-vocabulary merger (cross-vocab model merging)."""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.vocabulary.cross_vocab_merger import (
    CrossVocabMergeConfig,
    CrossVocabMerger,
)
from modelcypher.core.domain.vocabulary.embedding_projector import (
    ProjectionConfig,
    ProjectionStrategy,
)


class TestCrossVocabMergeConfig:
    """Tests for CrossVocabMergeConfig dataclass."""

    def test_defaults(self):
        config = CrossVocabMergeConfig()
        assert config.projection_strategy == ProjectionStrategy.PROCRUSTES
        assert config.high_similarity_threshold == 0.95
        assert config.similarity_threshold == 0.8
        assert config.confidence_threshold == 0.5
        assert config.blend_alpha == 0.5
        assert config.preserve_special_tokens is True
        assert config.use_embedding_similarity is True
        assert config.exact_match_only is False
        assert config.max_alignments_per_token == 3
        assert config.anchor_count == 1000
        assert config.regularization == 1e-6

    def test_custom_projection_strategy(self):
        config = CrossVocabMergeConfig(projection_strategy=ProjectionStrategy.PCA)
        assert config.projection_strategy == ProjectionStrategy.PCA

    def test_custom_thresholds(self):
        config = CrossVocabMergeConfig(
            high_similarity_threshold=0.99,
            similarity_threshold=0.9,
            confidence_threshold=0.6,
        )
        assert config.high_similarity_threshold == 0.99
        assert config.similarity_threshold == 0.9
        assert config.confidence_threshold == 0.6

    def test_custom_blend_alpha(self):
        config = CrossVocabMergeConfig(blend_alpha=0.7)
        assert config.blend_alpha == 0.7

    def test_custom_preserve_special_tokens(self):
        config = CrossVocabMergeConfig(preserve_special_tokens=False)
        assert config.preserve_special_tokens is False

    def test_custom_exact_match_only(self):
        config = CrossVocabMergeConfig(exact_match_only=True)
        assert config.exact_match_only is True

    def test_custom_anchor_count(self):
        config = CrossVocabMergeConfig(anchor_count=500)
        assert config.anchor_count == 500


class TestCrossVocabMergeConfigFromSimilarityDistribution:
    """Tests for from_similarity_distribution class method."""

    def test_empty_similarities_returns_defaults(self):
        config = CrossVocabMergeConfig.from_similarity_distribution([])
        assert config.projection_strategy == ProjectionStrategy.PROCRUSTES
        assert config.blend_alpha == 0.5

    def test_custom_strategy_preserved(self):
        config = CrossVocabMergeConfig.from_similarity_distribution(
            [], projection_strategy=ProjectionStrategy.CCA
        )
        assert config.projection_strategy == ProjectionStrategy.CCA

    def test_custom_blend_alpha_preserved(self):
        config = CrossVocabMergeConfig.from_similarity_distribution(
            [], blend_alpha=0.8
        )
        assert config.blend_alpha == 0.8

    def test_derives_thresholds_from_distribution(self):
        # Create a distribution from 0.0 to 1.0
        similarities = [i / 100.0 for i in range(101)]

        config = CrossVocabMergeConfig.from_similarity_distribution(similarities)

        # With 101 values (0.00, 0.01, ..., 1.00), percentiles are:
        # 95th percentile should be around 0.95
        # 80th percentile should be around 0.80
        # 50th percentile should be around 0.50
        assert abs(config.high_similarity_threshold - 0.95) < 0.02
        assert abs(config.similarity_threshold - 0.80) < 0.02
        assert abs(config.confidence_threshold - 0.50) < 0.02

    def test_custom_percentiles(self):
        similarities = [i / 100.0 for i in range(101)]

        config = CrossVocabMergeConfig.from_similarity_distribution(
            similarities,
            high_similarity_percentile=0.90,
            similar_percentile=0.70,
            approximate_percentile=0.30,
        )

        assert abs(config.high_similarity_threshold - 0.90) < 0.02
        assert abs(config.similarity_threshold - 0.70) < 0.02
        assert abs(config.confidence_threshold - 0.30) < 0.02

    def test_single_value_distribution(self):
        # Edge case: single value
        config = CrossVocabMergeConfig.from_similarity_distribution([0.5])

        # All percentiles should be 0.5
        assert config.high_similarity_threshold == 0.5
        assert config.similarity_threshold == 0.5
        assert config.confidence_threshold == 0.5


class TestCrossVocabMergeConfigToProjectionConfig:
    """Tests for to_projection_config method."""

    def test_returns_projection_config(self):
        merge_config = CrossVocabMergeConfig()
        proj_config = merge_config.to_projection_config()

        assert isinstance(proj_config, ProjectionConfig)

    def test_strategy_transferred(self):
        merge_config = CrossVocabMergeConfig(projection_strategy=ProjectionStrategy.PCA)
        proj_config = merge_config.to_projection_config()

        assert proj_config.strategy == ProjectionStrategy.PCA

    def test_regularization_transferred(self):
        merge_config = CrossVocabMergeConfig(regularization=1e-4)
        proj_config = merge_config.to_projection_config()

        assert proj_config.regularization == 1e-4

    def test_anchor_count_transferred(self):
        merge_config = CrossVocabMergeConfig(anchor_count=500)
        proj_config = merge_config.to_projection_config()

        assert proj_config.anchor_count == 500

    def test_preserve_norms_always_true(self):
        merge_config = CrossVocabMergeConfig()
        proj_config = merge_config.to_projection_config()

        assert proj_config.preserve_norms is True


class TestCrossVocabMergerInit:
    """Tests for CrossVocabMerger initialization."""

    def test_default_init(self):
        merger = CrossVocabMerger()
        assert merger.config.projection_strategy == ProjectionStrategy.PROCRUSTES
        assert merger._backend is not None

    def test_custom_config(self):
        config = CrossVocabMergeConfig(projection_strategy=ProjectionStrategy.CCA)
        merger = CrossVocabMerger(config=config)
        assert merger.config.projection_strategy == ProjectionStrategy.CCA

    def test_custom_backend(self):
        backend = get_default_backend()
        merger = CrossVocabMerger(backend=backend)
        assert merger._backend is backend

    def test_has_analyzer(self):
        merger = CrossVocabMerger()
        assert merger._analyzer is not None

    def test_has_projector(self):
        merger = CrossVocabMerger()
        assert merger._projector is not None


class TestCrossVocabMergerIsSpecialToken:
    """Tests for _is_special_token method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

    def test_bos_token(self, merger):
        assert merger._is_special_token("<bos>") is True

    def test_eos_token(self, merger):
        assert merger._is_special_token("<eos>") is True

    def test_pad_token(self, merger):
        assert merger._is_special_token("<pad>") is True

    def test_unk_token(self, merger):
        assert merger._is_special_token("<unk>") is True

    def test_start_token(self, merger):
        assert merger._is_special_token("<s>") is True

    def test_end_token(self, merger):
        assert merger._is_special_token("</s>") is True

    def test_cls_token(self, merger):
        assert merger._is_special_token("[CLS]") is True

    def test_sep_token(self, merger):
        assert merger._is_special_token("[SEP]") is True

    def test_mask_token(self, merger):
        assert merger._is_special_token("[MASK]") is True

    def test_bert_pad_token(self, merger):
        assert merger._is_special_token("[PAD]") is True

    def test_bert_unk_token(self, merger):
        assert merger._is_special_token("[UNK]") is True

    def test_pipe_style_token(self, merger):
        # <| and |> patterns
        assert merger._is_special_token("<|endoftext|>") is True
        assert merger._is_special_token("<|im_start|>") is True

    def test_regular_token_not_special(self, merger):
        assert merger._is_special_token("hello") is False
        assert merger._is_special_token("the") is False
        assert merger._is_special_token("Ġ") is False  # BPE prefix

    def test_case_insensitive(self, merger):
        assert merger._is_special_token("<PAD>") is True
        assert merger._is_special_token("<BOS>") is True
        assert merger._is_special_token("[cls]") is True


class TestCrossVocabMergerBuildIndexAlignment:
    """Tests for _build_index_alignment method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

    def test_same_vocab_sizes(self, merger):
        alignment = merger._build_index_alignment(100, 100)

        assert alignment.source_vocab_size == 100
        assert alignment.target_vocab_size == 100
        assert alignment.exact_matches == 100
        assert alignment.unmapped_count == 0

    def test_source_larger(self, merger):
        alignment = merger._build_index_alignment(150, 100)

        assert alignment.source_vocab_size == 150
        assert alignment.target_vocab_size == 100
        # First 100 are exact matches, remaining 50 are unmapped
        assert alignment.exact_matches == 100
        assert alignment.unmapped_count == 50

    def test_target_larger(self, merger):
        alignment = merger._build_index_alignment(100, 150)

        assert alignment.source_vocab_size == 100
        assert alignment.target_vocab_size == 150
        # All 100 source tokens have exact matches
        assert alignment.exact_matches == 100
        assert alignment.unmapped_count == 0

    def test_small_vocab(self, merger):
        alignment = merger._build_index_alignment(10, 10)

        assert alignment.source_vocab_size == 10
        assert alignment.exact_matches == 10

    def test_alignment_quality_exact_for_shared(self, merger):
        from modelcypher.core.domain.vocabulary.alignment_map import AlignmentQuality

        alignment = merger._build_index_alignment(10, 20)

        for a in alignment.iter_alignments():
            if a.source_id < 10:
                assert a.quality == AlignmentQuality.EXACT
                assert a.confidence == 1.0


class TestCrossVocabMergerGetSharedIndices:
    """Tests for _get_shared_indices method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

    def test_returns_tuple(self, merger):
        alignment = merger._build_index_alignment(10, 10)
        result = merger._get_shared_indices(alignment)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_lists(self, merger):
        alignment = merger._build_index_alignment(10, 10)
        source_indices, target_indices = merger._get_shared_indices(alignment)

        assert isinstance(source_indices, list)
        assert isinstance(target_indices, list)

    def test_exact_matches_included(self, merger):
        alignment = merger._build_index_alignment(10, 10)
        source_indices, target_indices = merger._get_shared_indices(alignment)

        # All 10 exact matches should be included
        assert len(source_indices) == 10
        assert len(target_indices) == 10

    def test_indices_match_one_to_one(self, merger):
        alignment = merger._build_index_alignment(5, 5)
        source_indices, target_indices = merger._get_shared_indices(alignment)

        # For index alignment, source_id == target_id
        for src, tgt in zip(source_indices, target_indices):
            assert src == tgt


class TestCrossVocabMergerMerge:
    """Tests for merge method."""

    @pytest.fixture
    def merger(self):
        config = CrossVocabMergeConfig(anchor_count=20)
        return CrossVocabMerger(config=config)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_merge_same_dimensions(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((50, 64))
        target = backend.random_normal((50, 64))

        result = merger.merge(source, target)

        assert result.merged_embeddings.shape == (50, 64)
        assert result.output_vocab_size == 50
        assert result.output_hidden_dim == 64

    def test_merge_different_vocab_sizes(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((30, 64))
        target = backend.random_normal((50, 64))

        result = merger.merge(source, target)

        # Output size matches target
        assert result.merged_embeddings.shape == (50, 64)
        assert result.output_vocab_size == 50

    def test_merge_different_dimensions(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 64))

        result = merger.merge(source, target)

        # Output dimension matches target
        assert result.merged_embeddings.shape == (50, 64)
        assert result.output_hidden_dim == 64

    def test_merge_returns_all_fields(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((20, 32))
        target = backend.random_normal((20, 32))

        result = merger.merge(source, target)

        assert result.merged_embeddings is not None
        assert result.output_vocab_size > 0
        assert result.output_hidden_dim > 0
        assert result.alignment_map is not None
        assert result.projection_result is not None
        assert result.compatibility is not None
        assert result.source_stats is not None
        assert result.target_stats is not None
        assert isinstance(result.tokens_preserved_from_source, int)
        assert isinstance(result.tokens_preserved_from_target, int)
        assert isinstance(result.tokens_interpolated, int)
        assert isinstance(result.warnings, list)

    def test_merge_with_vocab_dicts(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 32))

        source_vocab = {f"token{i}": i for i in range(10)}
        target_vocab = {f"token{i}": i for i in range(10)}

        result = merger.merge(
            source, target, source_vocab=source_vocab, target_vocab=target_vocab
        )

        assert result.merged_embeddings.shape == (10, 32)

    def test_merge_warnings_without_vocab_dicts(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 32))

        result = merger.merge(source, target)

        # Should warn about using index-based alignment
        assert any("index-based" in w.lower() for w in result.warnings)


class TestCrossVocabMergeResultToDict:
    """Tests for CrossVocabMergeResult.to_dict method."""

    @pytest.fixture
    def merger(self):
        config = CrossVocabMergeConfig(anchor_count=10)
        return CrossVocabMerger(config=config)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_to_dict_returns_dict(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        d = result.to_dict()

        assert isinstance(d, dict)

    def test_to_dict_contains_output_info(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        d = result.to_dict()

        assert "output_vocab_size" in d
        assert "output_hidden_dim" in d
        assert d["output_vocab_size"] == 10
        assert d["output_hidden_dim"] == 16

    def test_to_dict_contains_summaries(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        d = result.to_dict()

        assert "alignment_summary" in d
        assert "projection_summary" in d
        assert "compatibility_summary" in d
        assert "source_stats" in d
        assert "target_stats" in d

    def test_to_dict_contains_token_counts(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        d = result.to_dict()

        assert "tokens_preserved_from_source" in d
        assert "tokens_preserved_from_target" in d
        assert "tokens_interpolated" in d
        assert "warnings" in d


class TestCrossVocabMergerAnalyzeMergeQuality:
    """Tests for analyze_merge_quality method."""

    @pytest.fixture
    def merger(self):
        config = CrossVocabMergeConfig(anchor_count=10)
        return CrossVocabMerger(config=config)

    @pytest.fixture
    def backend(self):
        return get_default_backend()

    def test_returns_dict(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert isinstance(quality, dict)

    def test_contains_alignment_metrics(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "alignment_coverage" in quality
        assert "alignment_confidence" in quality
        assert "alignment_quality_distribution" in quality

    def test_contains_projection_metrics(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "projection_alignment_score" in quality
        assert "projection_reconstruction_error" in quality

    def test_contains_compatibility_metrics(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "compatibility_score" in quality
        assert "vocab_overlap_ratio" in quality

    def test_contains_overall_score(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "overall_quality_score" in quality
        assert 0.0 <= quality["overall_quality_score"] <= 1.0

    def test_contains_warnings_count(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "warnings_count" in quality
        assert isinstance(quality["warnings_count"], int)
