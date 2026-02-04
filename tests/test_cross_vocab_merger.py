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
from modelcypher.experimental.vocabulary.cross_vocab_merger import (
    AlignmentMethod,
    CrossVocabMerger,
)


class TestCrossVocabMergerInit:
    """Tests for CrossVocabMerger initialization."""

    def test_default_init(self):
        merger = CrossVocabMerger()
        assert merger._backend is not None

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


class TestCrossVocabMergerMerge:
    """Tests for merge method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

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

        assert result.merged_embeddings.shape == (50, 64)
        assert result.output_vocab_size == 50

    def test_merge_different_dimensions(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 64))

        result = merger.merge(source, target)

        assert result.merged_embeddings.shape == (50, 64)
        assert result.output_hidden_dim == 64

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
        assert result.alignment_map.exact_matches == 10

    def test_merge_alignment_method_without_vocab_dicts(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 32))
        target = backend.random_normal((10, 32))

        result = merger.merge(source, target)

        assert result.alignment_method == AlignmentMethod.INDEX


class TestCrossVocabMergeResultToDict:
    """Tests for CrossVocabMergeResult.to_dict method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

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
        assert "vocabulary_alignment" in d
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
        assert "alignment_method" in d


class TestCrossVocabMergerAnalyzeMergeQuality:
    """Tests for analyze_merge_quality method."""

    @pytest.fixture
    def merger(self):
        return CrossVocabMerger()

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

    def test_contains_match_metrics(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "coverage_ratio" in quality
        assert "exact_match_ratio" in quality
        assert "match_quality_distribution" in quality

    def test_contains_projection_metrics(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "projection_reconstruction_error" in quality
        assert "projection_mean_cosine_similarity" in quality
        assert "projection_norm_preservation_ratio" in quality

    def test_contains_alignment_fields(self, merger, backend):
        backend.random_seed(42)
        source = backend.random_normal((10, 16))
        target = backend.random_normal((10, 16))

        result = merger.merge(source, target)
        quality = merger.analyze_merge_quality(result)

        assert "vocab_overlap_ratio" in quality
        assert "dimension_ratio" in quality
        assert "requires_projection" in quality
        assert "requires_vocab_mapping" in quality
        assert "alignment_method" in quality
