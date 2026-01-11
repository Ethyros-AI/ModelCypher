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

"""Tests for unified weight stitching module.

Verifies the geometric principle:
    W_target = F_out @ W_source @ F_in.T

All weight transforms follow this single pattern - there are no special cases.
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge.stages.weight_stitcher import (
    ActivationSpace,
    StitchRegistry,
    build_registry_from_stitches,
    detect_weight_spaces,
    stitch_weight,
)


class TestStitchRegistry:
    """Tests for StitchRegistry class."""

    def test_register_and_retrieve(self) -> None:
        """Register a stitch and retrieve it."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)

        # Create a stitch: [tgt_dim=64, src_dim=128]
        output_transform = b.random_normal((64, 128))
        input_transform = b.random_normal((128, 64))
        b.eval(output_transform, input_transform)

        registry.register(ActivationSpace.HIDDEN, output_transform, input_transform)

        stitch = registry.get(ActivationSpace.HIDDEN)
        assert stitch is not None
        assert stitch.src_dim == 128
        assert stitch.tgt_dim == 64

    def test_detect_space_by_dimension(self) -> None:
        """Registry should detect space by dimension."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)

        # Register hidden: src=128, tgt=64
        registry.register(
            ActivationSpace.HIDDEN,
            b.random_normal((64, 128)),
            b.random_normal((128, 64)),
        )

        # Register intermediate: src=512, tgt=256
        registry.register(
            ActivationSpace.INTERMEDIATE,
            b.random_normal((256, 512)),
            b.random_normal((512, 256)),
        )

        # Detect dimensions
        assert registry.detect_space(128, "source") == ActivationSpace.HIDDEN
        assert registry.detect_space(512, "source") == ActivationSpace.INTERMEDIATE
        assert registry.detect_space(999, "source") is None  # Unknown

    def test_get_dims(self) -> None:
        """Get dimensions for a space."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)
        registry.register(
            ActivationSpace.ATTENTION,
            b.random_normal((896, 960)),  # tgt=896, src=960
            b.random_normal((960, 896)),
        )

        dims = registry.get_dims(ActivationSpace.ATTENTION)
        assert dims == (960, 896)

        assert registry.get_dims(ActivationSpace.HIDDEN) is None  # Not registered


class TestDetectWeightSpaces:
    """Tests for weight space detection."""

    def test_detect_mlp_gate_up(self) -> None:
        """Detect MLP gate/up weight spaces: [intermediate, hidden]."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)
        # Hidden: src=128, tgt=64
        registry.register(
            ActivationSpace.HIDDEN,
            b.random_normal((64, 128)),
            b.random_normal((128, 64)),
        )
        # Intermediate: src=512, tgt=256
        registry.register(
            ActivationSpace.INTERMEDIATE,
            b.random_normal((256, 512)),
            b.random_normal((512, 256)),
        )

        # gate_proj shape: [intermediate=512, hidden=128] (source dims)
        out_space, in_space = detect_weight_spaces((512, 128), registry)
        assert out_space == ActivationSpace.INTERMEDIATE
        assert in_space == ActivationSpace.HIDDEN

    def test_detect_mlp_down(self) -> None:
        """Detect MLP down_proj weight spaces: [hidden, intermediate]."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)
        registry.register(
            ActivationSpace.HIDDEN,
            b.random_normal((64, 128)),
            b.random_normal((128, 64)),
        )
        registry.register(
            ActivationSpace.INTERMEDIATE,
            b.random_normal((256, 512)),
            b.random_normal((512, 256)),
        )

        # down_proj shape: [hidden=128, intermediate=512] (source dims)
        out_space, in_space = detect_weight_spaces((128, 512), registry)
        assert out_space == ActivationSpace.HIDDEN
        assert in_space == ActivationSpace.INTERMEDIATE

    def test_unknown_dimensions(self) -> None:
        """Unknown dimensions return None."""
        b = get_default_backend()

        registry = StitchRegistry(b)
        registry.register(
            ActivationSpace.HIDDEN,
            b.random_normal((64, 128)),
            b.random_normal((128, 64)),
        )

        out_space, in_space = detect_weight_spaces((999, 777), registry)
        assert out_space is None
        assert in_space is None


class TestStitchWeight:
    """Tests for the unified stitch_weight function."""

    def test_stitch_mlp_weight(self) -> None:
        """Stitch an MLP weight matrix through both dimensions."""
        b = get_default_backend()
        b.random_seed(42)

        # Source: [512, 128] (intermediate, hidden)
        # Target: [256, 64] (intermediate, hidden)
        src_inter, tgt_inter = 512, 256
        src_hidden, tgt_hidden = 128, 64

        registry = StitchRegistry(b)

        # Hidden stitch
        hidden_out = b.random_normal((tgt_hidden, src_hidden))
        hidden_in = b.random_normal((src_hidden, tgt_hidden))
        registry.register(ActivationSpace.HIDDEN, hidden_out, hidden_in)

        # Intermediate stitch
        inter_out = b.random_normal((tgt_inter, src_inter))
        inter_in = b.random_normal((src_inter, tgt_inter))
        registry.register(ActivationSpace.INTERMEDIATE, inter_out, inter_in)

        b.eval(hidden_out, hidden_in, inter_out, inter_in)

        # Source weight [intermediate, hidden]
        source_weight = b.random_normal((src_inter, src_hidden))
        b.eval(source_weight)

        result = stitch_weight(
            source_weight=source_weight,
            registry=registry,
            backend=b,
            output_space=ActivationSpace.INTERMEDIATE,
            input_space=ActivationSpace.HIDDEN,
        )

        assert result is not None
        assert b.shape(result) == (tgt_inter, tgt_hidden)

    def test_stitch_single_dimension(self) -> None:
        """Stitch only one dimension (e.g., attention → hidden)."""
        b = get_default_backend()
        b.random_seed(42)

        src_hidden, tgt_hidden = 128, 64
        registry = StitchRegistry(b)

        hidden_out = b.random_normal((tgt_hidden, src_hidden))
        hidden_in = b.random_normal((src_hidden, tgt_hidden))
        registry.register(ActivationSpace.HIDDEN, hidden_out, hidden_in)
        b.eval(hidden_out, hidden_in)

        # Attention weight where only input dim matches hidden
        # [attn_dim, hidden] - only stitch hidden (input)
        source_weight = b.random_normal((96, src_hidden))
        b.eval(source_weight)

        result = stitch_weight(
            source_weight=source_weight,
            registry=registry,
            backend=b,
            output_space=None,  # No stitch for output
            input_space=ActivationSpace.HIDDEN,
        )

        assert result is not None
        # Output dim unchanged, input dim stitched
        assert b.shape(result) == (96, tgt_hidden)

    def test_no_stitch_returns_none(self) -> None:
        """If no stitch is needed/available, return None."""
        b = get_default_backend()
        b.random_seed(42)

        registry = StitchRegistry(b)  # Empty registry

        source_weight = b.random_normal((64, 32))
        b.eval(source_weight)

        result = stitch_weight(
            source_weight=source_weight,
            registry=registry,
            backend=b,
            output_space=None,
            input_space=None,
        )

        assert result is None

    def test_stitch_preserves_values(self) -> None:
        """Stitched weight should be a linear transform of source."""
        b = get_default_backend()
        b.random_seed(42)

        src_dim, tgt_dim = 128, 64
        registry = StitchRegistry(b)

        # Use identity-like transform for predictable output
        # Create [tgt, src] by selecting first tgt_dim rows of identity
        out_transform = b.eye(src_dim)[:tgt_dim, :]  # [tgt, src]
        in_transform = b.transpose(out_transform)  # [src, tgt]
        b.eval(out_transform, in_transform)
        registry.register(ActivationSpace.HIDDEN, out_transform, in_transform)

        source_weight = b.random_normal((src_dim, src_dim))
        b.eval(source_weight)

        result = stitch_weight(
            source_weight=source_weight,
            registry=registry,
            backend=b,
            output_space=ActivationSpace.HIDDEN,
            input_space=ActivationSpace.HIDDEN,
        )

        assert result is not None
        assert b.shape(result) == (tgt_dim, tgt_dim)


class TestBuildRegistry:
    """Tests for build_registry_from_stitches convenience function."""

    def test_build_full_registry(self) -> None:
        """Build registry from all stitch types."""
        b = get_default_backend()
        b.random_seed(42)

        hidden_stitch = (b.random_normal((64, 128)), b.random_normal((128, 64)))
        inter_stitch = (b.random_normal((256, 512)), b.random_normal((512, 256)))
        attn_stitch = (b.random_normal((896, 960)), b.random_normal((960, 896)))
        k_stitch = (b.random_normal((320, 384)), b.random_normal((384, 320)))
        v_stitch = (b.random_normal((320, 384)), b.random_normal((384, 320)))

        registry = build_registry_from_stitches(
            hidden_stitch=hidden_stitch,
            intermediate_stitch=inter_stitch,
            attention_stitch=attn_stitch,
            k_stitch=k_stitch,
            v_stitch=v_stitch,
            backend=b,
        )

        assert registry.get(ActivationSpace.HIDDEN) is not None
        assert registry.get(ActivationSpace.INTERMEDIATE) is not None
        assert registry.get(ActivationSpace.ATTENTION) is not None
        assert registry.get(ActivationSpace.KV) is not None
        assert registry.get(ActivationSpace.V) is not None

    def test_build_partial_registry(self) -> None:
        """Build registry with only some stitches."""
        b = get_default_backend()
        b.random_seed(42)

        hidden_stitch = (b.random_normal((64, 128)), b.random_normal((128, 64)))

        registry = build_registry_from_stitches(
            hidden_stitch=hidden_stitch,
            intermediate_stitch=None,
            attention_stitch=None,
            k_stitch=None,
            v_stitch=None,
            backend=b,
        )

        assert registry.get(ActivationSpace.HIDDEN) is not None
        assert registry.get(ActivationSpace.INTERMEDIATE) is None
        assert registry.get(ActivationSpace.ATTENTION) is None
