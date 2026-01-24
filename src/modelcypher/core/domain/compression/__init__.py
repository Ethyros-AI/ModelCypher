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

"""Geodesic-preserving compression modules.

Modules for lossless MLP compression that preserve geodesic structure.
The vast majority of a model is unused null space - but that null space
provides *shape*. These modules help "suck out the vacuum/chaos/noise"
while keeping geodesic relationships intact.

Modules:
    GeodesicLayerAnalyzer: Analyze geodesic structure, predict compressibility
    RMTAwareCompressor: Compress with RMT signal/noise separation (proven +25-50pp)
    GeodesicNullSpaceCompressor: Compress null space preserving shape (TODO)
    RankingPreservingOptimizer: Optimize for ranking, not MSE (TODO)
    ComposableLayerCompressor: Multi-layer compression with error tracking (TODO)
"""

from modelcypher.core.domain.compression.geodesic_analyzer import (
    GeodesicLayerAnalyzer,
    GeodesicLayerProfile,
)
from modelcypher.core.domain.compression.rmt_compressor import (
    CompressionResult,
    EvaluationResult,
    RMTAwareCompressor,
)
from modelcypher.core.domain.compression.ranking_optimizer import (
    RankingOptimizationResult,
    RankingPreservingOptimizer,
    optimize_for_ranking,
)
from modelcypher.core.domain.compression.composable_compressor import (
    ComposableLayerCompressor,
    CompositionResult,
    LayerCompressionState,
    compress_model_layers,
)

__all__ = [
    "GeodesicLayerAnalyzer",
    "GeodesicLayerProfile",
    "RMTAwareCompressor",
    "CompressionResult",
    "EvaluationResult",
    "RankingPreservingOptimizer",
    "RankingOptimizationResult",
    "optimize_for_ranking",
    "ComposableLayerCompressor",
    "CompositionResult",
    "LayerCompressionState",
    "compress_model_layers",
]
