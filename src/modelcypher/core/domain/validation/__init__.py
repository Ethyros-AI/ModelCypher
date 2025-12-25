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

"""Validation domain modules.

This package contains modules for dataset validation and quality scoring,
including automatic dataset repair and quality metrics.
"""

from modelcypher.core.domain.validation.auto_fix_engine import (
    AutoFixEngine,
    AutoFixResult,
    Fix,
    FixType,
    UnfixableLine,
)
from modelcypher.core.domain.validation.dataset_file_enumerator import (
    CompressionType,
    DatasetFileEnumerator,
    DatasetFileFormat,
    EnumeratedSample,
    EnumerationError,
    FileMetadata,
)
from modelcypher.core.domain.validation.dataset_format_analyzer import (
    DatasetFormatAnalyzer,
    FormatAnalysisResult,
)
from modelcypher.core.domain.validation.dataset_quality_scorer import (
    DatasetQualityScorer,
    QualityScore,
    ScoreRange,
)
from modelcypher.core.domain.validation.dataset_text_extractor import (
    DatasetTextExtractor,
    ExtractedText,
)
from modelcypher.core.domain.validation.dataset_validation_models import (
    DatasetContentFormat,
    DatasetPreparationError,
    DatasetPreparationErrorKind,
    DatasetStats,
    DatasetValidationProgress,
    QuickValidationResult,
    ValidatedTextDataset,
    ValidationError,
    ValidationErrorKind,
    ValidationResult,
    ValidationStatus,
    ValidationWarning,
    ValidationWarningKind,
)
from modelcypher.core.domain.validation.dataset_validator import (
    DatasetValidator,
    ProgressCallback,
    ValidationCache,
)
from modelcypher.core.domain.validation.intrinsic_identity_linter import (
    DatasetIdentityScanner,
    IdentityFinding,
    IntrinsicIdentityLinter,
)

__all__ = [
    # Auto-fix
    "AutoFixEngine",
    "AutoFixResult",
    "Fix",
    "FixType",
    "UnfixableLine",
    # Quality scoring
    "DatasetQualityScorer",
    "QualityScore",
    "ScoreRange",
    # Validation models
    "DatasetContentFormat",
    "DatasetPreparationError",
    "DatasetPreparationErrorKind",
    "DatasetStats",
    "DatasetValidationProgress",
    "QuickValidationResult",
    "ValidatedTextDataset",
    "ValidationError",
    "ValidationErrorKind",
    "ValidationResult",
    "ValidationStatus",
    "ValidationWarning",
    "ValidationWarningKind",
    # Format analyzer
    "DatasetFormatAnalyzer",
    "FormatAnalysisResult",
    # Validator
    "DatasetValidator",
    "ProgressCallback",
    "ValidationCache",
    # Identity linter
    "DatasetIdentityScanner",
    "IdentityFinding",
    "IntrinsicIdentityLinter",
    # Text extractor
    "DatasetTextExtractor",
    "ExtractedText",
    # File enumerator
    "CompressionType",
    "DatasetFileEnumerator",
    "DatasetFileFormat",
    "EnumeratedSample",
    "EnumerationError",
    "FileMetadata",
]
