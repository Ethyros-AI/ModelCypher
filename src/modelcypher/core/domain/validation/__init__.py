"""Validation domain modules.

This package contains modules for dataset validation and quality scoring,
including automatic dataset repair and quality metrics.
"""

from modelcypher.core.domain.validation.dataset_quality_scorer import (
    DatasetQualityScorer,
    QualityScore,
    ScoreRange,
)
from modelcypher.core.domain.validation.auto_fix_engine import (
    AutoFixEngine,
    AutoFixResult,
    Fix,
    FixType,
    UnfixableLine,
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
from modelcypher.core.domain.validation.dataset_format_analyzer import (
    DatasetFormatAnalyzer,
    FormatAnalysisResult,
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
from modelcypher.core.domain.validation.dataset_text_extractor import (
    DatasetTextExtractor,
    ExtractedText,
)
from modelcypher.core.domain.validation.dataset_file_enumerator import (
    CompressionType,
    DatasetFileEnumerator,
    DatasetFileFormat,
    EnumeratedSample,
    EnumerationError,
    FileMetadata,
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
