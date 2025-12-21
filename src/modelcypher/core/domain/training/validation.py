from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from .types import Hyperparameters

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class Violation:
    field: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None

class TrainingHyperparameterValidator:
    """
    Validates training hyperparameters against known best practices for Apple Silicon (MLX).
    Ported from Swift TrainingHyperparameterValidator.
    """
    
    # Thresholds ported from Swift ParameterThresholds.swift
    BATCH_SIZE_RANGE = range(1, 9) # 1-8 reasonable for local
    BATCH_SIZE_INFO_THRESHOLD = 4
    
    SEQUENCE_MIN = 128
    SEQUENCE_MAX = 4096 # MLX limitation/memory constraint
    SEQUENCE_WARNING = 2048
    
    LR_MIN = 1e-6
    LR_MAX = 1e-3
    LR_WARN_HIGH = 5e-4
    LR_INFO_LOW = 1e-5
    
    EPOCHS_MIN = 1
    EPOCHS_MAX_REC = 10
    
    GRAD_ACCUM_MAX = 16
    GRAD_ACCUM_WARN = 8

    @classmethod
    def comprehensive_violations(cls, params: Hyperparameters) -> List[Violation]:
        violations = []
        
        # Batch Size
        if params.batch_size <= 0:
             violations.append(Violation("batch_size", "Batch size must be > 0", ValidationSeverity.ERROR, "Start with 2 or 4"))
        elif params.batch_size not in cls.BATCH_SIZE_RANGE:
             violations.append(Violation("batch_size", "Batch size must be between 1 and 8 for Apple Silicon", ValidationSeverity.ERROR, "Start with 2 or 4"))
        elif params.batch_size > cls.BATCH_SIZE_INFO_THRESHOLD:
             violations.append(Violation("batch_size", "Batch sizes > 4 require significant memory", ValidationSeverity.INFO, "Monitor memory usage"))

        # Sequence Length
        if params.sequence_length < cls.SEQUENCE_MIN:
            violations.append(Violation("sequence_length", f"Sequence length must be at least {cls.SEQUENCE_MIN}", ValidationSeverity.ERROR))
        elif params.sequence_length > cls.SEQUENCE_MAX:
             violations.append(Violation("sequence_length", f"Sequence length > {cls.SEQUENCE_MAX} exceeds tested limits", ValidationSeverity.ERROR, "Reduce to 4096 or lower"))
        elif params.sequence_length > cls.SEQUENCE_WARNING:
             violations.append(Violation("sequence_length", f"Sequences > {cls.SEQUENCE_WARNING} increase memory usage significantly", ValidationSeverity.WARNING))

        # Learning Rate
        if not (cls.LR_MIN <= params.learning_rate <= cls.LR_MAX):
             violations.append(Violation("learning_rate", "Learning rate must be between 1e-6 and 1e-3", ValidationSeverity.ERROR, "Start with 3e-5 for LoRA"))
        elif params.learning_rate > cls.LR_WARN_HIGH:
             violations.append(Violation("learning_rate", "Learning rate above 5e-4 can destabilize training", ValidationSeverity.WARNING))
        elif params.learning_rate < cls.LR_INFO_LOW:
             violations.append(Violation("learning_rate", "Learning rate is very low", ValidationSeverity.INFO))

        # Epochs
        if params.epochs < cls.EPOCHS_MIN:
            violations.append(Violation("epochs", "Epochs must be positive", ValidationSeverity.ERROR))
        elif params.epochs > cls.EPOCHS_MAX_REC:
            violations.append(Violation("epochs", "Epochs > 10 often overfit small datasets", ValidationSeverity.WARNING, "Keep within 3-8 epochs"))

        return violations

    @classmethod
    def validate_for_engine(cls, params: Hyperparameters) -> None:
        """Throws ValueError on first ERROR severity violation."""
        violations = cls.comprehensive_violations(params)
        errors = [v for v in violations if v.severity == ValidationSeverity.ERROR]
        if errors:
            raise ValueError(f"Invalid Configuration: {errors[0].message}")
