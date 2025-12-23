from __future__ import annotations


class EstimatorError(Exception):
    """Exception raised for errors in geometric projectors or estimators."""
    
    def __init__(self, kind: str, message: str, count: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.count = count

    @staticmethod
    def insufficient_samples(count: int) -> "EstimatorError":
        return EstimatorError(
            "insufficientSamples",
            f"Intrinsic dimension estimation requires at least 3 samples (got {count}).",
            count=count,
        )

    @staticmethod
    def invalid_point_dimension(expected: int, found: int) -> "EstimatorError":
        return EstimatorError(
            "invalidPointDimension",
            f"All points must have the same dimensionality (expected {expected}, found {found}).",
        )

    @staticmethod
    def non_finite_point_value() -> "EstimatorError":
        return EstimatorError("nonFinitePointValue", "Points contain non-finite values (NaN/Inf).")

    @staticmethod
    def nearest_neighbor_degenerate() -> "EstimatorError":
        return EstimatorError(
            "nearestNeighborDegenerate",
            "Nearest-neighbor distances are degenerate (duplicates or zero distances).",
        )

    @staticmethod
    def regression_degenerate() -> "EstimatorError":
        return EstimatorError(
            "regressionDegenerate",
            "Regression is degenerate (insufficient variance in log(mu)).",
        )


class ProjectionError(Exception):
    """Exception raised for errors during dimensionality reduction or manifold projection."""
    pass
