from __future__ import annotations


class EstimatorError(Exception):
    """Exception raised for errors in geometric projectors or estimators."""
    pass


class ProjectionError(Exception):
    """Exception raised for errors during dimensionality reduction or manifold projection."""
    pass
