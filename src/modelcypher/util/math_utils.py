
import math

def compute_coefficient_of_variation(values: list[float]) -> float:
    """Compute the coefficient of variation (std / mean) for a list of values."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    return std / mean
