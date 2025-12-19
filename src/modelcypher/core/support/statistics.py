from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    clamped = max(0.0, min(1.0, p))
    position = clamped * float(len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    fraction = position - float(lower_index)
    return lower_value + (upper_value - lower_value) * fraction


def standard_deviation(values: list[float], mean_value: float | None = None) -> float:
    if len(values) < 2:
        return 0.0
    if mean_value is None:
        mean_value = mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(max(0.0, variance))


def standard_deviation_population(values: list[float], mean_value: float | None = None) -> float:
    if not values:
        return 0.0
    if mean_value is None:
        mean_value = mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return math.sqrt(max(0.0, variance))
