"""
Vector Math Utilities.

Provides common vector operations for geometry domain.
"""
from __future__ import annotations

import math
from typing import List, Optional, Union

class VectorMath:
    @staticmethod
    def dot(a: List[float], b: List[float]) -> Optional[float]:
        if len(a) != len(b):
            return None
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def l2_norm(a: List[float]) -> Optional[float]:
        if not a:
            return None
        return math.sqrt(sum(x * x for x in a))

    @staticmethod
    def l2_normalized(a: List[float]) -> List[float]:
        norm = VectorMath.l2_norm(a)
        if norm is None or norm == 0:
            return a
        return [x / norm for x in a]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
        if len(a) != len(b):
            return None
        
        dot_product = VectorMath.dot(a, b)
        if dot_product is None:
            return None
            
        norm_a = VectorMath.l2_norm(a)
        norm_b = VectorMath.l2_norm(b)
        
        if norm_a is None or norm_b is None or norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
