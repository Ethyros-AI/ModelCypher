# Adapter-side geometry utilities that may rely on NumPy/SciPy or other
# non-backend dependencies. These are intentionally kept out of the core
# domain layer to avoid CPU fallbacks in merge-critical paths.

__all__ = [
    "hash_analyzer",
    "lora_geometry_diagnostic",
]
