from __future__ import annotations

from modelcypher.ports.backend import Backend


def default_backend() -> Backend:
    try:
        from modelcypher.backends.mlx_backend import MLXBackend

        return MLXBackend()
    except Exception as exc:  # pragma: no cover - fallback only
        raise RuntimeError("MLX backend unavailable") from exc
