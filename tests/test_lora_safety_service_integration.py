# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for LoRA Safety Service per-direction integration."""

import pytest
from modelcypher.core.domain._backend import get_default_backend

@pytest.fixture
def backend():
    return get_default_backend()

class TestLoRASafetyPerDirection:
    """Tests verify_per_direction_bounds in LoRASafetyService."""

    def test_safe_adapter_passes(self, backend):
        """A safe adapter (small magnitude) passes verification."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
        
        m, n, rank = 64, 32, 4
        W = backend.random_normal((m, n))
        A = backend.random_normal((rank, n)) * 0.01
        B = backend.random_normal((m, rank)) * 0.01
        backend.eval(W, A, B)
        
        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)
        
        assert result.is_safe
        assert result.max_ratio < 1.0

    def test_unsafe_adapter_fails(self, backend):
        """An unsafe adapter (large magnitude) fails verification."""
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
        
        m, n, rank = 64, 32, 4
        
        # Create W with small singular values
        W = backend.random_normal((m, n)) * 0.1
        backend.eval(W)
        
        # Create huge LoRA delta
        A = backend.random_normal((rank, n)) * 10.0
        B = backend.random_normal((m, rank)) * 10.0
        backend.eval(A, B)
        
        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)
        
        assert not result.is_safe
        assert len(result.violations) > 0
        assert result.max_ratio > 1.0

    def test_orthogonal_adapter_passes(self, backend):
        """Adapter orthogonal to W passes (since it doesn't affect W's directions).
        
        Note: This tests the current diagonal-only check. A full-matrix check
        would flag off-diagonal interactions, but our current scope is protecting
        existing singular values.
        """
        from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService
        
        m, n, rank = 64, 32, 4
        
        # Create W with defined spectrum to separate signal from noise
        # m=64, n=32. Rank=4.
        # Signal: singular values [10, 10, 10, 10]
        # Noise: singular values [1e-8, ...]
        U_full, _, Vt_full = backend.svd(backend.random_normal((m, n)), compute_uv=True)
        
        # Construct S with gap
        S_vals = [10.0] * rank + [1e-8] * (n - rank)
        S = backend.array(S_vals)
        S_mat = backend.zeros((m, n))
        # Place diagonal
        # This is a bit manual in MLX if we can't just assign diagonal.
        # Easier: W = U[:,:n] @ diag(S) @ Vt
        # But U is [m, m], Vt is [n, n], S is [n] (conceptually)
        
        # W = U[:, :n] @ (S * Vt)
        # S * Vt scales rows of Vt
        SVt = backend.reshape(S, (-1, 1)) * Vt_full
        W = backend.matmul(U_full[:, :n], SVt)
        backend.eval(W)
        
        U, S, Vt = backend.svd(W, compute_uv=True)
        U_k = U[:, :rank]
        backend.eval(U_k)
        
        # Create Delta orthogonal to BOTH U_k and V_k
        # Delta = (I - U_k U_k^T) @ Random @ (I - V_k V_k^T)
        
        # Get V_k from Vt (rows of Vt are v_j^T, so columns of V are v_j)
        V = backend.transpose(Vt)
        V_k = V[:, :rank]
        backend.eval(V_k)
        
        Rand = backend.random_normal((m, n))
        
        # Project out U components
        Proj_U = backend.matmul(U_k, backend.matmul(backend.transpose(U_k), Rand))
        Res_U = Rand - Proj_U
        
        # Project out V components (from the result)
        # Res_U @ V_k @ V_k^T
        Proj_V = backend.matmul(Res_U, backend.matmul(V_k, backend.transpose(V_k)))
        Delta = Res_U - Proj_V
        
        # Scale it up - it's large but orthogonal
        Delta = Delta * 100.0
        backend.eval(Delta)
        
        # Factorize Delta into B, A for the API
        # (Just use SVD to get factors)
        U_d, S_d, Vt_d = backend.svd(Delta, compute_uv=True)
        # B = U_d @ sqrt(S_d), A = sqrt(S_d) @ Vt_d
        S_sqrt = backend.sqrt(S_d[:rank])
        B = U_d[:, :rank] * S_sqrt
        A = Vt_d[:rank, :] * backend.reshape(S_sqrt, (-1, 1))
        backend.eval(B, A)
        
        result = LoRASafetyService.verify_per_direction_bounds(B, A, W, backend)
        
        # Should be safe because projection U^T @ Delta @ V is near zero on diagonal
        # (Diagonal entries measure alignment)
        
        assert result.is_safe, f"Orthogonal delta should be safe, max ratio: {result.max_ratio}"
