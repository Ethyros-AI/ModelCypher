# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Per-head QK spectral analysis for softcap equivalence.

Measures sigma_max(W_Q_h) and sigma_max(W_K_h) per attention head, then
applies the derived bound from :mod:`qk_spectral_bound` to determine whether
logit softcapping is geometrically active or redundant.

Follows the layer-iteration pattern from
:func:`perturbation_bound.compute_layer_lipschitz_bounds`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.perturbation_bound import (
    _get_weight,
    _power_iter_sigma_max,
)
from modelcypher.core.domain.geometry.qk_spectral_bound import (
    HeadCompositionChange,
    HeadSpectralBound,
    composition_significant,
    max_logit_magnitude,
    qk_projection_scale,
    qk_spectral_product,
    softcap_equivalent_bound,
    softcap_utilization,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QKSpectralReport:
    """Model-level QK spectral analysis."""

    d_model: int
    d_k: int
    num_heads: int
    num_kv_heads: int
    soft_cap: float | None
    derived_bound: float | None
    per_head: list[HeadSpectralBound]
    heads_softcap_active: int
    heads_total: int
    mean_utilization: float
    max_utilization: float
    equivalent_softcap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_k": self.d_k,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "soft_cap": self.soft_cap,
            "derived_bound": self.derived_bound,
            "heads_softcap_active": self.heads_softcap_active,
            "heads_total": self.heads_total,
            "mean_utilization": self.mean_utilization,
            "max_utilization": self.max_utilization,
            "equivalent_softcap": self.equivalent_softcap,
        }


_EPS_F32 = 2.0**-23  # IEEE 754 float32 machine epsilon


@dataclass(frozen=True)
class QKCompositionReport:
    """Comparison of QK spectral products between two model states."""

    per_head: list[HeadCompositionChange]
    heads_total: int
    heads_significant: int
    mean_relative_change: float
    max_relative_change: float
    max_absolute_change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "heads_total": self.heads_total,
            "heads_significant": self.heads_significant,
            "mean_relative_change": self.mean_relative_change,
            "max_relative_change": self.max_relative_change,
            "max_absolute_change": self.max_absolute_change,
        }


class QKSpectralService:
    """Analyzes per-head QK spectral products against the softcap-equivalent bound."""

    def __init__(self, backend: "Backend") -> None:
        self._backend = backend

    def analyze_model(
        self,
        model: Any,
        soft_cap: float | None = None,
    ) -> QKSpectralReport:
        """Measure per-head QK spectral products and compare to derived bound.

        Parameters
        ----------
        model :
            Loaded model with ``model.config`` attributes and layer weights.
        soft_cap :
            Softcap value c. If None, reports measurements without bound comparison
            and computes the equivalent softcap that would match the natural constraint.
        """
        b = self._backend
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model has no config attribute")

        d_model = getattr(config, "hidden_size", 0)
        num_heads = getattr(config, "num_attention_heads", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        if d_model == 0 or num_heads == 0:
            raise ValueError(
                f"Cannot extract dimensions: hidden_size={d_model}, "
                f"num_attention_heads={num_heads}"
            )

        d_k = d_model // num_heads
        heads_per_kv_group = num_heads // num_kv_heads

        bound: float | None = None
        if soft_cap is not None:
            bound = softcap_equivalent_bound(soft_cap, d_k, d_model)

        base = getattr(model, "model", model)
        per_head: list[HeadSpectralBound] = []

        for layer_idx, layer in enumerate(base.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            w_q = _get_weight(attn, ["q_proj"])
            w_k = _get_weight(attn, ["k_proj"])
            if w_q is None or w_k is None:
                logger.debug("Layer %d: missing q_proj or k_proj, skipping", layer_idx)
                continue

            # Per-head sigma_max via power iteration on [head_dim, hidden_dim] slices.
            # Q weight shape: [num_heads * head_dim, hidden_dim]
            # K weight shape: [num_kv_heads * head_dim, hidden_dim]
            q_f32 = b.astype(w_q, "float32")
            k_f32 = b.astype(w_k, "float32")
            b.eval(q_f32)
            b.eval(k_f32)

            hidden_dim = int(q_f32.shape[1]) if len(q_f32.shape) > 1 else int(q_f32.shape[0])

            # Reshape to [n_heads, head_dim, hidden_dim] for per-head slicing
            q_reshaped = b.reshape(q_f32, (num_heads, d_k, hidden_dim))
            k_reshaped = b.reshape(k_f32, (num_kv_heads, d_k, hidden_dim))
            b.eval(q_reshaped)
            b.eval(k_reshaped)

            # Compute sigma_max for each K head once (shared across Q group)
            k_sigmas: list[float] = []
            for g in range(num_kv_heads):
                k_h = k_reshaped[g]  # [head_dim, hidden_dim]
                k_sigmas.append(_power_iter_sigma_max(k_h, b))

            # Per Q head: compute sigma_max and pair with its K group
            for h in range(num_heads):
                q_h = q_reshaped[h]  # [head_dim, hidden_dim]
                sig_q = _power_iter_sigma_max(q_h, b)

                kv_group = h // heads_per_kv_group
                sig_k = k_sigmas[kv_group]

                product = qk_spectral_product(sig_q, sig_k)

                if bound is not None:
                    util = softcap_utilization(sig_q, sig_k, bound)
                    proj = qk_projection_scale(sig_q, sig_k, bound)
                    active = util > 1.0
                else:
                    util = 0.0
                    proj = 1.0
                    active = False

                ml = max_logit_magnitude(sig_q, sig_k, d_k, d_model)

                per_head.append(
                    HeadSpectralBound(
                        layer_idx=layer_idx,
                        head_idx=h,
                        sigma_q=sig_q,
                        sigma_k=sig_k,
                        spectral_product=product,
                        bound=bound if bound is not None else 0.0,
                        utilization=util,
                        projection_scale=proj,
                        max_logit=ml,
                        softcap_active=active,
                    )
                )

        # Aggregate
        heads_total = len(per_head)
        heads_active = sum(1 for h in per_head if h.softcap_active)
        mean_util = (
            sum(h.utilization for h in per_head) / heads_total
            if heads_total > 0
            else 0.0
        )
        max_util = max((h.utilization for h in per_head), default=0.0)

        # Equivalent softcap: the minimum c that would deactivate all heads.
        # From max_logit = d_model * sigma_Q * sigma_K / sqrt(d_k),
        # the equivalent softcap is the max logit across all heads.
        eq_softcap = max((h.max_logit for h in per_head), default=0.0)

        return QKSpectralReport(
            d_model=d_model,
            d_k=d_k,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            soft_cap=soft_cap,
            derived_bound=bound,
            per_head=per_head,
            heads_softcap_active=heads_active,
            heads_total=heads_total,
            mean_utilization=mean_util,
            max_utilization=max_util,
            equivalent_softcap=eq_softcap,
        )

    def compare_models(
        self,
        base_model: Any,
        modified_model: Any,
        eps: float = _EPS_F32,
    ) -> QKCompositionReport:
        """Compare per-head QK spectral products between base and modified model.

        Measures the composition change Δ(σ_Q × σ_K) per head to detect
        attention selectivity drift invisible to per-matrix Weyl monitoring.

        Parameters
        ----------
        base_model :
            Original (unmodified) model.
        modified_model :
            Model after modification (training, correction, merge, etc.).
        eps :
            Machine epsilon for significance testing (default float32).
        """
        base_report = self.analyze_model(base_model)
        mod_report = self.analyze_model(modified_model)

        if base_report.heads_total != mod_report.heads_total:
            raise ValueError(
                f"Head count mismatch: base={base_report.heads_total}, "
                f"modified={mod_report.heads_total}"
            )

        changes: list[HeadCompositionChange] = []
        for base_h, mod_h in zip(base_report.per_head, mod_report.per_head):
            abs_change = abs(mod_h.spectral_product - base_h.spectral_product)
            rel_change = (
                abs_change / base_h.spectral_product
                if base_h.spectral_product > 0
                else 0.0
            )
            sig = composition_significant(rel_change, eps)

            changes.append(
                HeadCompositionChange(
                    layer_idx=base_h.layer_idx,
                    head_idx=base_h.head_idx,
                    base_product=base_h.spectral_product,
                    modified_product=mod_h.spectral_product,
                    absolute_change=abs_change,
                    relative_change=rel_change,
                    significant=sig,
                )
            )

        n_total = len(changes)
        n_sig = sum(1 for c in changes if c.significant)
        mean_rel = (
            sum(c.relative_change for c in changes) / n_total
            if n_total > 0
            else 0.0
        )
        max_rel = max((c.relative_change for c in changes), default=0.0)
        max_abs = max((c.absolute_change for c in changes), default=0.0)

        return QKCompositionReport(
            per_head=changes,
            heads_total=n_total,
            heads_significant=n_sig,
            mean_relative_change=mean_rel,
            max_relative_change=max_rel,
            max_absolute_change=max_abs,
        )
