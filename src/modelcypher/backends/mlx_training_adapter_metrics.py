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

"""Epoch-level MLX training telemetry types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


def _serialize_metric_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _serialize_metric_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_metric_value(item) for item in value]
    return value


@dataclass
class EpochMetrics:
    """Per-epoch diagnostic metrics for mechanism analysis."""

    epoch: int
    train_loss: float
    val_loss: float | None
    eta: float
    update_norm: float | None
    max_spectral_ratio: float | None
    mean_token_entropy: float | None
    repetition_rate: float | None
    elapsed_seconds: float
    spectral_ratio_growth_per_iter: float | None = None
    eta_ceiling: float | None = None
    adapter_saturation_median_ratio: float | None = None
    expert_saturation_map: dict[str, float] | None = None
    n_saturated_experts: int | None = None
    n_total_target_experts: int | None = None
    # MASS (Measured-Adaptive Step Size) diagnostics
    # Preconditioner P removed (falsification 2026-02-23: P ≈ I).
    # Cayley constraint preserved in NBLoRALinear.
    displacement: float | None = None  # eta_step * ||g||
    eta_sps: float | None = None       # Stochastic Polyak step-size (Loizou et al. 2020)
    eta_weyl: float | None = None      # Per-step Weyl displacement bound
    eta_step: float | None = None      # Actual per-step η = min(SPS, Weyl, ceiling)
    d_norm: float | None = None        # Active update norm for eta_weyl
    # Conformal margin rate (Sahraee-Ardakan, Delbracio & Milanfar 2026)
    eta_margin: float | None = None               # remaining_budget / ||g||
    remaining_budget: float | None = None          # sigma_k_min - ||DeltaW||_2
    max_displacement_to_remaining: float | None = None  # max(disp / remaining) in epoch
    max_effective_gain_ratio: float | None = None  # max(eta_step / eta_ceiling) in epoch
    # Geometric stopping certificate
    cert_grad_norm: float | None = None
    cert_alignment: float | None = None
    cert_curvature: float | None = None
    cert_delta_max_val: float | None = None
    cert_val_ci_half_width: float | None = None
    cert_delta_max_worst: float | None = None
    cert_task_improvement_met: bool | None = None
    cert_all_met: bool | None = None
    # Topological phase diagnostics (optional, computed when topo_monitor=True)
    topo_betti_0: int | None = None
    topo_betti_1: int | None = None
    topo_persistence_entropy: float | None = None
    topo_mean_ricci_curvature: float | None = None
    topo_ricci_curvature_std: float | None = None
    # Detailed dimensional expansion monitoring (emitted when dim_monitor=True)
    dim_expansion_ratio: float | None = None
    dim_peak_dim: float | None = None
    dim_final_dim: float | None = None
    dim_final_used_fraction: float | None = None
    dim_final_null_fraction: float | None = None
    dim_delta_from_baseline: float | None = None
    dim_null_recruitment_from_baseline: float | None = None
    dim_is_contracting: bool | None = None
    # Constrained training diagnostics (optional, when constraint_config provided)
    constraint_mu_inv: float | None = None
    constraint_mu_sep: float | None = None
    constraint_mu_geo: float | None = None
    constraint_C_inv: float | None = None
    constraint_C_sep: float | None = None
    constraint_C_geo: float | None = None
    # Geometric reshaping diagnostics (optional, when geometric_reshape=True)
    reshape_ce_norm: float | None = None
    reshape_expand_norm: float | None = None
    reshape_contrast_norm: float | None = None
    reshape_n_cf_pairs: int | None = None
    reshape_n_inv_pairs: int | None = None
    # Online correctness evaluation (optional, when eval_problems provided)
    online_eval_accuracy: float | None = None
    online_eval_n_correct: int | None = None
    online_eval_n_total: int | None = None
    online_eval_degraded: bool | None = None
    online_eval_degraded_raw: bool | None = None
    online_eval_degraded_significant: bool | None = None
    online_eval_alpha: float | None = None
    online_eval_current_ci_lower: float | None = None
    online_eval_current_ci_upper: float | None = None
    online_eval_baseline_ci_lower: float | None = None
    online_eval_baseline_ci_upper: float | None = None
    online_eval_pre_accuracy: float | None = None
    online_eval_pre_n_correct: int | None = None
    online_eval_pre_n_total: int | None = None
    online_eval_pre_degraded: bool | None = None
    online_eval_pre_degraded_raw: bool | None = None
    online_eval_pre_degraded_significant: bool | None = None
    online_eval_pre_alpha: float | None = None
    online_eval_pre_current_ci_lower: float | None = None
    online_eval_pre_current_ci_upper: float | None = None
    online_eval_pre_baseline_ci_lower: float | None = None
    online_eval_pre_baseline_ci_upper: float | None = None
    online_eval_pre_n_lost: int | None = None
    online_eval_pre_n_gained: int | None = None
    online_eval_pre_per_type_correct: dict[str, int] | None = None
    online_eval_pre_per_type_total: dict[str, int] | None = None
    online_eval_post_accuracy: float | None = None
    online_eval_post_n_correct: int | None = None
    online_eval_post_n_total: int | None = None
    online_eval_post_degraded: bool | None = None
    online_eval_post_degraded_raw: bool | None = None
    online_eval_post_degraded_significant: bool | None = None
    online_eval_post_alpha: float | None = None
    online_eval_post_current_ci_lower: float | None = None
    online_eval_post_current_ci_upper: float | None = None
    online_eval_post_baseline_ci_lower: float | None = None
    online_eval_post_baseline_ci_upper: float | None = None
    online_eval_post_n_lost: int | None = None
    online_eval_post_n_gained: int | None = None
    online_eval_post_per_type_correct: dict[str, int] | None = None
    online_eval_post_per_type_total: dict[str, int] | None = None
    online_eval_stop_basis_accuracy: float | None = None
    online_eval_stop_basis_n_correct: int | None = None
    online_eval_stop_basis_n_total: int | None = None
    online_eval_stop_basis_degraded: bool | None = None
    online_eval_stop_basis_degraded_raw: bool | None = None
    online_eval_stop_basis_degraded_significant: bool | None = None
    online_eval_stop_basis_alpha: float | None = None
    online_eval_stop_basis_current_ci_lower: float | None = None
    online_eval_stop_basis_current_ci_upper: float | None = None
    online_eval_stop_basis_baseline_ci_lower: float | None = None
    online_eval_stop_basis_baseline_ci_upper: float | None = None
    online_eval_stop_basis_stage: str | None = None
    gate_confound_event: bool | None = None
    # Outer similarity monitoring (optional, when rss_monitor=True)
    # Kucukahmetler et al. (2026) TMLR — base vs adapted relative representations
    rss_cosine: float | None = None
    rss_spearman: float | None = None
    rss_top1_agreement: float | None = None
    # Projected residual diagnostic (tighter than spectral norm ratio)
    projected_residual_max: float | None = None
    # Answer margin time series (P2: decision boundary confidence)
    margin_median: float | None = None
    margin_mean: float | None = None
    margin_min: float | None = None
    margin_n_near_zero: int | None = None
    margin_n_flipped: int | None = None
    # Adapter stable rank per layer (P3: memorization detection)
    stable_rank_median: float | None = None
    stable_rank_min: float | None = None
    per_layer_stable_rank: dict[str, float] | None = None
    # Token-weighted loss (P4: LongPPL-style capability-weighted observation)
    token_weighted_val_loss: float | None = None
    # Effective rank trend (P5: plasticity loss detection)
    effective_rank: float | None = None
    effective_rank_declining_streak: int | None = None
    # Research-only controller tracing (raw measurements, no interpretation)
    controller_mode: str | None = None
    optimizer_research_mode: str | None = None
    controller_trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _serialize_metric_value(value)
            for key, value in self.__dict__.items()
        }


__all__ = ["EpochMetrics"]
