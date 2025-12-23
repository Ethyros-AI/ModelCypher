
from typing import Dict, Any, Optional, List, Union
import mlx.core as mx
from modelcypher.core.domain.geometry.types import (
    MergerConfig, MergerResult, BatchMergerResult
)
from modelcypher.infrastructure.adapters.mlx.optimal_transport import GromovWassersteinSolver

class TransportGuidedMerger:
    """
    Implements Transport-Guided Merger logic.
    """
    
    @staticmethod
    async def merge_models(
        source_weights: Any,
        target_weights: Any,
        source_activations: Any,
        target_activations: Any,
        config: MergerConfig
    ) -> Union[MergerResult, BatchMergerResult]:
        
        # Check if batch (dict of layers) or single layer
        is_batch = isinstance(source_weights, dict) and isinstance(source_activations, dict)
        
        if is_batch:
            return await TransportGuidedMerger._merge_batch(
                source_weights, target_weights, 
                source_activations, target_activations, 
                config
            )
        else:
            return await TransportGuidedMerger._merge_single(
                source_weights, target_weights, 
                source_activations, target_activations, 
                config
            )

    @staticmethod
    async def _merge_single(
        source_weights: Any,
        target_weights: Any,
        source_activations: Any,
        target_activations: Any,
        config: MergerConfig
    ) -> MergerResult:
        
        S_acts = mx.array(source_activations)
        T_acts = mx.array(target_activations)
        S_w = mx.array(source_weights)
        T_w = mx.array(target_weights)
        
        n_samples = S_acts.shape[0]
        if n_samples < config.min_samples:
            raise ValueError(f"Insufficient samples: {n_samples} < {config.min_samples}")
            
        # Compute Intra-space distances
        C1 = GromovWassersteinSolver.compute_pairwise_distances(S_acts)
        C2 = GromovWassersteinSolver.compute_pairwise_distances(T_acts)
        
        # Marginals (uniform)
        p = mx.ones((n_samples,)) / n_samples
        q = mx.ones((n_samples,)) / n_samples # Assumption: same number of samples
        
        # Solve GW
        T, dist, iters = GromovWassersteinSolver.solve(
            C1, C2, p, q, 
            epsilon=config.gw_config.epsilon,
            max_iter=config.gw_config.max_iter,
            threshold=config.gw_config.threshold
        )
        
        # Transport Plan T maps samples to samples. 
        # But we need to map Neurons (features) or Weights?
        # Swift code aligns activations to weights first.
        # "alignActivationsToWeights": Transposes if needed so acts are [Neurons, Samples] ?
        # Wait, GW in Swift compares "Source Points" and "Target Points".
        # If we map Weights, the points should be Neurons (rows of W), and distance is computed based on their Activations (response profile).
        # Correct approach:
        # 1. Represent each Neuron as a point in Activation Space (over samples).
        #    X_source: [N_neurons, N_samples]
        # 2. Compute Dist matrix between Neurons: C1 [N_neurons, N_neurons].
        # 3. Solve GW to get T [N_neurons_S, N_neurons_T].
        # 4. Use T to transport W_source -> W_target dimensions.
        
        # Let's adjust inputs. source_activations usually [Samples, Neurons].
        # We need to transpose to [Neurons, Samples] to treat Neurons as the "points" being transported.
        
        # Check dimensions
        # S_w: [Out, In] or [N_neurons, D_emb]
        # S_acts: [Samples, N_neurons]
        
        n_neurons_s = S_w.shape[0]
        if S_acts.shape[1] == n_neurons_s:
            X_s = S_acts.T # [N_neurons, Samples]
        else:
             # Maybe already transposed?
             X_s = S_acts
             
        # Same for target
        n_neurons_t = T_w.shape[0]
        if T_acts.shape[1] == n_neurons_t:
             X_t = T_acts.T
        else:
             X_t = T_acts
             
        # Compute Kernel Matrices (Correlation/Distance between Neurons)
        # Using correlation or euclidean distance on activation profiles?
        # Swift uses Euclidean on aligned points.
        C1 = GromovWassersteinSolver.compute_pairwise_distances(X_s) # [Ns, Ns]
        C2 = GromovWassersteinSolver.compute_pairwise_distances(X_t) # [Nt, Nt]
        
        p = mx.ones((n_neurons_s,)) / n_neurons_s
        q = mx.ones((n_neurons_t,)) / n_neurons_t
        
        T_plan, dist, iters = GromovWassersteinSolver.solve(
            C1, C2, p, q,
            epsilon=config.gw_config.epsilon
        )
        # T_plan is [Ns, Nt]
        
        # Synthesize: W_merged = T_plan.T @ S_w  (Mapping Source neurons to Target structure)
        # Wait, shape math:
        # W_merged should match Target shape [Nt, D].
        # S_w is [Ns, D].
        # We want to form a linear combination of Source rows to populate Target rows.
        # If T_plan[i, j] is mass from S_i to T_j.
        # Target node j receives sum_i (T[i,j] * S_i).
        # So W_merged[j] = sum_i T[i,j] * S_w[i]
        # W_merged = T_plan.T @ S_w
        # [Nt, Ns] @ [Ns, D] = [Nt, D]. Correct.
        
        # Normalize rows of T?
        if config.normalize_rows:
             # Normalize T such that rows sum to 1? Or cols?
             # If we want T_j to be a valid average, sum_i T[i,j] should be 1.
             # This means columns of T should sum to 1.
             col_sums = mx.sum(T_plan, axis=0, keepdims=True) + 1e-9
             T_matrix = T_plan / col_sums # Normalize columns to sum to 1
        else:
             T_matrix = T_plan * n_neurons_s # Scale up??
             # Standard OT plan sums to 1 total. We need it to act as weight.
             # Usually we want barycentric projection.
             # Barycentric mapping: T_opt = T @ diag(1/q) ??
             # Let's use column normalization approach.
             pass
             
        W_aligned = T_matrix.T @ S_w
        
        # Blend
        alpha = config.blend_alpha
        W_merged = (1 - alpha) * W_aligned + alpha * T_w

        # Compute marginal error: how well T_plan marginals match p and q
        # Row sums should equal p, column sums should equal q
        row_marginal = mx.sum(T_plan, axis=1)  # Should equal p
        col_marginal = mx.sum(T_plan, axis=0)  # Should equal q
        row_error = float(mx.mean(mx.abs(row_marginal - p)).item())
        col_error = float(mx.mean(mx.abs(col_marginal - q)).item())
        marginal_error = (row_error + col_error) / 2.0

        return MergerResult(
            merged_weights=W_merged,
            gw_distance=dist,
            marginal_error=marginal_error,
            effective_rank=0,
            converged=True,
            iterations=iters,
            dimension_confidences=[],
        )

    @staticmethod
    async def _merge_batch(
        source_weights: Dict[str, Any],
        target_weights: Dict[str, Any],
        source_activations: Dict[str, Any],
        target_activations: Dict[str, Any],
        config: MergerConfig
    ) -> BatchMergerResult:
        results = {}
        failed = []
        total_dist = 0.0
        
        total_marginal_error = 0.0

        for k in source_weights.keys():
            if k in target_weights and k in source_activations and k in target_activations:
                try:
                    res = await TransportGuidedMerger._merge_single(
                        source_weights[k], target_weights[k],
                        source_activations[k], target_activations[k],
                        config
                    )
                    results[k] = res
                    total_dist += res.gw_distance
                    total_marginal_error += res.marginal_error
                except Exception as e:
                    failed.append(k)
            else:
                failed.append(k)

        quality = len(results) / (len(results) + len(failed)) if results else 0.0
        n_results = len(results)

        return BatchMergerResult(
            layer_results=results,
            mean_gw_distance=total_dist / n_results if n_results else 0.0,
            mean_marginal_error=total_marginal_error / n_results if n_results else 0.0,
            failed_layers=failed,
            quality_score=quality,
        )
