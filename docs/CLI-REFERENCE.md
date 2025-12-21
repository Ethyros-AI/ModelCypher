# CLI Reference

ModelCypher provides three primary CLI tools:
1.  **`mc-inspect`**: Analysis & Geometry
2.  **`mc-train`**: Training & Adaptation
3.  **`mc-dynamics`**: Training Physics & Monitoring

## Global Flags
-   `--json`: Output all results as JSON (essential for agent parsing).
-   `--verbose`: Enable debug logging.

---

## `mc-inspect`

Geometric analysis tools.

### `scan`
Scans a model for geometric signatures.
```bash
mc-inspect scan --model <path_or_hub_id> [--output <path>]
```
-   **Output**: JSON containing `intrinsic_dimension`, `refusal_vector_magnitude`, `topology_betti_numbers`.

### `intersection`
Computes the "Intersection Map" (Venn Diagram) between two models.
```bash
mc-inspect intersection --source <model_A> --target <model_B>
```
-   **Output**: JSON with `jaccard_similarity`, `rotation_matrix` (Procrustes), `aligned_subspace_rank`.

---

## `mc-train`

Training with geometric constraints.

### `lora`
Train a LoRA adapter.
```bash
mc-train lora \
    --model <base_model> \
    --data <train_data> \
    --rank <int> \
    --alpha <float> \
    --target-modules <list>
```

### `circuit-breaker`
Train a circuit breaker safety adapter.
```bash
mc-train circuit-breaker \
    --model <base> \
    --safe-data <safe> \
    --harmful-data <harmful> \
    --threshold <float>
```

---

## `mc-dynamics`

Physics of the training process.

### `analyze-gradients`
Compute Signal-to-Noise Ratio (SNR) and Smoothness.
```bash
mc-dynamics analyze-gradients --run-id <mlflow_run_id>
```

### `regime-detect`
Identify which "Phase" (Memorization, Generalization, Confusion) the training is in.
```bash
mc-dynamics regime-detect --loss-history <json_file>
```
