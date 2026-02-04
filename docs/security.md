# Security Policy

ModelCypher handles sensitive model weights and potential training data. Security is paramount.

## 1. Secrets Management

-   **Never commit API keys**. Use `.env` files.
-   ModelCypher respects `HF_TOKEN` environment variables for Hugging Face authentication.
-   **Do not hardcode** tokens in Python scripts.

## 2. Safe Tensors & Weights

-   **Prefer `safetensors`**: We default to `safetensors` for all model saving/loading to avoid pickle execution vulnerabilities.
-   **Pickle Warning**: Some workflows (e.g., LoRA adapter merge) may load `.bin`/`.pt` files using pickle-based loaders.
-   **Untrusted Models**: Do not load `.bin`/`.pt` weights from untrusted sources.

### 2.1 Remote Code Loading (Hugging Face)

ModelCypher disables `trust_remote_code` by default to avoid executing model-supplied code.

```bash
export MC_TRUST_REMOTE_CODE=1
```

Only enable this for models you trust.

## 3. Network Security

-   **TLS 1.2+**: All network connections (HF Hub, MLflow) default to strict TLS.
-   **No Telemetry**: ModelCypher does **not** phone home. All "training dynamics" data is stored locally or sent only to your configured MLflow server.

## 4. Reporting Vulnerabilities

If you discover a security vulnerability, please do not open a public issue. Email `jason@ethyros.ai`.
