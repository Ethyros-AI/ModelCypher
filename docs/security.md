# Security Policy

ModelCypher handles sensitive model weights and potential training data. Security is paramount.

## 1. Secrets Management

-   **Never commit API keys**. Use `.env` files.
-   ModelCypher respects `HF_TOKEN` environment variables for Hugging Face authentication.
-   **Do not hardcode** tokens in Python scripts.

## 2. Safe Tensors & Weights

-   **Prefer `safetensors`**: We default to `safetensors` for all model saving/loading to avoid pickle execution vulnerabilities.
-   **Pickle Warning**: ModelCypher will emit a warning if you load a `.bin` or `.CHECKPOINT` file that relies on `torch.load` (pickle).
-   **Untrusted Models**: Do not run `mc train …` or `mc model …` workflows on models from untrusted sources if they rely on pickle formats.

## 3. Network Security

-   **TLS 1.2+**: All network connections (HF Hub, MLflow) default to strict TLS.
-   **No Telemetry**: ModelCypher does **not** phone home. All "training dynamics" data is stored locally or sent only to your configured MLflow server.

## 4. Reporting Vulnerabilities

If you discover a security vulnerability, please do not open a public issue. Email `security@modelcypher.com`.
