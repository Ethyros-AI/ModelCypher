# Security Policy

ModelCypher handles sensitive model weights and potential training data. Security is paramount.

## 1. Secrets Management

-   **Never commit API keys**. Use `.env` files.
-   ModelCypher respects `HF_TOKEN` environment variables for Hugging Face authentication.
-   **Do not hardcode** tokens in Python scripts.

## 2. Safe Tensors & Weights

-   **Prefer `safetensors`**: We default to `safetensors` for all model saving/loading to avoid pickle execution vulnerabilities.
-   **Pickle Warning**: Some workflows (e.g., LoRA adapter merge) may use `torch.load` for `.bin`/`.pt` files, which relies on pickle.
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

## 4. MCP Server Security

The ModelCypher MCP server implements the MCP 2025-06-18 specification security features:

### 4.1 OAuth 2.1 Token Validation (Optional)

Enable OAuth 2.1 resource server validation for production deployments:

```bash
# Enable OAuth token validation
export MC_MCP_AUTH_ENABLED=1
export MC_MCP_AUTH_ISSUER=https://auth.example.com
export MC_MCP_AUTH_AUDIENCE=https://mcp.example.com
export MC_MCP_AUTH_JWKS_URI=https://auth.example.com/.well-known/jwks.json
```

**Supported Features:**
- RFC 8707 audience validation
- JWT validation via JWKS
- RS256/ES256 signature algorithms
- Automatic JWKS key rotation

**Requirements:**
- Install PyJWT: `poetry add "PyJWT[crypto]"`

### 4.2 Destructive Operation Confirmation

Enable explicit user consent for destructive operations (delete, cleanup):

```bash
# Enable confirmation for destructive operations
export MC_MCP_REQUIRE_CONFIRMATION=1
export MC_MCP_CONFIRMATION_TIMEOUT=300  # Optional, default 5 minutes
```

**How It Works:**

When enabled, destructive tools (`mc_job_delete`, `mc_model_delete`, `mc_checkpoint_delete`, `mc_rag_delete`, `mc_storage_cleanup`) require a two-step confirmation:

1. **First Call**: Returns a confirmation token and description
   ```json
   {
     "_schema": "mc.confirmation.required.v1",
     "status": "confirmation_required",
     "operation": "delete_model",
     "confirmationToken": "confirm_abc123...",
     "description": "Delete model 'my-model' from local registry",
     "expiresInSeconds": 300
   }
   ```

2. **Second Call**: Include the `confirmationToken` parameter to execute
   ```json
   {
     "modelId": "my-model",
     "confirmationToken": "confirm_abc123..."
   }
   ```

### 4.3 Tool Annotations

All MCP tools are annotated per MCP specification:

| Annotation | Meaning |
|------------|---------|
| `readOnlyHint: true` | Tool only reads data |
| `destructiveHint: true` | Tool deletes or modifies data irreversibly |
| `idempotentHint: true` | Safe to retry |
| `openWorldHint: true` | Makes external network calls |

### 4.4 Path Handling

ModelCypher accepts explicit file paths for model and dataset operations. It resolves
paths and validates existence, but does not impose a sandbox. Use OS permissions and
avoid exposing MCP on untrusted networks.

## 5. Reporting Vulnerabilities

If you discover a security vulnerability, please do not open a public issue. Email `jason@ethyros.ai`.
