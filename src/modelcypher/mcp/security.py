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

"""MCP Security Configuration and Middleware.

Implements MCP 2025-06-18 specification security features:
- OAuth 2.1 resource server validation (RFC 9728)
- Destructive operation confirmation
- Token audience validation (RFC 8707)

Usage:
    # Enable via environment variables:
    MC_MCP_AUTH_ENABLED=1
    MC_MCP_AUTH_ISSUER=https://auth.example.com
    MC_MCP_AUTH_AUDIENCE=https://mcp.example.com
    MC_MCP_REQUIRE_CONFIRMATION=1

    # Or programmatically:
    config = SecurityConfig.from_env()
    validator = TokenValidator(config)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

# Try to import JWT library, but make it optional
try:
    import jwt
    from jwt import PyJWKClient
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None  # type: ignore
    PyJWKClient = None  # type: ignore


class AuthError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str, code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class ConfirmationError(Exception):
    """Raised when confirmation is required but not provided."""

    def __init__(self, operation: str, confirmation_token: str):
        super().__init__(f"Confirmation required for {operation}")
        self.operation = operation
        self.confirmation_token = confirmation_token


class TokenStatus(Enum):
    """Token validation status."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_AUDIENCE = "invalid_audience"
    INVALID_ISSUER = "invalid_issuer"
    MISSING = "missing"
    DISABLED = "disabled"  # Auth not enabled


@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration for MCP server.

    Attributes:
        auth_enabled: Whether OAuth token validation is enabled
        issuer: Expected token issuer (iss claim)
        audience: Expected token audience (aud claim) - should be MCP server URI
        jwks_uri: URI to fetch JWKS for token verification
        require_confirmation: Whether destructive operations require confirmation
        confirmation_timeout_seconds: How long confirmation tokens are valid
        allowed_algorithms: JWT algorithms to accept
    """
    auth_enabled: bool = False
    issuer: str | None = None
    audience: str | None = None
    jwks_uri: str | None = None
    require_confirmation: bool = False
    confirmation_timeout_seconds: int = 300  # 5 minutes
    allowed_algorithms: tuple[str, ...] = ("RS256", "ES256")

    @classmethod
    def from_env(cls) -> SecurityConfig:
        """Load security configuration from environment variables.

        Environment variables:
            MC_MCP_AUTH_ENABLED: Set to "1" or "true" to enable OAuth
            MC_MCP_AUTH_ISSUER: OAuth token issuer URL
            MC_MCP_AUTH_AUDIENCE: Expected audience (your MCP server URI)
            MC_MCP_AUTH_JWKS_URI: JWKS endpoint for token verification
            MC_MCP_REQUIRE_CONFIRMATION: Set to "1" to require confirmation for destructive ops
            MC_MCP_CONFIRMATION_TIMEOUT: Confirmation token timeout in seconds (default: 300)
        """
        auth_enabled = os.environ.get("MC_MCP_AUTH_ENABLED", "").lower() in ("1", "true", "yes")
        require_confirmation = os.environ.get("MC_MCP_REQUIRE_CONFIRMATION", "").lower() in ("1", "true", "yes")

        return cls(
            auth_enabled=auth_enabled,
            issuer=os.environ.get("MC_MCP_AUTH_ISSUER"),
            audience=os.environ.get("MC_MCP_AUTH_AUDIENCE"),
            jwks_uri=os.environ.get("MC_MCP_AUTH_JWKS_URI"),
            require_confirmation=require_confirmation,
            confirmation_timeout_seconds=int(os.environ.get("MC_MCP_CONFIRMATION_TIMEOUT", "300")),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if self.auth_enabled:
            if not JWT_AVAILABLE:
                issues.append("OAuth enabled but PyJWT not installed. Run: pip install PyJWT[crypto]")
            if not self.issuer:
                issues.append("MC_MCP_AUTH_ISSUER required when auth is enabled")
            if not self.audience:
                issues.append("MC_MCP_AUTH_AUDIENCE required when auth is enabled")
            if not self.jwks_uri:
                issues.append("MC_MCP_AUTH_JWKS_URI required when auth is enabled")
        return issues


@dataclass
class TokenValidationResult:
    """Result of token validation."""
    status: TokenStatus
    claims: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    subject: str | None = None
    scopes: list[str] = field(default_factory=list)


class TokenValidator:
    """OAuth 2.1 token validator for MCP servers.

    Implements RFC 8707 audience validation and standard JWT verification.
    """

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._jwks_client: Any = None

    def _get_jwks_client(self) -> Any:
        """Lazily initialize JWKS client."""
        if self._jwks_client is None and self.config.jwks_uri and JWT_AVAILABLE:
            self._jwks_client = PyJWKClient(self.config.jwks_uri)
        return self._jwks_client

    def validate_token(self, token: str | None) -> TokenValidationResult:
        """Validate an OAuth access token.

        Args:
            token: Bearer token from Authorization header (without "Bearer " prefix)

        Returns:
            TokenValidationResult with status and claims
        """
        if not self.config.auth_enabled:
            return TokenValidationResult(status=TokenStatus.DISABLED)

        if not token:
            return TokenValidationResult(
                status=TokenStatus.MISSING,
                error="Authorization token required"
            )

        if not JWT_AVAILABLE:
            return TokenValidationResult(
                status=TokenStatus.INVALID_SIGNATURE,
                error="PyJWT not installed - cannot validate tokens"
            )

        try:
            jwks_client = self._get_jwks_client()
            if jwks_client is None:
                return TokenValidationResult(
                    status=TokenStatus.INVALID_SIGNATURE,
                    error="JWKS client not configured"
                )

            signing_key = jwks_client.get_signing_key_from_jwt(token)

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(self.config.allowed_algorithms),
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={
                    "require": ["exp", "iat", "iss", "aud"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                }
            )

            return TokenValidationResult(
                status=TokenStatus.VALID,
                claims=claims,
                subject=claims.get("sub"),
                scopes=claims.get("scope", "").split() if isinstance(claims.get("scope"), str) else [],
            )

        except jwt.ExpiredSignatureError:
            return TokenValidationResult(
                status=TokenStatus.EXPIRED,
                error="Token has expired"
            )
        except jwt.InvalidAudienceError:
            return TokenValidationResult(
                status=TokenStatus.INVALID_AUDIENCE,
                error=f"Token audience does not match expected: {self.config.audience}"
            )
        except jwt.InvalidIssuerError:
            return TokenValidationResult(
                status=TokenStatus.INVALID_ISSUER,
                error=f"Token issuer does not match expected: {self.config.issuer}"
            )
        except jwt.InvalidSignatureError:
            return TokenValidationResult(
                status=TokenStatus.INVALID_SIGNATURE,
                error="Token signature verification failed"
            )
        except Exception as e:
            return TokenValidationResult(
                status=TokenStatus.INVALID_SIGNATURE,
                error=f"Token validation failed: {str(e)}"
            )


@dataclass
class PendingConfirmation:
    """A pending destructive operation awaiting confirmation."""
    operation: str
    tool_name: str
    parameters: dict[str, Any]
    token: str
    created_at: float
    expires_at: float
    description: str


class ConfirmationManager:
    """Manages confirmation flow for destructive operations.

    When a destructive operation is called without confirmation:
    1. A confirmation token is generated
    2. The operation details are stored
    3. User must call the tool again with the confirmation token

    This implements MCP spec requirement for explicit user consent
    before destructive operations.
    """

    def __init__(self, config: SecurityConfig, secret_key: str | None = None):
        self.config = config
        self._secret_key = secret_key or os.environ.get("MC_MCP_SECRET_KEY", os.urandom(32).hex())
        self._pending: dict[str, PendingConfirmation] = {}

    def _generate_token(self, operation: str, params_hash: str) -> str:
        """Generate a confirmation token."""
        timestamp = str(int(time.time()))
        data = f"{operation}:{params_hash}:{timestamp}"
        signature = hmac.new(
            self._secret_key.encode() if isinstance(self._secret_key, str) else self._secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        return f"confirm_{signature}"

    def _hash_params(self, params: dict[str, Any]) -> str:
        """Create a hash of parameters for verification."""
        import json
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def require_confirmation(
        self,
        operation: str,
        tool_name: str,
        parameters: dict[str, Any],
        description: str,
        confirmation_token: str | None = None,
    ) -> PendingConfirmation | None:
        """Check if confirmation is required and valid.

        Args:
            operation: Operation identifier (e.g., "delete_model")
            tool_name: Name of the MCP tool
            parameters: Tool parameters
            description: Human-readable description of what will happen
            confirmation_token: Token from previous call, if confirming

        Returns:
            None if confirmation is valid or not required
            PendingConfirmation if confirmation is needed

        Raises:
            ConfirmationError: If confirmation is required (contains token for retry)
        """
        if not self.config.require_confirmation:
            return None

        # Clean expired confirmations
        now = time.time()
        expired = [k for k, v in self._pending.items() if v.expires_at < now]
        for k in expired:
            del self._pending[k]

        params_hash = self._hash_params(parameters)

        # Check if valid confirmation provided
        if confirmation_token and confirmation_token in self._pending:
            pending = self._pending[confirmation_token]
            if pending.operation == operation and self._hash_params(pending.parameters) == params_hash:
                del self._pending[confirmation_token]
                return None  # Confirmed!

        # Generate new confirmation requirement
        token = self._generate_token(operation, params_hash)
        pending = PendingConfirmation(
            operation=operation,
            tool_name=tool_name,
            parameters=parameters,
            token=token,
            created_at=now,
            expires_at=now + self.config.confirmation_timeout_seconds,
            description=description,
        )
        self._pending[token] = pending

        raise ConfirmationError(operation, token)

    def get_pending(self, token: str) -> PendingConfirmation | None:
        """Get details of a pending confirmation."""
        return self._pending.get(token)


# Type for tool functions
F = TypeVar("F", bound=Callable[..., Any])


def destructive_operation(
    operation: str,
    description_template: str,
) -> Callable[[F], F]:
    """Decorator to mark a tool as requiring confirmation.

    Usage:
        @destructive_operation("delete_model", "Delete model {modelId}")
        def mc_model_delete(modelId: str, confirmationToken: str | None = None) -> dict:
            ...

    The decorated function MUST accept a `confirmationToken` parameter.
    """
    def decorator(func: F) -> F:
        func._destructive_operation = operation  # type: ignore
        func._description_template = description_template  # type: ignore
        return func
    return decorator


def create_confirmation_response(
    error: ConfirmationError,
    description: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Create a standardized confirmation-required response.

    Returns a response that tells the client to retry with confirmation.
    """
    return {
        "_schema": "mc.confirmation.required.v1",
        "status": "confirmation_required",
        "operation": error.operation,
        "confirmationToken": error.confirmation_token,
        "description": description,
        "expiresInSeconds": timeout_seconds,
        "message": f"This is a destructive operation. To confirm, call this tool again with confirmationToken='{error.confirmation_token}'",
        "nextActions": [
            f"Call the same tool with confirmationToken='{error.confirmation_token}' to confirm",
            "Or cancel by not retrying within {timeout_seconds} seconds",
        ],
    }


# Convenience function to check auth on server startup
def validate_security_config() -> tuple[SecurityConfig, list[str]]:
    """Load and validate security configuration.

    Returns:
        Tuple of (config, issues) where issues is empty if valid
    """
    config = SecurityConfig.from_env()
    issues = config.validate()
    return config, issues
