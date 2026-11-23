"""Secure secret retrieval backed by Google Cloud Secret Manager with local env fallback.

All external API keys, ELN tokens, and model endpoints are resolved through this
provider so that zero credentials are hardcoded in source code or logged to stdout.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from medchem_agent.config.settings import AppSettings, get_settings


class SecretNotFoundError(RuntimeError):
    """Raised when a required secret cannot be resolved from Secret Manager or env."""


@dataclass
class _CachedSecret:
    value: str
    expires_at: float


class SecretProvider:
    """Retrieves secrets from Google Cloud Secret Manager with TTL caching.

    Resolution order:
    1. In-memory TTL cache (avoids redundant Secret Manager RPCs per turn).
    2. Google Cloud Secret Manager (`projects/{project}/secrets/{secret_id}/versions/latest`)
       when `enable_secret_manager` is True and `gcp_project_id` is configured.
    3. Environment variable fallback (`env_var`) for local development & CI tests.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        ttl_seconds: int = 300,
        client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, _CachedSecret] = {}
        self._client = client

    def _get_gcp_client(self) -> Any:
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(
        self,
        secret_id: str,
        *,
        env_var: str | None = None,
        version: str = "latest",
        required: bool = False,
    ) -> str | None:
        """Resolve a secret without ever logging or exposing its raw value.

        Args:
            secret_id: Secret name in Google Cloud Secret Manager (e.g. ``eln-api-token``).
            env_var: Optional fallback environment variable name for local dev.
            version: Secret Manager version alias or number (defaults to ``latest``).
            required: If True, raises ``SecretNotFoundError`` when missing.

        Returns:
            The resolved secret string, or ``None`` if optional and unavailable.
        """
        cache_key = f"{secret_id}:{version}"
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.value

        secret_value: str | None = None

        if self._settings.enable_secret_manager and self._settings.gcp_project_id:
            try:
                client = self._get_gcp_client()
                resource_name = (
                    f"projects/{self._settings.gcp_project_id}"
                    f"/secrets/{secret_id}/versions/{version}"
                )
                response = client.access_secret_version(request={"name": resource_name})
                secret_value = response.payload.data.decode("utf-8").strip()
            except Exception as exc:
                if required and not env_var:
                    raise SecretNotFoundError(
                        f"Failed to access secret '{secret_id}' in Secret Manager: {type(exc).__name__}"
                    ) from exc

        if not secret_value and env_var:
            raw_env = os.getenv(env_var)
            if raw_env and raw_env.strip():
                secret_value = raw_env.strip()

        if secret_value:
            self._cache[cache_key] = _CachedSecret(
                value=secret_value,
                expires_at=now + self._ttl_seconds,
            )
            return secret_value

        if required:
            raise SecretNotFoundError(
                f"Required secret '{secret_id}' (env fallback '{env_var}') is not configured. "
                "Provision it in Google Cloud Secret Manager or export it in your local .env file."
            )
        return None

    def clear_cache(self) -> None:
        """Invalidate cached secret values (useful during key rotation or tests)."""
        self._cache.clear()


_default_provider: SecretProvider | None = None


def get_secret_provider() -> SecretProvider:
    """Return the process-wide SecretProvider singleton."""
    global _default_provider
    if _default_provider is None:
        _default_provider = SecretProvider()
    return _default_provider
