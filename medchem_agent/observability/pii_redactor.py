"""Active PII, PHI, and credential scrubbing for logging, tracing, and memory pipelines.

Combines:
1. Optional Google Cloud Sensitive Data Protection (Cloud DLP) API de-identification
   when ``enable_cloud_dlp=True`` and ``gcp_project_id`` is configured.
2. Always-on deterministic local scrubbing for emails, phone numbers, SSNs,
   clinical subject/patient MRNs (e.g., ``PAT-123456``, ``MRN: 88219``),
   and bearer tokens / API keys so sensitive data is never persisted or logged.
"""

from __future__ import annotations

import re
from typing import Any

from medchem_agent.config.settings import AppSettings, get_settings

# Ordered local redaction rules (deterministic, fast, zero network dependency)
_LOCAL_REDACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "[REDACTED_API_KEY]",
        re.compile(
            r"(?i)(?:AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
            r"Bearer\s+[A-Za-z0-9._\-]{16,})"
        ),
    ),
    (
        "[REDACTED_SSN]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "[REDACTED_EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "[REDACTED_PHONE]",
        re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    ),
    (
        "[REDACTED_PATIENT_ID]",
        re.compile(r"\b(?:PAT|SUBJ|MRN)[-_:\s]?\d{4,12}\b", re.IGNORECASE),
    ),
    (
        "[REDACTED_CREDIT_CARD]",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
]

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "patient_name",
        "ssn",
        "mrn",
        "email",
        "phone",
    }
)


class PiiRedactor:
    """Scrubs PII/PHI and secrets from strings and nested structured payloads."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        dlp_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._dlp_client = dlp_client

    def _get_dlp_client(self) -> Any:
        if self._dlp_client is None:
            from google.cloud import dlp_v2

            self._dlp_client = dlp_v2.DlpServiceClient()
        return self._dlp_client

    def _redact_via_cloud_dlp(self, text: str) -> str:
        """Invoke Google Cloud DLP deidentify_content when enabled on GCP."""
        client = self._get_dlp_client()
        parent = f"projects/{self._settings.gcp_project_id}/locations/{self._settings.gcp_location}"
        inspect_config = {
            "info_types": [
                {"name": "EMAIL_ADDRESS"},
                {"name": "PHONE_NUMBER"},
                {"name": "US_SOCIAL_SECURITY_NUMBER"},
                {"name": "PERSON_NAME"},
                {"name": "MEDICAL_RECORD_NUMBER"},
                {"name": "CREDIT_CARD_NUMBER"},
            ]
        }
        deidentify_config = {
            "info_type_transformations": {
                "transformations": [
                    {
                        "primitive_transformation": {
                            "replace_with_info_type_config": {}
                        }
                    }
                ]
            }
        }
        response = client.deidentify_content(
            request={
                "parent": parent,
                "deidentify_config": deidentify_config,
                "inspect_config": inspect_config,
                "item": {"value": text},
            }
        )
        return str(response.item.value)

    def redact_text(self, text: str) -> str:
        """Redact sensitive entities from a string using Cloud DLP (optional) + local rules."""
        if not text:
            return text

        result = text
        if self._settings.enable_cloud_dlp and self._settings.gcp_project_id:
            try:
                result = self._redact_via_cloud_dlp(result)
            except Exception:
                # Fail closed to local deterministic scrubber; never drop scrubbing on RPC error
                pass

        for replacement, pattern in _LOCAL_REDACTION_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def redact_payload(self, data: Any) -> Any:
        """Recursively scrub PII, PHI, and credential fields from dicts/lists/strings."""
        if isinstance(data, str):
            return self.redact_text(data)
        if isinstance(data, dict):
            scrubbed: dict[Any, Any] = {}
            for key, value in data.items():
                key_lower = str(key).strip().lower()
                if key_lower in _SENSITIVE_FIELD_NAMES or key_lower.endswith(
                    ("_secret", "_token", "_api_key", "_password")
                ):
                    scrubbed[key] = "[REDACTED_SECRET]"
                else:
                    scrubbed[key] = self.redact_payload(value)
            return scrubbed
        if isinstance(data, list):
            return [self.redact_payload(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self.redact_payload(item) for item in data)
        return data


_default_redactor: PiiRedactor | None = None


def get_pii_redactor() -> PiiRedactor:
    """Return the process-wide PiiRedactor singleton."""
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = PiiRedactor()
    return _default_redactor
