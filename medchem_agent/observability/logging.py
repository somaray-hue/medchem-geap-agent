"""Structured JSON logging via structlog with automatic PII redaction & trace correlation.

All logs are emitted as single-line JSON compatible with Google Cloud Logging
(including ``severity``, ``logging.googleapis.com/trace``, and ``logging.googleapis.com/spanId``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

import structlog

from medchem_agent.config.settings import AppSettings, get_settings
from medchem_agent.observability.pii_redactor import get_pii_redactor
from medchem_agent.observability.tracing import get_active_trace_context

_SEVERITY_MAP = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "error": "ERROR",
    "critical": "CRITICAL",
}


def _add_gcp_severity(_logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Map structlog levels to Google Cloud Logging `severity` field."""
    event_dict["severity"] = _SEVERITY_MAP.get(method_name.lower(), "INFO")
    return event_dict


def _inject_otel_trace_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Attach active OpenTelemetry trace_id and span_id to every JSON log entry."""
    trace_ctx = get_active_trace_context()
    if trace_ctx["trace_id"]:
        event_dict["trace_id"] = trace_ctx["trace_id"]
        event_dict["span_id"] = trace_ctx["span_id"]
        settings = get_settings()
        if settings.gcp_project_id:
            event_dict["logging.googleapis.com/trace"] = (
                f"projects/{settings.gcp_project_id}/traces/{trace_ctx['trace_id']}"
            )
            event_dict["logging.googleapis.com/spanId"] = trace_ctx["span_id"]
    return event_dict


def _redact_pii_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Scrub PII/PHI and secrets from every log event before JSON serialization."""
    redactor = get_pii_redactor()
    return redactor.redact_payload(event_dict)


def configure_structured_logging(
    settings: AppSettings | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure structlog and stdlib logging for structured JSON output."""
    cfg = settings or get_settings()
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    target_stream = stream or sys.stdout

    logging.basicConfig(
        format="%(message)s",
        stream=target_stream,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_gcp_severity,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp_utc"),
            _inject_otel_trace_context,
            _redact_pii_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=target_stream),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "medchem_agent") -> Any:
    """Return a lazy structured JSON logger bound with component metadata.

    Passing ``component=name`` directly to ``structlog.get_logger`` preserves
    ``BoundLoggerLazyProxy`` semantics so module-level loggers imported before
    ``configure_structured_logging()`` automatically inherit JSON rendering,
    OpenTelemetry trace correlation, and PII redaction.
    """
    return structlog.get_logger(name, component=name)
