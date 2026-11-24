"""OpenTelemetry distributed tracing configuration for GEAP / Cloud Trace / OTLP.

Follows OpenTelemetry GenAI semantic conventions so agent runs, specialist
delegations, LLM calls, RDKit/TxGemma tool invocations, and async memory jobs
appear in a single correlated waterfall trace.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

from medchem_agent.config.settings import AppSettings, get_settings

_configured_provider: TracerProvider | None = None
_attached_exporters: set[str] = set()


def configure_tracing(
    settings: AppSettings | None = None,
    *,
    custom_exporter: SpanExporter | None = None,
    console_export: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry TracerProvider with GCP Cloud Trace, OTLP, or test exporter.

    Args:
        settings: Optional application settings override.
        custom_exporter: Optional exporter (e.g. ``InMemorySpanExporter`` in tests).
        console_export: If True, attaches a ``ConsoleSpanExporter`` for local debugging.

    Returns:
        The configured ``TracerProvider``.
    """
    global _configured_provider
    cfg = settings or get_settings()

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        provider = existing
        _configured_provider = provider
    else:
        resource = Resource.create(
            {
                "service.name": cfg.otel_service_name,
                "service.version": "0.1.0",
                "deployment.environment": cfg.app_env,
                "gen_ai.system": "google_adk_geap",
            }
        )
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        _configured_provider = provider
        _attached_exporters.clear()

    if custom_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(custom_exporter))
    else:
        if cfg.enable_cloud_trace and cfg.gcp_project_id:
            cloud_key = f"cloud_trace:{cfg.gcp_project_id}"
            if cloud_key not in _attached_exporters:
                try:
                    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                    provider.add_span_processor(
                        BatchSpanProcessor(CloudTraceSpanExporter(project_id=cfg.gcp_project_id))
                    )
                    _attached_exporters.add(cloud_key)
                except Exception:
                    pass

        if cfg.otel_exporter_otlp_endpoint:
            otlp_key = f"otlp:{cfg.otel_exporter_otlp_endpoint}"
            if otlp_key not in _attached_exporters:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(endpoint=cfg.otel_exporter_otlp_endpoint)
                        )
                    )
                    _attached_exporters.add(otlp_key)
                except Exception:
                    pass

        if console_export and "console" not in _attached_exporters:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            _attached_exporters.add("console")

    return provider


def get_tracer(instrumentation_name: str = "medchem_agent") -> trace.Tracer:
    """Return an OpenTelemetry tracer bound to the active provider."""
    global _configured_provider
    if _configured_provider is None:
        configure_tracing()
    return trace.get_tracer(instrumentation_name, "0.1.0")


def get_active_trace_context() -> dict[str, Any]:
    """Return hex-encoded active OpenTelemetry trace_id and span_id for log correlation."""
    span = trace.get_current_span()
    span_ctx = span.get_span_context() if span else None
    if span_ctx and span_ctx.is_valid:
        return {
            "trace_id": format(span_ctx.trace_id, "032x"),
            "span_id": format(span_ctx.span_id, "016x"),
        }
    return {"trace_id": None, "span_id": None}
