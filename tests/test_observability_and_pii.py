"""Tests for Category 4: Observability & Tracing (Structured JSON, Intent-vs-Outcome, OTel, PII)."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medchem_agent.observability.logging import configure_structured_logging, get_logger
from medchem_agent.observability.pii_redactor import PiiRedactor
from medchem_agent.observability.tracing import configure_tracing, get_tracer
from medchem_agent.plugins.observability_plugin import ChemObservabilityPlugin


def test_pii_redactor_scrubs_phi_pii_and_credentials() -> None:
    """Verify deterministic PII, PHI, and credential scrubbing on strings and nested payloads."""
    redactor = PiiRedactor()
    raw = (
        "Contact dr.alice@hospital.org or 555-867-5309 for subject PAT-901234 "
        "(SSN 123-45-6789) using Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    )
    scrubbed = redactor.redact_text(raw)
    for sensitive in ("dr.alice@hospital.org", "555-867-5309", "PAT-901234", "123-45-6789", "eyJhbGci"):
        assert sensitive not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_PATIENT_ID]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed

    nested = redactor.redact_payload(
        {"compound_id": "LORL", "api_key": "top-secret-value", "notes": "owner: bob@corp.io"}
    )
    assert nested["compound_id"] == "LORL"
    assert nested["api_key"] == "[REDACTED_SECRET]"
    assert "[REDACTED_EMAIL]" in nested["notes"]


def test_structured_json_logging_with_trace_correlation_and_pii_scrubbing() -> None:
    """Verify structlog emits valid Cloud-Logging-compatible JSON with trace_id and redacted PII."""
    exporter = InMemorySpanExporter()
    configure_tracing(custom_exporter=exporter)

    buffer = io.StringIO()
    configure_structured_logging(stream=buffer)
    log = get_logger("test.observability")

    tracer = get_tracer("test.observability")
    with tracer.start_as_current_span("test.root_span"):
        log.info(
            "chem.test.event",
            chemist_email="chemist@pharma.com",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
        )

    raw_line = buffer.getvalue().strip().splitlines()[-1]
    record = json.loads(raw_line)
    assert record["event"] == "chem.test.event"
    assert record["severity"] == "INFO"
    assert "timestamp_utc" in record
    assert record["trace_id"] is not None and len(record["trace_id"]) == 32
    assert record["span_id"] is not None and len(record["span_id"]) == 16
    assert record["chemist_email"] == "[REDACTED_EMAIL]"
    assert record["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"

    spans = exporter.get_finished_spans()
    assert any(s.name == "test.root_span" for s in spans)


@pytest.mark.asyncio
async def test_observability_plugin_captures_intent_vs_outcome_and_otel_spans() -> None:
    """Verify ChemObservabilityPlugin logs paired INTENT & OUTCOME records and linked OTel spans."""
    exporter = InMemorySpanExporter()
    configure_tracing(custom_exporter=exporter)

    plugin = ChemObservabilityPlugin(enable_async_memory_on_run_end=False)
    fake_tool = SimpleNamespace(name="standardize_and_compute_molecular_descriptors")
    fake_ctx = SimpleNamespace(agent_name="sar_mmpa_modeling_agent", function_call_id="call-42")

    # 1. Successful tool call records INTENT then matching OUTCOME
    await plugin.before_tool_callback(
        tool=fake_tool,
        tool_args={"molecules": [{"compound_id": "LORL", "smiles": "CCO"}], "email": "x@y.com"},
        tool_context=fake_ctx,
    )
    await plugin.after_tool_callback(
        tool=fake_tool,
        tool_args={"molecules": [{"compound_id": "LORL", "smiles": "CCO"}]},
        tool_context=fake_ctx,
        result={"status": "success", "valid_count": 1},
    )

    # 2. Failed tool call records intent_outcome_match=False
    fake_ctx_err = SimpleNamespace(agent_name="sar_mmpa_modeling_agent", function_call_id="call-43")
    await plugin.before_tool_callback(
        tool=fake_tool,
        tool_args={"molecules": [{"compound_id": "BAD", "smiles": "((("}]},
        tool_context=fake_ctx_err,
    )
    await plugin.after_tool_callback(
        tool=fake_tool,
        tool_args={"molecules": [{"compound_id": "BAD", "smiles": "((("}]},
        tool_context=fake_ctx_err,
        result={"status": "error", "error_code": "ALL_SMILES_INVALID"},
    )

    records = plugin.audit_records
    assert len(records) == 4
    assert records[0]["event_phase"] == "INTENT"
    assert records[0]["intended_args"]["email"] == "[REDACTED_SECRET]"
    assert records[1]["event_phase"] == "OUTCOME"
    assert records[1]["intent_outcome_match"] is True
    assert records[3]["event_phase"] == "OUTCOME"
    assert records[3]["intent_outcome_match"] is False
    assert records[3]["error_code"] == "ALL_SMILES_INVALID"

    finished_spans = exporter.get_finished_spans()
    tool_spans = [s for s in finished_spans if s.name == "adk.tool.standardize_and_compute_molecular_descriptors"]
    assert len(tool_spans) == 2
    assert tool_spans[0].attributes["agentops.intent_outcome_match"] is True
    assert tool_spans[1].attributes["agentops.intent_outcome_match"] is False
