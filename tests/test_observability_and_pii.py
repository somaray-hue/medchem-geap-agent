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
        "(SSN 123-45-6789) using Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 "
        "or key " + "AQ." + "DummyKeyTokenValue1234567890."
    )
    scrubbed = redactor.redact_text(raw)
    for sensitive in (
        "dr.alice@hospital.org",
        "555-867-5309",
        "PAT-901234",
        "123-45-6789",
        "eyJhbGci",
        "DummyKeyTokenValue1234567890",
    ):
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
    """Verify ChemObservabilityPlugin logs paired INTENT & OUTCOME records and linked parent-child OTel spans."""
    exporter = InMemorySpanExporter()
    configure_tracing(custom_exporter=exporter)

    log_buf = io.StringIO()
    configure_structured_logging(stream=log_buf)

    plugin = ChemObservabilityPlugin(enable_async_memory_on_run_end=False)
    fake_agent = SimpleNamespace(name="sar_mmpa_modeling_agent")
    fake_cb_ctx = SimpleNamespace(invocation_id="inv-trace-01")
    fake_tool = SimpleNamespace(name="standardize_and_compute_molecular_descriptors")
    fake_ctx = SimpleNamespace(agent_name="sar_mmpa_modeling_agent", function_call_id="call-42")
    inner_tracer = get_tracer("medchem_agent.tools.rdkit")

    # 1. Agent -> Tool -> Inner tool execution forms one linked parent-child OTel trace tree
    await plugin.before_agent_callback(agent=fake_agent, callback_context=fake_cb_ctx)
    await plugin.before_tool_callback(
        tool=fake_tool,
        tool_args={"molecules": [{"compound_id": "LORL", "smiles": "CCO"}], "email": "x@y.com"},
        tool_context=fake_ctx,
    )
    with inner_tracer.start_as_current_span("tool.standardize_and_compute_molecular_descriptors"):
        pass
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
    await plugin.after_agent_callback(agent=fake_agent, callback_context=fake_cb_ctx)

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
    agent_spans = [s for s in finished_spans if s.name == "adk.agent.sar_mmpa_modeling_agent"]
    tool_spans = [s for s in finished_spans if s.name == "adk.tool.standardize_and_compute_molecular_descriptors"]
    inner_spans = [s for s in finished_spans if s.name == "tool.standardize_and_compute_molecular_descriptors"]
    assert len(agent_spans) == 1
    assert len(tool_spans) == 2
    assert len(inner_spans) == 1

    # Verify true parent-child trace linkage: adk.agent -> adk.tool -> tool.*
    agent_span = agent_spans[0]
    first_tool_span = tool_spans[0]
    inner_span = inner_spans[0]
    assert first_tool_span.context.trace_id == agent_span.context.trace_id
    assert first_tool_span.parent is not None
    assert first_tool_span.parent.span_id == agent_span.context.span_id
    assert inner_span.context.trace_id == agent_span.context.trace_id
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == first_tool_span.context.span_id
    assert first_tool_span.attributes["agentops.intent_outcome_match"] is True
    assert tool_spans[1].attributes["agentops.intent_outcome_match"] is False

    # Verify emitted INTENT & OUTCOME JSON logs carry correlated trace_id & span_id
    emitted_logs = [json.loads(line) for line in log_buf.getvalue().strip().splitlines()]
    intent_logs = [entry for entry in emitted_logs if entry.get("event") == "agent.tool.intent"]
    outcome_logs = [entry for entry in emitted_logs if entry.get("event") == "agent.tool.outcome"]
    expected_trace_hex = format(agent_span.context.trace_id, "032x")
    assert intent_logs and intent_logs[0]["trace_id"] == expected_trace_hex
    assert outcome_logs and outcome_logs[0]["trace_id"] == expected_trace_hex
