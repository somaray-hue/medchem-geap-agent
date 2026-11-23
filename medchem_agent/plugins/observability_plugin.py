"""Observability, Intent-vs-Outcome Capture, OpenTelemetry Tracing, & Async Memory ADK Plugin.

Hooks into ADK's full invocation lifecycle:
- ``before_tool_callback``: Records structured JSON **Intent** (`agent.tool.intent`) and starts
  an OpenTelemetry child span before execution.
- ``after_tool_callback`` / ``on_tool_error_callback``: Records structured JSON **Outcome**
  (`agent.tool.outcome`) with latency, actual status, and ``intent_outcome_match`` audit flag.
- ``after_run_callback``: Schedules non-blocking background session-to-memory consolidation via
  ``AsyncMemoryConsolidationManager`` so memory generation never blocks user responses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry.trace import Span, Status, StatusCode

from medchem_agent.memory.async_consolidator import get_memory_consolidator
from medchem_agent.observability.logging import get_logger
from medchem_agent.observability.pii_redactor import get_pii_redactor
from medchem_agent.observability.tracing import get_tracer

logger = get_logger("plugins.observability")
tracer = get_tracer("medchem_agent.observability")

_TOOL_EXPECTED_EFFECTS: dict[str, str] = {
    "resolve_compound_identifiers_from_pubchem": "Resolve canonical SMILES and InChIKey from PubChem.",
    "fetch_target_bioactivities_from_chembl": "Retrieve curated target IC50/pChEMBL records from ChEMBL.",
    "search_medicinal_chemistry_literature": "Retrieve peer-reviewed SAR citations from Europe PMC.",
    "retrieve_internal_eln_series_and_tpp": "Query internal enterprise ELN TPP and proprietary series.",
    "standardize_and_compute_molecular_descriptors": "Compute deterministic RDKit descriptors, CNS-MPO, and PAINS.",
    "run_matched_molecular_pair_rigidification_analysis": "Quantify MMPA rigidification impact on RotBonds & potency.",
    "train_scaffold_split_potency_qsar_model": "Train leak-free scaffold/chronological split QSAR model.",
    "predict_txgemma_admet_and_toxicity_profile": "Score TDC ADMET & toxicity endpoints via DeepMind TxGemma.",
    "validate_and_score_candidate_macrocycle_design": "Verify candidate valence/sanitization and TPP fit in RDKit.",
    "export_validated_synthesis_candidates_to_eln": "HITL-gated export of SDF and provenance manifest to ELN.",
    "submit_external_cro_assay_order": "HITL-gated external CRO wet-lab assay order.",
    "commit_design_decision_to_campaign_memory": "Schedule async background campaign memory indexing.",
}


@dataclass
class _PendingToolIntent:
    tool_name: str
    agent_name: str
    expected_effect: str
    started_monotonic: float
    span: Span


class ChemObservabilityPlugin(BasePlugin):
    """ADK Plugin for Intent-vs-Outcome structured logging, OTel spans, and async memory dispatch."""

    def __init__(
        self,
        name: str = "chem_observability_plugin",
        enable_async_memory_on_run_end: bool = True,
    ) -> None:
        super().__init__(name=name)
        self._enable_async_memory = enable_async_memory_on_run_end
        self._in_flight_intents: dict[str, _PendingToolIntent] = {}
        self._agent_spans: dict[str, Span] = {}
        self._audit_records: list[dict[str, Any]] = []

    @property
    def audit_records(self) -> list[dict[str, Any]]:
        """Return captured Intent-vs-Outcome audit events (for telemetry inspection & tests)."""
        return list(self._audit_records)

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Start an OpenTelemetry span for each specialist agent execution."""
        key = f"{callback_context.invocation_id}:{agent.name}"
        span = tracer.start_span(
            f"adk.agent.{agent.name}",
            attributes={
                "gen_ai.agent.name": agent.name,
                "adk.invocation_id": str(callback_context.invocation_id),
            },
        )
        self._agent_spans[key] = span
        logger.info(
            "agent.execution.started",
            agent_name=agent.name,
            invocation_id=str(callback_context.invocation_id),
        )
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """End the specialist agent's OpenTelemetry span."""
        key = f"{callback_context.invocation_id}:{agent.name}"
        span = self._agent_spans.pop(key, None)
        if span is not None:
            span.set_status(Status(StatusCode.OK))
            span.end()
        logger.info(
            "agent.execution.completed",
            agent_name=agent.name,
            invocation_id=str(callback_context.invocation_id),
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Capture structured JSON Intent + start OTel span BEFORE tool execution."""
        call_id = str(getattr(tool_context, "function_call_id", None) or f"{tool.name}:{time.monotonic_ns()}")
        redacted_args = get_pii_redactor().redact_payload(tool_args)
        expected_effect = _TOOL_EXPECTED_EFFECTS.get(
            tool.name, f"Execute tool '{tool.name}' with validated arguments."
        )

        span = tracer.start_span(
            f"adk.tool.{tool.name}",
            attributes={
                "gen_ai.tool.name": tool.name,
                "gen_ai.tool.call.id": call_id,
                "gen_ai.agent.name": str(getattr(tool_context, "agent_name", "unknown")),
                "agentops.intent": expected_effect,
            },
        )
        self._in_flight_intents[call_id] = _PendingToolIntent(
            tool_name=tool.name,
            agent_name=str(getattr(tool_context, "agent_name", "unknown")),
            expected_effect=expected_effect,
            started_monotonic=time.monotonic(),
            span=span,
        )

        intent_event = {
            "event_phase": "INTENT",
            "call_id": call_id,
            "agent_name": str(getattr(tool_context, "agent_name", "unknown")),
            "tool_name": tool.name,
            "expected_effect": expected_effect,
            "intended_args": redacted_args,
        }
        self._audit_records.append(intent_event)
        logger.info("agent.tool.intent", **intent_event)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Capture structured JSON Outcome + Intent-vs-Outcome match AFTER tool execution."""
        call_id = str(getattr(tool_context, "function_call_id", None) or "")
        pending = self._in_flight_intents.pop(call_id, None)
        if pending is None:
            # Match fallback if function_call_id was synthetic
            for k, v in list(self._in_flight_intents.items()):
                if v.tool_name == tool.name:
                    pending = self._in_flight_intents.pop(k)
                    call_id = k
                    break

        duration_ms = (
            round((time.monotonic() - pending.started_monotonic) * 1000.0, 2)
            if pending
            else 0.0
        )
        outcome_status = str(result.get("status", "success")) if isinstance(result, dict) else "success"
        intent_outcome_match = outcome_status in {
            "success",
            "exported",
            "pending_confirmation",
            "scheduled_async",
            "queued_for_dispatch",
        }
        error_code = result.get("error_code") if isinstance(result, dict) else None

        if pending is not None:
            pending.span.set_attribute("agentops.outcome.status", outcome_status)
            pending.span.set_attribute("agentops.intent_outcome_match", intent_outcome_match)
            pending.span.set_attribute("agentops.duration_ms", duration_ms)
            if error_code:
                pending.span.set_attribute("agentops.error_code", str(error_code))
                pending.span.set_status(Status(StatusCode.ERROR, str(error_code)))
            else:
                pending.span.set_status(Status(StatusCode.OK))
            pending.span.end()

        outcome_event = {
            "event_phase": "OUTCOME",
            "call_id": call_id,
            "agent_name": pending.agent_name if pending else str(getattr(tool_context, "agent_name", "unknown")),
            "tool_name": tool.name,
            "expected_effect": pending.expected_effect if pending else "",
            "actual_status": outcome_status,
            "intent_outcome_match": intent_outcome_match,
            "error_code": error_code,
            "duration_ms": duration_ms,
        }
        self._audit_records.append(outcome_event)
        logger.info("agent.tool.outcome", **outcome_event)
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Record failed tool outcome & close OTel span cleanly with error semantics."""
        call_id = str(getattr(tool_context, "function_call_id", None) or "")
        pending = self._in_flight_intents.pop(call_id, None)
        duration_ms = (
            round((time.monotonic() - pending.started_monotonic) * 1000.0, 2)
            if pending
            else 0.0
        )
        if pending is not None:
            pending.span.record_exception(error)
            pending.span.set_status(Status(StatusCode.ERROR, str(error)))
            pending.span.end()

        outcome_event = {
            "event_phase": "OUTCOME",
            "call_id": call_id,
            "tool_name": tool.name,
            "actual_status": "exception",
            "intent_outcome_match": False,
            "error_type": type(error).__name__,
            "duration_ms": duration_ms,
        }
        self._audit_records.append(outcome_event)
        logger.error("agent.tool.outcome.exception", **outcome_event)
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        """Trigger non-blocking background session-to-memory consolidation after run completes."""
        if not self._enable_async_memory:
            return
        session = getattr(invocation_context, "session", None)
        memory_service = getattr(invocation_context, "memory_service", None)
        if session is not None and memory_service is not None:
            consolidator = get_memory_consolidator()
            consolidator.bind_memory_service(memory_service)
            consolidator.schedule_session_consolidation(session)
