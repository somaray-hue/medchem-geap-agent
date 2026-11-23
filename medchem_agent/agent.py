"""ADK App & root_agent entry point discovered by `adk web`, `adk run`, `adk eval`, and GEAP Runtime.

Configures:
- ``root_agent``: Multi-agent Medicinal Chemistry Coordinator (Coordinator + Sequential + Parallel)
- ``events_compaction_config``: Token-based + sliding-window history compaction with FAST Gemini summarizer
- ``context_cache_config``: Gemini / Vertex AI context caching for large chemistry prompts/skills
- ``resumability_config``: Enables ADK native Human-in-the-Loop (HITL) tool confirmations
- ``plugins``:
    1. ``ChemGuardrailsAndPolicyPlugin`` (Biosecurity, Model Armor, IP egress fence, self-eval)
    2. ``ChemObservabilityPlugin`` (Intent vs. Outcome JSON logs, OpenTelemetry spans, async memory)
    3. ``ReflectAndRetryToolPlugin`` (Guided automatic reflection on recoverable tool errors)
"""

from __future__ import annotations

from google.adk.apps.app import (
    App,
    ContextCacheConfig,
    EventsCompactionConfig,
    ResumabilityConfig,
)
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.google_llm import Gemini
from google.adk.plugins import ReflectAndRetryToolPlugin

from medchem_agent.agents.pipeline import build_medchem_coordinator_agent
from medchem_agent.config.model_router import ModelRouter, TaskComplexity
from medchem_agent.config.settings import AppSettings, get_settings
from medchem_agent.observability.logging import configure_structured_logging
from medchem_agent.observability.tracing import configure_tracing
from medchem_agent.plugins.guardrails_plugin import ChemGuardrailsAndPolicyPlugin
from medchem_agent.plugins.observability_plugin import ChemObservabilityPlugin
from medchem_agent.prompts.constitution import COMPACTION_SUMMARY_PROMPT


def create_medchem_app(settings: AppSettings | None = None) -> App:
    """Build the fully configured enterprise ADK App instance."""
    cfg = settings or get_settings()
    configure_structured_logging(cfg)
    configure_tracing(cfg)

    router = ModelRouter(cfg)
    compaction_model = router.route(TaskComplexity.SUMMARIZATION).model_id

    compaction_config = EventsCompactionConfig(
        summarizer=LlmEventSummarizer(
            llm=Gemini(model=compaction_model),
            prompt_template=COMPACTION_SUMMARY_PROMPT,
        ),
        token_threshold=cfg.compaction_token_threshold,
        event_retention_size=cfg.compaction_event_retention_size,
        compaction_interval=cfg.compaction_interval,
        overlap_size=cfg.compaction_overlap_size,
    )

    context_cache_config = ContextCacheConfig(
        min_tokens=cfg.context_cache_min_tokens,
        ttl_seconds=cfg.context_cache_ttl_seconds,
        cache_intervals=10,
    )

    coordinator = build_medchem_coordinator_agent(router)

    return App(
        name=cfg.app_name,
        root_agent=coordinator,
        events_compaction_config=compaction_config,
        context_cache_config=context_cache_config,
        resumability_config=ResumabilityConfig(is_resumable=True),
        plugins=[
            ChemGuardrailsAndPolicyPlugin(settings=cfg),
            ChemObservabilityPlugin(enable_async_memory_on_run_end=True),
            ReflectAndRetryToolPlugin(
                max_retries=2,
                throw_exception_if_retry_exceeded=False,
            ),
        ],
    )


app = create_medchem_app()
root_agent = app.root_agent
