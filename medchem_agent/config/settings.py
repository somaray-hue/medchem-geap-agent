"""Typed application settings loaded from environment variables (no hardcoded secrets)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AppSettings(BaseModel):
    """Runtime configuration for the MedChem GEAP Enterprise Agent."""

    app_name: str = Field(
        default="medchem_geap_agent",
        description="Canonical ADK application name used across sessions, memory, and traces.",
    )
    app_env: Literal["local", "staging", "production"] = Field(
        default="local",
        description="Deployment environment controlling persistence and cloud adapters.",
    )
    log_level: str = Field(default="INFO", description="Structured logging verbosity.")

    # Google Cloud / GEAP settings
    gcp_project_id: str | None = Field(default=None, description="Google Cloud project ID.")
    gcp_location: str = Field(default="us-central1", description="GCP region for GEAP/Vertex AI.")
    agent_engine_id: str | None = Field(
        default=None,
        description="Vertex AI Agent Engine / GEAP Agent Runtime ID when deployed to GCP.",
    )
    use_vertex_ai: bool = Field(
        default=False,
        description="Whether Gemini calls route through Vertex AI / GEAP endpoints.",
    )

    # Strategic model routing tiers
    model_fast: str = Field(
        default="gemini-2.5-flash",
        description="Low-latency model tier for extraction, database curation, and compaction.",
    )
    model_reasoning: str = Field(
        default="gemini-2.5-pro",
        description="High-reasoning model tier for scientific planning, SAR synthesis, and critique.",
    )
    txgemma_endpoint_id: str | None = Field(
        default=None,
        description="Deployed Vertex AI / GEAP Model Garden endpoint ID for DeepMind TxGemma.",
    )

    # Persistent session, memory & enterprise ELN storage
    session_db_url: str = Field(
        default="sqlite+aiosqlite:///./data/medchem_sessions.db",
        description="SQLAlchemy async URL for persistent ADK DatabaseSessionService.",
    )
    memory_db_path: Path = Field(
        default=Path("./data/medchem_memory.db"),
        description="SQLite path for persistent local FTS-indexed ADK memory service.",
    )
    eln_db_path: Path = Field(
        default=Path("./data/enterprise_eln.db"),
        description="SQLite path for persistent enterprise internal ELN compound store.",
    )
    export_dir: Path = Field(
        default=Path("./data/exports"),
        description="Directory for HITL-approved SDF and provenance export packages.",
    )

    # History compaction & context caching thresholds
    compaction_token_threshold: int = Field(
        default=24000,
        ge=1000,
        description="Token threshold triggering ADK tail-retention history compaction.",
    )
    compaction_event_retention_size: int = Field(
        default=6,
        ge=1,
        description="Recent raw events preserved uncompacted after token-based compaction.",
    )
    compaction_interval: int = Field(
        default=4,
        ge=1,
        description="Number of invocations triggering sliding-window compaction.",
    )
    compaction_overlap_size: int = Field(
        default=1,
        ge=0,
        description="Number of overlapping invocations carried across compacted summaries.",
    )
    context_cache_min_tokens: int = Field(default=2048, ge=0)
    context_cache_ttl_seconds: int = Field(default=1800, ge=60)

    # Governance, security, and observability toggles
    enable_cloud_dlp: bool = Field(
        default=False,
        description="Enable Google Cloud Sensitive Data Protection (DLP) PII scrubbing.",
    )
    enable_model_armor: bool = Field(
        default=False,
        description="Enable Google Cloud Model Armor prompt/response screening.",
    )
    model_armor_template_id: str | None = Field(
        default=None,
        description="Model Armor template resource ID for enterprise safety screening.",
    )
    enable_secret_manager: bool = Field(
        default=False,
        description="Fetch API keys/tokens from Google Cloud Secret Manager.",
    )
    otel_service_name: str = Field(default="medchem-geap-agent")
    otel_exporter_otlp_endpoint: str | None = Field(default=None)
    enable_cloud_trace: bool = Field(default=False)

    # HITL thresholds
    hitl_cost_threshold_usd: float = Field(
        default=5000.0,
        ge=0.0,
        description="Synthesis/assay order cost above which explicit human confirmation is required.",
    )
    hitl_max_auto_compounds: int = Field(
        default=5,
        ge=1,
        description="Batch size above which human confirmation is strictly required.",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Build and cache application settings from environment variables."""
    app_env_raw = os.getenv("APP_ENV", "local").strip().lower()
    app_env: Literal["local", "staging", "production"] = (
        app_env_raw if app_env_raw in {"local", "staging", "production"} else "local"  # type: ignore[assignment]
    )
    return AppSettings(
        app_env=app_env,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        gcp_project_id=os.getenv("GCP_PROJECT_ID") or None,
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        agent_engine_id=os.getenv("AGENT_ENGINE_ID") or None,
        use_vertex_ai=_as_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), default=False),
        model_fast=os.getenv("MODEL_FAST", "gemini-2.5-flash"),
        model_reasoning=os.getenv("MODEL_REASONING", "gemini-2.5-pro"),
        txgemma_endpoint_id=os.getenv("TXGEMMA_ENDPOINT_ID") or None,
        session_db_url=os.getenv(
            "SESSION_DB_URL",
            "sqlite+aiosqlite:///./data/medchem_sessions.db",
        ),
        memory_db_path=Path(os.getenv("MEMORY_DB_PATH", "./data/medchem_memory.db")),
        eln_db_path=Path(os.getenv("ELN_DB_PATH", "./data/enterprise_eln.db")),
        export_dir=Path(os.getenv("EXPORT_DIR", "./data/exports")),
        enable_cloud_dlp=_as_bool(os.getenv("ENABLE_CLOUD_DLP"), default=False),
        enable_model_armor=_as_bool(os.getenv("ENABLE_MODEL_ARMOR"), default=False),
        model_armor_template_id=os.getenv("MODEL_ARMOR_TEMPLATE_ID") or None,
        enable_secret_manager=_as_bool(
            os.getenv("ENABLE_SECRET_MANAGER"),
            default=(app_env == "production"),
        ),
        otel_service_name=os.getenv("OTEL_SERVICE_NAME", "medchem-geap-agent"),
        otel_exporter_otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        enable_cloud_trace=_as_bool(os.getenv("ENABLE_CLOUD_TRACE"), default=False),
    )
