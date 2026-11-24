"""Persistent Session & Long-Term Memory service factories for ADK / GEAP.

Production (GEAP / Vertex AI Agent Engine):
  - Sessions: ``VertexAiSessionService`` (or Cloud SQL PostgreSQL via ``DatabaseSessionService``)
  - Memory:   ``VertexAiMemoryBankService`` (GEAP Memory Bank)

Local Development / CI:
  - Sessions: ``DatabaseSessionService`` backed by persistent async SQLite
              (``sqlite+aiosqlite:///./data/medchem_sessions.db``)
  - Memory:   ``SqliteMemoryService`` with SQLite FTS5 full-text indexing enabled
"""

from __future__ import annotations

from pathlib import Path

from google.adk.memory import BaseMemoryService, SqliteMemoryService, VertexAiMemoryBankService
from google.adk.sessions import BaseSessionService, DatabaseSessionService, VertexAiSessionService

from medchem_agent.config.secrets import get_secret_provider
from medchem_agent.config.settings import AppSettings, get_settings
from medchem_agent.observability.logging import get_logger

logger = get_logger("memory.persistence")


def _ensure_sqlite_parent_dir(db_url_or_path: str | Path) -> None:
    """Create parent directory for file-backed SQLite databases if needed."""
    raw = str(db_url_or_path)
    if raw.startswith("sqlite"):
        # Format: sqlite+aiosqlite:///./data/medchem_sessions.db or sqlite:////abs/path.db
        if ":////" in raw:
            file_part = "/" + raw.split(":////", 1)[1]
        elif ":///" in raw:
            file_part = raw.split(":///", 1)[1]
        else:
            return
        if file_part != ":memory:":
            Path(file_part).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(raw).parent.mkdir(parents=True, exist_ok=True)


def build_session_service(settings: AppSettings | None = None) -> BaseSessionService:
    """Instantiate the persistent ADK SessionService appropriate for the environment."""
    cfg = settings or get_settings()

    if cfg.use_vertex_ai and cfg.gcp_project_id and cfg.agent_engine_id:
        logger.info(
            "session_service.initialized",
            backend="VertexAiSessionService",
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
        )
        return VertexAiSessionService(
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
            agent_engine_id=cfg.agent_engine_id,
        )

    db_url = cfg.session_db_url
    if cfg.enable_secret_manager and cfg.gcp_project_id:
        secret_db_url = get_secret_provider().get_secret("session-db-url", env_var="SESSION_DB_URL")
        if secret_db_url:
            db_url = secret_db_url

    _ensure_sqlite_parent_dir(db_url)
    logger.info(
        "session_service.initialized",
        backend="DatabaseSessionService",
        db_scheme=db_url.split("://", 1)[0],
    )
    return DatabaseSessionService(db_url=db_url)


def build_memory_service(settings: AppSettings | None = None) -> BaseMemoryService:
    """Instantiate the persistent ADK MemoryService (GEAP Memory Bank or FTS SQLite)."""
    cfg = settings or get_settings()

    if cfg.use_vertex_ai and cfg.gcp_project_id and cfg.agent_engine_id:
        logger.info(
            "memory_service.initialized",
            backend="VertexAiMemoryBankService",
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
        )
        return VertexAiMemoryBankService(
            project=cfg.gcp_project_id,
            location=cfg.gcp_location,
            agent_engine_id=cfg.agent_engine_id,
        )

    _ensure_sqlite_parent_dir(cfg.memory_db_path)
    logger.info(
        "memory_service.initialized",
        backend="SqliteMemoryService",
        fts="on",
    )
    return SqliteMemoryService(db_path=cfg.memory_db_path, fts="on")
