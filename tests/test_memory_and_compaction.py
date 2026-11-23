"""Tests for Category 2: Context & Memory (Constitution, Compaction, Persistent State, Async Memory)."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.adk.memory import SqliteMemoryService
from google.adk.sessions import DatabaseSessionService

from medchem_agent.agent import create_medchem_app
from medchem_agent.config.settings import AppSettings
from medchem_agent.memory.async_consolidator import AsyncMemoryConsolidationManager
from medchem_agent.memory.enterprise_store import EnterpriseElnStore
from medchem_agent.memory.persistence import build_memory_service, build_session_service
from medchem_agent.prompts.constitution import COORDINATOR_INSTRUCTION, MEDCHEM_CONSTITUTION


def test_robust_system_constitution_contains_core_guardrails() -> None:
    """Verify the system constitution encodes division of labor, IP fence, biosecurity, and wet-lab caveats."""
    for phrase in (
        "No Margin for Error",
        "Proprietary Structure Fence",
        "Biosecurity / CBRN Refusal",
        "Bemis-Murcko scaffold splits",
        "wet-lab",
        "Human-in-the-Loop",
    ):
        assert phrase in MEDCHEM_CONSTITUTION
    assert MEDCHEM_CONSTITUTION in COORDINATOR_INSTRUCTION


def test_history_compaction_and_context_cache_configured_on_adk_app() -> None:
    """Verify ADK App mounts EventsCompactionConfig, ContextCacheConfig, and ResumabilityConfig."""
    app = create_medchem_app()
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.token_threshold == 24000
    assert app.events_compaction_config.event_retention_size == 6
    assert app.events_compaction_config.compaction_interval == 4
    assert app.events_compaction_config.overlap_size == 1
    assert app.events_compaction_config.summarizer is not None

    assert app.context_cache_config is not None
    assert app.context_cache_config.ttl_seconds == 1800

    assert app.resumability_config is not None
    assert app.resumability_config.is_resumable is True


@pytest.mark.asyncio
async def test_persistent_database_session_and_sqlite_memory_services(tmp_path: Path) -> None:
    """Verify sessions and long-term memories persist cleanly to disk via SQLite backends."""
    session_db = tmp_path / "sessions.db"
    memory_db = tmp_path / "memory.db"
    cfg = AppSettings(
        session_db_url=f"sqlite+aiosqlite:///{session_db}",
        memory_db_path=memory_db,
        eln_db_path=tmp_path / "eln.db",
    )

    session_service = build_session_service(cfg)
    memory_service = build_memory_service(cfg)
    assert isinstance(session_service, DatabaseSessionService)
    assert isinstance(memory_service, SqliteMemoryService)

    created = await session_service.create_session(
        app_name="medchem_geap_agent",
        user_id="chemist_1",
        session_id="sess_persist_01",
        state={"active_target": "ALK", "tpp_min_cns_mpo": 4.0},
    )
    assert created.id == "sess_persist_01"

    # Re-open fresh DatabaseSessionService against same DB file to prove persistence
    session_service_reloaded = build_session_service(cfg)
    loaded = await session_service_reloaded.get_session(
        app_name="medchem_geap_agent",
        user_id="chemist_1",
        session_id="sess_persist_01",
    )
    assert loaded is not None
    assert loaded.state["active_target"] == "ALK"
    assert loaded.state["tpp_min_cns_mpo"] == 4.0


@pytest.mark.asyncio
async def test_enterprise_eln_store_and_async_memory_consolidation(tmp_path: Path) -> None:
    """Verify background async memory consolidation does not block caller and scrubs PII before storage."""
    eln_db = tmp_path / "eln.db"
    memory_db = tmp_path / "mem.db"
    store = EnterpriseElnStore(db_path=eln_db)

    tpp = store.get_tpp("ALK")
    assert tpp is not None and tpp["uniprot_id"] == "Q9UM73"
    assert len(store.get_proprietary_compounds("ALK")) >= 2

    memory_service = SqliteMemoryService(db_path=memory_db, fts="on")
    consolidator = AsyncMemoryConsolidationManager(
        memory_service=memory_service,
        eln_store=store,
    )

    # Schedule non-blocking background note indexing containing PII that must be redacted
    job_id = consolidator.schedule_campaign_note_indexing(
        campaign_id="ALK-2026-Q4",
        target_symbol="ALK",
        summary=(
            "Cyclization improved CNS-MPO for ALK L1196M; reviewed by dr.jane@biopharma.com "
            "using subject PAT-441122."
        ),
        metadata={"api_key": "AIzaSySecretKeyValue1234567890", "rot_bonds": 0},
    )
    assert job_id.startswith("mem-note-")

    # Drain background task and verify persistence + PII redaction
    results = await consolidator.drain(timeout_seconds=5.0)
    assert len(results) == 1
    assert results[0].status == "completed"

    hits = store.search_campaign_notes("ALK macrocycle cyclization L1196M")
    assert len(hits) >= 1
    stored_summary = hits[0]["summary"]
    assert "dr.jane@biopharma.com" not in stored_summary
    assert "PAT-441122" not in stored_summary
    assert "[REDACTED_EMAIL]" in stored_summary
    assert "[REDACTED_PATIENT_ID]" in stored_summary
    assert hits[0]["metadata"]["api_key"] == "[REDACTED_SECRET]"
