"""Non-blocking asynchronous memory consolidation manager.

Offloads expensive long-term memory generation, PII scrubbing, session-to-memory
ingestion (`memory_service.add_session_to_memory`), and campaign note indexing to
background `asyncio` tasks so medicinal chemist UI responses are never blocked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from google.adk.memory import BaseMemoryService
from google.adk.sessions import Session

from medchem_agent.memory.enterprise_store import EnterpriseElnStore, get_enterprise_eln_store
from medchem_agent.observability.logging import get_logger
from medchem_agent.observability.pii_redactor import get_pii_redactor
from medchem_agent.observability.tracing import get_tracer

logger = get_logger("memory.async_consolidator")
tracer = get_tracer("medchem_agent.memory")


@dataclass
class MemoryJobResult:
    """Status record for an asynchronous background memory consolidation job."""

    job_id: str
    job_type: str
    status: str
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error: str | None = None


class AsyncMemoryConsolidationManager:
    """Queues and executes background memory consolidation without blocking caller turns."""

    def __init__(
        self,
        memory_service: BaseMemoryService | None = None,
        eln_store: EnterpriseElnStore | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self._memory_service = memory_service
        self._eln_store = eln_store or get_enterprise_eln_store()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._pending_tasks: set[asyncio.Task[MemoryJobResult]] = set()
        self._completed_jobs: list[MemoryJobResult] = []
        self._job_counter = 0

    def bind_memory_service(self, memory_service: BaseMemoryService) -> None:
        """Attach or update the backing ADK BaseMemoryService."""
        self._memory_service = memory_service

    def _next_job_id(self, prefix: str) -> str:
        self._job_counter += 1
        return f"{prefix}-{self._job_counter:05d}"

    def schedule_session_consolidation(self, session: Session) -> str:
        """Schedule a non-blocking background task to consolidate a session into long-term memory.

        Returns immediately with a ``job_id`` so the interactive turn finishes without waiting
        for LLM memory extraction or database indexing.
        """
        job_id = self._next_job_id("mem-session")

        async def _run_job() -> MemoryJobResult:
            async with self._semaphore:
                with tracer.start_as_current_span("memory.async_consolidate_session") as span:
                    span.set_attribute("memory.job_id", job_id)
                    span.set_attribute("session.id", session.id)
                    try:
                        if self._memory_service is not None:
                            await self._memory_service.add_session_to_memory(session)
                        result = MemoryJobResult(
                            job_id=job_id,
                            job_type="session_to_memory",
                            status="completed",
                        )
                        logger.info(
                            "memory.consolidation.completed",
                            job_id=job_id,
                            session_id=session.id,
                            backend=type(self._memory_service).__name__
                            if self._memory_service
                            else "none",
                        )
                    except Exception as exc:
                        result = MemoryJobResult(
                            job_id=job_id,
                            job_type="session_to_memory",
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        logger.error(
                            "memory.consolidation.failed",
                            job_id=job_id,
                            session_id=session.id,
                            error=result.error,
                        )
                    self._completed_jobs.append(result)
                    return result

        task = asyncio.create_task(_run_job(), name=job_id)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        logger.info("memory.consolidation.scheduled", job_id=job_id, session_id=session.id)
        return job_id

    def schedule_campaign_note_indexing(
        self,
        *,
        campaign_id: str,
        target_symbol: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Schedule non-blocking PII-redacted campaign insight persistence in background."""
        job_id = self._next_job_id("mem-note")
        redactor = get_pii_redactor()
        safe_summary = redactor.redact_text(summary)
        safe_meta = redactor.redact_payload(metadata or {})

        async def _run_note_job() -> MemoryJobResult:
            async with self._semaphore:
                with tracer.start_as_current_span("memory.async_index_campaign_note") as span:
                    span.set_attribute("memory.job_id", job_id)
                    span.set_attribute("medchem.campaign_id", campaign_id)
                    span.set_attribute("medchem.target_symbol", target_symbol)
                    try:
                        note_id = await asyncio.to_thread(
                            self._eln_store.save_campaign_note,
                            campaign_id,
                            target_symbol,
                            safe_summary,
                            safe_meta,
                        )
                        result = MemoryJobResult(
                            job_id=job_id,
                            job_type="campaign_note_index",
                            status="completed",
                        )
                        logger.info(
                            "memory.campaign_note.indexed",
                            job_id=job_id,
                            note_id=note_id,
                            campaign_id=campaign_id,
                            target_symbol=target_symbol,
                        )
                    except Exception as exc:
                        result = MemoryJobResult(
                            job_id=job_id,
                            job_type="campaign_note_index",
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        logger.error(
                            "memory.campaign_note.failed",
                            job_id=job_id,
                            error=result.error,
                        )
                    self._completed_jobs.append(result)
                    return result

        task = asyncio.create_task(_run_note_job(), name=job_id)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return job_id

    @property
    def pending_count(self) -> int:
        """Return the number of background consolidation tasks currently in flight."""
        return len(self._pending_tasks)

    @property
    def completed_jobs(self) -> list[MemoryJobResult]:
        """Return completed job history for observability and testing."""
        return list(self._completed_jobs)

    async def drain(self, timeout_seconds: float = 10.0) -> list[MemoryJobResult]:
        """Await all in-flight background memory jobs (used on shutdown and in tests)."""
        if not self._pending_tasks:
            return []
        tasks = list(self._pending_tasks)
        done, _ = await asyncio.wait(tasks, timeout=timeout_seconds)
        return [t.result() for t in done if not t.cancelled() and t.exception() is None]


_default_consolidator: AsyncMemoryConsolidationManager | None = None


def get_memory_consolidator() -> AsyncMemoryConsolidationManager:
    """Return the process-wide AsyncMemoryConsolidationManager singleton."""
    global _default_consolidator
    if _default_consolidator is None:
        _default_consolidator = AsyncMemoryConsolidationManager()
    return _default_consolidator
