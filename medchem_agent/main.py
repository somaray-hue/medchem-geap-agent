"""Production FastAPI & ADK Runner entrypoint for GEAP Agent Runtime / Cloud Run.

Wires:
- Persistent Session Service (`VertexAiSessionService` on GEAP or `DatabaseSessionService` locally)
- Persistent Memory Service (`VertexAiMemoryBankService` on GEAP or `SqliteMemoryService` locally)
- Graceful draining of background async memory consolidation tasks on server shutdown
- Health, JSON Schema contract inspection, and streaming/turn-based `/query` endpoints
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from google.adk import Runner
from google.genai import types
from pydantic import BaseModel, Field

from medchem_agent.agent import app as adk_app
from medchem_agent.config.settings import get_settings
from medchem_agent.memory.async_consolidator import get_memory_consolidator
from medchem_agent.memory.persistence import build_memory_service, build_session_service
from medchem_agent.observability.logging import get_logger
from medchem_agent.schemas.chemistry import export_all_tool_json_schemas

logger = get_logger("main")

_runner: Runner | None = None


def build_production_runner() -> Runner:
    """Create an ADK Runner backed by persistent session and memory services."""
    settings = get_settings()
    session_service = build_session_service(settings)
    memory_service = build_memory_service(settings)
    get_memory_consolidator().bind_memory_service(memory_service)
    return Runner(
        app=adk_app,
        session_service=session_service,
        memory_service=memory_service,
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _runner
    _runner = build_production_runner()
    logger.info("service.started", app_name=adk_app.name)
    try:
        yield
    finally:
        await get_memory_consolidator().drain(timeout_seconds=10.0)
        logger.info("service.stopped", app_name=adk_app.name)


fastapi_app = FastAPI(
    title="MedChem GEAP Enterprise Agent",
    version="0.1.0",
    description=(
        "Enterprise Life Sciences Medicinal Chemistry Agent built on Google ADK, GEAP, "
        "DeepMind Agentic Cheminformatics Skills, and TxGemma."
    ),
    lifespan=_lifespan,
)


class QueryRequest(BaseModel):
    """Typed HTTP payload for interacting with the MedChem agent."""

    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    prompt: str = Field(..., min_length=1, max_length=8000)


class QueryResponse(BaseModel):
    """Structured HTTP response from the MedChem agent."""

    session_id: str
    response_text: str
    pending_memory_jobs: int


@fastapi_app.get("/health")
async def health_check() -> dict[str, Any]:
    """Liveness and readiness probe for GEAP Agent Runtime / Cloud Run."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": adk_app.name,
        "environment": settings.app_env,
        "models": {
            "fast": settings.model_fast,
            "reasoning": settings.model_reasoning,
            "txgemma_endpoint": settings.txgemma_endpoint_id or "physicochemical_fallback",
        },
        "pending_async_memory_jobs": get_memory_consolidator().pending_count,
    }


@fastapi_app.get("/schemas")
async def list_tool_schemas() -> dict[str, Any]:
    """Expose strict Pydantic v2 JSON Schemas for all chemistry tools."""
    return export_all_tool_json_schemas()


@fastapi_app.post("/query", response_model=QueryResponse)
async def run_agent_query(request: QueryRequest) -> QueryResponse:
    """Execute one medicinal chemistry agent turn with persistent session state."""
    if _runner is None:
        raise HTTPException(status_code=503, detail="Agent runner is not initialized.")

    existing = await _runner.session_service.get_session(
        app_name=adk_app.name,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    if existing is None:
        await _runner.session_service.create_session(
            app_name=adk_app.name,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    user_content = types.Content(role="user", parts=[types.Part.from_text(text=request.prompt)])
    final_chunks: list[str] = []
    async for event in _runner.run_async(
        user_id=request.user_id,
        session_id=request.session_id,
        new_message=user_content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_chunks.append(part.text)

    return QueryResponse(
        session_id=request.session_id,
        response_text="\n".join(final_chunks).strip(),
        pending_memory_jobs=get_memory_consolidator().pending_count,
    )


if __name__ == "__main__":
    uvicorn.run("medchem_agent.main:fastapi_app", host="0.0.0.0", port=8080, reload=False)
