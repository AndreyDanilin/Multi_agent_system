"""Litestar API facade.

The module imports without Litestar installed so local tests and offline demos
remain runnable. Installing the API extra enables the real ASGI app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from research_copilot.service import ResearchCopilotService
from research_copilot.types import RetrievalMode


class QueryRequest(BaseModel):
    question: str
    retrieval_mode: RetrievalMode = "hybrid_rerank"


class EvaluationRequest(BaseModel):
    modes: list[RetrievalMode] | None = None


@dataclass
class LocalApiApp:
    """Small route manifest used when Litestar is not installed."""

    service: ResearchCopilotService
    litestar_app: Any | None = None

    route_paths: tuple[str, ...] = (
        "/api/health",
        "/api/ingest/ragbench",
        "/api/query",
        "/api/evaluations/run",
        "/api/runs/{run_id}",
    )

    def health(self) -> dict[str, str]:
        return health()


def health() -> dict[str, str]:
    return {"status": "ok", "service": "agentic-research-copilot"}


def create_app(service: ResearchCopilotService | None = None) -> LocalApiApp:
    service = service or ResearchCopilotService.create_demo()
    try:
        from litestar import Litestar, get, post
    except Exception:
        return LocalApiApp(service=service)

    @get("/api/health")
    async def health_route() -> dict[str, str]:
        return health()

    @post("/api/ingest/ragbench")
    async def ingest_ragbench() -> dict[str, str | int]:
        return service.ingest_ragbench(sample_only=True)

    @post("/api/query")
    async def query(data: QueryRequest) -> dict[str, Any]:
        return service.query(
            question=data.question,
            retrieval_mode=data.retrieval_mode,
        ).model_dump(mode="json")

    @post("/api/evaluations/run")
    async def run_evaluation(data: EvaluationRequest) -> dict[str, Any]:
        return service.run_evaluation(modes=data.modes).model_dump(mode="json")

    @get("/api/runs/{run_id:str}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run = service.get_run(run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        return run.model_dump(mode="json")

    litestar_app = Litestar(
        route_handlers=[
            health_route,
            ingest_ragbench,
            query,
            run_evaluation,
            get_run,
        ]
    )
    return LocalApiApp(service=service, litestar_app=litestar_app)


app = create_app().litestar_app
