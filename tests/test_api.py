from research_copilot.api import create_app
from research_copilot.retrieval import (
    DeterministicEmbeddingModel,
    InMemoryRetrievalRepository,
    RetrievalService,
)
from research_copilot.service import ResearchCopilotService


def create_test_service() -> ResearchCopilotService:
    repository = InMemoryRetrievalRepository(
        embedding_model=DeterministicEmbeddingModel(dimensions=64)
    )
    return ResearchCopilotService(retrieval_service=RetrievalService(repository=repository))


def test_api_factory_exposes_required_routes():
    service = create_test_service()
    app = create_app(service=service)

    assert set(app.route_paths) >= {
        "/api/health",
        "/api/ingest/ragbench",
        "/api/query",
        "/api/evaluations/run",
        "/api/runs/{run_id}",
    }


def test_service_query_endpoint_contract_works_without_external_keys():
    service = create_test_service()
    service.ingest_sample()

    response = service.query(
        question="DNS lookup fails after I connect to VPN. What should I check?",
        retrieval_mode="hybrid_rerank",
    )

    assert response.answer
    assert response.citations
    assert response.retrieval_mode == "hybrid_rerank"
