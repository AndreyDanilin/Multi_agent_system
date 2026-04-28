from research_copilot.data.fixtures import (
    load_sample_techqa_documents,
    load_sample_techqa_questions,
)
from research_copilot.evaluation import EvaluationRunner
from research_copilot.retrieval.embeddings import DeterministicEmbeddingModel
from research_copilot.retrieval.repository import LanceDBRepository
from research_copilot.retrieval.service import RetrievalService


def test_evaluation_reports_retrieval_quality_metrics(tmp_path):
    repository = LanceDBRepository(
        path=tmp_path / "lancedb",
        embedding_model=DeterministicEmbeddingModel(dimensions=64),
    )
    retrieval = RetrievalService(repository=repository)
    retrieval.ingest_documents(load_sample_techqa_documents())
    runner = EvaluationRunner(retrieval_service=retrieval)

    report = runner.run(load_sample_techqa_questions(), modes=["lexical", "hybrid_rerank"], limit=3)

    assert report.dataset == "galileo-ai/ragbench/techqa"
    assert set(report.mode_metrics) == {"lexical", "hybrid_rerank"}
    assert report.mode_metrics["hybrid_rerank"].hit_at_k >= 0.5
    assert report.mode_metrics["hybrid_rerank"].mrr > 0
