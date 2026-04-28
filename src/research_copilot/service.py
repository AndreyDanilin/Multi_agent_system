"""Application service layer used by API, CLI and UI."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from research_copilot.data.fixtures import (
    load_sample_techqa_documents,
    load_sample_techqa_questions,
)
from research_copilot.evaluation import EvaluationRunner
from research_copilot.graph import ResearchCopilotGraph
from research_copilot.retrieval import (
    DeterministicEmbeddingModel,
    LanceDBRepository,
    RetrievalService,
)
from research_copilot.types import AnswerResponse, EvaluationReport, RetrievalMode, SourceDocument


class ResearchCopilotService:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service
        self.graph = ResearchCopilotGraph(retrieval_service=retrieval_service)
        self.evaluator = EvaluationRunner(retrieval_service=retrieval_service)
        self.runs: dict[str, AnswerResponse | EvaluationReport] = {}

    @classmethod
    def create_demo(
        cls,
        data_dir: str | Path = ".cache/research-copilot",
    ) -> ResearchCopilotService:
        repository = LanceDBRepository(
            path=Path(data_dir) / "lancedb",
            embedding_model=DeterministicEmbeddingModel(dimensions=128),
        )
        return cls(retrieval_service=RetrievalService(repository=repository))

    def ingest_sample(self) -> int:
        return self.ingest_documents(load_sample_techqa_documents())

    def ingest_documents(self, documents: list[SourceDocument]) -> int:
        return self.retrieval_service.ingest_documents(documents)

    def ingest_ragbench(self, *, sample_only: bool = True) -> dict[str, str | int]:
        if not sample_only:
            raise RuntimeError(
                "Full Hugging Face ingestion requires installing the optional 'datasets' extra. "
                "The portfolio demo intentionally defaults to the committed techqa fixture."
            )
        chunks = self.ingest_sample()
        return {
            "dataset": "galileo-ai/ragbench",
            "subset": "techqa",
            "documents": len(load_sample_techqa_documents()),
            "chunks": chunks,
        }

    def query(
        self,
        *,
        question: str,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
    ) -> AnswerResponse:
        response = self.graph.query(question, retrieval_mode=retrieval_mode)
        run_id = str(uuid4())
        response.metadata["run_id"] = run_id
        self.runs[run_id] = response
        return response

    def run_evaluation(
        self,
        *,
        modes: list[RetrievalMode] | None = None,
    ) -> EvaluationReport:
        report = self.evaluator.run(load_sample_techqa_questions(), modes=modes)
        run_id = str(uuid4())
        report.metadata["run_id"] = run_id
        self.runs[run_id] = report
        return report

    def get_run(self, run_id: str) -> AnswerResponse | EvaluationReport | None:
        return self.runs.get(run_id)
