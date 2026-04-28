"""Deterministic retrieval evaluation."""

from __future__ import annotations

from research_copilot.retrieval.service import RetrievalService
from research_copilot.types import (
    EvaluationMetrics,
    EvaluationQuestion,
    EvaluationReport,
    RetrievalMode,
)


class EvaluationRunner:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def run(
        self,
        questions: list[EvaluationQuestion],
        *,
        modes: list[RetrievalMode] | None = None,
        limit: int = 5,
    ) -> EvaluationReport:
        selected_modes = modes or ["lexical", "vector", "hybrid", "hybrid_rerank"]
        metrics: dict[str, EvaluationMetrics] = {}

        for mode in selected_modes:
            hits = 0
            reciprocal_rank_sum = 0.0
            citation_coverage = 0
            latency_sum = 0.0

            for item in questions:
                results = self.retrieval_service.search(item.question, mode=mode, limit=limit)
                latency_sum += self.retrieval_service.last_latency_ms
                rank = next(
                    (
                        index + 1
                        for index, chunk in enumerate(results)
                        if chunk.document_id == item.expected_document_id
                    ),
                    None,
                )
                if rank is not None:
                    hits += 1
                    reciprocal_rank_sum += 1 / rank
                    citation_coverage += 1

            total = max(1, len(questions))
            metrics[mode] = EvaluationMetrics(
                mode=mode,
                hit_at_k=round(hits / total, 4),
                mrr=round(reciprocal_rank_sum / total, 4),
                citation_coverage=round(citation_coverage / total, 4),
                average_latency_ms=round(latency_sum / total, 3),
            )

        dataset = questions[0].dataset if questions else "galileo-ai/ragbench/techqa"
        return EvaluationReport(
            dataset=dataset,
            mode_metrics=metrics,
            total_questions=len(questions),
        )
