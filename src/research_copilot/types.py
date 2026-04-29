"""Public data contracts for the Agentic Research Copilot."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

RetrievalMode = Literal["lexical", "vector", "hybrid", "hybrid_rerank"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class SourceDocument(BaseModel):
    """A source document imported from RAGBench or uploaded by a user."""

    document_id: str
    title: str
    text: str
    source: str = "galileo-ai/ragbench/techqa"
    answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A retrievable text chunk with retrieval scores and provenance."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    source: str
    lexical_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A user-facing citation attached to an answer."""

    chunk_id: str
    document_id: str
    title: str
    quote: str
    source: str
    score: float


class AgentEvent(BaseModel):
    """Trace event emitted by a graph node."""

    node: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class FunctionCall(BaseModel):
    """OpenAI Responses-style function call item requested by an agent."""

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str = "{}"

    @classmethod
    def from_arguments(cls, *, name: str, arguments: dict[str, Any]) -> FunctionCall:
        return cls(name=name, arguments=json.dumps(arguments, separators=(",", ":")))

    def parsed_arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments or "{}")


class FunctionCallOutput(BaseModel):
    """OpenAI Responses-style function output item returned to the graph."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

    @classmethod
    def from_output(cls, *, call_id: str, output: dict[str, Any]) -> FunctionCallOutput:
        return cls(call_id=call_id, output=json.dumps(output, separators=(",", ":"), default=str))

    def parsed_output(self) -> dict[str, Any]:
        return json.loads(self.output or "{}")


class FunctionTool(BaseModel):
    """OpenAI Responses-style function tool declaration."""

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True


class AgentState(BaseModel):
    """State passed through the research copilot graph."""

    question: str
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    route: str | None = None
    plan: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str | None = None
    assessment: Literal["complete", "needs_evidence", "failed"] | None = None
    confidence: float = 0.0
    events: list[AgentEvent] = Field(default_factory=list)
    function_calls: list[FunctionCall] = Field(default_factory=list)
    function_call_outputs: list[FunctionCallOutput] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    """Final response returned by API, UI and SDK calls."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[AgentEvent] = Field(default_factory=list)
    confidence: float = 0.0
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationQuestion(BaseModel):
    """A deterministic evaluation item derived from RAGBench."""

    question_id: str
    question: str
    expected_document_id: str
    reference_answer: str
    dataset: str = "galileo-ai/ragbench/techqa"


class EvaluationMetrics(BaseModel):
    """Retrieval quality metrics for one retrieval mode."""

    mode: RetrievalMode
    hit_at_k: float
    mrr: float
    citation_coverage: float
    average_latency_ms: float


class EvaluationReport(BaseModel):
    """Comparison report across retrieval modes."""

    dataset: str
    mode_metrics: dict[str, EvaluationMetrics]
    total_questions: int
    generated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
