"""Agentic Research Copilot.

A portfolio-grade multi-agent research assistant where RAG is implemented as
one tool in a broader orchestration graph.
"""

from .graph import ResearchCopilotGraph
from .service import ResearchCopilotService
from .types import (
    AgentEvent,
    AgentState,
    AnswerResponse,
    Citation,
    EvaluationReport,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RetrievedChunk,
)

__all__ = [
    "AgentEvent",
    "AgentState",
    "AnswerResponse",
    "Citation",
    "EvaluationReport",
    "FunctionCall",
    "FunctionCallOutput",
    "FunctionTool",
    "ResearchCopilotGraph",
    "ResearchCopilotService",
    "RetrievedChunk",
]
