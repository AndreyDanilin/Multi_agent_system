"""OpenAI Responses-style model client boundary."""

from __future__ import annotations

from typing import Any, Protocol

from research_copilot.types import FunctionTool

DEFAULT_RESPONSE_MODEL = "gpt-4.1-mini"


class OpenAIResponsesClient(Protocol):
    """Small subset of the OpenAI Responses API used by the graph."""

    def create(
        self,
        *,
        model: str,
        input: list[dict[str, Any]],
        tools: list[FunctionTool] | None = None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        ...


class DeterministicOpenAIResponsesClient:
    """Offline Responses-style synthesizer used by tests and no-key demos."""

    def create(
        self,
        *,
        model: str = DEFAULT_RESPONSE_MODEL,
        input: list[dict[str, Any]],
        tools: list[FunctionTool] | None = None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        context = [
            str(item.get("content", ""))
            for item in input
            if item.get("role") == "developer" and item.get("content")
        ]
        if not context:
            return {"output_text": (
                "I do not have enough retrieved evidence to answer this question. "
                "Try ingesting a corpus or using a broader retrieval mode."
            )}
        joined = " ".join(context[:2])
        first_sentence = joined.split(".")[0].strip()
        if first_sentence:
            return {"output_text": f"Based on the retrieved technical notes, {first_sentence}."}
        return {"output_text": f"Based on the retrieved technical notes, {joined[:220].strip()}."}
