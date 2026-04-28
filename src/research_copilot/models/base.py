"""Provider-neutral LLM adapter boundary."""

from __future__ import annotations

from typing import Protocol


class LLMAdapter(Protocol):
    def answer(self, question: str, context: list[str]) -> str:
        ...


class DeterministicLLM:
    """Offline fallback synthesizer used by tests and no-key demos."""

    def answer(self, question: str, context: list[str]) -> str:
        if not context:
            return (
                "I do not have enough retrieved evidence to answer this question. "
                "Try ingesting a corpus or using a broader retrieval mode."
            )
        joined = " ".join(context[:2])
        first_sentence = joined.split(".")[0].strip()
        if first_sentence:
            return f"Based on the retrieved technical notes, {first_sentence}."
        return f"Based on the retrieved technical notes, {joined[:220].strip()}."
