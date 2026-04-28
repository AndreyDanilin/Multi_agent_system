"""Small attributed RAGBench techqa fixture for offline demos and tests."""

from __future__ import annotations

import json
from importlib import resources

from research_copilot.types import EvaluationQuestion, SourceDocument

DATASET = "galileo-ai/ragbench/techqa"


def _load_fixture() -> dict:
    with resources.files(__package__).joinpath("sample_techqa.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def load_sample_techqa_documents() -> list[SourceDocument]:
    payload = _load_fixture()
    return [SourceDocument(**item) for item in payload["documents"]]


def load_sample_techqa_questions() -> list[EvaluationQuestion]:
    payload = _load_fixture()
    return [EvaluationQuestion(**item) for item in payload["questions"]]
