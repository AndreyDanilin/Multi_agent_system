# ADR 0005: Prioritize Deterministic Retrieval Evaluation

## Status
Accepted

## Context
LLM-as-judge evaluation is useful but hard to reproduce without keys and can obscure whether retrieval improved. The project's advanced showcase is retrieval quality, so deterministic metrics should be the default.

## Decision
Evaluate retrieval modes with hit@k, mean reciprocal rank, citation coverage and latency. Optional LLM judge evaluation can be added later as a separate enhancement.

## Consequences
- Regression tests can run in CI without secrets.
- Retrieval improvements can be compared directly.
- The UI can explain why hybrid reranking improves or fails on a sample corpus.
