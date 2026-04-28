# ADR 0004: Use a Provider-Neutral Model Layer

## Status
Accepted

## Context
Portfolio reviewers should be able to run the project without API keys, while the architecture should still support hosted or local LLM providers.

## Decision
Define a small LLM adapter boundary for answer synthesis and use a deterministic local fallback for tests and no-key demos. Hosted models can be added behind the same interface without affecting graph, retrieval or API contracts.

## Consequences
- Tests are deterministic and cost-free.
- The demo runs without secrets.
- Model providers remain an implementation detail, not a system-wide dependency.
