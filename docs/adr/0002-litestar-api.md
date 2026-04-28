# ADR 0002: Use Litestar for API Layer

## Status
Accepted

## Context
The portfolio goal is to show modern, typed backend architecture rather than only a notebook-style demo. FastAPI would be highly recognizable, but Litestar offers a strong typed framework with dependency injection, controllers, OpenAPI support and a clean separation between application services and HTTP transport.

## Decision
Use Litestar as the API framework for production/runtime installs. Keep the service layer framework-neutral and expose a lightweight route manifest when Litestar is not installed, so tests and offline demos can still import the package.

## Consequences
- The API layer remains thin and replaceable.
- OpenAPI and typed request models are part of the intended runtime surface.
- Local tests do not need to install the full web stack.
