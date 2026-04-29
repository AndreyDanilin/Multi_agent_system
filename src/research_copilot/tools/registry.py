"""Typed tool registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_copilot.types import FunctionTool

ToolHandler = Callable[..., dict[str, Any]]


class ToolRegistry:
    """Minimal registry used by graph nodes to invoke typed tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._schemas: dict[str, FunctionTool] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        if not name:
            raise ValueError("tool name cannot be empty")
        self._tools[name] = handler
        self._schemas[name] = FunctionTool(
            name=name,
            description=description,
            parameters=parameters,
        )

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool is not registered: {name}")
        return self._tools[name](**kwargs)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def schemas(self) -> list[FunctionTool]:
        return [self._schemas[name] for name in self.names]
