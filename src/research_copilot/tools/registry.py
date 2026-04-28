"""Typed tool registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ToolHandler = Callable[..., dict[str, Any]]


class ToolRegistry:
    """Minimal registry used by graph nodes to invoke typed tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if not name:
            raise ValueError("tool name cannot be empty")
        self._tools[name] = handler

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool is not registered: {name}")
        return self._tools[name](**kwargs)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)
