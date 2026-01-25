"""Base tool interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field


class ToolBase(ABC, BaseModel):
    """Abstract base class for tools."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description for LLM")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Dictionary with execution results

        Raises:
            ValueError: If arguments are invalid
            RuntimeError: If execution fails
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool arguments.

        Returns:
            JSON schema dictionary compatible with OpenAI function calling
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema(),
            },
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters.

        Returns:
            JSON schema for parameters
        """
        pass

