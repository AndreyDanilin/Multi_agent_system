"""Base memory interface."""

from abc import ABC, abstractmethod
from typing import Any, List

from pydantic import BaseModel

from ..schemas.messages import AgentMessage


class MemoryBase(ABC, BaseModel):
    """Abstract base class for memory implementations."""

    @abstractmethod
    async def add(self, message: AgentMessage) -> None:
        """Add a message to memory.

        Args:
            message: Message to store
        """
        pass

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> List[AgentMessage]:
        """Retrieve recent messages.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of recent messages
        """
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[AgentMessage]:
        """Search memory by query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching messages
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored messages."""
        pass

