"""Short-term memory implementation (conversation context)."""

from typing import List

from ..schemas.messages import AgentMessage
from .base import MemoryBase


class ShortTermMemory(MemoryBase):
    """In-memory conversation context.

    Stores recent messages in a simple list. Suitable for single-session
    conversations where full context is maintained.
    """

    _messages: List[AgentMessage] = []

    async def add(self, message: AgentMessage) -> None:
        """Add message to short-term memory.

        Args:
            message: Message to store
        """
        self._messages.append(message)

    async def get_recent(self, limit: int = 10) -> List[AgentMessage]:
        """Get most recent messages.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of recent messages, most recent last
        """
        return self._messages[-limit:]

    async def search(self, query: str, limit: int = 5) -> List[AgentMessage]:
        """Search messages by content (simple text matching).

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching messages
        """
        query_lower = query.lower()
        matches = [
            msg
            for msg in self._messages
            if query_lower in msg.content.lower()
        ]
        return matches[:limit]

    async def clear(self) -> None:
        """Clear all stored messages."""
        self._messages.clear()

    async def get_all(self) -> List[AgentMessage]:
        """Get all stored messages.

        Returns:
            Complete message history
        """
        return self._messages.copy()

