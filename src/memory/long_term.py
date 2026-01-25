"""Long-term memory interface (pluggable vector store)."""

from typing import List

from ..schemas.messages import AgentMessage
from .base import MemoryBase


class LongTermMemory(MemoryBase):
    """Abstract interface for long-term memory.

    This is a placeholder for future vector store integration.
    In production, this would connect to:
    - Pinecone, Weaviate, or Chroma for vector storage
    - Embedding models for semantic search
    - Persistent storage for cross-session memory

    Current implementation is a no-op to maintain interface compatibility.
    """

    async def add(self, message: AgentMessage) -> None:
        """Add message to long-term memory.

        Args:
            message: Message to store

        Note:
            Currently a no-op. Implement vector storage integration here.
        """
        # TODO: Implement vector store integration
        # 1. Generate embeddings for message content
        # 2. Store in vector database with metadata
        # 3. Index for semantic search
        pass

    async def get_recent(self, limit: int = 10) -> List[AgentMessage]:
        """Retrieve recent messages from long-term memory.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            Empty list (not implemented)

        Note:
            In production, this would query vector store by recency.
        """
        return []

    async def search(self, query: str, limit: int = 5) -> List[AgentMessage]:
        """Semantic search in long-term memory.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            Empty list (not implemented)

        Note:
            In production, this would:
            1. Generate query embedding
            2. Perform similarity search in vector store
            3. Return most relevant messages
        """
        return []

    async def clear(self) -> None:
        """Clear long-term memory.

        Note:
            In production, this would clear the vector store.
        """
        pass

