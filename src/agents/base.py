"""Base agent class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..schemas.messages import AgentMessage, MessageRole, MessageType
from ..utils.logger import get_logger


class AgentBase(ABC, BaseModel):
    """Abstract base class for all agents.

    Provides common functionality:
    - Message creation
    - Logging
    - LLM interaction interface
    """

    name: str
    role: MessageRole
    llm: Optional[Any] = None  # LangChain LLM instance
    logger: Any = None

    def __init__(self, **data: Any):
        """Initialize agent with logging."""
        super().__init__(**data)
        if self.logger is None:
            self.logger = get_logger(self.__class__.__name__)

    def create_message(
        self,
        content: str,
        message_type: MessageType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Create a structured message.

        Args:
            content: Message content
            message_type: Type of message
            metadata: Optional metadata

        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            role=self.role,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
        )

    @abstractmethod
    async def process(
        self, state: Any, **kwargs: Any
    ) -> AgentMessage:
        """Process input and generate response.

        Args:
            state: Current agent state
            **kwargs: Additional context

        Returns:
            AgentMessage with response
        """
        pass

    def log_decision(self, decision: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log agent decision for observability.

        Args:
            decision: Description of the decision
            context: Additional context data
        """
        self.logger.info(
            "agent_decision",
            agent=self.name,
            decision=decision,
            context=context or {},
        )

