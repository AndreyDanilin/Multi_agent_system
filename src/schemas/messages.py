"""Message schemas for agent communication."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    MANAGER = "manager"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Type of message content."""

    TEXT = "text"
    PLAN = "plan"
    EXECUTION_RESULT = "execution_result"
    CRITIQUE = "critique"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Structured message for agent communication."""

    role: MessageRole = Field(..., description="Role of the message sender")
    message_type: MessageType = Field(..., description="Type of message content")
    content: str = Field(..., description="Main message content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: Optional[float] = Field(default=None, description="Message timestamp")

    class Config:
        """Pydantic config."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "role": "planner",
                "message_type": "plan",
                "content": "Execute step 1: Calculate 2+2",
                "metadata": {"step_id": 1},
            }
        }

