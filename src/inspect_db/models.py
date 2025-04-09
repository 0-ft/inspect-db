from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, Content, ModelOutput
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_ai.util import SandboxEnvironmentSpec
from sqlmodel import JSON, Column, Relationship, SQLModel, Field
from typing import Any, Optional, List
from datetime import datetime
import uuid
from enum import Enum
from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessage
from uuid import UUID
class MessageRole(str, Enum):
    """Enum for chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class DBChatMessage(SQLModel, table=True):
    """Database model for a chat message."""
    
    # Database fields
    uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    sample_uuid: UUID = Field(foreign_key="dbevalsample.uuid")
    
    # Relationships
    sample: "DBEvalSample" = Relationship(back_populates="messages")

    # Inspect fields
    id: str | None = Field(default=None) # Original inspect-ai message ID
    index_in_sample: int
    role: MessageRole
    content: str | list[Content] = Field(sa_column=Column(JSON))
    source: str | None = Field(default=None)

    # For assistant messages
    model: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None, sa_column=Column(JSON))

    # For tool messages
    tool_call_id: str | None = Field(default=None)
    function: str | None = Field(default=None)
    error: ToolCallError | None = Field(default=None, sa_column=Column(JSON))
    
    @classmethod
    def from_inspect(cls, message: ChatMessage, sample_uuid: UUID, index_in_sample: int) -> "DBChatMessage":
        return cls(
            id=message.id,
            sample_uuid=sample_uuid,
            index_in_sample=index_in_sample,
            role=MessageRole(message.role),
            content=message.content,
            source=message.source,
            model=message.model if isinstance(message, ChatMessageAssistant) else None,
            tool_calls=message.tool_calls if isinstance(message, ChatMessageAssistant) else None,
            tool_call_id=message.tool_call_id if isinstance(message, ChatMessageTool) else None,
            function=message.function if isinstance(message, ChatMessageTool) else None,
            error=message.error if isinstance(message, ChatMessageTool) else None
        )
    

class DBEvalSample(SQLModel, table=True):
    """Database model for an eval sample header."""
    
    # Database fields
    uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    log_uuid: UUID = Field(foreign_key="dbevallog.uuid")
    
    # Relationships
    log: "DBEvalLog" = Relationship(back_populates="samples")
    messages: list[DBChatMessage] = Relationship(back_populates="sample")

    # Inspect fields
    id: str  # Original inspect-ai sample ID
    epoch: int

    input: str | list[ChatMessage] = Field(sa_column=Column(JSON))
    choices: list[str] | None = Field(default=None, sa_column=Column(JSON))
    target: str | list[str] = Field(sa_column=Column(JSON))

    sandbox: Any = Field(default=None, sa_column=Column(JSON))
    files: list[str] | None = Field(default=None, sa_column=Column(JSON))
    setup: str | None = Field(default=None, sa_column=Column(JSON))
    output: ModelOutput | None = Field(default=None, sa_column=Column(JSON))
    scores: dict[str, Score] | None = Field(default=None, sa_column=Column(JSON))
    
    @classmethod
    def from_inspect(cls, sample: EvalSample, log_uuid: UUID) -> "DBEvalSample":
        return cls(
            id=str(sample.id),
            epoch=sample.epoch,
            input=sample.input,
            target=sample.target,
            sandbox=sample.sandbox,
            files=sample.files,
            setup=sample.setup,
            output=sample.output,
            scores=sample.scores,
            log_uuid=log_uuid,
        )


class DBEvalLog(SQLModel, table=True):
    """Database model for an eval log."""
    
    # Database fields
    uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    location: str = Field(unique=True)  # Unique constraint on location
    inserted: datetime = Field(default_factory=datetime.now)
    
    # Relationships
    samples: List["DBEvalSample"] = Relationship(back_populates="log")
