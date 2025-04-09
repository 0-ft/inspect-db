from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_ai.util import SandboxEnvironmentSpec
from pydantic import TypeAdapter
from sqlmodel import JSON, Column, Relationship, SQLModel, Field, TypeDecorator
from typing import Any, List
from datetime import datetime
import uuid
from enum import Enum
from inspect_ai.log import EvalError, EvalSample, Event, EvalSampleLimit
from uuid import UUID


class MessageRole(str, Enum):
    """Enum for chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PydanticJson(TypeDecorator):
    impl = JSON()
    cache_ok = True

    def __init__(self, pt):
        super().__init__()
        self.pt = TypeAdapter(pt)
        self.coerce_compared_value = self.impl.coerce_compared_value  # type: ignore

    def bind_processor(self, dialect):
        return lambda value: self.pt.dump_json(value) if value is not None else None

    def result_processor(self, dialect, coltype):
        return lambda value: self.pt.validate_json(value) if value is not None else None


class DBChatMessage(SQLModel, table=True):
    """Database model for a chat message."""

    # Database fields
    db_uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    db_sample_uuid: UUID = Field(foreign_key="dbevalsample.db_uuid")

    # Relationships
    sample: "DBEvalSample" = Relationship(back_populates="messages")

    # Inspect fields
    id: str | None = Field(default=None)  # Original inspect-ai message ID
    index_in_sample: int
    role: MessageRole
    content: str | list[Content] = Field(
        sa_column=Column(PydanticJson(str | list[Content]))
    )
    source: str | None = Field(default=None)

    # For assistant messages
    model: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(
        default=None, sa_column=Column(PydanticJson(list[ToolCall]))
    )

    # For tool messages
    tool_call_id: str | None = Field(default=None)
    function: str | None = Field(default=None)
    error: ToolCallError | None = Field(
        default=None, sa_column=Column(PydanticJson(ToolCallError))
    )

    @classmethod
    def from_inspect(
        cls, message: ChatMessage, sample_uuid: UUID, index_in_sample: int
    ) -> "DBChatMessage":
        return cls(
            id=message.id,
            db_sample_uuid=sample_uuid,
            index_in_sample=index_in_sample,
            role=MessageRole(message.role),
            content=message.content,
            source=message.source,
            model=message.model if isinstance(message, ChatMessageAssistant) else None,
            tool_calls=message.tool_calls
            if isinstance(message, ChatMessageAssistant)
            else None,
            tool_call_id=message.tool_call_id
            if isinstance(message, ChatMessageTool)
            else None,
            function=message.function if isinstance(message, ChatMessageTool) else None,
            error=message.error if isinstance(message, ChatMessageTool) else None,
        )

    def to_inspect(self) -> ChatMessage:
        assert (
            self.source is None or self.source == "input" or self.source == "generate"
        )
        if self.role == MessageRole.SYSTEM:
            return ChatMessageSystem(
                id=self.id,
                content=self.content,
                source=self.source,
            )
        elif self.role == MessageRole.USER:
            return ChatMessageUser(
                id=self.id,
                content=self.content,
                source=self.source,
            )
        elif self.role == MessageRole.ASSISTANT:
            return ChatMessageAssistant(
                id=self.id,
                content=self.content,
                model=self.model,
                tool_calls=self.tool_calls,
                source=self.source,
            )
        elif self.role == MessageRole.TOOL:
            return ChatMessageTool(
                id=self.id,
                content=self.content,
                tool_call_id=self.tool_call_id,
                function=self.function,
                error=self.error,
                source=self.source,
            )
        else:
            raise ValueError(f"Unknown message role: {self.role}")


class DBEvalSample(SQLModel, table=True):
    """Database model for an eval sample header."""

    # Database fields
    db_uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    db_log_uuid: UUID = Field(foreign_key="dbevallog.db_uuid")

    # Relationships
    log: "DBEvalLog" = Relationship(back_populates="samples")
    messages: list[DBChatMessage] = Relationship(back_populates="sample")

    # Inspect fields
    id: str  # Original inspect-ai sample ID
    epoch: int

    input: str | list[ChatMessage] = Field(sa_column=Column(JSON))
    choices: list[str] | None = Field(default=None, sa_column=Column(JSON))
    target: str | list[str] = Field(sa_column=Column(JSON))

    sandbox: SandboxEnvironmentSpec | None = Field(default=None, sa_column=Column(JSON))
    files: list[str] | None = Field(default=None, sa_column=Column(JSON))
    setup: str | None = Field(default=None, sa_column=Column(JSON))
    output: ModelOutput = Field(sa_column=Column(PydanticJson(ModelOutput)))
    scores: dict[str, Score] | None = Field(
        default=None, sa_column=Column(PydanticJson(dict[str, Score]))
    )

    sample_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    events: list[Event] = Field(sa_column=Column(PydanticJson(list[Event])))
    model_usage: dict[str, ModelUsage] = Field(
        sa_column=Column(PydanticJson(dict[str, ModelUsage]))
    )

    total_time: float | None = Field(default=None)
    working_time: float | None = Field(default=None)
    uuid: str | None = Field(default=None)
    error: EvalError | None = Field(
        default=None, sa_column=Column(PydanticJson(EvalError))
    )

    attachments: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    limit: EvalSampleLimit | None = Field(
        default=None, sa_column=Column(PydanticJson(EvalSampleLimit))
    )

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
            sample_metadata=sample.metadata,
            events=sample.events,
            model_usage=sample.model_usage,
            total_time=sample.total_time,
            working_time=sample.working_time,
            attachments=sample.attachments,
            limit=sample.limit,
            error=sample.error,
            uuid=sample.uuid,
            db_log_uuid=log_uuid,
        )

    def to_inspect(self) -> EvalSample:
        return EvalSample(
            id=self.id,
            epoch=self.epoch,
            input=self.input,
            target=self.target,
            sandbox=self.sandbox,
            files=self.files,
            setup=self.setup,
            output=self.output,
            scores=self.scores,
            metadata=self.sample_metadata,
            events=self.events,
            model_usage=self.model_usage,
            total_time=self.total_time,
            working_time=self.working_time,
            uuid=self.uuid,
            error=self.error,
            attachments=self.attachments,
            limit=self.limit,
            messages=[message.to_inspect() for message in self.messages],
        )


class DBEvalLog(SQLModel, table=True):
    """Database model for an eval log."""

    # Database fields
    db_uuid: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    location: str = Field(unique=True)  # Unique constraint on location
    inserted: datetime = Field(default_factory=datetime.now)

    # Relationships
    samples: List["DBEvalSample"] = Relationship(back_populates="log")
