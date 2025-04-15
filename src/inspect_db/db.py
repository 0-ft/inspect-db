from collections.abc import Iterator
from uuid import UUID
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ChatMessage
from inspect_ai.scorer import Score
from sqlalchemy import Engine
from sqlmodel import SQLModel, String, cast, col, create_engine, Session, func, select
from typing import Any, Literal, Protocol
from contextlib import contextmanager

from inspect_db.common import EvalSampleLocator
from rich.table import Table
from rich.console import RenderableType

from .models import DBEvalLog, DBEvalSample, DBChatMessage
import logging

logger = logging.getLogger(__name__)


class EvalSource(Protocol):
    """Protocol for eval sources."""

    def get_logs(
        self, task: str | None = None, task_id: str | None = None
    ) -> Iterator[EvalLog]:
        """Get logs from the eval source."""
        ...

    def get_samples(
        self,
        log_task: str | None = None,
        log_task_id: str | None = None,
        sample_id: str | None = None,
    ) -> Iterator[tuple[EvalSampleLocator, EvalSample]]:
        """Get samples from the eval source."""
        ...

    def get_messages(
        self,
        role: Literal["system", "user", "assistant", "tool"] | None = None,
        pattern: str | None = None,
    ) -> Iterator[tuple[EvalSampleLocator, ChatMessage]]:
        """Get messages from the eval source."""
        ...

    def get_scores(self) -> Iterator[tuple[EvalSampleLocator, dict[str, Score] | None]]:
        """Get scores from the eval source."""
        ...


class EvalDB(EvalSource):
    """Low-level database operations that work directly with database models."""

    def __init__(self, db: str | Engine):
        """Initialize the database connection.

        Args:
            database_url: SQLAlchemy database URL (e.g. 'sqlite:///eval.db')
        """
        if isinstance(db, str):
            self.engine = create_engine(db)
        else:
            self.engine = db
        SQLModel.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        with Session(self.engine) as session:
            SQLModel.metadata.create_all(self.engine)
            yield session

    def ingest(
        self,
        log: EvalLog,
        tags: list[str] | None = None,
        session: Session | None = None,
    ) -> UUID:
        """Insert a log and its associated samples and messages into the database.

        Args:
            log: EvalLog object

        Returns:
            The UUID of the inserted log
        """
        # Create database models
        with session or self.session() as session:
            db_log = DBEvalLog.from_inspect(log, tags=tags)
            log_uuid = db_log.db_uuid
            samples = []
            messages = []
            for sample in log.samples or []:
                db_sample = DBEvalSample.from_inspect(sample, log_uuid)
                samples.append(db_sample)
                for index, msg in enumerate(sample.messages or []):
                    db_msg = DBChatMessage.from_inspect(
                        msg, db_sample.db_uuid, log_uuid, index
                    )
                    messages.append(db_msg)
            session.add(db_log)
            session.add_all(samples)
            session.add_all(messages)
            session.commit()

        return log_uuid

    @staticmethod
    def log_query(
        log_uuid: UUID | None = None,
        task: str | None = None,
        task_id: str | None = None,
        tags: list[str] | None = None,
    ):
        query = select(DBEvalLog)
        if log_uuid:
            query = query.where(DBEvalLog.db_uuid == log_uuid)
        if task:
            query = query.where(DBEvalLog.eval.task == task)
        if task_id:
            query = query.where(DBEvalLog.eval.task_id == task_id)
        for tag in tags or []:
            query = query.where(col(DBEvalLog.db_tags).contains(tag))
        return query

    @staticmethod
    def sample_query(
        sample_id: str | None = None,
        log_task: str | None = None,
        log_task_id: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
    ):
        query = select(DBEvalSample)
        if sample_id:
            query = query.where(DBEvalSample.id == sample_id)
        if log_task:
            query = query.where(DBEvalSample.log.eval.task == log_task)
        if log_task_id:
            query = query.where(DBEvalSample.log.eval.task_id == log_task_id)
        if log_uuid:
            query = query.where(DBEvalSample.db_log_uuid == log_uuid)
        if sample_uuid:
            query = query.where(DBEvalSample.db_uuid == sample_uuid)
        return query

    def get_db_logs(
        self,
        log_uuid: UUID | None = None,
        task: str | None = None,
        task_id: str | None = None,
        tags: list[str] | None = None,
        session: Session | None = None,
    ) -> Iterator[DBEvalLog]:
        """Get all logs in the database.

        Args:
            log_uuid: UUID of the log to retrieve

        Returns:
            The DBEvalLog object, or None if not found
        """
        query = EvalDB.log_query(
            log_uuid=log_uuid,
            task=task,
            task_id=task_id,
            tags=tags,
        )
        with session or self.session() as session:
            yield from session.exec(query)

    def get_logs(
        self,
        task: str | None = None,
        task_id: str | None = None,
        log_uuid: UUID | None = None,
        tags: list[str] | None = None,
        session: Session | None = None,
    ) -> Iterator[EvalLog]:
        for db_log in self.get_db_logs(
            task=task, task_id=task_id, log_uuid=log_uuid, tags=tags, session=session
        ):
            yield db_log.to_inspect()

    def get_db_samples(
        self,
        log_task: str | None = None,
        log_task_id: str | None = None,
        sample_id: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        session: Session | None = None,
    ) -> Iterator[DBEvalSample]:
        query = EvalDB.sample_query(
            sample_id=sample_id,
            log_task=log_task,
            log_task_id=log_task_id,
            log_uuid=log_uuid,
            sample_uuid=sample_uuid,
        )
        with session or self.session() as session:
            yield from session.exec(query)

    def get_samples(
        self,
        log_task: str | None = None,
        log_task_id: str | None = None,
        sample_id: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        session: Session | None = None,
    ) -> Iterator[tuple[EvalSampleLocator, EvalSample]]:
        """Get matching samples in inspect EvalSample format.

        Args:
            log_uuid: UUID of the log (optional)
            sample_uuid: UUID of the sample (optional)

        Returns:
            List of EvalSample objects
        """
        for db_sample in self.get_db_samples(
            sample_id=sample_id,
            log_task=log_task,
            log_task_id=log_task_id,
            log_uuid=log_uuid,
            sample_uuid=sample_uuid,
            session=session,
        ):
            yield (db_sample.locator(), db_sample.to_inspect())

    def get_db_messages(
        self,
        role: Literal["system", "user", "assistant", "tool"] | None = None,
        pattern: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        has_tool_calls: bool | None = None,
        session: Session | None = None,
    ) -> Iterator[DBChatMessage]:
        query = select(DBChatMessage).order_by(
            col(DBChatMessage.db_log_uuid),
            col(DBChatMessage.db_sample_uuid),
            col(DBChatMessage.index_in_sample),
        )
        if log_uuid:
            query = query.where(col(DBChatMessage.sample.db_log_uuid) == log_uuid)
        if sample_uuid:
            query = query.where(col(DBChatMessage.db_sample_uuid) == sample_uuid)
        if role:
            query = query.where(DBChatMessage.role == role)
        if pattern:
            query = query.where(
                cast(col(DBChatMessage.content), String).like(f"%{pattern}%")
            )  # TODO: duckdb-engine doesn't seem to support regexp_match
        if has_tool_calls is not None:
            if has_tool_calls:
                query = query.where(col(DBChatMessage.tool_calls) != None)  # noqa: E711
            else:
                query = query.where(col(DBChatMessage.tool_calls) == None)  # noqa: E711
        with session or self.session() as session:
            for db_msg in session.exec(query):
                yield db_msg

    def get_messages(
        self,
        role: Literal["system", "user", "assistant", "tool"] | None = None,
        pattern: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        has_tool_calls: bool | None = None,
        session: Session | None = None,
    ) -> Iterator[tuple[EvalSampleLocator, ChatMessage]]:
        for db_msg in self.get_db_messages(
            role=role,
            pattern=pattern,
            log_uuid=log_uuid,
            sample_uuid=sample_uuid,
            has_tool_calls=has_tool_calls,
            session=session,
        ):
            yield (db_msg.sample.locator(), db_msg.to_inspect())

    def get_scores(
        self,
        sample_id: str | None = None,
        log_task: str | None = None,
        log_task_id: str | None = None,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
    ) -> Iterator[tuple[EvalSampleLocator, dict[str, Score] | None]]:
        query = EvalDB.sample_query(
            sample_id=sample_id,
            log_task=log_task,
            log_task_id=log_task_id,
            log_uuid=log_uuid,
            sample_uuid=sample_uuid,
        )
        with self.session() as session:
            for db_sample in session.exec(query):
                yield (db_sample.locator(), db_sample.scores)

    def stats(self) -> dict[str, Any]:
        with self.session() as session:
            # Count logs
            log_count = session.exec(select(func.count()).select_from(DBEvalLog)).one()

            # Count samples
            sample_count = session.exec(
                select(func.count()).select_from(DBEvalSample)
            ).one()

            # Count messages
            message_count = session.exec(
                select(func.count()).select_from(DBChatMessage)
            ).one()

            # Get average samples per log
            avg_samples = sample_count / log_count if log_count > 0 else 0

            # Get average messages per sample
            avg_messages = message_count / sample_count if sample_count > 0 else 0

            # Get message role distribution
            role_counts = session.exec(
                select(DBChatMessage.role, func.count()).group_by(DBChatMessage.role)
            ).all()

            return {
                "log_count": log_count,
                "sample_count": sample_count,
                "message_count": message_count,
                "avg_samples_per_log": round(avg_samples, 2),
                "avg_messages_per_sample": round(avg_messages, 2),
                "role_distribution": dict(role_counts),
            }

    def stats_table(self) -> RenderableType:
        stats = self.stats()

        # Create and display main stats table
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Logs", str(stats["log_count"]))
        table.add_row("Total Samples", str(stats["sample_count"]))
        table.add_row("Total Messages", str(stats["message_count"]))

        table.add_section()
        table.add_row("System Messages", str(stats["role_distribution"]["system"]))
        table.add_row("User Messages", str(stats["role_distribution"]["user"]))
        table.add_row(
            "Assistant Messages", str(stats["role_distribution"]["assistant"])
        )
        table.add_row("Tool Messages", str(stats["role_distribution"]["tool"]))
        table.add_section()

        table.add_row("Avg Samples per Log", str(stats["avg_samples_per_log"]))
        table.add_row("Avg Messages per Sample", str(stats["avg_messages_per_sample"]))

        return table
