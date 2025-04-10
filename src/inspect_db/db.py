from collections.abc import Iterator
from uuid import UUID
from inspect_ai.log import EvalLog, EvalSample
from sqlalchemy import Engine
from sqlmodel import SQLModel, String, cast, col, create_engine, Session, func, select
from typing import Any, Literal
from contextlib import contextmanager
from .models import DBEvalLog, DBEvalSample, DBChatMessage
import logging

logger = logging.getLogger(__name__)


class EvalDB:
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

    def insert_log_header(self, log: EvalLog) -> UUID:
        """Insert a log header into the database.

        Args:
            log: EvalLog object

        Returns:
            The UUID of the inserted log
        """
        # Create database models
        db_log = DBEvalLog.from_inspect(log)

        # Insert log header
        with self.session() as session:
            session.add(db_log)
            session.commit()
            session.refresh(db_log)  # Ensure we have the UUID
            log_uuid = db_log.db_uuid

        return log_uuid

    def insert_log_sample(self, sample: EvalSample, log_uuid: UUID) -> UUID:
        """Insert a log sample into the database.

        Args:
            sample: EvalSample object

        Returns:
            The UUID of the inserted log sample
        """
        db_sample = DBEvalSample.from_inspect(sample, log_uuid)
        with self.session() as session:
            session.add(db_sample)
            session.commit()
            session.refresh(db_sample)  # Ensure we have the UUID
            sample_uuid = db_sample.db_uuid

        return sample_uuid

    def insert_log_and_samples(self, log: EvalLog) -> UUID:
        """Insert a log and its associated samples and messages into the database.

        Args:
            log_data: Dictionary containing log data

        Returns:
            The UUID of the inserted log
        """
        # Create database models
        db_log = DBEvalLog.from_inspect(log)

        # Insert log and samples
        with self.session() as session:
            session.add(db_log)
            session.commit()
            session.refresh(db_log)  # Ensure we have the UUID
            log_uuid = db_log.db_uuid

            # Insert samples and messages
            for sample in log.samples or []:
                db_sample = DBEvalSample.from_inspect(sample, log_uuid)
                session.add(db_sample)

                # Insert messages
                for index, msg in enumerate(sample.messages):
                    db_msg = DBChatMessage.from_inspect(
                        msg, db_sample.db_uuid, log_uuid, index
                    )
                    session.add(db_msg)

            session.commit()

        return log_uuid

    def get_db_logs(self, log_uuid: UUID | None = None) -> Iterator[DBEvalLog]:
        """Get all logs in the database.

        Args:
            log_uuid: UUID of the log to retrieve

        Returns:
            The DBEvalLog object, or None if not found
        """
        query = select(DBEvalLog)
        if log_uuid:
            query = query.where(DBEvalLog.db_uuid == log_uuid)
        with self.session() as session:
            yield from session.exec(query)

    def get_db_samples(
        self, log_uuid: UUID | None = None, sample_uuid: UUID | None = None
    ) -> Iterator[DBEvalSample]:
        """Get all samples for a log.

        Args:
            log_id: ID of the log

        Returns:
            List of DBEvalSample objects
        """
        query = select(DBEvalSample)
        if log_uuid:
            query = query.where(DBEvalSample.db_log_uuid == log_uuid)
        if sample_uuid:
            query = query.where(DBEvalSample.db_uuid == sample_uuid)
        with self.session() as session:
            yield from session.exec(query)

    def get_db_messages(
        self,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        role: Literal["system", "user", "assistant", "tool"] | None = None,
        pattern: str | None = None,
    ) -> Iterator[DBChatMessage]:
        """Get database messages for a sample, optionally filtered by role.

        Args:
            sample_id: ID of the sample
            role: Optional role to filter messages by

        Returns:
            List of DBChatMessage objects
        """
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
        with self.session() as session:
            yield from session.exec(query)

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
            avg_samples = (
                session.exec(
                    select(
                        func.avg(
                            select(func.count())
                            .select_from(DBEvalSample)
                            .where(DBEvalSample.db_log_uuid == DBEvalLog.db_uuid)
                            .scalar_subquery()
                        )
                    )
                ).one()
                or 0
            )

            # Get average messages per sample
            avg_messages = (
                session.exec(
                    select(
                        func.avg(
                            select(func.count())
                            .select_from(DBChatMessage)
                            .where(DBChatMessage.db_sample_uuid == DBEvalSample.db_uuid)
                            .scalar_subquery()
                        )
                    )
                ).one()
                or 0
            )

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
