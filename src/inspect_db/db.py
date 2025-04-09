from uuid import UUID
from inspect_ai.log import EvalLog
from sqlmodel import SQLModel, create_engine, Session, select
from typing import Literal, Sequence
from contextlib import contextmanager
from .models import DBEvalLog, DBEvalSample, DBChatMessage, MessageRole
import logging

logger = logging.getLogger(__name__)


class EvalDB:
    """Low-level database operations that work directly with database models."""

    def __init__(self, database_url: str):
        """Initialize the database connection.

        Args:
            database_url: SQLAlchemy database URL (e.g. 'sqlite:///eval.db')
        """
        self.engine = create_engine(database_url)
        SQLModel.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        with Session(self.engine) as session:
            yield session

    def insert_log(self, log: EvalLog) -> UUID:
        """Insert a log and its associated samples and messages into the database.

        Args:
            log_data: Dictionary containing log data

        Returns:
            The UUID of the inserted log
        """
        # Create database models
        db_log = DBEvalLog(
            location=log.location,
        )

        # Insert log and samples
        with self.session() as session:
            session.add(db_log)
            session.commit()
            session.refresh(db_log)  # Ensure we have the UUID
            log_uuid = db_log.uuid

            # Insert samples and messages
            for sample in log.samples or []:
                db_sample = DBEvalSample.from_inspect(sample, log_uuid)
                session.add(db_sample)

                # Insert messages
                for index, msg in enumerate(sample.messages):
                    db_msg = DBChatMessage.from_inspect(msg, db_sample.uuid, index)
                    session.add(db_msg)

            session.commit()

        return log_uuid

    def get_db_logs(self, log_uuid: UUID | None = None) -> Sequence[DBEvalLog]:
        """Get all logs in the database.

        Args:
            log_uuid: UUID of the log to retrieve

        Returns:
            The DBEvalLog object, or None if not found
        """
        query = select(DBEvalLog)
        if log_uuid:
            query = query.where(DBEvalLog.uuid == log_uuid)
        with self.session() as session:
            return session.exec(query).all()

    def get_db_samples(
        self, log_uuid: UUID | None = None, sample_uuid: UUID | None = None
    ) -> Sequence[DBEvalSample]:
        """Get all samples for a log.

        Args:
            log_id: ID of the log

        Returns:
            List of DBEvalSample objects
        """
        query = select(DBEvalSample)
        if log_uuid:
            query = query.where(DBEvalSample.log_uuid == log_uuid)
        if sample_uuid:
            query = query.where(DBEvalSample.uuid == sample_uuid)
        with self.session() as session:
            return session.exec(query).all()

    def get_db_messages(
        self,
        log_uuid: UUID | None = None,
        sample_uuid: UUID | None = None,
        role: Literal["system", "user", "assistant", "tool"] | None = None,
    ) -> Sequence[DBChatMessage]:
        """Get database messages for a sample, optionally filtered by role.

        Args:
            sample_id: ID of the sample
            role: Optional role to filter messages by

        Returns:
            List of DBChatMessage objects
        """
        query = select(DBChatMessage).order_by(DBChatMessage.index_in_sample)
        if log_uuid:
            query = query.where(DBChatMessage.sample_uuid == sample_uuid)
        if sample_uuid:
            query = query.where(DBChatMessage.sample_uuid == sample_uuid)
        if role:
            query = query.where(DBChatMessage.role == MessageRole(role))
        with self.session() as session:
            return session.exec(query).all()
