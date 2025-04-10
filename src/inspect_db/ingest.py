from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Literal, Optional, Dict, Any, Protocol
import queue
from concurrent.futures import ThreadPoolExecutor
from inspect_ai.log import EvalLog, read_eval_log
from .db import EvalDB
from sqlmodel import select, func
from .models import DBEvalLog, DBEvalSample, DBChatMessage
from sqlalchemy.exc import IntegrityError


class IngestionProgressListener(Protocol):
    """Protocol for reporting ingestion progress."""

    def progress(self) -> ContextManager[Any]:
        """Context manager for progress reporting."""
        ...

    def on_ingestion_started(self, workers: int) -> None:
        """Called when ingestion starts.

        Args:
            workers: Number of worker threads
        """
        ...

    def on_log_started(self, file_path: Path) -> None:
        """Called when a file starts being processed.

        Args:
            file_path: Path to the file being processed

        Returns:
            An identifier for this file's progress tracking
        """
        ...

    def on_log_loaded(
        self,
        file_path: Path,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        """Called when a file has been processed.

        Args:
            file_id: Identifier returned by on_file_started
            status: Status of the file processing
            message: Description of the outcome
        """
        ...

    def on_log_completed(
        self,
        file_path: Path,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        """Called when a file has been inserted."""

    def on_ingestion_complete(self) -> None:
        """Called when all files have been processed."""
        ...


class NullProgressListener(IngestionProgressListener):
    """Progress listener that does nothing."""

    def progress(self) -> ContextManager[Any]:
        return nullcontext()

    def on_ingestion_started(self, workers: int) -> None:
        pass

    def on_log_started(self, file_path: Path) -> Any:
        return None

    def on_log_loaded(
        self,
        file_path: Path,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        pass

    def on_ingestion_complete(self) -> None:
        pass


def get_db_stats(database_uri: str) -> Dict[str, Any]:
    """Get statistics about the database.

    Args:
        database_uri: SQLAlchemy database URI

    Returns:
        Dictionary containing database statistics
    """
    db = EvalDB(database_uri)

    with db.session() as session:
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


def insert_log(
    db: EvalDB, progress: IngestionProgressListener, log_path: Path, log: EvalLog
):
    try:
        with db.session() as session:
            db_log = DBEvalLog.from_inspect(log)
            session.add(db_log)

            for sample in log.samples or []:
                db_sample = DBEvalSample.from_inspect(sample, db_log.db_uuid)
                session.add(db_sample)

                for index_in_sample, message in enumerate(sample.messages or []):
                    db_message = DBChatMessage.from_inspect(
                        message, db_sample.db_uuid, index_in_sample
                    )
                    session.add(db_message)
            session.commit()
            progress.on_log_completed(log_path, "success", f"Inserted {log_path}")
    except IntegrityError:
        progress.on_log_completed(
            log_path, "skipped", f"Log already exists: {log_path}"
        )
    except Exception as e:
        progress.on_log_completed(log_path, "error", str(e))


def load_log_worker(
    log_queue: queue.Queue[Path], db: EvalDB, progress: IngestionProgressListener
):
    """Worker thread that processes eval files from the queue and puts them in the insert queue.

    Args:
        log_queue: Queue containing eval files to load
        db: Database instance
        progress: Progress listener
    """
    while True:
        try:
            log_path = log_queue.get_nowait()
            progress.on_log_started(log_path)
            try:
                log = read_eval_log(str(log_path))
                progress.on_log_loaded(log_path, "success", f"Loaded {log_path}")
                insert_log(db, progress, log_path, log)
            except Exception as e:
                progress.on_log_loaded(log_path, "error", str(e))
                progress.on_log_completed(log_path, "error", str(e))
            finally:
                log_queue.task_done()
        except queue.Empty:
            break


def ingest_eval_files(
    database_uri: str,
    eval_paths: list[str] | list[Path],
    workers: int = 4,
    progress_listener: Optional[IngestionProgressListener] = None,
) -> None:
    """Ingest eval files into the database.

    Args:
        database_uri: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
        eval_paths: List of glob patterns matching .eval files
        workers: Number of worker threads for log loading
        progress_listener: Optional progress listener for custom progress display
    """
    # Initialize database
    db = EvalDB(database_uri)

    eval_paths = [Path(path) for path in eval_paths]

    if not eval_paths:
        print("No .eval files found matching the given patterns")
        return

    # Create queues
    log_queue = queue.Queue[Path]()

    # Fill log queue
    for eval_file in eval_paths:
        log_queue.put(eval_file)

    # Start workers
    progress_listener = progress_listener or NullProgressListener()
    progress_listener.on_ingestion_started(workers)

    with progress_listener.progress():
        # Start log loading workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            load_futures = [
                executor.submit(load_log_worker, log_queue, db, progress_listener)
                for _ in range(workers)
            ]

            # Wait for all workers to complete
            for future in load_futures:
                future.result()

    # Show summary
    progress_listener.on_ingestion_complete()
