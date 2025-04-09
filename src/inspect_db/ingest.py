from contextlib import nullcontext
from pathlib import Path
from glob import glob
from typing import ContextManager, Literal, Optional, Dict, Any, Protocol
import queue
from concurrent.futures import ThreadPoolExecutor
from inspect_ai.log import read_eval_log
from .db import EvalDB
from sqlmodel import select, func
from .models import DBEvalLog, DBEvalSample, DBChatMessage
from sqlalchemy.exc import IntegrityError


class IngestionProgressListener[FileId](Protocol):
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

    def on_file_started(self, file_path: Path) -> FileId:
        """Called when a file starts being processed.

        Args:
            file_path: Path to the file being processed

        Returns:
            An identifier for this file's progress tracking
        """
        ...

    def on_file_finished(
        self,
        file_id: FileId,
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

    def on_ingestion_complete(self) -> None:
        """Called when all files have been processed."""
        ...


class NullProgressListener(IngestionProgressListener):
    """Progress listener that does nothing."""

    def progress(self) -> ContextManager[Any]:
        return nullcontext()

    def on_ingestion_started(self, workers: int) -> None:
        pass

    def on_file_started(self, file_path: Path) -> Any:
        return None

    def on_file_finished(
        self, file_id: Any, status: Literal["success", "skipped", "error"], message: str
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


def process_eval_file(
    eval_file: Path, db: EvalDB, progress: IngestionProgressListener
) -> bool:
    """Process a single eval file and insert it into the database.

    Args:
        eval_file: Path to the eval file
        db: RawEvalDB instance
        progress: Progress listener

    Returns:
        Tuple of (success, message) where success is True if the file was processed successfully
        and message describes the outcome
    """
    file_id = progress.on_file_started(eval_file)
    try:
        log = read_eval_log(str(eval_file))
        log_uuid = db.insert_log(log)
        progress.on_file_finished(
            file_id, "success", f"Ingested {eval_file.name} ({log_uuid})"
        )
        return True
    except IntegrityError:
        progress.on_file_finished(
            file_id, "skipped", f"Skipped {eval_file.name} (already exists)"
        )
        return False
    except Exception as e:
        progress.on_file_finished(
            file_id, "error", f"Error processing {eval_file.name}: {str(e)}"
        )
        return False


def worker(q: queue.Queue, db: EvalDB, progress: IngestionProgressListener):
    """Worker thread that processes eval files from the queue.

    Args:
        q: Queue containing eval files
        db: RawEvalDB instance
        progress: Progress listener

    Returns:
        Tuple of (success_count, skipped_count, error_count)
    """
    while True:
        try:
            eval_file = q.get_nowait()
            process_eval_file(eval_file, db, progress)
            q.task_done()
        except queue.Empty:
            break


def ingest_eval_files(
    database_uri: str,
    eval_paths: list[str | Path],
    workers: int = 4,
    progress_listener: Optional[IngestionProgressListener] = None,
) -> None:
    """Ingest eval files into the database.

    Args:
        database_uri: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
        eval_paths: List of glob patterns matching .eval files
        workers: Number of worker threads
        progress_listener: Optional progress listener for custom progress display
    """
    # Initialize database
    db = EvalDB(database_uri)

    # Find all eval files
    eval_files = []
    for path in eval_paths:
        eval_files.extend(
            [Path(f) for f in glob(str(path), recursive=True) if f.endswith(".eval")]
        )

    if not eval_files:
        print("No .eval files found matching the given patterns")
        return

    # Create queue and worker pool
    q = queue.Queue()
    for eval_file in eval_files:
        q.put(eval_file)

    # Start workers
    progress_listener = progress_listener or NullProgressListener()

    progress_listener.on_ingestion_started(workers)

    with progress_listener.progress():
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(worker, q, db, progress_listener)
                for _ in range(workers)
            ]
            for future in futures:
                future.result()

    # Show summary
    progress_listener.on_ingestion_complete()
