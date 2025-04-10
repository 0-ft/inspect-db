from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import ContextManager, Literal, Optional, Any
import queue
from concurrent.futures import ThreadPoolExecutor
from inspect_ai.log import EvalLog, read_eval_log
from .db import EvalDB
from sqlalchemy.exc import IntegrityError
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.console import Console


class IngestionProgressListener:
    """Protocol for reporting ingestion progress."""

    def progress(self) -> ContextManager[Any]:
        """Context manager for progress reporting."""
        return nullcontext()

    def on_ingestion_started(self, workers: int) -> None:
        """Called when ingestion starts.

        Args:
            workers: Number of worker threads
        """
        pass

    def on_log_started(self, file_path: Path) -> None:
        """Called when a file starts being processed.

        Args:
            file_path: Path to the file being processed

        Returns:
            An identifier for this file's progress tracking
        """
        pass

    def on_log_loaded(
        self,
        file_path: Path,
        status: Literal["success", "error"],
        message: str,
    ) -> None:
        """Called when a file has been processed.

        Args:
            file_id: Identifier returned by on_file_started
            status: Status of the file processing
            message: Description of the outcome
        """
        pass

    def on_log_completed(
        self,
        file_path: Path,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        """Called when a file has been inserted."""
        pass

    def on_ingestion_complete(self) -> None:
        """Called when all files have been processed."""
        pass


class RichProgressListener(IngestionProgressListener):
    """Rich-based implementation of IngestionProgressListener."""

    started_count = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    workers = 0

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.rich_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.path_tasks = {}

    def progress(self) -> ContextManager[Any]:
        return self.rich_progress

    def on_ingestion_started(self, workers: int) -> None:
        self.workers = workers

    def on_log_started(self, file_path: Path) -> None:
        """Start tracking progress for a file."""
        task_id = self.rich_progress.add_task(f"Processing {file_path.name}", total=1)
        self.path_tasks[file_path] = task_id
        self.started_count += 1

    def on_log_completed(
        self,
        file_path: Path,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        """Update progress for a completed file."""
        self.rich_progress.update(
            self.path_tasks[file_path], advance=1, description=message
        )
        if status == "success":
            self.success_count += 1
        elif status == "skipped":
            self.skipped_count += 1
        elif status == "error":
            self.error_count += 1

    def on_ingestion_complete(self) -> None:
        """Show ingestion summary."""
        self.console.print()
        summary = Table(title="Ingestion Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Total Files Found", str(self.started_count))
        summary.add_row("Successfully Ingested", str(self.success_count))
        summary.add_row("Skipped (Already Exists)", str(self.skipped_count))
        summary.add_row("Errors", str(self.error_count))
        summary.add_row("Workers", str(self.workers))
        self.console.print(summary)


def insert_log(
    db: EvalDB, progress: IngestionProgressListener, log_path: Path, log: EvalLog
):
    try:
        db.insert_log(log)
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
                log = EvalLog.model_validate(log)
                progress.on_log_loaded(log_path, "success", f"Loaded {log_path}")
                insert_log(db, progress, log_path, log)
            except Exception as e:
                progress.on_log_loaded(log_path, "error", str(e))
                progress.on_log_completed(log_path, "error", str(e))
            finally:
                log_queue.task_done()
        except queue.Empty:
            break


def ingest_logs(
    database_uri: str,
    path_patterns: list[str],
    workers: int = 4,
    progress_listener: Optional[IngestionProgressListener] = None,
) -> None:
    """Ingest eval files into the database.

    Args:
        database_uri: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
        path_patterns: List of glob patterns matching .eval files
        workers: Number of worker threads for log loading
        progress_listener: Optional progress listener for custom progress display
    """
    # Initialize database
    db = EvalDB(database_uri)

    eval_paths = [Path(f) for pattern in path_patterns for f in glob(pattern)]

    if not eval_paths:
        print("No .eval files found matching the given patterns")
        return

    # Create queue
    log_queue = queue.Queue[Path]()
    for eval_path in eval_paths:
        log_queue.put(eval_path)

    # Start workers
    progress_listener = progress_listener or RichProgressListener()
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

    progress_listener.on_ingestion_complete()
