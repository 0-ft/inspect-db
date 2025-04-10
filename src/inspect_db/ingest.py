from contextlib import nullcontext
from glob import glob
from pathlib import Path
import re
from typing import ContextManager, Literal, Optional, Any
import queue
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID
import zipfile
from inspect_ai.log import EvalSample, read_eval_log
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
        """Called when ingestion starts."""
        pass

    def on_log_started(self, file_path: Path) -> None:
        """Called when a file starts being processed."""
        pass

    def on_log_samples_counted(self, file_path: Path, samples_count: int) -> None:
        """Called when the header of a file has been read."""
        pass

    def on_log_sample_completed(
        self,
        file_path: Path,
        sample_id: str,
        epoch: int,
        status: Literal["success", "error"],
        message: str,
    ) -> None:
        """Called when a sample of a file has been read."""
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
        task_id = self.rich_progress.add_task(file_path.name, start=False)
        self.path_tasks[file_path] = task_id
        self.started_count += 1

    def on_log_samples_counted(self, file_path: Path, samples_count: int) -> None:
        """Called when the header of a file has been read."""
        self.rich_progress.start_task(self.path_tasks[file_path])
        self.rich_progress.update(
            self.path_tasks[file_path],
            total=samples_count,
        )

    def on_log_sample_completed(
        self,
        file_path: Path,
        sample_id: str,
        epoch: int,
        status: Literal["success", "skipped", "error"],
        message: str,
    ) -> None:
        """Called when a sample of a file has been read."""
        self.rich_progress.update(
            self.path_tasks[file_path],
            advance=1,
        )

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


# def _get_log_sample_ids(log_path: Path) -> list[tuple[str, int]]:
#     """Get the sample ids and epochs from a log file."""
#     # format is samples/<id>_epoch_<epoch>.json
#     with zipfile.ZipFile(log_path, "r") as zip_file:
#         pattern = re.compile(r"samples/(.*)_epoch_(\d+).json")
#         files = list(zip_file.namelist())
#         matches = [pattern.match(file) for file in files]
#         return [(match.group(1), int(match.group(2))) for match in matches if match]


def insert_log_samples(
    db: EvalDB,
    progress: IngestionProgressListener,
    log_path: Path,
    log_uuid: UUID,
):
    sample_pattern = re.compile(r"samples/(.*)_epoch_(\d+).json")
    with zipfile.ZipFile(log_path, "r") as zip_file:
        files = list(zip_file.namelist())
        matches = [sample_pattern.match(file) for file in files]
        sample_ids = [
            (match.group(1), int(match.group(2))) for match in matches if match
        ]
        progress.on_log_samples_counted(log_path, len(sample_ids))

        for id, epoch in sample_ids:
            try:
                with zip_file.open(f"samples/{id}_epoch_{epoch}.json", "r") as f:
                    sample = EvalSample.model_validate_json(f.read())
                    db.insert_sample_and_messages(sample, log_uuid)
                    progress.on_log_sample_completed(
                        log_path, str(sample.id), sample.epoch, "success", "Inserted"
                    )
            except Exception as e:
                progress.on_log_sample_completed(
                    log_path, str(sample.id), sample.epoch, "error", str(e)
                )


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
        except queue.Empty:
            break

        progress.on_log_started(log_path)
        try:
            log_header = read_eval_log(str(log_path), header_only=True)
            log_uuid = db.insert_log_header(log_header)
            insert_log_samples(db, progress, log_path, log_uuid)
            progress.on_log_completed(log_path, "success", f"Inserted {log_path}")
        except IntegrityError:
            progress.on_log_completed(
                log_path, "skipped", f"Log already exists: {log_path}"
            )
        except Exception as e:
            progress.on_log_completed(log_path, "error", str(e))
        finally:
            log_queue.task_done()


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
