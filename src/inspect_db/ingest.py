from __future__ import annotations
from glob import glob
from pathlib import Path
from queue import Queue
from queue import Empty
from threading import Thread
from typing import List, Dict

from sqlmodel import select, col
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskID,
    ProgressColumn,
)
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from inspect_ai.log import read_eval_log
from inspect_db.models import DBChatMessage, DBEvalLog, DBEvalSample
from inspect_db.util import iter_inspect_samples_fast, read_eval_log_header
from .db import EvalDB


class StatusColumn(ProgressColumn):
    """A column that shows a spinner, tick, or cross based on task status."""

    def render(self, task) -> RenderableType:
        status = task.fields.get("status", "")
        if status == "inserted":
            return Text("✅", style="green")
        elif status == "error":
            return Text("❌", style="red")
        elif status == "reading":
            return Spinner("dots12", style="blue bold")
        elif status == "inserting":
            return Spinner("dots12", style="green bold")
        elif status == "queued":
            return Spinner("dots12", style="dim")
        else:
            return " "


def read_log_worker(
    job_queue: Queue[tuple[Path, TaskID] | None],
    insert_queue: Queue[tuple[Path, List[DBEvalLog | DBEvalSample | DBChatMessage]]],
    progress: Progress,
    tags: list[str] | None,
):
    """Process logs from a job queue and return the DB objects to insert."""
    while True:
        job = job_queue.get(block=True)
        if job is None:  # Sentinel value to stop worker
            return

        log_path, task_id = job
        try:
            progress.update(task_id, status="reading")
            # Count samples
            log_header, sample_ids = read_eval_log_header(log_path)
            progress.update(task_id, total=len(sample_ids))

            # Create DB objects
            to_insert = []
            db_log = DBEvalLog.from_inspect(log_header, tags=tags)
            to_insert.append(db_log)

            # Process each sample
            for sample in iter_inspect_samples_fast(log_path, sample_ids):
                db_sample = DBEvalSample.from_inspect(sample, db_log.db_uuid)
                to_insert.append(db_sample)

                for i, message in enumerate(sample.messages):
                    db_message = DBChatMessage.from_inspect(
                        message, db_sample.db_uuid, db_log.db_uuid, i
                    )
                    to_insert.append(db_message)

                progress.update(task_id, advance=1)

            insert_queue.put((log_path, to_insert), block=True)
            progress.update(task_id, status="inserting")

        except Exception as e:
            progress.update(
                task_id,
                status="error",
                error=str(e),
                total=0,
            )


def ingest_logs(database_uri, path_patterns, workers=4, tags: list[str] | None = None):
    """Ingest logs from files matching path_patterns into the database."""
    console = Console()
    db = EvalDB(database_uri)

    # Find all log files
    log_paths = [
        Path(path)
        for pattern in path_patterns
        for path in glob(pattern, recursive=True)
    ]

    # Setup queues
    job_queue: Queue[tuple[Path, TaskID] | None] = Queue()
    insert_queue: Queue[tuple[Path, List[DBEvalLog | DBEvalSample | DBChatMessage]]] = (
        Queue(maxsize=8)
    )

    # Setup progress tracking
    progress = Progress(
        StatusColumn(),
        TextColumn("{task.description}"),
        # TextColumn("{task.fields[error]}", style="red"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Create tasks for each log file
    task_ids: Dict[Path, TaskID] = {}
    # for log_path in log_paths:
    #     task_ids[log_path] = progress.add_task(
    #         f"{log_path.name[:8]}...{log_path.name[-12:]}",
    #         status="queued",
    #         error=None,
    #     )

    # Stats tracking
    stats = {
        "logs": 0,
        "samples": 0,
        "messages": 0,
        "failed": 0,
    }

    with progress:
        # Check which logs are already in the database and queue up new ones
        log_locations: Dict[Path, str] = {}
        with db.session() as session:
            for log_path in progress.track(
                log_paths, description="Checking existing logs"
            ):
                task_ids[log_path] = progress.add_task(
                    f"{log_path.name[:8]}...{log_path.name[-12:]}",
                    status="queued",
                    error=None,
                    visible=False,
                )

                try:
                    log_header = read_eval_log(str(log_path), header_only=True)
                    log_locations[log_path] = log_header.location
                except Exception as e:
                    console.log(f"Error reading {log_path}: {e}")
                    progress.update(
                        task_ids[log_path],
                        status="error",
                        error=e,
                        total=0,
                        visible=True,
                    )
                    stats["failed"] += 1
                    continue

        existing = session.exec(
            select(DBEvalLog.location).where(
                col(DBEvalLog.location).in_(log_locations.values())
            )
        ).all()

        for log_path, log_location in log_locations.items():
            if log_location in existing:
                console.log(f"Skipping {log_path} - already in database")
                progress.update(task_ids[log_path], status="skipped", visible=False)
            else:
                job_queue.put((log_path, task_ids[log_path]))
                progress.update(task_ids[log_path], status="queued", visible=True)

        if job_queue.empty():
            console.log("No new logs to process.")
            return

        # Add sentinel values to stop workers
        for _ in range(workers):
            job_queue.put(None)

        # Start workers
        load_workers = []
        for _ in range(workers):
            thread = Thread(
                target=read_log_worker,
                args=(job_queue, insert_queue, progress, tags),
            )
            thread.start()
            load_workers.append(thread)

        with db.session() as session:
            # Process results while workers are running
            while any(t.is_alive() for t in load_workers) or not insert_queue.empty():
                try:
                    log_path, result = insert_queue.get(block=True, timeout=0.1)
                    session.add_all(result)
                    session.commit()
                    progress.update(
                        task_ids[log_path],
                        status="inserted",
                        visible=False,
                    )

                    # Update stats
                    stats["logs"] += 1
                    stats["samples"] += sum(
                        1 for r in result if isinstance(r, DBEvalSample)
                    )
                    stats["messages"] += sum(
                        1 for r in result if isinstance(r, DBChatMessage)
                    )
                except Empty:
                    continue
                except Exception as e:
                    console.log(f"Error inserting {log_path}: {e}")
                    progress.update(
                        task_ids[log_path],
                        status="error",
                        error=e,
                        total=0,
                    )
                    stats["failed"] += 1

            # Clean up
            for thread in load_workers:
                thread.join()

            session.commit()

    # Display summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Logs Processed", str(stats["logs"]))
    table.add_row("Samples Inserted", str(stats["samples"]))
    table.add_row("Messages Inserted", str(stats["messages"]))
    table.add_row("Failed Logs", str(stats["failed"]))
    console.print(table)
