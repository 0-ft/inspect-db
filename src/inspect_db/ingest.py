from __future__ import annotations
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from multiprocessing import Queue, Process
import queue
import time
from typing import Literal

from sqlmodel import select, col
from rich.progress import Progress
from rich.console import Console
from rich.table import Table

from inspect_ai.log import read_eval_log
from inspect_db.models import DBChatMessage, DBEvalLog, DBEvalSample
from inspect_db.util import iter_inspect_samples_fast, read_eval_log_header
from .db import EvalDB

type StatusUpdate = (
    tuple[Literal["samples_counted"], Path, int]
    | tuple[Literal["sample_read"], Path, int]
    | tuple[Literal["error"], Path, Exception]
    | tuple[Literal["pending", "started", "inserted", "skipped"], Path]
    | tuple[Literal["finished"]]
)

type LogResult = tuple[Path, Sequence[DBEvalLog | DBEvalSample | DBChatMessage]]


def read_log_worker(
    job_queue: Queue[Path | None],
    result_queue: Queue[LogResult],
    update_queue: Queue[StatusUpdate],
):
    """Process logs from a job queue and return the DB objects to insert."""
    while True:
        # Get next job from queue
        log_path = job_queue.get(block=True)
        if log_path is None:  # Sentinel value to stop worker
            break

        try:
            update_queue.put(("started", log_path))
            # Count samples
            log_header, sample_ids = read_eval_log_header(log_path)
            update_queue.put(("samples_counted", log_path, len(sample_ids)))

            # Create DB objects
            to_insert = []
            db_log = DBEvalLog.from_inspect(log_header)
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

                update_queue.put(("sample_read", log_path, len(sample.messages)))

            result_queue.put((log_path, to_insert))

        except Exception as e:
            update_queue.put(("error", log_path, e))
            raise e


def progress_view_worker(update_queue: Queue[StatusUpdate], console: Console):
    """Display progress updates in a live view."""
    path_tasks = {}
    progress = Progress(console=console)
    total_messages = 0
    with progress:
        while True:
            update = update_queue.get(block=True)
            if update[0] == "finished":
                break
            elif update[0] == "pending":
                path_tasks[update[1]] = progress.add_task(
                    update[1].name, total=None, status="pending"
                )
            elif update[0] == "sample_read":
                progress.update(path_tasks[update[1]], advance=1)
                total_messages += update[2]
            elif update[0] == "samples_counted":
                progress.update(path_tasks[update[1]], total=update[2])
            elif update[0] in ("started", "inserted", "skipped", "error"):
                progress.update(path_tasks[update[1]], status=update[0])

    table = Table(title="Ingestion Stats")
    table.add_column("Stat", justify="right")
    table.add_column("Value", justify="left")
    stats = {
        "total_logs": 0,
        "logs_ingested": 0,
        "logs_skipped": 0,
        "logs_errors": 0,
        "samples_ingested": 0,
        "messages_ingested": total_messages,
    }
    for task in progress.tasks:
        if task.fields["status"] == "inserted":
            stats["logs_ingested"] += 1
            stats["samples_ingested"] += int(task.total or 0)
        elif task.fields["status"] == "skipped":
            stats["logs_skipped"] += 1
        elif task.fields["status"] == "error":
            stats["logs_errors"] += 1

    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


def ingest_logs(database_uri, path_patterns, workers=4):
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
    job_queue: Queue[Path | None] = Queue()
    result_queue: Queue[LogResult] = Queue()
    progress_queue: Queue[StatusUpdate] = Queue()
    # Check which logs are already in the database
    with db.session() as session:
        for log_path in log_paths:
            progress_queue.put(("pending", log_path))
            log_header = read_eval_log(str(log_path), header_only=True)
            query = select(DBEvalLog).where(
                col(DBEvalLog.location) == log_header.location
            )

            if session.exec(query).first():
                progress_queue.put(("skipped", log_path))
            else:
                job_queue.put(log_path)

    if job_queue.empty():
        console.print("No new logs to process.")
        return

    # Add sentinel values to stop workers
    for _ in range(workers):
        job_queue.put(None)

    # Start workers
    load_workers = []
    for _ in range(workers):
        process = Process(
            target=read_log_worker, args=(job_queue, result_queue, progress_queue)
        )
        process.start()
        load_workers.append(process)

    progress_process = Process(
        target=progress_view_worker, args=(progress_queue, console)
    )
    progress_process.start()

    with db.session() as session:
        # Process events and results while workers are running
        while any(p.is_alive() for p in load_workers) or not result_queue.empty():
            # Handle results
            try:
                log_path, result = result_queue.get_nowait()
                try:
                    session.add_all(result)
                    session.commit()
                    progress_queue.put(("inserted", log_path))
                except Exception as e:
                    progress_queue.put(("error", log_path, e))
            except queue.Empty:
                time.sleep(0.1)

        # Clean up
        for process in load_workers:
            process.join()

        progress_queue.put(("finished",))
        progress_process.join()
        session.commit()
