from __future__ import annotations
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from multiprocessing import Queue, Process
from typing import Literal

from sqlmodel import select, col
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box
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


def progress_view_worker(
    console: Console,
    update_queue: Queue[StatusUpdate],
    result_queue: Queue[LogResult],
):
    """Display progress updates in a live view."""

    class IngestProgress(Progress):
        def __init__(self):
            self.total_messages = 0
            self.path_tasks = {}
            self.status_counts = {}
            super().__init__(
                TextColumn("{task.description}"),
                TextColumn("({task.fields[status]})", style="dim italic"),
                BarColumn(finished_style="dim"),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            )

        def summary_table(self):
            table = Table(box=box.SIMPLE)
            table.add_column("messages parsed")
            table.add_column("logs inserted")
            table.add_row(
                str(self.total_messages), str(self.status_counts.get("inserted", 0))
            )
            return table
            # return Panel(table, title="Ingestion Stats", expand=False)

        def get_renderables(self):
            yield self.summary_table()
            yield from super().get_renderables()

        def read_from_queue(self, update_queue: Queue[StatusUpdate]):
            while True:
                update = update_queue.get(block=True)
                self.status_counts[update[0]] = self.status_counts.get(update[0], 0) + 1
                if update[0] == "finished":
                    break
                elif update[0] == "pending":
                    path = update[1]
                    self.path_tasks[path] = self.add_task(
                        f"{path.name[:8]}[dim]...[/dim]{path.name[-12:]}",
                        total=None,
                        status="pending",
                    )
                elif update[0] == "sample_read":
                    self.update(self.path_tasks[update[1]], advance=1)
                    self.total_messages += update[2]
                elif update[0] == "samples_counted":
                    self.update(self.path_tasks[update[1]], total=update[2])
                elif update[0] == "error":
                    self.update(self.path_tasks[update[1]], total=0, status="error")
                elif update[0] in ("started", "inserted", "skipped"):
                    self.update(self.path_tasks[update[1]], status=update[0])

    progress = IngestProgress()
    with Live(progress, refresh_per_second=10):
        progress.read_from_queue(update_queue)


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
            try:
                log_header = read_eval_log(str(log_path), header_only=True)
            except Exception as e:
                progress_queue.put(("error", log_path, e))
                continue

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
        target=progress_view_worker,
        args=(console, progress_queue, result_queue),
    )
    progress_process.start()

    with db.session() as session:
        # Process events and results while workers are running
        while any(p.is_alive() for p in load_workers) or not result_queue.empty():
            # Handle results
            log_path, result = result_queue.get(block=True)
            try:
                session.add_all(result)
                session.commit()
                progress_queue.put(("inserted", log_path))
            except Exception as e:
                progress_queue.put(("error", log_path, e))

        # Clean up
        for process in load_workers:
            process.join()

        progress_queue.put(("finished",))
        progress_process.join()
        session.commit()
