from __future__ import annotations
from glob import glob
from pathlib import Path
from multiprocessing import Queue, Process
from typing import List, Optional
from dataclasses import dataclass

from sqlmodel import select
from rich.progress import (
    ProgressColumn,
)
from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from inspect_ai.log import EvalLog, read_eval_log
from inspect_db.models import DBEvalLog
from .db import EvalDB
from tqdm import tqdm


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


@dataclass
class LogReadResult:
    log_path: Path
    log: Optional[EvalLog] = None
    error: Optional[Exception] = None

    def __post_init__(self):
        pass  # No longer needed since we use field(default_factory=list)


def read_log_worker(
    job_queue: Queue[Path | None],
    results_queue: Queue[LogReadResult | None],
) -> None:
    """Process a single log file and return the DB objects to insert."""
    while True:
        print("Getting log path")
        log_path = job_queue.get(block=True)
        if log_path is None:
            job_queue.put(None)
            print("Received sentinel value, exiting")
            break

        try:
            print(f"Reading log: {log_path}")
            # Read full log since it's new
            log = read_eval_log(str(log_path))
            if not isinstance(log, object) or not hasattr(log, "samples"):
                raise ValueError(f"Invalid log format: {log_path}")
            print(f"Read log: {log_path}")
            results_queue.put(LogReadResult(log_path=log_path, log=log, error=None))
        except Exception as e:
            print(f"Error reading log: {log_path}")
            results_queue.put(LogReadResult(log_path=log_path, log=None, error=e))
    results_queue.put(None)
    print("Worker finished")


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

    # Stats tracking
    stats = {
        "logs": 0,
        "samples": 0,
        "messages": 0,
        "failed": 0,
    }

    # First get all existing log locations
    with db.session() as session:
        existing_locations = set(session.exec(select(DBEvalLog.location)).all())

    job_queue: Queue[Path | None] = Queue()
    results_queue: Queue[LogReadResult | None] = Queue()

    for log_path in log_paths:
        if log_path.name in existing_locations:
            continue
        job_queue.put(log_path)

    job_queue.put(None)

    worker_processes = []
    for _ in range(workers):
        worker = Process(target=read_log_worker, args=(job_queue, results_queue))
        worker.start()
        worker_processes.append(worker)

    results: List[LogReadResult] = []
    workers_finished = 0
    while workers_finished < workers:
        result = results_queue.get(block=True)
        if result is None:
            workers_finished += 1
        else:
            results.append(result)

    print("All logs read, inserting to database")

    # Insert all results in a single transaction
    with db.session() as session:
        for result in tqdm(results, desc="Inserting logs"):
            if result.log is not None and not result.error:  # Check if log exists
                db.ingest(result.log, tags=tags, session=session, commit=False)

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
