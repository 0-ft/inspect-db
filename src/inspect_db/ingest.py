from dataclasses import dataclass
from glob import glob
from pathlib import Path
import queue
from multiprocessing import Queue, Process
from typing import Literal
from inspect_ai.log import read_eval_log
from pydantic import BaseModel
from sqlmodel import select, col
import threading
import time

from inspect_db.models import DBChatMessage, DBEvalLog, DBEvalSample
from inspect_db.util import iter_inspect_samples_fast, read_eval_log_header
from .db import EvalDB
from rich.progress import (
    Progress,
    TaskID,
    TimeRemainingColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.console import Console


@dataclass
class InsertJob:
    log_path: Path
    log: DBEvalLog
    samples: list[DBEvalSample]
    messages: list[DBChatMessage]


class IngestionProgress(Progress):
    class LogStartedEvent(BaseModel):
        type: Literal["log_started"] = "log_started"
        log_path: Path

    class LogSamplesCountedEvent(BaseModel):
        type: Literal["log_samples_counted"] = "log_samples_counted"
        log_path: Path
        samples_count: int

    class LogSampleReadEvent(BaseModel):
        type: Literal["log_sample_read"] = "log_sample_read"
        log_path: Path
        messages_count: int

    class LogCompletedEvent(BaseModel):
        type: Literal["log_completed"] = "log_completed"
        log_path: Path
        status: Literal["skipped", "inserted", "error"]
        samples_count: int
        messages_count: int

    IngestionEvent = (
        LogStartedEvent
        | LogSamplesCountedEvent
        | LogSampleReadEvent
        | LogCompletedEvent
    )

    logs_skipped: int = 0
    logs_inserted: int = 0
    logs_errors: int = 0
    samples_inserted: int = 0
    messages_inserted: int = 0
    log_tasks: dict[Path, TaskID] = {}
    event_queue: Queue = Queue()
    _processing_thread: threading.Thread | None = None
    _stop_processing: threading.Event = threading.Event()

    def __init__(self, console: Console):
        super().__init__(
            TextColumn("{task.description}{task.fields[note]}"),
            BarColumn(finished_style="dim"),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def stats(self) -> dict[str, int]:
        return {
            "logs_skipped": self.logs_skipped,
            "logs_inserted": self.logs_inserted,
            "logs_errors": self.logs_errors,
            "samples_inserted": self.samples_inserted,
            "messages_inserted": self.messages_inserted,
        }

    def process_events(self):
        while not self.event_queue.empty():
            event: IngestionProgress.IngestionEvent = self.event_queue.get_nowait()
            if event.type == "log_started":
                self.log_tasks[event.log_path] = self.add_task(
                    f"{event.log_path.name[:10]}[dim]...[/dim]{event.log_path.name[-13:]}",
                    total=None,
                    note="",
                )
            elif event.type == "log_samples_counted":
                self.update(self.log_tasks[event.log_path], total=event.samples_count)
            elif event.type == "log_sample_read":
                self.update(self.log_tasks[event.log_path], advance=1)
            elif event.type == "log_completed":
                if event.status == "skipped":
                    self.update(
                        self.log_tasks[event.log_path],
                        total=0,
                        completed=0,
                        note=" (skipped)",
                    )
                    self.logs_skipped += 1
                elif event.status == "inserted":
                    self.logs_inserted += 1
                    self.samples_inserted += event.samples_count
                    self.messages_inserted += event.messages_count
                elif event.status == "error":
                    self.logs_errors += 1

    def _process_queue_loop(self):
        while not self._stop_processing.is_set():
            self.process_events()
            time.sleep(0.01)  # Small sleep to prevent busy waiting

    def start(self):
        """Start the background thread to process events"""
        super().start()
        self._stop_processing.clear()
        self._processing_thread = threading.Thread(target=self._process_queue_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()

    def stop(self):
        """Stop the background thread and process any remaining events"""
        super().stop()
        if self._processing_thread is not None:
            self._stop_processing.set()
            self._processing_thread.join()
            # Process any remaining events
            self.process_events()

    def stats_table(self) -> Table:
        table = Table(title="Ingestion Stats")
        table.add_column("Stat", justify="right")
        table.add_column("Value", justify="left")
        for key, value in self.stats().items():
            table.add_row(key, str(value))
        return table


def read_logs_worker(
    log_queue: Queue,
    insert_queue: Queue,
    progress_queue: Queue,
) -> None:
    while True:
        try:
            log_path = log_queue.get_nowait()
        except queue.Empty:
            break

        log_header, sample_ids = read_eval_log_header(log_path)
        progress_queue.put(
            IngestionProgress.LogSamplesCountedEvent(
                log_path=log_path, samples_count=len(sample_ids)
            )
        )
        db_log = DBEvalLog.from_inspect(log_header)
        db_samples = []
        db_messages = []
        for sample in iter_inspect_samples_fast(log_path, sample_ids):
            db_sample = DBEvalSample.from_inspect(sample, db_log.db_uuid)
            db_samples.append(db_sample)
            for message in sample.messages:
                db_message = DBChatMessage.from_inspect(
                    message, db_sample.db_uuid, db_log.db_uuid, len(db_samples)
                )
                db_messages.append(db_message)
            progress_queue.put(
                IngestionProgress.LogSampleReadEvent(
                    log_path=log_path, messages_count=len(db_messages)
                )
            )
        insert_queue.put(InsertJob(log_path, db_log, db_samples, db_messages))


def ingest_logs(
    database_uri: str,
    path_patterns: list[str],
    workers: int,
) -> None:
    db = EvalDB(database_uri)
    log_paths = [
        Path(path)
        for pattern in path_patterns
        for path in glob(pattern, recursive=True)
    ]

    # Use multiprocessing.Queue for inter-process communication
    log_queue = Queue()
    insert_queue = Queue()

    console = Console()
    progress = IngestionProgress(console=console)

    progress.start()

    # First pass: check existing logs and queue tasks
    with db.session() as session:
        for log_path in log_paths:
            progress.event_queue.put(
                IngestionProgress.LogStartedEvent(log_path=log_path)
            )
            log_header = read_eval_log(str(log_path), header_only=True)
            query = select(DBEvalLog).where(
                col(DBEvalLog.location) == log_header.location
            )
            if session.exec(query).first():
                progress.event_queue.put(
                    IngestionProgress.LogCompletedEvent(
                        log_path=log_path,
                        status="skipped",
                        samples_count=0,
                        messages_count=0,
                    )
                )
            else:
                log_queue.put(log_path)

    # Start worker processes
    worker_processes: list[Process] = []
    for _ in range(workers):
        worker = Process(
            target=read_logs_worker,
            args=(log_queue, insert_queue, progress.event_queue),
        )
        worker.start()
        worker_processes.append(worker)

    # Process insert queue while workers are running
    while True:
        # Handle insert jobs (non-blocking)
        try:
            job: InsertJob = insert_queue.get_nowait()
            try:
                with db.session() as session:
                    session.add(job.log)
                    session.add_all(job.samples)
                    session.add_all(job.messages)
                    session.commit()
                    console.log(
                        f"Inserted {job.log_path}", highlight=False, style="dim"
                    )
                    progress.event_queue.put(
                        IngestionProgress.LogCompletedEvent(
                            log_path=job.log_path,
                            status="inserted",
                            samples_count=len(job.samples),
                            messages_count=len(job.messages),
                        )
                    )
            except Exception as e:
                console.log(f"Error inserting {job.log_path}: {e}")
                progress.event_queue.put(
                    IngestionProgress.LogCompletedEvent(
                        log_path=job.log_path,
                        status="error",
                        samples_count=len(job.samples),
                        messages_count=len(job.messages),
                    )
                )
        except queue.Empty:
            pass

        if (
            not any(worker.is_alive() for worker in worker_processes)
            and log_queue.empty()
            and insert_queue.empty()
        ):
            break

    # Stop the progress processing thread and clean up
    progress.stop()

    # Clean up worker processes
    for worker in worker_processes:
        worker.join()

    # Print final stats
    console.print(progress.stats_table())
