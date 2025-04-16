from __future__ import annotations
from glob import glob
from pathlib import Path

from sqlmodel import select
from rich.console import Console
from rich.table import Table
from inspect_ai.log import read_eval_log
from inspect_db.models import DBEvalLog
from .db import EvalDB
from tqdm import tqdm


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
        "logs_inserted": 0,
        "samples_inserted": 0,
        "messages_inserted": 0,
        "logs_skipped": 0,
        "logs_failed": 0,
    }

    # First get all existing log locations
    with db.session() as session:
        existing_locations = set(session.exec(select(DBEvalLog.location)).all())

        for log_path in tqdm(log_paths, desc="Processing logs"):
            if log_path.name in existing_locations:
                stats["logs_skipped"] += 1
                tqdm.write(f"Skipping log {log_path} because it already exists")
                continue

            try:
                log = read_eval_log(str(log_path))
                db.ingest(log, tags=tags, session=session, commit=False)
                session.commit()
                stats["logs_inserted"] += 1
                stats["samples_inserted"] += len(log.samples or [])
                stats["messages_inserted"] += sum(
                    len(sample.messages) for sample in log.samples or []
                )
                tqdm.write(f"Ingested log {log_path}")
            except Exception as e:
                stats["logs_failed"] += 1
                tqdm.write(f"Failed to ingest log {log_path}: {e}")

    # Display summary table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Logs Processed", str(stats["logs_inserted"]))
    table.add_row("Logs Skipped", str(stats["logs_skipped"]))
    table.add_row("Logs Failed", str(stats["logs_failed"]))
    table.add_row("Samples Inserted", str(stats["samples_inserted"]))
    table.add_row("Messages Inserted", str(stats["messages_inserted"]))
    console.print(table)
