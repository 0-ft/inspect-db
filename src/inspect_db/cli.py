from pathlib import Path
from typing import Any, ContextManager, Literal
import click
from rich.table import Table
from .ingest import IngestionProgressListener, ingest_eval_files, get_db_stats
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()


class RichProgressListener(IngestionProgressListener):
    """Rich-based implementation of IngestionProgressListener."""

    started_count = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    workers = 0

    def __init__(self):
        self.console = Console()
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

    def on_log_loaded(
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


@click.group()
def cli():
    """Inspect DB CLI for managing eval logs."""
    pass


@cli.command()
@click.argument("database_uri", type=str)
@click.argument("path_patterns", type=str, nargs=-1)
@click.option("--workers", "-w", type=int, default=4, help="Number of worker threads")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def ingest(
    database_uri: str, path_patterns: list[str], workers: int, quiet: bool
) -> None:
    """Ingest eval files into the database.

    DATABASE_URI: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
    PATH_PATTERNS: One or more glob patterns matching .eval files
    """
    progress_listener = None if quiet else RichProgressListener()
    ingest_eval_files(database_uri, path_patterns, workers, progress_listener)


@cli.command()
@click.argument("database_uri", type=str)
def stats(database_uri: str) -> None:
    """Show statistics about the database.

    DATABASE_URI: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
    """
    stats = get_db_stats(database_uri)

    # Create and display main stats table
    main_table = Table(title="Database Statistics")
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green")

    main_table.add_row("Total Logs", str(stats["log_count"]))
    main_table.add_row("Total Samples", str(stats["sample_count"]))
    main_table.add_row("Total Messages", str(stats["message_count"]))
    main_table.add_row("Avg Samples per Log", str(stats["avg_samples_per_log"]))
    main_table.add_row("Avg Messages per Sample", str(stats["avg_messages_per_sample"]))

    console.print(main_table)

    # Create and display role distribution table
    role_table = Table(title="Message Role Distribution")
    role_table.add_column("Role", style="cyan")
    role_table.add_column("Count", style="green")

    for role, count in stats["role_distribution"].items():
        role_table.add_row(role, str(count))

    console.print(role_table)
