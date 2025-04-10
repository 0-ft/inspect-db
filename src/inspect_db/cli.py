import click

from inspect_db.db import EvalDB
from .ingest import RichProgressListener, ingest_logs
from rich.console import Console
from rich.table import Table

console = Console()


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
    progress_listener = None if quiet else RichProgressListener(console=console)
    ingest_logs(database_uri, path_patterns, workers, progress_listener)


@cli.command()
@click.argument("database_uri", type=str)
def stats(database_uri: str) -> None:
    """Show statistics about the database.

    DATABASE_URI: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
    """
    db = EvalDB(database_uri)
    stats = db.stats()

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
