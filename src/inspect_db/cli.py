import itertools
import click
import json
from typing import Literal

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

from inspect_db.db import EvalDB
from .ingest import ingest_logs
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from .models import DBChatMessage, DBEvalLog, DBEvalSample

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
    ingest_logs(database_uri, path_patterns, workers)


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


def format_message_json(message: DBChatMessage) -> str:
    """Format a message as a JSON string.

    Args:
        message: Message to format

    Returns:
        JSON string representation of the message
    """
    return message.to_inspect().model_dump_json()


def create_message_panel(message: DBChatMessage, pattern: str | None = None) -> Panel:
    """Create a rich panel for a message.

    Args:
        message: Message to create panel for
        pattern: Optional search string to highlight

    Returns:
        Rich Panel containing the message
    """
    inspect_message = message.to_inspect()

    # Create message header
    header_parts: list[Text] = []
    header_parts.append(Text(f"#{message.index_in_sample}", style="cyan"))
    header_parts.append(Text(f"{inspect_message.role}", style="bold"))
    if inspect_message.source:
        header_parts.append(Text(f"[source={inspect_message.source}]", style="dim"))
    if isinstance(inspect_message, ChatMessageAssistant) and inspect_message.model:
        header_parts.append(Text(f"[model={inspect_message.model}]", style="dim"))

    if isinstance(inspect_message, ChatMessageTool):
        if inspect_message.tool_call_id:
            header_parts.append(
                Text(f"[tool_call_id={inspect_message.tool_call_id}]", style="dim")
            )
        if inspect_message.function:
            header_parts.append(
                Text(f"[function={inspect_message.function}]", style="dim")
            )
        if inspect_message.error:
            header_parts.append(Text(f"[error={inspect_message.error}]", style="red"))

    header = Text(" ").join(header_parts)
    # Create message content with highlighting
    message_text = Text(inspect_message.text)
    if pattern:
        message_text.highlight_regex(pattern, style="red bold")

    content = Group(message_text)
    # Add tool calls if present
    if isinstance(inspect_message, ChatMessageAssistant) and inspect_message.tool_calls:
        table = Table(title="Tool Calls")
        table.add_column("ID", style="cyan")
        table.add_column("Function", style="cyan")
        table.add_column("Arguments", style="dim")
        for tool_call in inspect_message.tool_calls:
            table.add_row(
                tool_call.id, tool_call.function, json.dumps(tool_call.arguments)
            )

        content = Group(message_text, table)
    return Panel(content, title=header, border_style="blue")


def create_sample_panel(sample: DBEvalSample, message_panels: list[Panel]) -> Panel:
    """Format a sample and its messages using rich formatting.

    Args:
        sample_messages: List of messages in the sample
        pattern: Optional search string to highlight
    """

    # Create sample panel with messages nested inside

    sample_panel = Panel(
        Group(*message_panels),
        title=Text(f"sample id: {sample.id} | epoch: {sample.epoch}"),
        border_style="green",
    )

    # Print the nested structure
    return sample_panel


def create_log_panel(log: DBEvalLog, sample_panels: list[Panel]) -> Panel:
    """Format a log and its samples using rich formatting.

    Args:
        log: Log to format
        sample_panels: List of sample panels to include in the log panel
    """
    log_panel = Panel(
        Group(*sample_panels),
        title=Text(f"{log.location} | inserted: {log.inserted}"),
        border_style="yellow",
    )
    return log_panel


@cli.command()
@click.argument("database_uri", type=str)
@click.option(
    "--pattern", "-p", type=str, help="Search string to look for in message content"
)
@click.option(
    "--role",
    "-r",
    type=click.Choice(["system", "user", "assistant", "tool"]),
    help="Filter messages by role",
)
@click.option("--collect-logs", "-l", is_flag=True, help="Collect messages by log")
@click.option("--json-output", "-j", is_flag=True, help="Output results as JSON lines")
def grep(
    database_uri: str,
    pattern: str | None,
    role: Literal["system", "user", "assistant", "tool"] | None,
    collect_logs: bool,
    json_output: bool,
) -> None:
    """Search through messages in the database.

    DATABASE_URI: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
    """
    db = EvalDB(database_uri)

    # Get matching messages
    messages = db.get_db_messages(pattern=pattern, role=role)

    # Format output
    if json_output:
        for message in messages:
            print(format_message_json(message))
    else:
        if collect_logs:
            by_log = itertools.groupby(messages, key=lambda x: x.sample.log)
            for log, messages in by_log:
                sample_panels = [
                    create_sample_panel(
                        sample,
                        [
                            create_message_panel(message, pattern)
                            for message in messages
                        ],
                    )
                    for sample, messages in itertools.groupby(
                        messages, key=lambda x: x.sample
                    )
                ]
                log_panel = create_log_panel(log, sample_panels)
                console.print(log_panel)
        else:
            by_sample = itertools.groupby(
                messages, key=lambda x: (x.sample.log, x.sample)
            )
            for (log, sample), messages in by_sample:
                sample_panel = create_sample_panel(
                    sample,
                    [create_message_panel(message, pattern) for message in messages],
                )
                log_panel = create_log_panel(log, [sample_panel])
                console.print(log_panel)
