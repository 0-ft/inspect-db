from collections.abc import Sequence
import itertools
import click
from typing import Literal

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall

from inspect_db.db import EvalDB
from .ingest import ingest_logs
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import rich.box as box
from rich.json import JSON
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

    console.print(db.stats_table())


def create_tool_call_table(tool_calls: Sequence[ToolCall]) -> RenderableType:
    def create_tool_call_panel(tool_call: ToolCall) -> Panel:
        table = Table(show_header=False, box=box.MINIMAL, pad_edge=False)
        for key, value in tool_call.arguments.items():
            table.add_row(key, JSON.from_data(value, indent=2))
            table.add_section()
        title = Text(tool_call.function).append(f" {tool_call.id}", style="dim italic")
        return Panel(table, title=title, border_style="red")

    return Group(*[create_tool_call_panel(tool_call) for tool_call in tool_calls])


@click.pass_context
def create_message_panel(
    ctx: click.Context,
    message: DBChatMessage,
    pattern: str | None = None,
) -> Panel:
    """Create a rich panel for a message.

    Args:
        ctx: Click context containing display options
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
    # Add tool calls if present and enabled
    if (
        ctx.params.get("show_tool_calls", True)
        and isinstance(inspect_message, ChatMessageAssistant)
        and inspect_message.tool_calls
    ):
        content = Group(
            message_text, create_tool_call_table(inspect_message.tool_calls)
        )
    return Panel(content, title=header, border_style="blue")


@click.pass_context
def create_sample_panel(
    ctx: click.Context,
    sample: DBEvalSample,
    message_panels: list[Panel],
) -> Panel:
    """Format a sample and its messages using rich formatting.

    Args:
        ctx: Click context containing display options
        sample: Sample to format
        message_panels: List of message panels to include
    """
    # Create sample panel with messages nested inside
    content_elements = []

    # Create score information if available and enabled
    if ctx.params.get("show_scores", False) and sample.scores:
        score_table = Table(
            "Score",
            "Value",
            "Reason",
            box=box.SIMPLE,
            style="green",
            header_style="green",
        )

        for score_name, score in sample.scores.items():
            score_table.add_row(
                score_name,
                JSON.from_data(score.value, indent=2),
                score.explanation if ctx.params.get("show_score_reasons", True) else "",
            )
        content_elements.append(score_table)

    content_elements.extend(message_panels)
    sample_panel = Panel(
        Group(*content_elements),
        title=Text(f"sample id: {sample.id} | epoch: {sample.epoch}"),
        border_style="green",
    )

    return sample_panel


def create_log_panel(log: DBEvalLog, sample_panels: list[Panel]) -> Panel:
    """Format a log and its samples using rich formatting.

    Args:
        log: Log to format
        sample_panels: List of sample panels to include in the log panel
    """
    log_panel = Panel(
        Group(*sample_panels),
        title=Text(log.eval.task).append(f" {log.eval.created}", style="dim"),
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
@click.option(
    "--has-tool-calls",
    "-t",
    type=click.Choice(["true", "false"]),
    help="Filter messages by whether they have tool calls",
)
@click.option(
    "--show-tool-calls/--hide-tool-calls",
    default=True,
    help="Show or hide tool calls in output",
)
@click.option(
    "--show-scores/--hide-scores",
    default=True,
    help="Show or hide scores in output",
)
@click.option(
    "--show-score-reasons/--hide-score-reasons",
    default=True,
    help="Show or hide score reasons in output",
)
@click.option(
    "--log-task",
    type=str,
    help="Filter messages by log task name",
)
@click.option(
    "--log-task-id",
    type=str,
    help="Filter messages by log task ID",
)
def grep(
    database_uri: str,
    pattern: str | None,
    role: Literal["system", "user", "assistant", "tool"] | None,
    collect_logs: bool,
    json_output: bool,
    has_tool_calls: str | None,
    log_task: str | None,
    log_task_id: str | None,
    show_tool_calls: bool,
    show_scores: bool,
    show_score_reasons: bool,
) -> None:
    """Search through messages in the database.

    DATABASE_URI: SQLAlchemy database URI (e.g. 'sqlite:///eval.db')
    """
    db = EvalDB(database_uri)

    # Convert has_tool_calls string to bool if specified
    has_tool_calls_bool = None if has_tool_calls is None else has_tool_calls == "true"

    # Get matching messages
    messages = db.get_db_messages(
        pattern=pattern,
        role=role,
        has_tool_calls=has_tool_calls_bool,
        log_task=log_task,
        log_task_id=log_task_id,
    )

    # Format output
    if json_output:
        for message in messages:
            print(message.to_inspect().model_dump_json())
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
