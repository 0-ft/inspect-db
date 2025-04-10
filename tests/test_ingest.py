from inspect_ai.log import read_eval_log
from pathlib import Path
from inspect_db.ingest import ingest_logs
from inspect_db.db import EvalDB
from inspect_db.models import DBChatMessage, DBEvalLog, DBEvalSample
from sqlmodel import select


def test_ingest_single_file(
    sample_eval_log_path: Path,
    db_uri: str,
    mock_progress_listener,
):
    """Test ingesting a single eval file."""
    # Load log for reference
    log = read_eval_log(str(sample_eval_log_path))
    n_samples = len(log.samples or [])
    n_messages = sum(len(sample.messages or []) for sample in log.samples or [])

    # Ingest the file
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(sample_eval_log_path)],
        workers=1,
        progress_listener=mock_progress_listener,
    )

    # Verify progress listener was called correctly
    mock_progress_listener.on_ingestion_started.assert_called_once_with(1)
    mock_progress_listener.on_log_started.assert_called_once_with(sample_eval_log_path)
    mock_progress_listener.on_log_completed.assert_called_once()
    mock_progress_listener.on_ingestion_complete.assert_called_once()

    # Verify the log was inserted into the database
    db = EvalDB(db_uri)
    with db.session() as session:
        logs = session.exec(select(DBEvalLog)).all()
        assert len(logs) == 1
        assert logs[0].location == str(sample_eval_log_path)
        samples = session.exec(select(DBEvalSample)).all()
        assert len(samples) == n_samples
        messages = session.exec(select(DBChatMessage)).all()
        assert len(messages) == n_messages


def test_ingest_parallel(
    sample_eval_log_paths: list[Path],
    db_uri: str,
):
    """Test parallel ingestion with multiple workers."""
    # Ingest with multiple workers
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(path) for path in sample_eval_log_paths],
        workers=2,
    )

    # Verify all logs were inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        logs = session.exec(select(DBEvalLog)).all()
        assert len(logs) == 3


def test_ingest_duplicate(
    sample_eval_log_path: Path,
    db_uri: str,
    mock_progress_listener,
):
    """Test handling of duplicate log files."""
    # Ingest the same file twice
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(sample_eval_log_path), str(sample_eval_log_path)],
        workers=1,
        progress_listener=mock_progress_listener,
    )

    # Verify progress listener was called correctly for duplicate
    assert mock_progress_listener.on_log_completed.call_count == 2
    duplicate_call = mock_progress_listener.on_log_completed.call_args_list[1][0]
    assert "skipped" in duplicate_call
    assert "already exists" in duplicate_call[2]

    # Verify only one log was inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        logs = session.exec(select(DBEvalLog)).all()
        assert len(logs) == 1


def test_ingest_invalid_file(
    temp_dir: Path,
    db_uri: str,
    mock_progress_listener,
):
    """Test handling of invalid eval files."""
    # Create an invalid eval file
    invalid_path = temp_dir / "invalid.eval"
    invalid_path.write_text("not a valid eval file")

    # Ingest the invalid file
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(invalid_path)],
        workers=1,
        progress_listener=mock_progress_listener,
    )

    # Verify error was reported
    mock_progress_listener.on_log_completed.assert_called_once()
    error_call = mock_progress_listener.on_log_completed.call_args
    assert "error" in error_call[0][1]

    # Verify nothing was inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        logs = session.exec(select(DBEvalLog)).all()
        assert len(logs) == 0


def test_ingest_no_files(
    db_uri: str,
    mock_progress_listener,
):
    """Test handling when no files are found."""
    # Try to ingest non-existent files
    ingest_logs(
        database_uri=db_uri,
        path_patterns=["nonexistent/*.eval"],
        workers=1,
        progress_listener=mock_progress_listener,
    )

    # Verify no progress events were called
    mock_progress_listener.on_ingestion_started.assert_not_called()
    mock_progress_listener.on_log_started.assert_not_called()
    mock_progress_listener.on_log_completed.assert_not_called()
    mock_progress_listener.on_ingestion_complete.assert_not_called()


def test_ingest_null_progress_listener(
    temp_dir: Path,
    sample_eval_log_path: Path,
    db_uri: str,
):
    """Test that ingestion works with null progress listener."""
    # Ingest with null progress listener
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(sample_eval_log_path)],
        workers=1,
        progress_listener=None,
    )

    # Verify the log was inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        logs = session.exec(select(DBEvalLog)).all()
        assert len(logs) == 1
