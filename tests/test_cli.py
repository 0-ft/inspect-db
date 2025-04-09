from unittest import mock
import pytest
from pathlib import Path
from click.testing import CliRunner
from inspect_db.cli import cli
from inspect_ai.log import EvalLog, write_eval_log
from inspect_db.db import EvalDB
from sqlmodel import select, func
from inspect_db.models import DBEvalLog
from inspect_db.ingest import IngestionProgressListener


def test_ingest_command(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test the ingest command."""
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    result = runner.invoke(cli, ["ingest", db_uri, str(sample_eval_log_path)])
    
    assert result.exit_code == 0
    
    # Verify progress listener was used correctly
    mock_progress_listener.on_file_started.assert_called_once_with(sample_eval_log_path)
    mock_progress_listener.on_file_finished.assert_called_once_with(
        "mock_file_id", "success", mocker.ANY
    )
    mock_progress_listener.on_ingestion_complete.assert_called_once()
    
    # Verify database was created and contains data
    db = EvalDB(db_uri)
    with db.session() as session:
        count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert count == 1

def test_ingest_command_multiple_files(db_uri: str, temp_dir: Path, sample_eval_log: EvalLog, mock_progress_listener, mocker):
    """Test the ingest command with multiple files."""
    # Create multiple eval files
    for i in range(3):
        file_path = temp_dir / f"test_{i}.eval"
        write_eval_log(sample_eval_log, str(file_path))
    
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    result = runner.invoke(cli, ["ingest", db_uri, str(temp_dir / "*.eval")])
    
    assert result.exit_code == 0
    
    # Verify progress listener was used correctly
    assert mock_progress_listener.on_file_started.call_count == 3
    assert mock_progress_listener.on_file_finished.call_count == 3
    mock_progress_listener.on_ingestion_complete.assert_called_once()
    
    # Verify database contains all records
    db = EvalDB(db_uri)
    with db.session() as session:
        count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert count == 3


def test_ingest_command_duplicate_files(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test the ingest command with duplicate files."""
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    
    # First ingestion
    result = runner.invoke(cli, ["ingest", db_uri, str(sample_eval_log_path)])
    assert result.exit_code == 0
        
    # Second ingestion of same file
    result = runner.invoke(cli, ["ingest", db_uri, str(sample_eval_log_path)])
    assert result.exit_code == 0
    
    # Verify progress listener was used correctly for second ingestion
    mock_progress_listener.on_file_started.assert_called_with(sample_eval_log_path)
    mock_progress_listener.on_file_finished.assert_called_with(
        "mock_file_id", "skipped", mocker.ANY
    )
    mock_progress_listener.on_ingestion_complete.assert_called()
    
    # Verify only one record was inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert count == 1

def test_ingest_command_invalid_file(db_uri: str, temp_dir: Path, mock_progress_listener, mocker):
    """Test the ingest command with an invalid file."""
    # Create invalid eval file
    invalid_file = temp_dir / "invalid.eval"
    invalid_file.write_text("invalid content")
    
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    result = runner.invoke(cli, ["ingest", db_uri, str(invalid_file)])
    
    assert result.exit_code == 0
    
    # Verify progress listener was used correctly
    mock_progress_listener.on_file_started.assert_called_once_with(invalid_file)
    mock_progress_listener.on_file_finished.assert_called_once_with(
        "mock_file_id", "error", mocker.ANY
    )
    mock_progress_listener.on_ingestion_complete.assert_called_once()

def test_ingest_command_with_workers(db_uri: str, temp_dir: Path, sample_eval_log: EvalLog, mock_progress_listener, mocker):
    """Test the ingest command with custom number of workers."""
    # Create multiple eval files
    for i in range(5):
        file_path = temp_dir / f"test_{i}.eval"
        write_eval_log(sample_eval_log, str(file_path))
    
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    result = runner.invoke(cli, ["ingest", "--workers", "2", db_uri, str(temp_dir / "*.eval")])
    
    assert result.exit_code == 0
    
    # Verify progress listener was used correctly
    assert mock_progress_listener.on_file_started.call_count == 5
    assert mock_progress_listener.on_file_finished.call_count == 5
    mock_progress_listener.on_ingestion_complete.assert_called_once()
    
    # Verify all records were inserted
    db = EvalDB(db_uri)
    with db.session() as session:
        count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert count == 5

def test_ingest_command_quiet(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test the ingest command with quiet mode."""
    runner = CliRunner()
    
    mocker.patch('inspect_db.cli.RichProgressListener', return_value=mock_progress_listener)
    result = runner.invoke(cli, ["ingest", "--quiet", db_uri, str(sample_eval_log_path)])
    
    assert result.exit_code == 0
    
    # Verify progress listener was not used
    mock_progress_listener.on_file_started.assert_not_called()
    mock_progress_listener.on_file_finished.assert_not_called()
    mock_progress_listener.on_ingestion_complete.assert_not_called()
    
    # Verify database was still updated
    db = EvalDB(db_uri)
    with db.session() as session:
        count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert count == 1

def test_stats_command(db_uri: str, sample_eval_log_path: Path):
    """Test the stats command."""
    # First ingest some data
    runner = CliRunner()
    runner.invoke(cli, ["ingest", db_uri, str(sample_eval_log_path)])
    
    # Then get stats
    result = runner.invoke(cli, ["stats", db_uri])
    
    assert result.exit_code == 0
    assert "Total Logs" in result.output
    assert "Total Samples" in result.output
    assert "Total Messages" in result.output

def test_stats_command_empty_db(db_uri: str):
    """Test the stats command with an empty database."""
    runner = CliRunner()
    result = runner.invoke(cli, ["stats", db_uri])
    
    assert result.exit_code == 0
    assert "Total Logs" in result.output
    assert "0" in result.output  # Should show zero counts
