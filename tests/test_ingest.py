from pathlib import Path
from inspect_db.ingest import process_eval_file, worker, ingest_eval_files, get_db_stats
from inspect_db.db import RawEvalDB
from inspect_db.models import DBEvalLog
import queue
from sqlmodel import select, func


def test_process_eval_file_success(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test successful processing of an eval file."""
    db = RawEvalDB(db_uri)
    
    success = process_eval_file(sample_eval_log_path, db, mock_progress_listener)
    
    assert success is True
    mock_progress_listener.on_file_started.assert_called_once_with(sample_eval_log_path)
    mock_progress_listener.on_file_finished.assert_called_once_with(
        "mock_file_id", "success", mocker.ANY
    )
    
    # Verify data was inserted
    with db.session() as session:
        log_count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert log_count == 1

def test_process_eval_file_duplicate(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test handling of duplicate eval files."""
    db = RawEvalDB(db_uri)
    
    # Process file first time
    process_eval_file(sample_eval_log_path, db, mock_progress_listener)
    
    # Process same file again
    success = process_eval_file(sample_eval_log_path, db, mock_progress_listener)
    
    assert success is False
    assert mock_progress_listener.on_file_started.call_count == 2
    assert mock_progress_listener.on_file_finished.call_count == 2
    mock_progress_listener.on_file_finished.assert_called_with(
        "mock_file_id", "skipped", mocker.ANY
    )

def test_process_eval_file_invalid(db_uri: str, temp_dir: Path, mock_progress_listener, mocker):
    """Test handling of invalid eval files."""
    db = RawEvalDB(db_uri)
    
    # Create invalid eval file
    invalid_file = temp_dir / "invalid.eval"
    invalid_file.write_text("invalid content")
    
    success = process_eval_file(invalid_file, db, mock_progress_listener)
    
    assert success is False
    mock_progress_listener.on_file_started.assert_called_once_with(invalid_file)
    mock_progress_listener.on_file_finished.assert_called_once_with(
        "mock_file_id", "error", mocker.ANY
    )

def test_worker(db_uri: str, sample_eval_log_path: Path, mock_progress_listener, mocker):
    """Test the worker function."""
    db = RawEvalDB(db_uri)
    q = queue.Queue()
    
    # Add task to queue
    q.put(sample_eval_log_path)
    
    # Run worker
    worker(q, db, mock_progress_listener)
    
    mock_progress_listener.on_file_started.assert_called_once_with(sample_eval_log_path)
    mock_progress_listener.on_file_finished.assert_called_once_with(
        "mock_file_id", "success", mocker.ANY
    )

def test_get_db_stats(db_uri: str, sample_eval_log_path: Path, mock_progress_listener):
    """Test the database statistics function."""
    # Insert some data
    db = RawEvalDB(db_uri)
    process_eval_file(sample_eval_log_path, db, mock_progress_listener)
    
    # Get stats
    stats = get_db_stats(db_uri)
    
    assert stats["log_count"] == 1
    assert stats["sample_count"] > 0  # Sample count from the actual log
    assert stats["message_count"] > 0  # Message count from the actual log
    assert stats["avg_samples_per_log"] > 0
    assert stats["avg_messages_per_sample"] > 0
    assert "user" in stats["role_distribution"]
    assert "assistant" in stats["role_distribution"] 
    
def test_ingest_eval_file(db_uri: str, sample_eval_log_path: Path, mock_progress_listener):
    """Test ingestion of a single eval file."""
    ingest_eval_files(db_uri, [sample_eval_log_path], progress_listener=mock_progress_listener)
    
    # Verify progress listener was used correctly
    assert mock_progress_listener.on_file_started.call_count == 1
    assert mock_progress_listener.on_file_finished.call_count == 1
    mock_progress_listener.on_ingestion_complete.assert_called_once()
    
    # Verify database contains one record
    db = RawEvalDB(db_uri)
    with db.session() as session:
        log_count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert log_count == 1


def test_ingest_eval_files_with_duplicates(db_uri: str, temp_dir: Path, sample_eval_log_path: Path, mock_progress_listener):
    """Test ingestion with duplicate files."""
    # Ingest same file twice
    ingest_eval_files(db_uri, [sample_eval_log_path, str(sample_eval_log_path)], progress_listener=mock_progress_listener)
    
    # Verify progress listener was used correctly
    assert mock_progress_listener.on_file_started.call_count == 2
    assert mock_progress_listener.on_file_finished.call_count == 2
    mock_progress_listener.on_ingestion_complete.assert_called_once()
    
    # Verify only one record was inserted
    db = RawEvalDB(db_uri)
    with db.session() as session:
        log_count = session.exec(select(func.count()).select_from(DBEvalLog)).one()
        assert log_count == 1
