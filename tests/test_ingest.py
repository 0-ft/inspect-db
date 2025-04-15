from pathlib import Path
from inspect_db.ingest import (
    ingest_logs,
)
from inspect_db.db import EvalDB
from inspect_ai.log import read_eval_log


def test_ingest_logs_skip_existing(
    db_uri: str, sample_eval_log_paths: list[Path], temp_dir: Path
):
    """Test that ingest_logs skips logs that already exist in the database"""
    raw_db = EvalDB(db_uri)
    # First ingest a log
    with raw_db.session() as session:
        log = read_eval_log(str(sample_eval_log_paths[0]))
        raw_db.ingest(log, session=session)
        session.commit()

    # Try to ingest the same log again
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(sample_eval_log_paths[0])],
        workers=1,
    )

    # Verify the log was only inserted once
    with raw_db.session() as session:
        logs = list(raw_db.get_db_logs(session=session))
        assert len(logs) == 1


def test_ingest_logs_multiple_workers(db_uri: str, sample_eval_log_paths: list[Path]):
    """Test that ingest_logs works with multiple workers"""
    # Ingest logs with multiple workers
    ingest_logs(
        database_uri=db_uri,
        path_patterns=[str(path) for path in sample_eval_log_paths],
        workers=2,
    )

    # Verify all logs were inserted
    raw_db = EvalDB(db_uri)
    with raw_db.session() as session:
        logs = list(raw_db.get_db_logs(session=session))
        assert len(logs) == len(sample_eval_log_paths)

        # Verify samples and messages were inserted
        for log in logs:
            log_uuid = log.db_uuid
            samples = list(raw_db.get_samples(log_uuid=log_uuid))
            assert len(samples) > 0
            for locator, sample in samples:
                assert len(sample.messages) > 0
