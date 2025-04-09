from inspect_ai.log import EvalLog
from inspect_db.db import EvalDB
from inspect_db.models import DBEvalLog, MessageRole


def test_insert_log(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test inserting a log with samples and messages."""
    # print("output glarb", json.dumps(sample_log.samples[0].output))
    log_uuid = raw_db.insert_log(sample_eval_log)

    # Verify log was inserted
    with raw_db.session() as session:
        db_log = session.get(DBEvalLog, log_uuid)
        assert db_log is not None

        samples = db_log.samples
        print(samples)
        assert len(samples) == len(sample_eval_log.samples or [])


def test_get_db_log(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test retrieving a log by UUID."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_eval_log)

    # Test getting the log
    db_logs = list(raw_db.get_db_logs(log_uuid=log_uuid))
    assert len(db_logs) == 1
    db_log = db_logs[0]
    assert db_log is not None
    assert db_log.location == sample_eval_log.location


def test_get_db_samples(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test retrieving samples for a log."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_eval_log)

    # Test getting samples
    samples = list(raw_db.get_db_samples(log_uuid=log_uuid))
    assert len(samples) == len(sample_eval_log.samples or [])

    # Verify sample data
    for db_sample, original_sample in zip(samples, sample_eval_log.samples or []):
        assert db_sample.id == str(original_sample.id)
        assert db_sample.epoch == original_sample.epoch
        assert db_sample.input == original_sample.input
        assert db_sample.target == original_sample.target


def test_get_db_sample(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test retrieving a single sample by UUID."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_eval_log)

    # Get all samples to get a sample UUID
    samples = list(raw_db.get_db_samples(log_uuid, None))
    assert len(samples) > 0

    # Test getting a single sample
    sample_uuid = samples[0].db_uuid
    db_sample = list(raw_db.get_db_samples(log_uuid=log_uuid, sample_uuid=sample_uuid))[
        0
    ]
    assert db_sample is not None
    assert db_sample.db_uuid == sample_uuid


def test_get_db_messages(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test retrieving messages for a sample."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_eval_log)

    # Get a sample to get its UUID
    samples = list(raw_db.get_db_samples(log_uuid=log_uuid))
    assert len(samples) > 0
    sample_uuid = samples[0].db_uuid

    # Test getting all messages
    messages = list(raw_db.get_db_messages(sample_uuid=sample_uuid))
    assert len(messages) > 0

    # Test getting messages filtered by role
    assistant_messages = list(
        raw_db.get_db_messages(sample_uuid=sample_uuid, role="assistant")
    )
    assert all(msg.role == MessageRole.ASSISTANT for msg in assistant_messages)

    # Verify message data
    original_sample = sample_eval_log.samples[0] if sample_eval_log.samples else None
    if original_sample:
        for db_msg, original_msg in zip(messages, original_sample.messages):
            assert db_msg.role == MessageRole(original_msg.role)
            assert db_msg.content == original_msg.content
            assert db_msg.index_in_sample == original_sample.messages.index(
                original_msg
            )
