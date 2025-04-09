from inspect_ai.log import EvalLog, read_eval_log
import pytest
from pathlib import Path
from inspect_db.db import RawEvalDB
from inspect_db.models import DBEvalLog, DBEvalSample, DBChatMessage, MessageRole

@pytest.fixture
def raw_db(db_uri: str):
    """Create a temporary database for testing."""
    return RawEvalDB(db_uri)

@pytest.fixture
def sample_log():
    """Load sample log data from file."""
    log_path = Path(__file__).parent / "sample_logs" / "2025-03-28T23-01-38+00-00_agentharm_Nxk5Uxvp8fxJ6ecWsUah43.eval"
    return read_eval_log(str(log_path))

def test_insert_log(raw_db: RawEvalDB, sample_log: EvalLog):
    """Test inserting a log with samples and messages."""
    log_uuid = raw_db.insert_log(sample_log)
    
    # Verify log was inserted
    with raw_db.session() as session:
        db_log = session.get(DBEvalLog, log_uuid)
        assert db_log is not None
        
        samples = db_log.samples
        print(samples)
        assert len(samples) == len(sample_log.samples or [])

def test_get_db_log(raw_db: RawEvalDB, sample_log: EvalLog):
    """Test retrieving a log by UUID."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_log)
    
    # Test getting the log
    db_log = raw_db.get_db_log(log_uuid)
    assert db_log is not None
    assert db_log.location == sample_log.location

def test_get_db_samples(raw_db: RawEvalDB, sample_log: EvalLog):
    """Test retrieving samples for a log."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_log)
    
    # Test getting samples
    samples = raw_db.get_db_samples(log_uuid)
    assert len(samples) == len(sample_log.samples or [])
    
    # Verify sample data
    for db_sample, original_sample in zip(samples, sample_log.samples or []):
        assert db_sample.id == str(original_sample.id)
        assert db_sample.epoch == original_sample.epoch
        assert db_sample.input == original_sample.input
        assert db_sample.target == original_sample.target

def test_get_db_sample(raw_db: RawEvalDB, sample_log: EvalLog):
    """Test retrieving a single sample by UUID."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_log)
    
    # Get all samples to get a sample UUID
    samples = raw_db.get_db_samples(log_uuid)
    assert len(samples) > 0
    
    # Test getting a single sample
    sample_uuid = samples[0].uuid
    db_sample = raw_db.get_db_sample(sample_uuid)
    assert db_sample is not None
    assert db_sample.uuid == sample_uuid

def test_get_db_messages(raw_db: RawEvalDB, sample_log: EvalLog):
    """Test retrieving messages for a sample."""
    # First insert a log
    log_uuid = raw_db.insert_log(sample_log)
    
    # Get a sample to get its UUID
    samples = raw_db.get_db_samples(log_uuid)
    assert len(samples) > 0
    sample_uuid = samples[0].uuid
    
    # Test getting all messages
    messages = raw_db.get_db_messages(sample_uuid)
    assert len(messages) > 0
    
    # Test getting messages filtered by role
    assistant_messages = raw_db.get_db_messages(sample_uuid, role="assistant")
    assert all(msg.role == MessageRole.ASSISTANT for msg in assistant_messages)
    
    # Verify message data
    original_sample = sample_log.samples[0] if sample_log.samples else None
    if original_sample:
        for db_msg, original_msg in zip(messages, original_sample.messages):
            assert db_msg.role == MessageRole(original_msg.role)
            assert db_msg.content == original_msg.content
            assert db_msg.index_in_sample == original_sample.messages.index(original_msg)
