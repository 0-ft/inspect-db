from contextlib import nullcontext
import pytest
from pathlib import Path
import tempfile
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from pytest_mock import MockerFixture

from inspect_db.db import EvalDB
from inspect_db.ingest import IngestionProgressListener


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_eval_log_path() -> Path:
    """Get the path to a sample eval log file."""
    return (
        Path(__file__).parent
        / "sample_logs"
        / "2025-03-28T23-01-38+00-00_agentharm_Nxk5Uxvp8fxJ6ecWsUah43.eval"
    )


@pytest.fixture
def sample_eval_log(sample_eval_log_path: Path) -> EvalLog:
    """Load a sample eval log from the test directory."""
    return read_eval_log(str(sample_eval_log_path))


@pytest.fixture
def raw_db(db_uri: str):
    """Create a temporary database for testing."""
    return EvalDB(db_uri)


@pytest.fixture
def sample_eval_sample(sample_eval_log: EvalLog) -> EvalSample:
    """Get the first sample from the sample eval log."""
    assert sample_eval_log.samples is not None
    assert len(sample_eval_log.samples) > 0
    return sample_eval_log.samples[0]


@pytest.fixture
def db_path(temp_dir: Path) -> Path:
    """Create a path for the test database."""
    return temp_dir / "test.db"


@pytest.fixture
def db_uri(db_path: Path) -> str:
    """Create a SQLAlchemy URL for the test database.

    Note: For testing purposes we use SQLite, but the library supports any SQLAlchemy-compatible database.
    In production, you should use a proper database like PostgreSQL.
    """
    return f"sqlite:///{db_path}"


@pytest.fixture
def mock_progress_listener(mocker: MockerFixture):
    """Create a mock progress listener for testing."""
    mock = mocker.Mock(spec=IngestionProgressListener)

    mock.on_file_started.return_value = (
        "mock_file_id"  # Return a consistent ID for testing
    )
    mock.progress.return_value = nullcontext()
    return mock
