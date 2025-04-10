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
def sample_eval_log_paths() -> list[Path]:
    """Get paths to all sample eval log files."""
    return [
        Path(__file__).parent
        / "sample_logs"
        / "2025-03-28T13-47-14+00-00_gpqa-diamond_4dLhGxA36WnSauR29xehc7.eval",
        Path(__file__).parent
        / "sample_logs"
        / "2025-04-10T09-09-00+01-00_commonsense-qa_BvqVogcx7wZWDpvRYnci6b.eval",
        Path(__file__).parent
        / "sample_logs"
        / "2025-03-28T23-01-38+00-00_agentharm_Nxk5Uxvp8fxJ6ecWsUah43.eval",
    ]


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
def sample_eval_logs(sample_eval_log_paths: list[Path]) -> list[EvalLog]:
    """Load all sample eval logs from the test directory."""
    return [read_eval_log(str(path)) for path in sample_eval_log_paths]


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


@pytest.fixture(
    params=[
        "sqlite",
        "duckdb",
    ]
)
def db_uri(request, db_path: Path) -> str:
    """Create a SQLAlchemy URL for the test database.

    Tests will run once for each database type.
    For file-based databases, uses the temp_dir.
    """
    return f"{request.param}:///{db_path}"


@pytest.fixture
def mock_progress_listener(mocker: MockerFixture):
    """Create a mock progress listener for testing."""
    mock = mocker.Mock(spec=IngestionProgressListener)

    mock.progress.return_value = nullcontext()
    return mock
