from collections.abc import Iterator
from pathlib import Path
import re
import zipfile

from inspect_ai.log import EvalSample, read_eval_log, read_eval_log_sample, EvalLog


def get_inspect_log_sample_ids(log_path: Path) -> list[tuple[str, int]]:
    """Get the sample ids and epochs from a log file."""
    pattern = re.compile(r"samples/(.*)_epoch_(\d+).json")
    with zipfile.ZipFile(log_path, "r") as zip_file:
        files = list(zip_file.namelist())
        matches = [pattern.match(file) for file in files]
        return [(match.group(1), int(match.group(2))) for match in matches if match]


def iter_inspect_samples(
    log_path: Path, sample_ids: list[tuple[str, int]]
) -> Iterator[EvalSample]:
    """Iterate over the samples in an inspect log."""
    for id, epoch in sample_ids:
        yield read_eval_log_sample(str(log_path), id, epoch)


def iter_inspect_samples_fast(
    log_path: Path, sample_ids: list[tuple[str, int]]
) -> Iterator[EvalSample]:
    """Iterate over the samples in an inspect log."""
    with zipfile.ZipFile(log_path, "r") as zip_file:
        for id, epoch in sample_ids:
            with zip_file.open(f"samples/{id}_epoch_{epoch}.json", "r") as f:
                yield EvalSample.model_validate_json(f.read())


def read_eval_log_header(log_path: Path) -> tuple[EvalLog, list[tuple[str, int]]]:
    """Read the header of an inspect log and find the sample ids and epochs."""
    log = read_eval_log(str(log_path), header_only=True)
    return log, get_inspect_log_sample_ids(log_path)
