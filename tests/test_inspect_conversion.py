from inspect_ai.log import EvalLog
import pytest
from inspect_db.db import EvalDB


def dict_diff(d1: dict, d2: dict, path: str = "") -> None:
    for key in set(d1.keys()) | set(d2.keys()):
        assert key in d1, f"{path}.{key} not in d1"
        assert key in d2, f"{path}.{key} not in d2"
        if isinstance(d1[key], float):
            assert d1[key] == pytest.approx(d2[key]), f"{path}.{key} doesn't match"
        elif isinstance(d1[key], dict):
            dict_diff(d1[key], d2[key], f"{path}.{key}")
        else:
            assert d1[key] == d2[key], f"{path}.{key} doesn't match"


def test_dict_diff():
    d1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    d2 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
    dict_diff(d1, d2)

    with pytest.raises(AssertionError):
        dict_diff(d1, {"a": 1, "b": 2, "c": {"d": 3, "e": 5}})
    with pytest.raises(AssertionError):
        dict_diff(d1, {"a": 1, "b": 2, "c": {"d": 3, "f": 4}})
    with pytest.raises(AssertionError):
        dict_diff(d1, {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": 5}})


def test_convert_sample(raw_db: EvalDB, sample_eval_log: EvalLog):
    """Test converting an Inspect EvalSample to DBEvalSample and back"""

    assert sample_eval_log.samples is not None
    assert len(sample_eval_log.samples) > 0

    log_uuid = raw_db.insert_log(sample_eval_log)

    for orig_sample, as_db_sample in zip(
        sample_eval_log.samples, raw_db.get_db_samples(log_uuid=log_uuid)
    ):
        as_inspect_sample = as_db_sample.to_inspect()
        assert as_inspect_sample is not None
        dict_diff(as_inspect_sample.model_dump(), orig_sample.model_dump())
