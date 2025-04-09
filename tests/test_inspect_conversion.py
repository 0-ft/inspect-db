from inspect_ai.log import EvalLog
from inspect_db.db import EvalDB


def dict_diff(d1: dict, d2: dict) -> None:
    for key in set(d1.keys()) | set(d2.keys()):
        assert d1[key] == d2[key], f"{key} !="


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
        # dict_diff(as_inspect_sample.model_dump(), orig_sample.model_dump())
        assert as_inspect_sample.model_dump() == orig_sample.model_dump()
