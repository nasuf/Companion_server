import asyncio
from types import SimpleNamespace

import pytest

from evals.context_embedding.run_eval import repeat_judgements


def verdict(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish)], usage=None)


def run(responses, repeats=1):
    queue = list(responses)
    calls = []

    async def create():
        calls.append(1)
        return queue.pop(0)

    row = {"judge_repeats": [{"pass": True}]}
    asyncio.run(repeat_judgements(create, row, repeats))
    return row, len(calls)


def test_valid_first_attempt_needs_no_retry():
    row, calls = run([verdict('{"pass": false}')])
    assert calls == 1
    assert row["judge_repeats"][1] == {"pass": False}
    assert "judge_repeat_retries" not in row


def test_truncated_output_is_retried_once_and_kept_as_evidence():
    row, calls = run([verdict('{"pass": tr', "length"), verdict('{"pass": true}')])
    assert calls == 2
    assert row["judge_repeats"][1] == {"pass": True}
    assert row["judge_repeat_retries"] == 1
    assert [(r["repeat"], r["attempt"], r["finish_reason"]) for r in row["judge_repeat_raw"]] == [
        (0, 0, "length"), (0, 1, "stop")]


def test_invalid_pass_field_is_retried():
    row, calls = run([verdict('{"pass": "yes"}'), verdict('{"pass": false}')])
    assert calls == 2 and row["judge_repeats"][1] == {"pass": False}


def test_second_failure_raises_with_both_attempts_recorded():
    row = {"judge_repeats": []}
    queue = [verdict("{bad"), verdict(None)]

    async def create():
        return queue.pop(0)

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(repeat_judgements(create, row, 1))
    assert len(row["judge_repeat_raw"]) == 2
    assert row["judge_repeats"] == []


def test_each_repeat_gets_its_own_retry_budget():
    row, calls = run([verdict("{"), verdict('{"pass": true}'),
                      verdict("{"), verdict('{"pass": false}')], repeats=2)
    assert calls == 4
    assert row["judge_repeat_retries"] == 2
    assert [v["pass"] for v in row["judge_repeats"][1:]] == [True, False]
