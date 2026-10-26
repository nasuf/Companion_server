from copy import deepcopy

import pytest

from evals.context_embedding.merge_reports import merge
from evals.context_embedding.test_compare import state_reports


def chunks():
    report, _ = state_reports(36)
    report.update(schema_version=1, state_repeats=9, gate_policy={"v": 1}, layout_variant="B",
                  status="baseline_recorded", started_at="2026-10-26T01:00:00Z",
                  ended_at="2026-10-26T01:10:00Z")
    left, right = deepcopy(report), deepcopy(report)
    left["runs"], right["runs"] = report["runs"][:18], report["runs"][18:]
    right.update(started_at="2026-10-26T01:20:00Z", ended_at="2026-10-26T01:30:00Z")
    return left, right


def test_merge_keeps_every_row_and_spans_both_chunks():
    left, right = chunks()
    merged = merge([left, right])
    assert merged["completed_turns"] == 36 and merged["case_count"] == 36
    assert merged["started_at"].endswith("01:00:00Z") and merged["ended_at"].endswith("01:30:00Z")
    assert merged["status"] == "baseline_recorded" and len(merged["chunks"]) == 2


@pytest.mark.parametrize("field", ["prompt_snapshot_sha256", "judge_protocol", "layout_variant", "gate_policy"])
def test_chunks_with_different_protocols_are_refused(field):
    left, right = chunks()
    right[field] = "changed"
    with pytest.raises(ValueError, match=field):
        merge([left, right])


def test_duplicate_rows_and_incomplete_chunks_are_refused():
    left, right = chunks()
    with pytest.raises(ValueError, match="Duplicate"):
        merge([left, deepcopy(left)])
    right["status"] = "failed"
    with pytest.raises(ValueError, match="Incomplete"):
        merge([left, right])


def test_unjudged_chunk_marks_the_merged_report():
    left, right = chunks()
    right["status"] = "judge_incomplete"
    right["runs"][0]["judge"] = {"pass": None}
    merged = merge([left, right])
    assert merged["status"] == "judge_incomplete" and merged["unjudged_turns"] == 1


def failing(left, index=0):
    call = left["runs"][index]["calls"][0]
    call.update(error_type="ReadTimeout", measurement={"status": "request_failed"})
    return left["runs"][index]["case"], left["runs"][index]["sample"]


def rerun_of(left, case, sample=0):
    rerun = deepcopy(left)
    rerun.update(state_repeats=1, repeats=1, started_at="2026-10-26T02:00:00Z")
    rerun["runs"] = [deepcopy(r) for r in chunks()[0]["runs"] if r["case"] == case]
    for r in rerun["runs"]:
        r["sample"] = 0
    return rerun


def test_failed_call_without_rerun_stays_unqualified():
    left, right = chunks()
    failing(left)
    merged = merge([left, right])
    assert merged["status"] == "measurement_not_qualified" and merged["replaced_replays"] == []


def test_one_rerun_replaces_the_failed_replay_and_is_recorded():
    left, right = chunks()
    case, sample = failing(left)
    merged = merge([left, right], [(rerun_of(left, case), sample)])
    assert merged["status"] == "baseline_recorded" and merged["completed_turns"] == 36
    assert [(r["case"], r["sample"]) for r in merged["runs"] if r.get("rerun_of_failed_replay")] == [(case, sample)]
    assert merged["replaced_replays"][0]["failed_calls"][0]["error_type"] == "ReadTimeout"


def test_reruns_must_target_a_failed_replay_and_succeed():
    left, right = chunks()
    with pytest.raises(ValueError, match="without a failed call"):
        merge([left, right], [(rerun_of(left, left["runs"][1]["case"]), 0)])
    case, sample = failing(left)
    again = rerun_of(left, case)
    again["runs"][0]["calls"][0]["measurement"] = {"status": "request_failed"}
    assert merge([left, right], [(again, sample)])["status"] == "measurement_not_qualified"
    changed = rerun_of(left, case)
    changed["prompt_snapshot_sha256"] = "other"
    with pytest.raises(ValueError, match="Re-run protocol differs"):
        merge([left, right], [(changed, sample)])
