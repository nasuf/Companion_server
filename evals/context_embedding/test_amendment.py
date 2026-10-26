"""Amendment v2: hash-linked to the frozen policy, applied identically to every arm."""
from copy import deepcopy

import pytest

from evals.context_embedding.calibrate import apply_amendment, build_gate_policy, policy_sha256
from evals.context_embedding.compare import compare, select_variant
from evals.context_embedding.measurement import REPETITION_PROTOCOL_V2, previous_reply_repetition
from evals.context_embedding.test_compare import register, state_reports


def amend(policy, **overrides):
    amendment = {
        "version": "test-v2", "base_policy_sha256": policy_sha256(policy),
        "targets": {"tier": ["target"]},
        "judge_recovery": {"max_excluded_fraction": 0.03},
        "repetition": {"protocol": REPETITION_PROTOCOL_V2},
        "latency": {}, "input_tokens": {}, "binding": {}, "variant_selection": {},
    }
    amendment.update(overrides)
    return amendment


def bound(*reports, names=("baseline", "A", "B")):
    for report, name in zip(reports, names):
        report.update(status="baseline_recorded", case_count=36, layout_variant=name,
                      prompt_snapshot_sha256=f"prompt-{name}", state_repeats=9, measurement_qualified=True)
    return {"cases": 36, "expected_calls_per_arm": len(reports[0]["runs"]),
            "prompt_snapshot_sha256": {name: f"prompt-{name}" for name in names}}


def tier_reports(count=36, duration=(1.0, 1.0)):
    before, after = state_reports(count)
    for report, seconds in zip((before, after), duration):
        for row in report["runs"]:
            row["calls"][0].update(reply_path="tier", duration_seconds=seconds)
    return before, after


def setup(before, after, **overrides):
    manifest = bound(before, after)
    calibration = register(before, after)
    # register() derived the policy from the baseline's protocol; keep A/A aligned.
    calibration["protocol"]["prompt_snapshot_sha256"] = before["prompt_snapshot_sha256"]
    policy = build_gate_policy(calibration)
    before["gate_policy"] = after["gate_policy"] = deepcopy(policy)
    return calibration, amend(policy, **overrides), manifest


def run(before, after, calibration, amendment, manifest):
    return compare(before, after, None, calibration, amendment=amendment,
                   manifest=manifest, arms=("baseline", "A"))


def test_jaccard_is_stricter_than_dice_and_v1_stays_reproducible():
    reply, previous = "今天去看电影吧", "今天去看电影呢好吗"
    dice = previous_reply_repetition(reply, previous)["bigram_similarity"]
    jaccard = previous_reply_repetition(reply, previous, REPETITION_PROTOCOL_V2)["bigram_similarity"]
    assert jaccard < dice
    with pytest.raises(ValueError):
        previous_reply_repetition(reply, previous, "unknown")


def test_amendment_must_reference_the_frozen_base_policy():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    amendment["base_policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preregistered base"):
        run(before, after, calibration, amendment, manifest)


def test_reports_must_match_the_registered_arm():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    after["prompt_snapshot_sha256"] = "prompt-B"
    with pytest.raises(ValueError, match="registered arm A"):
        run(before, after, calibration, amendment, manifest)


def test_targets_come_from_the_amendment_not_the_command_line():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    with pytest.raises(ValueError, match="differ"):
        compare(before, after, ["something_else"], calibration, amendment=amendment,
                manifest=manifest, arms=("baseline", "A"))
    amendment["targets"] = {"tier": ["absent_category"]}
    with pytest.raises(ValueError, match="targets missing"):
        run(before, after, calibration, amendment, manifest)


def test_unjudged_identity_is_excluded_from_every_arm_within_bound():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    before["runs"][0]["judge"] = {"pass": None, "reason": "invalid_model_judgement"}
    result = run(before, after, calibration, amendment, manifest)
    exclusion = result["judge_exclusions"]
    assert exclusion["passed"] and len(exclusion["excluded"]) == 1
    assert result["quality"]["categories"][0]["turn_pairs"] == 35
    # Without the amendment the same gap stays a hard stop.
    with pytest.raises(ValueError, match="Unjudged"):
        compare(before, after, ["target"], calibration)


def test_unjudged_identities_over_bound_or_critical_fail():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    for row in before["runs"][:2]:
        row["judge"] = {"pass": None}
    assert not run(before, after, calibration, amendment, manifest)["passed"]  # 2/36 > 3%
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    before["runs"][0]["critical"] = after["runs"][0]["critical"] = True
    before["runs"][0]["judge"] = {"pass": None}
    result = run(before, after, calibration, amendment, manifest)
    assert not result["passed"] and result["judge_exclusions"]["critical_excluded"]


@pytest.mark.parametrize("candidate,passed", [(1.1, True), (1.11, False)])
def test_nonstream_paths_are_gated_on_full_response_time(candidate, passed):
    before, after = tier_reports(duration=(1.0, candidate))
    for row in after["runs"]:
        row["calls"][0]["ttft_seconds"] = 0.1  # a faster first token must not hide it
    calibration, amendment, manifest = setup(before, after)
    result = run(before, after, calibration, amendment, manifest)
    gate = result["resource_gates"][0]
    assert gate["latency_metric"] == "p90_complete_seconds"
    assert result["passed"] is passed


def test_state_path_uses_full_response_time_with_36_calls():
    before, after = state_reports(36)
    for report, seconds in ((before, 1.0), (after, 1.2)):
        for row in report["runs"]:
            row["calls"][0]["duration_seconds"] = seconds
    calibration, amendment, manifest = setup(before, after)
    amendment["targets"] = {"current_state_shortcircuit": ["target"]}
    result = run(before, after, calibration, amendment, manifest)
    population = result["resource_gates"][0]["latency_population"]
    assert population["metric"] == "duration_seconds" and population["candidate_samples"] == 36
    assert "p90_complete_seconds_over_10_percent" in result["resource_gates"][0]["reasons"]


def test_input_tokens_are_compared_as_totals():
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    for row in after["runs"]:
        extra = deepcopy(row["calls"][0])
        row["calls"].append(extra)  # same mean input per call, double the total
    result = run(before, after, calibration, amendment, manifest)
    reasons = result["resource_gates"][0]["reasons"]
    assert "measured_input_tokens_over_10_percent" in reasons


def selection_scenario():
    """B beats A on repetition and passes every baseline gate; only the
    critical case separates them (A passes it 3/3, B fails 2/3)."""
    before, variant_a = tier_reports()
    variant_b = deepcopy(variant_a)
    manifest = bound(before, variant_a, variant_b)
    calibration = register(before, variant_a, rate=0.05)
    calibration["protocol"]["prompt_snapshot_sha256"] = before["prompt_snapshot_sha256"]
    policy = build_gate_policy(calibration)
    for report in (before, variant_a, variant_b):
        report["gate_policy"] = deepcopy(policy)
        for sample, row in enumerate(report["runs"][:3]):
            row.update(case="crit", sample=sample, critical=True)
    for row in before["runs"]:
        row["judge"]["pass"] = False
        row["reply"] = row["previous_assistant_reply"]  # baseline repeats everywhere
    variant_a["runs"][5]["reply"] = variant_a["runs"][5]["previous_assistant_reply"]
    for sample in (1, 2):
        variant_b["runs"][sample]["judge"]["pass"] = False
    return before, variant_a, variant_b, calibration, amend(policy), manifest


def test_b_wins_on_repetition_when_nothing_else_separates_it():
    before, variant_a, variant_b, calibration, amendment, manifest = selection_scenario()
    for sample in (1, 2):
        variant_b["runs"][sample]["judge"]["pass"] = True
    result = select_variant(before, variant_a, variant_b, None, calibration,
                            amendment=amendment, manifest=manifest)
    assert result["selected"] == "B" and result["selected_passed"]


def test_b_is_not_chosen_with_a_critical_regression_against_a():
    before, variant_a, variant_b, calibration, amendment, manifest = selection_scenario()
    result = select_variant(before, variant_a, variant_b, None, calibration,
                            amendment=amendment, manifest=manifest)
    assert result["baseline_to_b"]["passed"] and result["b_repetition_improved"]
    assert result["b_other_categories_within_aa_noise"]
    assert result["a_to_b_critical_regressions"]
    assert result["selected"] == "A"


def test_apply_amendment_rejects_incomplete_or_loose_rules():
    before, after = tier_reports()
    calibration, amendment, _ = setup(before, after)
    policy = build_gate_policy(calibration)
    for broken in ({"judge_recovery": {"max_excluded_fraction": 0.2}},
                   {"targets": {}}, {"repetition": {"protocol": "dice"}}):
        with pytest.raises(ValueError):
            apply_amendment(policy, amend(policy, **broken))
    incomplete = amend(policy)
    del incomplete["binding"]
    with pytest.raises(ValueError, match="Incomplete"):
        apply_amendment(policy, incomplete)


def main_reports(hashes=("h", "h")):
    before, after = tier_reports()
    for report, digest in zip((before, after), hashes):
        for row in report["runs"]:
            row["calls"][0].update(reply_path="main_llm", ttft_seconds=1,
                                   request_sha256=None if digest is None else f"{digest}-{row['case']}")
            row["judge"]["pass"] = True
    after["runs"][0]["judge"]["pass"] = False  # one flip on the main path
    return before, after


@pytest.mark.parametrize("hashes,flag,passed", [
    (("h", "h"), True, True),    # identical requests: the flip is noise
    (("h", "h"), False, False),  # rule off: strict main non-regression applies
    (("h", "x"), True, False),   # requests differ: candidate owns the flip
    ((None, None), True, False), # no request evidence: never assume identical
])
def test_identical_requests_only_waive_paths_the_candidate_did_not_change(hashes, flag, passed):
    before, after = main_reports(hashes)
    calibration, amendment, manifest = setup(before, after, targets={"main_llm": ["target"]})
    if flag:
        amendment["unchanged_input_rule"] = True
    result = run(before, after, calibration, amendment, manifest)
    assert result["quality"]["passed"] is passed
    assert bool(result["request_identical_paths"]) is (passed and flag)


def test_older_judge_fingerprint_matches_only_the_settings_in_force():
    from evals.context_embedding.compare import judge_compatible
    from evals.context_embedding.judge import JUDGE_REQUEST, judge_signature
    old = {k: v for k, v in judge_signature("qwen3.5-plus").items() if k not in JUDGE_REQUEST}
    assert judge_compatible(old, judge_signature("qwen3.5-plus"))
    assert not judge_compatible(old, {**judge_signature("qwen3.5-plus"), "max_tokens": 4000})
    assert not judge_compatible(old, judge_signature("other-model"))


def test_old_fingerprint_is_checked_against_frozen_settings_not_current_ones(monkeypatch):
    from evals.context_embedding.compare import judge_compatible
    from evals.context_embedding.judge import JUDGE_REQUEST, judge_signature
    old = {k: v for k, v in judge_signature("qwen3.5-plus").items() if k not in JUDGE_REQUEST}
    monkeypatch.setitem(JUDGE_REQUEST, "max_tokens", 4000)
    assert not judge_compatible(old, judge_signature("qwen3.5-plus"))


def test_identical_requests_never_waive_unknown_usage():
    before, after = main_reports()
    after["runs"][0]["judge"]["pass"] = True
    calibration, amendment, manifest = setup(before, after, targets={"main_llm": ["target"]},
                                             unchanged_input_rule=True)
    for row in after["runs"]:
        row["calls"][0]["measurement"] = {**row["calls"][0]["measurement"], "status": "request_failed"}
    result = run(before, after, calibration, amendment, manifest)
    assert result["request_identical_paths"] and not result["passed"]
    assert "usage_or_price_unknown" in result["resource_gates"][0]["reasons"]


@pytest.mark.parametrize("field,value", [("measurement_qualified", False), ("measurement_qualified", None),
                                         ("status", "measurement_not_qualified")])
def test_unqualified_or_unregistered_reports_are_not_bound(field, value):
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    after[field] = value
    with pytest.raises(ValueError, match="registered arm A"):
        run(before, after, calibration, amendment, manifest)
    before, after = tier_reports()
    calibration, amendment, manifest = setup(before, after)
    del manifest["prompt_snapshot_sha256"]["A"]
    after["prompt_snapshot_sha256"] = None  # a missing hash never matches a missing registration
    with pytest.raises(ValueError, match="registered arm A"):
        run(before, after, calibration, amendment, manifest)


def test_later_turns_after_a_changed_path_are_not_request_identical():
    from evals.context_embedding.compare import unchanged_paths
    before, after = main_reports()
    assert unchanged_paths(before, after) == {("chat.completions", "main_llm")}
    for report in (before, after):
        extra = deepcopy(report["runs"][0])
        extra.update(turn=1)
        report["runs"].append(extra)
        report["runs"][0]["calls"][0].update(reply_path="tier", request_sha256=f"{id(report)}")
    # Case 0 now answers turn 0 on a changed path; its turn-1 main reply inherits that.
    assert unchanged_paths(before, after) == frozenset()


def test_amended_comparison_reads_only_the_manifest_the_amendment_names(tmp_path):
    from evals.context_embedding.compare import ROOT, load_bound_manifest
    named = "evidence/context-policy-review-20261026/confirm-v3/manifest.json"
    with pytest.raises(ValueError, match="applies to"):
        load_bound_manifest({"applies_to_manifest": named}, tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="does not name"):
        load_bound_manifest({}, ROOT / named)


def test_judge_gaps_never_mask_an_unqualified_measurement():
    from evals.context_embedding.run_eval import final_status
    assert final_status(qualified=False, unjudged=3) == "measurement_not_qualified"
    assert final_status(qualified=True, unjudged=3) == "judge_incomplete"
    assert final_status(qualified=True, unjudged=0) == "baseline_recorded"


@pytest.mark.parametrize("replaced,rule,passed", [
    ((0, 1), None, False),                                                     # not pre-registered
    ((0, 1), {"max_replaced_fraction": 0.03, "max_excess_fraction": 0.03}, True),
    ((0, 2), {"max_replaced_fraction": 0.03, "max_excess_fraction": 0.03}, False),  # 2/36 over 3%
    ((1, 1), {"max_replaced_fraction": 0.03, "max_excess_fraction": 0.0}, True),
])
def test_rerun_replacements_are_bounded_and_preregistered(replaced, rule, passed):
    before, after = tier_reports()
    overrides = {"transport_failure_rule": rule} if rule else {}
    calibration, amendment, manifest = setup(before, after, **overrides)
    for report, count in zip((before, after), replaced):
        report["replaced_replays"] = [{"case": str(i), "sample": 0} for i in range(count)]
    result = run(before, after, calibration, amendment, manifest)
    assert result["transport_failures"]["passed"] is passed
    if not passed:
        assert result["passed"] is False


def test_transport_bounds_must_be_explicit_and_small():
    policy = build_gate_policy(register(*tier_reports()))
    for rule in ({"max_replaced_fraction": 0.2, "max_excess_fraction": 0.01}, {"max_replaced_fraction": 0.01}):
        with pytest.raises(ValueError, match="Transport"):
            apply_amendment(policy, amend(policy, transport_failure_rule=rule))
