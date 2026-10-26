from copy import deepcopy
import pytest
from evals.context_embedding.compare import paired_quality, compare


def reports():
    a = {"runs": []}
    b = {"runs": []}
    for n in range(8):
        for sample in range(3):
            row = dict(
                case=str(n),
                sample=sample,
                turn=0,
                category="target",
                critical=False,
                judge={"pass": False},
            )
            a["runs"].append(row)
            row = deepcopy(row)
            row["judge"]["pass"] = True
            b["runs"].append(row)
    return a, b


def test_bootstrap_uses_conversations_and_no_quality_gain_fails():
    a, b = reports()
    q = paired_quality(a, b, ["target"], 100)
    assert q["passed"] and q["categories"][0]["scenario_count"] == 8
    assert q["categories"][0]["paired_cluster_95_ci"] == [1, 1]
    assert not paired_quality(b, b, ["target"], 100)["passed"]


def test_critical_pair_regression_is_not_hidden_by_average():
    a, b = reports()
    for rows in (a["runs"], b["runs"]):
        rows[0]["critical"] = True
    a["runs"][0]["judge"]["pass"] = True
    b["runs"][0]["judge"]["pass"] = False
    assert not paired_quality(a, b, ["target"], 100)["passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_incomplete_judgements_never_pass(mutation):
    a, b = reports()
    if mutation == "missing":
        b["runs"].pop()
    elif mutation == "duplicate":
        b["runs"].append(b["runs"][0])
    else:
        b["runs"][0]["judge"]["pass"] = None
    with pytest.raises(ValueError):
        paired_quality(a, b, ["target"], 100)


def test_extra_generation_cost_cannot_hide_in_per_call_average():
    from evals.context_embedding.measurement import ARK_MODEL, priced_call

    a, b = reports()
    m = dict(
        status="measured",
        input_tokens=1000,
        output_tokens=10,
        cached_input_tokens=500,
        web_search_calls=0,
    )
    call = dict(
        model=ARK_MODEL,
        route="chat.completions",
        reply_path="main_llm",
        measurement=m,
        price=priced_call(ARK_MODEL, "chat.completions", m),
        ttft_seconds=1,
    )
    for report in (a, b):
        report.update(
            cases_sha256="same",
            model=ARK_MODEL,
            repeats=3,
            model_options={"temperature": 0.5},
            judge_protocol={"model": "fixture", "instruction_sha256": "fixed"},
        )
        for row in report["runs"]:
            row["calls"] = [deepcopy(call)]
    assert compare(a, b, ["target"])["passed"]
    for row in b["runs"]:
        row["calls"].append(deepcopy(call))
    result = compare(a, b, ["target"])
    assert not result["passed"]
    assert (
        "normalized_cost_cny_over_10_percent" in result["resource_gates"][0]["reasons"]
    )


def test_changed_judge_cannot_be_compared_as_product_quality_gain():
    a, b = reports()
    for report in (a, b):
        report.update(cases_sha256="same", model="same", repeats=3, model_options={})
    a["judge_protocol"] = {"model": "old"}
    b["judge_protocol"] = {"model": "new"}
    with pytest.raises(ValueError, match="judge_protocol"):
        compare(a, b, ["target"])
    del b["judge_protocol"]
    with pytest.raises(ValueError, match="Missing evaluation protocol"):
        compare(a, b, ["target"])
