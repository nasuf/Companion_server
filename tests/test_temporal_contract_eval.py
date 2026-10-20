import pytest

from app.services.temporal_query import resolve_temporal_query
from evals.temporal_contract.run_eval import evaluate, load_cases

CASES, MANIFEST = load_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_frozen_temporal_contract(case):
    result = evaluate(case, resolve_temporal_query)
    assert result["ok"], result["errors"]


def test_dataset_reports_actual_independence_and_no_model_claim():
    assert len(CASES) == 240
    assert len({c["scenario_family"] for c in CASES}) == 48
    assert MANIFEST["model_evaluation"] is False and MANIFEST["holdout"] is False
