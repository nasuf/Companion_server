"""Evaluation must expose losses, mismatches and incomplete/unsafe evidence."""

from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

import pytest

from evals.memory_baseline.dataset import load_cases, split_cases, fingerprint
from evals.memory_baseline.metrics import grade, paired_interval, summarize
from evals.memory_baseline.safety import isolated_url, loopback_network_fence


def case():
    return deepcopy(load_cases()[0])


def observations(c):
    return [
        {
            "case": c["id"],
            "sample": 0,
            "arm": arm,
            "cache": cache,
            "conditions": "frozen",
            "stored": None,
            "candidates": c["expected"],
            "selected": c["expected"],
            "used": None,
            "latency_ms": 1.0,
            "db_calls": 1 if cache == "cold" else 0,
            "llm_cost_cny": 0.0,
            "error": None,
        }
        for arm in ("baseline", "candidate")
        for cache in ("cold", "warm")
    ]


def test_bank_has_stratified_family_holdout_and_stable_identity():
    cases = load_cases()
    split = split_cases(cases)
    assert len(cases) == 240 and len({c["family"] for c in cases}) == 240
    assert Counter(split.values()) == {"development": 168, "holdout": 72}
    for group in {c["group"] for c in cases}:
        assert (
            sum(split[c["id"]] == "holdout" for c in cases if c["group"] == group) == 6
        )
    assert split_cases(list(reversed(cases))) == split
    assert fingerprint(cases) == fingerprint(load_cases())


def test_absent_stage_is_not_an_empty_success():
    c = case()
    g = grade(c, observations(c)[0])
    assert (
        g["stages"]["stored"]["status"] == "not_run"
        and g["stages"]["used"]["recall"] is None
    )
    assert (
        g["losses"]["recording_miss"] is None and g["losses"]["answer_unused"] is None
    )
    row = observations(c)[0]
    row["stored"] = []
    assert grade(c, row)["losses"]["recording_miss"] == c["expected"]


def test_correct_answer_cannot_hide_retrieval_or_selection_loss():
    c = case()
    row = observations(c)[0]
    row.update(stored=c["expected"], candidates=[], selected=[], used=c["expected"])
    g = grade(c, row)
    assert g["losses"]["retrieval_miss"] == c["expected"]
    assert g["stages"]["selected"]["recall"] == 0 and g["stages"]["used"]["recall"] == 1
    row.update(candidates=c["expected"], used=[])
    assert grade(c, row)["losses"]["selection_miss"] == c["expected"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "unknown",
        "mismatch",
        "nan",
        "error",
        "no_candidates",
        "no_selected",
        "boolean_sample",
        "fence",
    ],
)
def test_incomplete_or_mismatched_runs_never_qualify(mutation):
    c = case()
    rows = observations(c)
    if mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    if mutation == "missing":
        rows.pop()
    if mutation == "unknown":
        rows[0]["case"] = "unknown"
    if mutation == "mismatch":
        rows[0]["conditions"] = "changed"
    if mutation == "nan":
        rows[0]["latency_ms"] = float("nan")
    if mutation == "error":
        rows[0]["error"] = "TimeoutError"
    if mutation == "no_candidates":
        rows[0]["candidates"] = None
    if mutation == "no_selected":
        rows[0]["selected"] = None
    if mutation == "boolean_sample":
        rows[0]["sample"] = False
    if mutation == "fence":
        rows[0]["network_violations"] = ["blocked"]
    result = summarize([c], rows)
    assert not result["complete"] and not result["instrumentation_passed"]
    assert (
        result["algorithm_quality_passed"] is None
        and result["model_quality"] == "not_run"
    )


def test_complete_self_comparison_does_not_claim_algorithm_or_model_quality():
    c = case()
    result = summarize([c], observations(c))
    assert result["complete"] and result["recall_interval"]["delta"] == 0
    assert (
        result["model_quality"] == "not_run"
        and result["algorithm_quality_passed"] is None
    )


def test_repeat_sampling_does_not_invent_independent_scenarios():
    one = paired_interval({"a": [1.0], "b": [-1.0]}, draws=1000)
    repeated = paired_interval({"a": [1.0] * 5, "b": [-1.0] * 5}, draws=1000)
    assert one["clusters"] == repeated["clusters"] == 2
    assert one["lower"] == repeated["lower"] and one["upper"] == repeated["upper"]
    assert repeated["paired_observations"] == 10


def test_side_qualified_identity_and_hard_tenant_failures():
    c = next(c for c in load_cases() if c["group"] == "multi_agent")
    rows = observations(c)
    rows[0]["selected"] = [c["forbidden"][0]]
    assert summarize([c], rows)["hard_failures"]
    c["memories"].append(
        {
            "key": "user:fact",
            "source": "user",
            "scope": "target",
            "text": "合成用户事实",
            "level": 1,
        }
    )
    g = grade(c, {"candidates": ["user:fact", "ai:fact"], "selected": ["user:fact"]})
    assert g["stages"]["selected"]["recall"] == 0


@pytest.mark.parametrize(
    "url,kind",
    [
        ("postgresql://u:p@106.52.115.80:5432/companion", "postgres"),
        ("postgresql://u:p@127.0.0.1:55441/companion", "postgres"),
        ("postgresql://u:p@localhost:55441/companion_memory_eval_test", "postgres"),
        ("postgresql://u:p@127.0.0.1/companion_memory_eval_test", "postgres"),
        (
            "postgresql://u:p@127.0.0.1:5432/companion_memory_eval_test?host=prod",
            "postgres",
        ),
        ("redis://127.0.0.1:6379/0", "redis"),
        ("redis://prod.test:6379/15", "redis"),
        ("redis://127.0.0.1:6379/15?db=0", "redis"),
        ("redis://127.0.0.1:6379/16", "redis"),
    ],
)
def test_production_and_ambiguous_urls_refused(url, kind):
    with pytest.raises(ValueError):
        isolated_url(url, kind)


def test_explicit_disposable_urls_allowed():
    assert isolated_url(
        "postgresql://u:p@127.0.0.1:55441/companion_memory_eval_unit", "postgres"
    )
    assert isolated_url("redis://127.0.0.1:56391/14", "redis")


def test_socket_fence_blocks_external_even_if_application_recovers():
    calls = []
    with patch.object(socket.socket, "connect", lambda s, a: calls.append(a)):
        with loopback_network_fence() as violations:
            with socket.socket() as conn:
                conn.connect(("127.0.0.1", 55441))
                with pytest.raises(RuntimeError):
                    conn.connect(("106.52.115.80", 5432))
                with pytest.raises(RuntimeError):
                    conn.connect(("203.0.113.9", 443))
    assert len(violations) == 2 and calls == [("127.0.0.1", 55441)]


def test_fresh_process_disables_pydantic_dotenv_and_inherited_credentials(tmp_path):
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=must-never-load\nDATABASE_URL=postgresql://prod:5432/companion\n"
    )
    source = """from evals.memory_baseline.safety import configure_isolation
configure_isolation('postgresql://u:p@127.0.0.1:55441/companion_memory_eval_unit','redis://127.0.0.1:56391/14')
from app.config import settings
assert settings.database_url.endswith('/companion_memory_eval_unit')
assert not settings.dashscope_api_key
print('isolated')
"""
    import os

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "DASHSCOPE_API_KEY": "inherited-secret",
    }
    p = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "isolated" and "must-never-load" not in p.stdout


def test_validation_does_not_import_application_or_require_credentials():
    source = """from evals.memory_baseline.dataset import load_cases,split_cases
import sys
assert len(load_cases())==240
assert len(split_cases(load_cases()))==240
assert 'app.db' not in sys.modules and 'app.config' not in sys.modules
"""
    assert (
        subprocess.run(
            [sys.executable, "-c", source], capture_output=True, timeout=10
        ).returncode
        == 0
    )


def test_unknown_and_unhashable_memory_identity_is_rejected():
    c = case()
    for value in (["ai:unknown"], [{"bad": "id"}]):
        row = observations(c)[0]
        row["selected"] = value
        with pytest.raises(ValueError):
            grade(c, row)


def test_fixture_seeded_retrieval_loss_does_not_claim_extraction():
    c = case()
    row = observations(c)[0]
    row.update(
        fixture_seeded=True, fixture_ids=c["expected"], candidates=[], selected=[]
    )
    g = grade(c, row)
    assert g["losses"]["retrieval_miss"] == c["expected"]
    assert g["losses"]["recording_miss"] is None


def test_manifest_rejects_source_drift():
    from evals.memory_baseline.manifest import verify_source

    with patch("evals.memory_baseline.manifest.source_files", return_value={"a": "v"}):
        verify_source(
            {"code_files": {"a": "v"}, "code_tree_sha256": fingerprint({"a": "v"})}
        )
        with pytest.raises(ValueError):
            verify_source({"code_files": {"a": "changed"}, "code_tree_sha256": "old"})


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"answer_correct":true,"stale_misuse":false,"persona_drift":false,"used":["ai:unknown"]}',
        '{"answer_correct":1,"stale_misuse":false,"persona_drift":false,"used":[]}',
        '{"answer_correct":true,"stale_misuse":false,"persona_drift":false,"used":["ai:fact","ai:fact"]}',
    ],
)
def test_live_judge_cannot_invent_evidence_or_coerce_boolean(raw):
    from evals.memory_baseline.live import parse_judgement

    with pytest.raises(ValueError):
        parse_judgement(raw, ["ai:fact"])


def test_live_requires_five_complete_pairs_and_preserves_usage_receipts():
    from evals.memory_baseline.live import summarize_live, receipt

    assert not summarize_live(load_cases(), [], 5)["complete"]
    assert not summarize_live(load_cases(), [], 4)["complete"]
    price = {"input_cny_per_million": 1.0, "output_cny_per_million": 2.0}
    assert (
        receipt(
            {
                "model": "returned",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
            price,
        )["cost_cny"]
        == 0.0002
    )
    with pytest.raises(ValueError):
        receipt({"model": "returned"}, price)


def test_baseline_self_comparison_cannot_meet_activation_or_missing_model_gate():
    from evals.memory_baseline.comparison import compare

    cases = load_cases()
    rows = [r for c in cases for r in observations(c)]
    result = compare(cases, rows, target_groups=["long_memory"])
    assert not result["quality_passed"] and not result["checks"]["live_evidence"]
    assert not result["checks"]["holdout.long_memory.recall"]
    assert result["checks"]["holdout.overall.recall"]
    rows.pop()
    assert (
        compare(cases, rows, target_groups=["long_memory"])["reason"]
        == "incomplete_retrieval_evidence"
    )


def test_existing_adapters_preserve_durative_direction_and_persona_anchor_rules():
    from evals.memory_baseline.adapters import grade_durative, grade_identity
    from evals.durative_state.cases import CASES

    c = next(c for c in CASES if c.kind == "expired")
    assert grade_durative(c, "你还在三亚吗")["violations"]
    assert grade_durative(c, "今天看看电影吧")["semantic_status"] == "not_run"
    assert not grade_identity({"answer_anchors": ["普洱"]}, "我在镇江")["anchored"]
    assert grade_identity({"answer_anchors": ["普洱"]}, "作为AI我在普洱")["leak"]


def test_partial_fixture_failure_cleans_only_created_owners():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from evals.memory_baseline.replay import fixture

    user = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="owned-eval-user")),
        delete_many=AsyncMock(),
    )
    db = SimpleNamespace(
        user=user,
        aiagent=SimpleNamespace(
            create=AsyncMock(side_effect=RuntimeError("fixture failed"))
        ),
        aimemory=SimpleNamespace(delete_many=AsyncMock()),
        usermemory=SimpleNamespace(delete_many=AsyncMock()),
    )
    with pytest.raises(RuntimeError, match="fixture failed"):
        asyncio.run(fixture(db, case(), {}))
    user.delete_many.assert_awaited_once_with(where={"id": {"in": ["owned-eval-user"]}})


def test_swallowed_database_failure_still_visible_to_measurement():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from evals.memory_baseline.replay import CountedDatabase

    counted = CountedDatabase(
        SimpleNamespace(query_raw=AsyncMock(side_effect=RuntimeError("read failed")))
    )
    with pytest.raises(RuntimeError):
        asyncio.run(counted.query_raw("synthetic query"))
    assert counted.calls == 1 and counted.failures == ["RuntimeError"]
