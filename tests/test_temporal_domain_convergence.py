"""Canonical implementations preserve module identity, clocks and frozen corpora."""

import ast
from datetime import datetime, timezone, timedelta
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "app.services.temporal": "timestamps",
    "app.services.temporal_query": "query",
    "app.services.temporal_binding": "binding",
    "app.services.temporal_operation": "operations",
    "app.services.chat.temporal_context": "context",
    "app.services.chat.temporal_recall": "recall",
    "app.services.chat.temporal_guard": "guard",
    "app.services.chat.followup_policy": "followup",
    "app.services.schedule_domain.time_service": "clock",
    "app.services.schedule_domain.time_parser": "parser",
    "app.services.schedule_domain.time_expression": "expression",
    "app.services.memory.retrieval.timeline": "timeline",
}


@pytest.mark.parametrize("old,new", ALIASES.items())
def test_legacy_module_is_identical_not_a_second_implementation(old, new):
    assert importlib.import_module(old) is importlib.import_module(
        "app.services.time_domain." + new
    )


def test_ntp_calibration_state_is_shared_across_old_and_new_imports(monkeypatch):
    old = importlib.import_module("app.services.schedule_domain.time_service")
    new = importlib.import_module("app.services.time_domain.clock")
    monkeypatch.setattr(old, "_NTP_DRIFT_SECONDS", 12.5)
    assert new.get_ntp_drift()[0] == 12.5
    monkeypatch.setattr(new, "_NTP_DRIFT_SECONDS", -2.5)
    assert old.get_ntp_drift()[0] == -2.5


def test_runtime_has_no_imports_back_to_legacy_temporal_modules():
    offenders = []
    legacy_paths = {ROOT / (name.replace(".", "/") + ".py") for name in ALIASES}
    for folder in ("app", "jobs"):
        for p in (ROOT / folder).rglob("*.py"):
            if p in legacy_paths:
                continue
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom):
                    if node.module in ALIASES or any(
                        (str(node.module) + "." + a.name) in ALIASES for a in node.names
                    ):
                        offenders.append(f"{p.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    if any(a.name in ALIASES for a in node.names):
                        offenders.append(str(p))
    assert not offenders, offenders


def test_six_evaluation_families_and_frozen_corpus_are_preserved():
    from evals.temporal.__main__ import inventory

    result = inventory()
    assert result["ok"] and len(result["files"]) == 11
    assert {
        "contract",
        "replay",
        "binding",
        "dialogue",
        "incident",
        "awareness",
        "recall",
    } == set(result["suites"])
    for row in result["files"]:
        assert (ROOT / row["old"]).read_bytes() == (ROOT / row["new"]).read_bytes()


@pytest.mark.parametrize("suite", ["binding", "dialogue", "incident"])
def test_model_layer_never_runs_without_explicit_opt_in(tmp_path, suite):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.temporal",
            "--suite",
            suite,
            "--output-dir",
            str(tmp_path / "new"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert run.returncode != 0 and "--live" in run.stderr
    assert not (tmp_path / "new").exists()


def test_old_and_new_eval_imports_share_the_frozen_contract():
    old = importlib.import_module("evals.temporal_contract.run_eval")
    new = importlib.import_module("evals.temporal.contract.run_eval")
    assert old is new
    assert len(new.load_cases()[0]) == 240


def test_half_open_dates_and_timezone_are_unchanged():
    from app.services.time_domain.query import resolve_temporal_query

    query = resolve_temporal_query(
        "昨天",
        anchor=datetime(2026, 11, 1, 0, 1, tzinfo=timezone(timedelta(hours=8))),
        anchor_source="persisted_message",
    )
    assert query.retrieval_range() == (
        datetime(2026, 10, 30, 16, tzinfo=timezone.utc),
        datetime(2026, 10, 31, 16, tzinfo=timezone.utc),
    )
