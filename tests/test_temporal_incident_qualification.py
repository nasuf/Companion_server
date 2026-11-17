"""The live qualification harness must reject a silently degraded fixture."""
import logging

import pytest

from evals.temporal_awareness.run_incident_eval import _LocationFailures, validate_location_evidence
from tests.g02_harness_support import configure_pair


def test_qualification_cli_can_run_by_file_path_outside_checkout(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "evals/temporal_awareness/run_incident_eval.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--prompt-snapshot" in result.stdout
    assert "--cases-json" in result.stdout


def test_time_context_uses_production_renderer_with_frozen_clock(monkeypatch):
    from datetime import datetime, timezone
    from app.services.chat import orchestrator as chat
    from evals.temporal_awareness.run_incident_eval import configure_time_context
    io = configure_pair(monkeypatch)
    io.now = datetime(2026, 10, 23, 9, 32, tzinfo=timezone.utc)
    configure_time_context(monkeypatch, chat, io)
    assert chat.build_time_context() == "当前时间：2026年10月23日 17时 星期五"


@pytest.mark.asyncio
async def test_fixture_reads_real_location_loader_with_correct_clock_and_ownership(monkeypatch):
    from app.services.chat import orchestrator as chat
    io = configure_pair(monkeypatch)
    io.agent.city = '浙江省 杭州市 西湖区'
    context = await chat.load_location_context(io.agent, 'u-1', 'w-1')
    assert io.location_queries.await_count == 2
    assert context.ai_city == '杭州市' and context.user_city == '南京市'
    assert context.age_days == 0 and not context.diagnostics()['user_stale']
    assert context.relation == 'different'
    assert context.web_search_location()['city'] == '南京市'
    validate_location_evidence('normal', {'location_context': context.diagnostics()}, 2, [])


@pytest.mark.asyncio
async def test_fixture_db_failure_is_not_qualified_as_success(monkeypatch):
    from app.services.chat import orchestrator as chat
    from app.services.chat import location_context
    io = configure_pair(monkeypatch)
    io.location_queries.side_effect = ConnectionError('synthetic unavailable')
    logger = logging.getLogger(location_context.__name__)
    failures = _LocationFailures()
    logger.addHandler(failures)
    try:
        result = await chat.load_location_context(io.agent, 'u-1', 'w-1')
        assert result.user_source == 'unknown'
        with pytest.raises(RuntimeError, match='failed during qualification'):
            validate_location_evidence('normal', {}, io.location_queries.await_count, failures.failures)
    finally:
        logger.removeHandler(failures)


def test_missing_reads_rejected_but_current_state_short_circuit_is_explicit():
    with pytest.raises(RuntimeError, match='not exercised'):
        validate_location_evidence('normal', {}, 0, [])
    validate_location_evidence('current_state', {}, 0, [])
    with pytest.raises(RuntimeError, match='not exercised'):
        validate_location_evidence('current_state', {'location_context': {'user_source': 'device'}}, 0, [])


async def test_deterministic_historical_fallback_is_judged_not_incomplete(tmp_path, monkeypatch):
    """Nightly regression: this case answered with the fixed historical-recall fallback (no model
    call) and the harness raised "Incomplete actual reply flow" every night, so the whole run was
    reported incomplete. Its validity reads also hit the unconnected database."""
    import json
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import openai

    from evals.quality import corpus
    from evals.temporal.awareness import run_incident_eval as harness
    from app.services.prompting.registry import PROMPT_DEFINITIONS

    for key in ("DATABASE_URL", "REDIS_URL", "TRACING_BACKEND"):  # run() overwrites these
        monkeypatch.setenv(key, os.environ.get(key, ""))
    for key in ("ARK_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(key, "synthetic")
    for key in ("ARK_BASE_URL", "DASHSCOPE_BASE_URL"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    judged = []

    class Completions:
        async def create(self, **kwargs):
            judged.append(kwargs["messages"][-1]["content"])
            content = json.dumps({"pass": True, "reason": "synthetic"})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)

    class Client:
        def __init__(self, *_a, **_k):
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    warnings = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    logging.getLogger("app.services.time_domain.validity_context").addHandler(handler)
    raw = next(json.loads(line) for line in (Path(__file__).resolve().parents[1] / "evals/temporal/dialogue/cases.jsonl")
               .read_text().splitlines() if '"development-identity-family_trip"' in line)
    snapshot = tmp_path / "prompts.json"
    snapshot.write_text(json.dumps([{"key": d.key, "content": d.default_text, "is_enabled": True}
                                    for d in PROMPT_DEFINITIONS], ensure_ascii=False))
    model_calls = []

    async def provider_call(messages, **_options):
        model_calls.append(messages)
        return {"text": "去西藏的是你弟弟呀。", "usage": {}}

    args = SimpleNamespace(source_root=tmp_path, prompt_snapshot=snapshot, output=tmp_path / "out.json",
                           model="synthetic", judge_model="synthetic", samples=1, provider_call=provider_call)
    try:
        await harness.run(args, cases=[corpus.normalise("temporal_dialogue", raw)])
    finally:
        logging.getLogger("app.services.time_domain.validity_context").removeHandler(handler)
    run = json.loads(args.output.read_text())["runs"][0]
    assert run["response_diagnostics"]["reply_path"] == "temporal_recall"
    assert run["calls"] == [] and model_calls == []      # 确定性回复, 没有模型调用
    assert run["judge"]["pass"] is True and judged        # 照常交给评审
    validity = run["response_diagnostics"]["temporal_context"]["validity"]
    assert validity["status"] != "unavailable" and validity["recent_source_count"] == 1
    assert not [w for w in warnings if "unavailable" in w]
