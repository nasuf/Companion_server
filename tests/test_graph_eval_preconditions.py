"""A quality gate cannot grade unsupported persona/capability assumptions."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evals.graph_equivalence.preconditions import (
    REQUIREMENTS, Fact, audit_bank, audit_case, fixture_facts, row_precondition_valid,
)
from evals.reply_register.cases import ALL_CASES, FALSE_PREMISE_CASES
from evals.reply_register.judge import FALSE_PREMISE_ASSUMPTIONS
from tests.g02_harness_support import synthetic_agent


def base_facts():
    return {key: value for key, value in fixture_facts(synthetic_agent()).items()
            if key in {"persona.age", "persona.city", "persona.occupation", "capability.text_only"}}


def case(case_id):
    return next(item for item in FALSE_PREMISE_CASES if item.id == case_id)


def test_every_original_false_premise_has_an_explicit_requirement():
    assert set(REQUIREMENTS) == {item.id for item in FALSE_PREMISE_CASES}
    assert len(ALL_CASES) == 99
    assert len({item.id for item in ALL_CASES}) == 99


def test_legacy_unproven_fixture_still_fails_closed():
    facts = base_facts()
    assert set(facts) == {"persona.age", "persona.city", "persona.occupation", "capability.text_only"}
    result = audit_bank(ALL_CASES, facts, FALSE_PREMISE_ASSUMPTIONS)
    assert not result["passed"]
    assert result["case_count"] == 99 and result["blocked_case_count"] == 12
    assert sum(issue["code"] == "missing_fact" for item in result["cases"] for issue in item["issues"]) == 9
    assert sum(issue["code"] == "judge_assumption_conflict" for item in result["cases"] for issue in item["issues"]) == 12


@pytest.mark.parametrize("case_id", ["fp_antarctica", "fp_three_sisters", "fp_has_kids", "fp_hates_cats", "fp_is_ai"])
def test_empty_history_is_not_proof_of_negative_persona_facts(case_id):
    result = audit_case(case(case_id), base_facts(), {})
    assert result["issues"] == [{"code": "missing_fact", "key": REQUIREMENTS[case_id].key}]


@pytest.mark.parametrize("case_id", ["fp_sent_photo", "fp_called_me", "fp_ordered_food", "fp_met_offline"])
def test_disabled_features_or_different_channels_do_not_disprove_delivery(case_id):
    facts = {"capability.photo_enabled": Fact(False, "fixture"),
             "delivery.voice_message_yesterday": Fact(True, "fixture")}
    result = audit_case(case(case_id), facts, {})
    assert result["issues"] == [{"code": "missing_fact", "key": REQUIREMENTS[case_id].key}]


@pytest.mark.parametrize("value", [None, "24", True, 35, 22])
def test_identity_precondition_rejects_unknown_wrong_types_and_fixture_drift(value):
    result = audit_case(case("fp_age_35"), {"persona.age": Fact(value, "fixture")}, {})
    assert not result["passed"]


def test_evidence_requires_a_source_and_current_judge_capability_assumptions():
    result = audit_case(case("fp_age_35"), {"persona.age": Fact(24, "")}, FALSE_PREMISE_ASSUMPTIONS)
    assert {issue["code"] for issue in result["issues"]} == {"missing_fact", "unverified_judge_assumption"}


def test_declared_fact_is_not_enough_if_it_never_reaches_reply_model():
    facts = fixture_facts(synthetic_agent())
    assert audit_case(case("fp_age_35"), facts, {})["passed"]
    result = audit_case(case("fp_age_35"), facts, {}, trusted_references=["小岚"], response_inputs=[["你今年多大"]])
    assert {issue["code"] for issue in result["issues"]} == {"fact_not_in_model_input", "fact_not_in_judge_input"}


def test_user_claim_alone_cannot_supply_authoritative_evidence():
    facts = fixture_facts(synthetic_agent())
    statement = facts["persona.age"].statement
    result = audit_case(case("fp_age_35"), facts, {}, trusted_references=[], response_inputs=[[statement]])
    assert not result["passed"]


def test_dropped_reference_or_one_uninformed_attempt_blocks_evidence():
    facts = fixture_facts(synthetic_agent())
    statement = facts["persona.age"].statement
    for inputs in ([], [["事实被模板丢弃"]], [[statement], ["另一分支没有事实"]]):
        assert not audit_case(case("fp_age_35"), facts, {}, trusted_references=[statement], response_inputs=inputs)["passed"]
    result = audit_case(case("fp_age_35"), facts, {}, trusted_references=[statement], response_inputs=[[statement]],
                        judge_references=[statement], judge_inputs=[[statement]])
    assert result["passed"] and "24" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", ["fp_age_35", "fp_lives_abroad", "fp_doctor_job"])
async def test_existing_main_identity_renderer_contains_authoritative_facts(case_id):
    from app.services.chat.prompt_builder import _build_personality_section
    from evals.graph_equivalence.run_eval import Patches
    from tests.g02_harness_support import configure_pair
    with Patches() as patches:
        io = configure_pair(patches)
        section = await _build_personality_section(io.agent)
        facts = fixture_facts(io.agent)
        result = audit_case(case(case_id), facts, {}, trusted_references=[section.body], response_inputs=[[section.body]],
                            judge_references=[section.body], judge_inputs=[[section.body]])
        assert result["passed"]


def test_unknown_cases_and_duplicate_ids_fail_closed():
    unknown = SimpleNamespace(id="new-false-case", group="falsepremise")
    assert not audit_bank([unknown], {}, {})["passed"]
    normal = ALL_CASES[0]
    assert not audit_bank([normal, normal], {}, {})["passed"]


@pytest.mark.parametrize("failure", ["missing", "wrong_case", "string_passed", "missing_requirements", "issues", "wrong_group", "preflight_only"])
def test_report_requires_explicit_evidence_for_original_case(failure):
    c = case("fp_age_35")
    row = {"group": c.group, "case_precondition": {"case": c.id, "stage": "model_input", "passed": True,
           "requirements": [REQUIREMENTS[c.id].key], "issues": []}}
    assert row_precondition_valid(row, c)
    if failure == "missing":
        row.pop("case_precondition")
    elif failure == "wrong_case":
        row["case_precondition"]["case"] = "other"
    elif failure == "string_passed":
        row["case_precondition"]["passed"] = "true"
    elif failure == "missing_requirements":
        row["case_precondition"]["requirements"] = []
    elif failure == "issues":
        row["case_precondition"]["issues"] = [{"code": "missing_fact"}]
    elif failure == "wrong_group":
        row["group"] = "chitchat"
    else:
        row["case_precondition"]["stage"] = "preflight"
    assert not row_precondition_valid(row, c)


def test_reply_or_user_text_cannot_supply_missing_judge_reference():
    from evals.reply_register.judge import build_judge_prompt
    c = case("fp_age_35")
    facts = fixture_facts(synthetic_agent())
    statement = facts["persona.age"].statement
    prompt = build_judge_prompt(c.group, "", c.message, statement)
    result = audit_case(c, facts, {}, trusted_references=[statement], response_inputs=[[statement]],
                        judge_references=[], judge_inputs=[[prompt]])
    assert result["issues"] == [{"code": "fact_not_in_judge_input", "key": "persona.age"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(REQUIREMENTS))
@pytest.mark.parametrize("relevance", ["weak", "medium", "strong"])
async def test_actual_tier_render_delivers_known_identity_to_both_executors(monkeypatch, case_id, relevance):
    from app.config import settings
    from app.services.chat import intent_replies, orchestrator as chat
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentType
    from tests.g02_harness_support import configure_pair
    from tests.test_chat_graph_equivalence import collect
    from evals.graph_equivalence.scenarios import history_for, witness_for
    from evals.reply_register.judge import build_judge_prompt
    c = case(case_id)
    for executor in ("legacy", "langgraph"):
        with monkeypatch.context() as patch:
            io = configure_pair(patch)
            history = [SimpleNamespace(id=f"history-{i}", role=role, content=text, metadata={}, createdAt=io.now)
                       for i, (role, text) in enumerate(history_for(c))]
            history.append(SimpleNamespace(id="u-new", role="user", content=c.message, metadata={}, createdAt=io.now))
            io.db.message.find_many.side_effect = lambda **_: list(reversed(history))
            patch.setattr(settings, "chat_executor", executor)
            patch.setattr(chat, "detect_relational_context", lambda *_: None)
            chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance=relevance)
            patch.setattr(intent_replies, "get_chat_model", lambda: "synthetic-model")
            invoke = AsyncMock(return_value="我想不起来了")
            patch.setattr(intent_replies, "invoke_text", invoke)
            references = [witness_for(c)] if witness_for(c) else []
            for kind in ("weak", "medium", "strong"):
                async def tier(_reply=getattr(intent_replies, f"memory_{kind}_reply"), **kwargs):
                    references.extend(str(kwargs[key]) for key in ("personality_brief", "ai_memory", "l3_memory")
                                      if kwargs.get(key))
                    return await _reply(**kwargs)
                patch.setattr(chat, f"_memory_{kind}_reply", tier)
            await collect(io, c.message, forced_intent=IntentType.NONE)
            invoke.assert_awaited_once()
            facts = fixture_facts(io.agent)
            statement = facts[REQUIREMENTS[c.id].key].statement
            judge_input = build_judge_prompt(c.group, "", c.message, "合成回复", known_facts=statement)
            result = audit_case(c, facts, {}, trusted_references=references,
                                response_inputs=[[invoke.await_args.args[1]]],
                                judge_references=[statement], judge_inputs=[[judge_input]])
            assert result["passed"]


def test_older_rows_without_case_evidence_cannot_pass_summary():
    from evals.graph_equivalence.run_eval import report_summary
    from tests.test_graph_eval_safety import valid_rows
    rows = valid_rows()
    rows[0].pop("case_precondition")
    result = report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)
    assert not result["case_preconditions_passed"] and not result["passed"]


@pytest.mark.parametrize("group,raw", [
    ("chitchat", '{"verdict":"unknown","reason":"natural"}'),
    ("chitchat", '{"reason":"natural"}'),
    ("chitchat", '{"verdict":"natural"'),
    ("chitchat", '{"verdict":"natural"} {"verdict":"off_topic"}'),
    ("chitchat", "not natural"),
    ("chitchat", "natural or off_topic"),
    ("chitchat", "This reply is natural"),
    ("chitchat", '{"verdict":null,"reason":"natural"}'),
    ("chitchat", '{"verdict":["natural"]}'),
    ("chitchat", '{"verdict":{"value":"natural"}}'),
    ("emotion", "I cannot determine providing_suggestions"),
    ("falsepremise", "The rubric mentions correct_pushback"),
])
def test_invalid_judge_output_cannot_gain_a_grade_from_explanation_words(group, raw):
    from evals.reply_register.judge import parse_verdict
    assert parse_verdict(group, raw) is None


@pytest.mark.parametrize("group,verdict", [
    ("fact", "companion"), ("chitchat", "natural"),
    ("emotion", "providing_suggestions"), ("outofwindow", "honest_uncertainty"),
    ("falsepremise", "correct_pushback"),
])
def test_valid_json_fences_and_exact_bare_judge_labels_remain_supported(group, verdict):
    from evals.reply_register.judge import parse_verdict
    encoded = json.dumps({"verdict": verdict, "reason": "test"})
    for raw in (encoded, "```json\n" + encoded + "\n```", " " + verdict + "\n"):
        assert parse_verdict(group, raw) == verdict


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [[], ["--preflight-only"], ["--paired-classifiers"]])
async def test_invalid_full_bank_aborts_before_db_redis_and_model_calls(monkeypatch, tmp_path, args):
    from evals.graph_equivalence import run_eval
    output = tmp_path / "report.json"
    snapshot = AsyncMock(side_effect=AssertionError("No production read allowed"))
    model = Mock(side_effect=AssertionError("No provider quota allowed"))
    monkeypatch.setattr(run_eval, "read_snapshot", snapshot)
    monkeypatch.setattr(run_eval, "fixture_facts", lambda _: base_facts())
    monkeypatch.setattr(run_eval.models, "get_chat_model", model)
    monkeypatch.setattr(run_eval.models, "get_utility_model", model)
    monkeypatch.setattr("sys.argv", ["g02", "--read-only-production-snapshot", "--output", str(output), *args])
    with pytest.raises(SystemExit) as rejected:
        await run_eval.main()
    assert rejected.value.code == 2
    snapshot.assert_not_awaited()
    model.assert_not_called()
    report = json.loads(output.read_text())
    assert report["schema_version"] == 6
    assert report["rows"] == [] and not report["qualification_passed"]
    assert report["case_preconditions"]["blocked_case_count"] == 9


@pytest.mark.asyncio
async def test_valid_subset_preflight_is_never_full_qualification(monkeypatch, tmp_path):
    from evals.graph_equivalence import run_eval
    output = tmp_path / "report.json"
    snapshot = AsyncMock(side_effect=AssertionError("No production read allowed"))
    monkeypatch.setattr(run_eval, "read_snapshot", snapshot)
    monkeypatch.setattr("sys.argv", ["g02", "--preflight-only", "--case-limit", "1", "--output", str(output)])
    await run_eval.main()
    snapshot.assert_not_awaited()
    report = json.loads(output.read_text())
    assert report["case_preconditions"]["passed"]
    assert not report["full_case_bank"] and not report["qualification_passed"]


def test_grounded_bank_has_explicit_facts_without_rewriting_original_cases():
    from evals.graph_equivalence.scenarios import DECLARED_FACTS, history_for, witness_for
    from evals.reply_register.judge import GROUNDED_FALSE_PREMISE_ASSUMPTIONS
    result = audit_bank(ALL_CASES, fixture_facts(synthetic_agent()), GROUNDED_FALSE_PREMISE_ASSUMPTIONS)
    assert result["passed"] and result["case_count"] == 99
    witnessed = [c for c in ALL_CASES if witness_for(c)]
    assert len(witnessed) == len(DECLARED_FACTS) == 9
    for c in ALL_CASES:
        original = c.history
        effective = history_for(c)
        if witness_for(c):
            assert effective == (("assistant", witness_for(c)), *original)
            assert fixture_facts(synthetic_agent())[REQUIREMENTS[c.id].key].statement == witness_for(c)
        else:
            assert effective == original
        assert c.history == original
        if c.group == "outofwindow":
            assert not witness_for(c)


@pytest.mark.parametrize("case_id", list(REQUIREMENTS))
def test_grounded_judge_receives_authoritative_fact_and_retains_verdicts(case_id):
    from evals.reply_register.judge import build_judge_prompt
    c = case(case_id)
    fact = fixture_facts(synthetic_agent())[REQUIREMENTS[c.id].key]
    prompt = build_judge_prompt(c.group, "", c.message, "合成回复", known_facts=fact.statement)
    assert fact.statement in prompt
    assert "她只能在这个聊天窗口里打字" not in prompt
    assert "能发语音/图片不等于已经打电话" in prompt
    for label in ("correct_pushback", "plays_along", "harsh", "evasive"):
        assert label in prompt


def test_grounded_judge_rejects_empty_facts_without_changing_legacy_caller():
    from evals.reply_register.judge import build_judge_prompt
    with pytest.raises(ValueError):
        build_judge_prompt("falsepremise", "", "合成用户", "合成回复", known_facts="")
    assert "她只能在这个聊天窗口里打字" in build_judge_prompt("falsepremise", "", "合成用户", "合成回复")


@pytest.mark.asyncio
async def test_calibration_uses_grounded_context_only_for_false_premises(monkeypatch):
    from evals.reply_register import judge, run_eval
    captured = []
    async def answer(_, prompt):
        index = len(captured)
        captured.append(prompt)
        return json.dumps({"verdict": judge.CALIBRATION[index][3]})
    monkeypatch.setattr(run_eval, "_ask_judge", answer)
    assert await run_eval.run_calibration(object(), concurrency=1, grounded_false_premises=True)
    assert len(captured) == len(judge.CALIBRATION) == 14
    for prompt, (group, _, _, _) in zip(captured, judge.CALIBRATION, strict=True):
        assert ("角色没有去过南极。" in prompt) == (group == "falsepremise")


def prompt_snapshot():
    from datetime import datetime, timezone
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    return {key: SimpleNamespace(key=key, content=d.default_text, defaultContent=d.default_text,
                                updatedAt=datetime(2026, 10, 4, tzinfo=timezone.utc), isEnabled=True)
            for key, d in PROMPT_DEFINITION_MAP.items()}


def test_prospective_sync_is_local_and_preserves_customized_web_policies():
    from copy import deepcopy
    from evals.graph_equivalence.run_eval import prospective_prompt_snapshot
    snapshot = prompt_snapshot()
    for key in ("chat.response_instruction", "chat.anti_hallucination_hard_rule"):
        snapshot[key].content = "WEB-定制规则-" + key
    for kind in ("weak", "medium", "strong", "l3"):
        row = snapshot[f"memory.{kind}_reply"]
        row.content = row.defaultContent = "原默认-" + kind
    before = deepcopy(snapshot)
    effective, changes = prospective_prompt_snapshot(snapshot)
    assert snapshot == before
    assert {row["key"] for row in changes} == {f"memory.{kind}_reply" for kind in ("weak", "medium", "strong", "l3")}
    assert all(not row["customized"] and "expected_updated_at" in row for row in changes)
    for key in ("chat.response_instruction", "chat.anti_hallucination_hard_rule"):
        assert effective[key].content == before[key].content
    assert effective["memory.weak_reply"] is not snapshot["memory.weak_reply"]


@pytest.mark.parametrize("failure", ["missing", "customized", "unreviewed"])
def test_prospective_sync_rejects_missing_customized_or_unreviewed_prompt_changes(failure):
    from evals.graph_equivalence.run_eval import prospective_prompt_snapshot
    snapshot = prompt_snapshot()
    if failure == "missing":
        snapshot.pop("memory.weak_reply")
    else:
        row = snapshot["memory.weak_reply" if failure == "customized" else "chat.system_base"]
        row.content = "Web 定制" if failure == "customized" else "原默认"
        row.defaultContent = "原默认"
    with pytest.raises(RuntimeError):
        prospective_prompt_snapshot(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["weak", "medium", "strong", "l3"])
async def test_tier_policy_prefix_uses_web_overrides_once(monkeypatch, kind):
    from app.services.prompting import store
    from tests.g02_harness_support import configure_pair
    configure_pair(monkeypatch)
    snapshot = prompt_snapshot()
    snapshot["chat.response_instruction"].content = "WEB-回复规则 {max_per} {total}"
    snapshot["chat.anti_hallucination_hard_rule"].content = "WEB-反幻觉规则"
    monkeypatch.setattr(store, "db", SimpleNamespace(prompttemplate=SimpleNamespace(
        find_unique=AsyncMock(side_effect=lambda *, where: snapshot.get(where["key"])),
    )))
    prompt = await store.get_prompt_text(f"memory.{kind}_reply")
    assert prompt.count("WEB-回复规则") == prompt.count("WEB-反幻觉规则") == 1
    prefix = prompt.split("\n\n", 1)[0]
    assert "{max_per}" not in prefix and "{total}" not in prefix


@pytest.mark.asyncio
async def test_tier_respects_personality_section_disable_without_loading_main_prompt(monkeypatch):
    from app.config import settings
    from app.services.chat import orchestrator as chat
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.prompting import store
    from tests.g02_harness_support import configure_pair
    from tests.test_chat_graph_equivalence import collect
    io = configure_pair(monkeypatch)
    monkeypatch.setattr(settings, "chat_executor", "langgraph")
    snapshot = prompt_snapshot()
    snapshot["chat.personality_section"].isEnabled = False
    monkeypatch.setattr(store, "db", SimpleNamespace(prompttemplate=SimpleNamespace(
        find_unique=AsyncMock(side_effect=lambda *, where: snapshot.get(where["key"])),
    )))
    chat.fetch_parallel_context.return_value = FetchedContext(memory_relevance="weak")
    factory = AsyncMock(side_effect=AssertionError("Tier must keep main prompt lazy"))
    monkeypatch.setattr(chat, "build_system_prompt", factory)
    await collect(io)
    assert io.tier_calls[0][1]["personality_brief"] == io.agent.name
    factory.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs,expected", [
    (["{invalid", '{"verdict":"natural"}'], "natural"),
    (["not natural", "unknown", '{"verdict":[]}'], None),
    (['{"verdict":"off_topic"}'], "off_topic"),
])
async def test_judge_retries_only_unparsed_outputs_and_keeps_every_attempt(outputs, expected):
    from evals.graph_equivalence.run_eval import REQUEST_PHASE, digest, grade_response
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[SimpleNamespace(content=raw) for raw in outputs]))
    verdict, attempts = await grade_response(model, "chitchat", "frozen judge input")
    assert verdict == expected
    assert model.ainvoke.await_count == len(outputs) == len(attempts)
    assert all(call.args == ("frozen judge input",) for call in model.ainvoke.await_args_list)
    assert all(row["input_hash"] == digest("frozen judge input") for row in attempts)
    assert [row["output_hash"] for row in attempts] == [digest(raw) for raw in outputs]
    assert REQUEST_PHASE.get() == "response"


@pytest.mark.asyncio
async def test_judge_provider_or_fence_exception_is_not_retried():
    from evals.graph_equivalence.run_eval import REQUEST_PHASE, grade_response
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("Evaluation blocked business address")))
    with pytest.raises(RuntimeError):
        await grade_response(model, "chitchat", "same input")
    model.ainvoke.assert_awaited_once()
    assert REQUEST_PHASE.get() == "response"
