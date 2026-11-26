from types import SimpleNamespace

from app.services.chat import token_budget
from app.services.llm import token_count

DOUBAO = "doubao-seed-character-260628"


def test_calibrated_counts_follow_the_production_fit():
    # The 11-07 search turn: ~11k CJK + 3k other chars over 94 messages.
    text = "你" * 1000 + "a" * 1000
    assert token_count.count_text(text, DOUBAO) == round(1000 * 0.795 + 1000 * 0.392)
    assert token_count.count_text(text, "ark/" + DOUBAO) == token_count.count_text(text, DOUBAO)
    messages = [{"role": "user", "content": text}, {"role": "assistant", "content": ""}]
    assert token_count.count_messages(messages, DOUBAO) == round(795 + 392 + 2 * 16.6)


def test_uncalibrated_model_uses_a_bound_above_every_calibrated_rate():
    rate = token_count.rate("some-new-model")
    assert rate.calibrated is False
    for known in token_count.RATES.values():
        assert rate.cjk >= known.cjk and rate.other >= known.other


def test_measure_splits_system_sections_history_and_current_turn():
    system = "## 回复要求\n像真人一样\n\n## 你记得的事情\n用户喜欢猫\n\n## 联网结果使用\n引用来源\n\n## 时间\n周三"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "早上好"},
        {"role": "assistant", "content": "早呀"},
        {"role": "system", "content": "## 情绪标记\n[EMO:]"},
        {"role": "user", "content": "今天有什么新闻"},
    ]
    measured = token_budget.measure(messages, DOUBAO)
    assert set(measured["sections"]) == {"回复要求", "你记得的事情", "联网结果使用", "时间", "情绪标记"}
    cats = measured["categories"]
    assert cats["persona"] == measured["sections"]["回复要求"]
    assert cats["memory"] == measured["sections"]["你记得的事情"]
    assert cats["tools"] == measured["sections"]["联网结果使用"]
    assert cats["other_system"] == measured["sections"]["时间"] + measured["sections"]["情绪标记"]
    assert measured["history_tokens"] == token_count.count_text("早上好", DOUBAO) + token_count.count_text("早呀", DOUBAO)
    assert measured["current_tokens"] == token_count.count_text("今天有什么新闻", DOUBAO)
    assert measured["calibrated"] is True


def _builder(system_chars, history_rows, row_chars):
    calls = []

    def build(budget):
        calls.append(budget)
        keep = min(history_rows, budget // round(row_chars * 1.5))
        rows = [{"role": "user", "content": "聊" * row_chars} for _ in range(keep)]
        return [{"role": "system", "content": "设" * system_chars}, *rows, {"role": "user", "content": "搜一下"}]
    return build, calls


def test_search_cap_leaves_requests_under_32k_alone():
    build, calls = _builder(7000, 50, 40)
    messages, cap = token_budget.fit_search_cap(build, 4000, DOUBAO)
    assert calls == [4000] and cap["trimmed"] is False and cap["over_cap"] is False
    assert cap["projected_input_tokens"] == token_budget.search_projection(messages, DOUBAO) <= 32768


def test_search_cap_trims_history_only_until_it_fits():
    # System ~9.5k and 4000-token history put the projection above 32k.
    build, calls = _builder(12000, 200, 40)
    messages, cap = token_budget.fit_search_cap(build, 4000, DOUBAO)
    assert cap["projected_before_trim"] > 32768
    assert cap["trimmed"] is True and cap["over_cap"] is False
    assert cap["projected_input_tokens"] <= 32768 and cap["history_budget"] < 4000
    assert messages[0]["content"] == "设" * 12000 and messages[-1]["content"] == "搜一下"
    assert len(calls) <= 1 + 6


def test_search_cap_reports_when_the_system_prompt_alone_is_too_long():
    build, _ = _builder(30000, 10, 40)
    _, cap = token_budget.fit_search_cap(build, 4000, DOUBAO)
    assert cap["over_cap"] is True and cap["history_budget"] == 0


def test_finish_main_messages_records_measurement_and_search_cap(monkeypatch):
    from app.services.chat import context_policy, graph_phases

    saved = {}
    monkeypatch.setattr(context_policy, "note_window", lambda name, metrics: saved.update(metrics))
    monkeypatch.setattr(graph_phases, "_main_reply_model", lambda web: DOUBAO)

    def build(system, rows, *, drop_older_than_seconds, token_budget, metrics):
        metrics.update(truncated=False, dropped_by_gap=0, token_budget=token_budget)
        return [{"role": "system", "content": "## 回复要求\n" + "设" * 14000},
                *[{"role": "user", "content": "聊" * 40}] * (token_budget // 60),
                {"role": "user", "content": "搜"}]

    services = SimpleNamespace(build_chat_messages=build, RECAP_GAP_SECONDS=10800,
                               logger=SimpleNamespace(warning=lambda *_: None))
    ctx = SimpleNamespace(recent_messages=[], history_fetch_limit=200, needs_web_search=True,
                          reengagement_gap_seconds=None)
    graph_phases._finish_main_messages(ctx, services, "SYS", [])
    assert saved["search_cap"]["trimmed"] is True and saved["search_cap"]["over_cap"] is False
    # The window describes the request actually sent, not the first attempt.
    assert saved["token_budget"] == saved["search_cap"]["history_budget"]
    assert saved["measured"]["categories"]["persona"] > 0 and saved["fetch_limited"] is False
    ctx.needs_web_search = False
    saved.clear()
    graph_phases._finish_main_messages(ctx, services, "SYS", [])
    assert "search_cap" not in saved and saved["token_budget"] == 4000


def test_calibration_summary_compares_chat_and_search_turns_separately():
    from app.services.effect import token_calibration

    def measured(est, old, model=DOUBAO):
        return {"model": "ark/" + model, "calibrated": True, "input_tokens": est, "heuristic_input_tokens": old}

    rows = [
        dict(measured=measured(10500, 15000), llm_name="ChatOpenAI", model_name=DOUBAO, prompt_tokens=10000),
        dict(measured=measured(9500, 14000), llm_name="ChatOpenAI", model_name=DOUBAO, prompt_tokens=10000),
        dict(measured=measured(500, 700), llm_name="ChatOpenAI", model_name=DOUBAO, prompt_tokens=600),
        # A fallback model answered: its billing says nothing about the doubao rate.
        dict(measured=measured(9000, 9000), llm_name="ChatOpenAI", model_name="qwen3.5-plus", prompt_tokens=8000),
        dict(measured=measured(9000, 9000), llm_name=None, model_name=None, prompt_tokens=None),
        dict(measured=json_dump(measured(14000, 20000)), llm_name="ArkResponsesAPI", model_name=DOUBAO,
             prompt_tokens=30000, search_cap=json_dump({"projected_input_tokens": 32000, "trimmed": True})),
        dict(measured=measured(14000, 20000), llm_name="ArkResponsesAPI", model_name=DOUBAO,
             prompt_tokens=33204, search_cap={"projected_input_tokens": 32700, "trimmed": False}),
    ]
    result = token_calibration.summarize(rows)
    [doubao] = result["models"]
    assert doubao["turns"] == 3 and doubao["turns_1000_plus"] == 2
    assert doubao["calibrated_error"]["p5"] == -0.05 and doubao["calibrated_error"]["p95"] == 0.05
    assert doubao["heuristic_error"]["p50"] in (0.4, 0.5)
    assert result["model_mismatch_turns"] == 1 and result["no_billing_turns"] == 1
    search = result["search"]
    assert search["turns"] == 2 and search["billed_over_32k"] == 1 and search["trimmed"] == 1
    assert search["under_projection"] == 1 and search["max_billed"] == 33204


def json_dump(value):
    import json
    return json.dumps(value)
