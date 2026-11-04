"""CHAT01.10.2 重复表述回放评测的纯函数部分: 用例构建、打分口径、配对比较、脱敏规则。"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import pytest

from evals.reply_repetition import cases as case_mod
from evals.reply_repetition import compare, run_replay, sanitize, score

T0 = datetime(2026, 10, 30, 11, 21, 0)


def _row(i, role, content, sec, **meta):
    return {"id": f"m{i}", "role": role, "content": content,
            "created_at": (T0 + timedelta(seconds=sec)).isoformat(), "metadata": meta}


def _window(rows, turn_id, label="paraphrase", situation="overlap", user="real-user"):
    return {"score": {"turn_message_id": turn_id, "user_id": user, "conversation_id": "conv",
                      "situation": situation, "label": label, "local_date": "2026-10-30",
                      "max_similarity": 0.8}, "rows": rows}


MAIN = {"response_diagnostics": {"reply_path": "main_llm"}}


def test_build_case_replays_only_the_turn_production_answered():
    rows = [
        _row(1, "assistant", "你今天忙不忙呀", 0, trace_id="t0", **MAIN),
        _row(2, "user", "今天加班到好晚", 20),
        _row(3, "user", "而且", 29),
        # 上一轮回应的是 "今天加班到好晚/而且"; "饭都没吃" 是那一轮生成期间到的, 落库比上一轮早: 交叠
        _row(4, "user", "饭都没吃", 31),
        _row(5, "assistant", "辛苦啦 加到几点呀", 32, trace_id="t1", achievement_turn_id="user-turn:m2,m3", **MAIN),
        _row(6, "assistant", "那快去吃点东西吧", 45, trace_id="t2", achievement_turn_id="user-turn:m4",
             response_diagnostics={"reply_path": "main_llm", "topic_continuation": {"status": "applied"}}),
    ]
    built = case_mod.build_case(_window(rows, "m6"), 7, "u01")
    assert built is not None
    case, mapping = built
    assert case["id"] == "rr-007" and case["kind"] == "positive"
    # 只回应 "饭都没吃": 把前两句也当成本轮, 模型会把上一轮已经接过的话再答一遍
    assert case["message"] == "饭都没吃" and case["current_ids"] == ["h3"]
    assert case["judge_user_text"] == "今天加班到好晚 而且 饭都没吃"  # 评判口径与 10.1 eligible_turns 一致
    assert [r["id"] for r in case["history_rows"]] == ["h0", "h1", "h2", "h3", "h4"]
    assert case["history_rows"][-1]["content"] == "辛苦啦 加到几点呀"
    assert case["received_at"].startswith("2026-10-30T11:21:31") and case["now"].startswith("2026-10-30T11:21:45")
    assert case["previous_ai_turns"] == [["你今天忙不忙呀"], ["辛苦啦 加到几点呀"]]
    assert set(case["tags"]) == {"overlap", "paraphrase", "topic_continuation", "partial_user_turn"}
    assert "m" not in json.dumps(case["history_rows"]).replace("metadata", "")  # 不带真实消息 ID
    assert mapping["ids"]["h4"] == "m5" and mapping["user_id"] == "real-user"
    # 回应的消息不在导出窗口里: 复现不了, 丢弃 (不退回 "最后一段用户消息")
    rows[5]["metadata"]["achievement_turn_id"] = "user-turn:gone"
    assert case_mod.build_case(_window(rows, "m6"), 1, "u01") is None


def test_build_case_tags_and_skips():
    rows = [
        _row(1, "assistant", "在忙吗 好久没聊了", 0, trace_id="p", proactive=True, trigger_type="followup_unfinished"),
        _row(2, "user", "嗯嗯", 600),
        _row(3, "assistant", "刚才在开会呢 现在有空啦", 640, trace_id="t", **MAIN),
    ]
    case, _ = case_mod.build_case(_window(rows, "m3", label="not_repeat", situation="ack"), 1, "u02")
    # 没有 achievement_turn_id 的老消息: 回应最后一段用户消息
    assert case["current_ids"] == ["h1"] and case["message"] == "嗯嗯"
    assert case["kind"] == "negative" and set(case["tags"]) == {"ack", "delay_section", "after_followup"}
    assert case_mod.build_case(_window(rows, "missing"), 1, "u02") is None
    assert case_mod.build_case(_window(rows[2:], "m3"), 1, "u02") is None  # 前面没有用户消息
    rows[2]["metadata"]["achievement_turn_id"] = "user-turn:m9"
    rows.insert(0, _row(9, "user", "很早之前那句", -10))
    windows = [_window(rows, "m3", user="a"), _window(rows, "x", user="b"), _window(rows[1:], "m3", user="a")]
    cases, maps, dropped = case_mod.build_cases({"windows": windows})
    # 第三个窗口没导出回应的 m9: 复现不了, 丢弃; 第一个窗口里有 m9, 正常建例
    assert [c["id"] for c in cases] == ["rr-001"] and dropped == ["x", "m3"] and len(maps) == 1


def test_replay_helpers():
    case = {"id": "rr-1", "now": "2026-10-30T11:21:45+00:00", "received_at": "2026-10-30T11:21:31+00:00",
            "history_rows": [], "message": "好的", "current_ids": ["h1", "h2"]}
    incident = run_replay.incident(case)
    assert incident["user_message_id"] == "h2" and incident["route"] == "ordinary"
    assert incident["reply_context"] == {"received_at": case["received_at"], "turn_message_ids": ["h1", "h2"]}
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 500}}
    assert run_replay.call_cost(usage) == pytest.approx((500 * 0.8 + 500 * 0.16 + 100 * 2.0) / 1e6)
    assert run_replay.call_cost(None) is None
    assert run_replay.detail_cost({"runs": [{"calls": [{"messages": [], "usage": usage}, {"kind": "judge"}]}]}) > 0
    items = [{"id": i} for i in range(5)]
    assert [x["id"] for x in run_replay.shard(items, "1/2")] == [1, 3]
    with pytest.raises(ValueError):
        run_replay.shard(items, "2/2")


def test_arm_spent_counts_partials_and_skips_unreadable(tmp_path):
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    detail = {"runs": [{"calls": [{"messages": [{"role": "system"}], "usage": usage}]}]}
    (tmp_path / "rr-001.json").write_text(json.dumps(detail))
    (tmp_path / "rr-002.json.partial0").write_text(json.dumps(detail))  # 中断留下的也花了钱
    (tmp_path / "rr-003.json").write_text('{"runs": [')  # 别的分片写到一半
    (tmp_path / "notes.json").write_text(json.dumps(detail))
    assert run_replay.arm_spent(tmp_path) == pytest.approx(1.6)


def test_completed_detail(tmp_path):
    path = tmp_path / "rr-1.json"
    assert not run_replay.completed(path)
    path.write_text("{bad")
    assert not run_replay.completed(path)
    path.write_text(json.dumps({"runs": []}))
    assert not run_replay.completed(path)
    path.write_text(json.dumps({"runs": [], "total": 0}))
    assert run_replay.completed(path)


async def test_production_context_reaches_the_real_builders():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from pytest import MonkeyPatch

    from app.services.chat.data_fetch_phase import FetchedContext

    case = {"id": "rr-1", "now": "2026-10-30T11:21:45+00:00", "received_at": "2026-10-30T11:21:31+00:00",
            "history_rows": [], "message": "好的", "current_ids": ["h1"]}
    context = {"ai_status": {"activity": "开会", "status": "very_busy", "upcoming": ""},
               "topic": {"category": "生活", "turns": 3},
               "delay": {"activity": "开会", "status": "very_busy", "seconds": 28, "reason_text": "在忙。"}}
    assert run_replay.incident(case, context)["reply_context"]["received_status"] == {
        "activity": "开会", "status": "very_busy"}
    assert "received_status" not in run_replay.incident(case, {"topic": context["topic"]})["reply_context"]
    assert "received_status" not in run_replay.incident(case)["reply_context"]

    chat = SimpleNamespace(fetch_parallel_context=AsyncMock(return_value=FetchedContext(memory_relevance="weak")),
                           push_topic=None, actual_delay_seconds=None, explain_delay_reason=None,
                           _fetch_intent_context=None, _generate_reply=AsyncMock(return_value="reply"))
    from app.services.prompting import store

    with MonkeyPatch.context() as patches:
        # 回放器 (configure_pair) 先把 store.db 换成内存桩, 钩子再改它的 find_unique
        patches.setattr(store, "db", SimpleNamespace(prompttemplate=SimpleNamespace(find_unique=None)))
        hook = run_replay.force_main({}, AsyncMock(), {"rr-1": context})
        hook(patches, chat, None, case)
        assert chat.fetch_parallel_context.return_value.ai_status == context["ai_status"]
        assert await chat.push_topic("c", "好的") == {"category": "生活", "turns": 3}
        assert chat.actual_delay_seconds({}) == 28
        assert await chat.explain_delay_reason("schedule_busy") == "在忙。"
        assert await chat._generate_reply() == "reply"
    # 没有作息/话题/延迟的用例: 三项保持回放器默认 (不渲染对应段)
    empty = SimpleNamespace(fetch_parallel_context=AsyncMock(return_value=FetchedContext(memory_relevance="weak")),
                            push_topic="stub", actual_delay_seconds="stub", explain_delay_reason="stub")
    with MonkeyPatch.context() as patches:
        run_replay.apply_production_context(patches, empty, {})
        assert empty.fetch_parallel_context.return_value.ai_status is None
        assert (empty.push_topic, empty.actual_delay_seconds, empty.explain_delay_reason) == ("stub",) * 3


def _vec(text):
    v = [0.0] * 8
    for ch in text:
        v[ord(ch) % 8] += 1
    return v


async def _embed(texts):
    return [_vec(t) for t in texts]


async def test_score_uses_production_definition(tmp_path):
    case = {"id": "rr-1", "situation": "ack", "message": "是呀", "judge_user_text": "嗯 是呀", "user": "u01",
            "kind": "positive", "tags": [], "previous_ai_turns": [["我刚还在跟导演走位呢"]]}
    rows = [{"arm": "a", "case": "rr-1", "sample": 0, "bubbles": ["我刚还在跟导演走位呢", "哈哈"],
             "responded": True, "main_calls": 1, "seconds": 1.0, "cost_cny": 0.001},
            {"arm": "a", "case": "rr-1", "sample": 1, "bubbles": ["那你快去忙吧晚点聊"],
             "responded": True, "main_calls": 1, "seconds": 1.0, "cost_cny": 0.001},
            {"arm": "a", "case": "rr-1", "sample": 2, "bubbles": ["我刚还在跟导演走位呢"],
             "responded": None, "main_calls": 1, "seconds": 1.0, "cost_cny": None}]
    calls = []

    async def judge(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("down")
        return "verbatim"

    cache = tmp_path / "cache.json"
    out = await score.score([case], rows, cache, _embed, judge)
    assert "【用户这一句】\n嗯 是呀\n" in calls[0]  # 评判看到的是 10.1 口径的用户原话
    assert [r["label"] for r in out] == ["verbatim", "none", "failed"]
    assert [r["repeated"] for r in out] == [True, False, None]  # 评判失败不当作不重复
    assert out[0]["bubble_count"] == 2 and out[0]["chars"] == len("我刚还在跟导演走位呢哈哈")
    # 缓存命中不再调用评判; 失败的那次没缓存, 重跑会补
    again = await score.score([case], rows, cache, _embed, judge)
    assert [r["label"] for r in again] == ["verbatim", "none", "verbatim"] and len(calls) == 3


def test_samples_reads_finished_details(tmp_path):
    (tmp_path / "a").mkdir()
    main = {"messages": [{"role": "system", "content": "s"}], "duration_seconds": 1.5, "usage": None}
    helper = {"messages": [{"role": "user", "content": "x"}], "duration_seconds": 0.5,
              "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
    detail = {"total": 2, "runs": [
        {"case": "rr-1", "sample": 0, "reply": "x y", "bubbles": ["x", "y"], "judge": {"pass": True},
         "calls": [main, helper, {"kind": "temporal_check"}]},
        {"case": "rr-other", "sample": 0, "reply": "z", "bubbles": ["z"], "judge": {"pass": False}, "calls": []}]}
    (tmp_path / "a" / "rr-1.json").write_text(json.dumps(detail))
    (tmp_path / "a" / "rr-2.json").write_text(json.dumps({"runs": detail["runs"]}))  # 没跑完: 不读
    rows = score.samples(tmp_path, "a", {"rr-1"})
    # 耗时只算主回复; 费用含辅助调用, 但有一次用量未知就整体未知
    assert rows == [{"arm": "a", "case": "rr-1", "sample": 0, "bubbles": ["x", "y"], "responded": True,
                     "main_calls": 1, "seconds": 1.5, "cost_cny": None}]
    detail["runs"][0].pop("bubbles")
    (tmp_path / "a" / "rr-1.json").write_text(json.dumps(detail))
    with pytest.raises(ValueError):  # 老回放器的结果不能混进来比
        score.samples(tmp_path, "a", {"rr-1"})


def _scores(arm, case, values, responded=True):
    return [{"arm": arm, "case": case, "sample": i, "repeated": v, "verbatim": False, "responded": responded,
             "chars": 10, "bubble_count": 2, "seconds": 1.0, "cost_cny": 0.001} for i, v in enumerate(values)]


def test_compare_pairs_cases_and_clusters_by_user():
    cases = [{"id": f"c{i}", "user": f"u{i % 3}", "kind": "positive" if i < 6 else "negative",
              "tags": ["ack"] if i % 2 else ["other"]} for i in range(8)]
    rows = []
    for c in cases:
        rows += _scores("new", c["id"], [False, False, True, False])
        rows += _scores("old", c["id"], [True, True, False, True])
    rows += _scores("new", "c0", [None])  # 没评到的采样不进均值
    report = compare.compare(cases, rows, "new", "old")
    diff = report["paired_diff"]["positive_repeated"]
    assert diff["diff"] == -0.5 and diff["cases"] == 6 and diff["users"] == 3
    # 每例差值都一样, bootstrap 区间缩成一点; 但每例只有 4 次采样, 采样噪声让真实区间宽得多
    assert diff["ci95_bootstrap"] == [-0.5, -0.5]
    # 每例两组合并组内方差 = (0.75 + 0.75) / 6 = 0.25, 差值方差 = 0.25 × (1/4 + 1/4) = 0.125, 6 例平均
    assert diff["se_sampling"] == pytest.approx(math.sqrt(6 * 0.125) / 6, abs=1e-4)
    assert diff["ci95"] == pytest.approx([-0.5 - 1.96 * diff["se_sampling"], -0.5 + 1.96 * diff["se_sampling"]], abs=2e-4)
    assert report["runs"] == {"new": 1, "old": 1} and report["calibration_run_sd"] == {}
    assert report["arms"]["new"]["unjudged_repetition"] == 1
    assert report["arms"]["old"]["positive_repeated"] == 0.75
    assert report["arms"]["new"]["by_tag_repeated"]["ack"]["cases"] == 4
    assert report["coverage"]["positive"] == {"cases": 6, "paired": 6, "ratio": 1.0}
    assert report["coverage"]["unequal_samples"] == ["c0"]  # new 多了一个 (没评到的) 采样
    assert report["coverage"]["min_samples"] == {"new": 4, "old": 4}
    gated = compare.gate(report)
    assert gated["checks"] == {"repetition_not_worse": True, "responded_not_worse": True,
                               "coverage": True, "equal_samples": False, "enough_runs": True,
                               "enough_samples": True, "same_calibration": True}
    assert not gated["passed"]
    rows = [r for r in rows if not (r["arm"] == "new" and r["case"] == "c0" and r["repeated"] is None)]
    assert compare.gate(compare.compare(cases, rows, "new", "old"))["passed"]
    worse = compare.compare(cases, rows, "old", "new")
    assert not compare.gate(worse)["checks"]["repetition_not_worse"]
    # 候选在难例上崩了、没出回复: 少了这些例不能反而显得更好
    crashed = [r for r in rows if not (r["arm"] == "new" and r["case"] == "c1")]
    report = compare.compare(cases, crashed, "new", "old")
    assert report["coverage"]["missing"] == {"new": ["c1"], "old": []}
    assert not compare.gate(report)["checks"]["coverage"]
    assert compare.paired(cases, {}, {}, "repeated", __import__("random").Random(0)) is None
    # 用 "+" 把两次回放并成一组: 每例采样数相加
    rows += _scores("old2", "c2", [True, False])
    pooled = compare.compare(cases, rows, "new", "old+old2")
    assert pooled["coverage"]["unequal_samples"] == ["c2"] and pooled["arms"]["old+old2"]["samples"] == 34
    assert pooled["runs"]["old+old2"] == 2
    # 回放次数按数据计: 重复写同一次回放不算多次, 点名了却没有数据的回放直接报错
    assert compare.compare(cases, rows, "new+new+new", "old")["runs"]["new+new+new"] == 1
    with pytest.raises(ValueError, match="No scores"):
        compare.compare(cases, rows, "new+typo", "old")


def test_sanitize_rules_and_shift():
    year = 2026
    text = "我电话13812345678 身份证11010119900307123X 住西湖区文三路90号3幢2单元 发我邮箱a.b@x.com 看https://t.cn/abc 10月30日见"
    out = sanitize.apply_rules(text, year)
    for secret in ("13812345678", "11010119900307123X", "文三路90号", "a.b@x.com", "https://t.cn"):
        assert secret not in out
    assert "[手机号]" in out and "[证件号]" in out and "[地址]" in out and "9月25日见" in out
    assert sanitize.apply_rules("住在翠苑小区5栋", year) == "住在[小区]"
    # 号码与网址的规则不吞中文
    assert sanitize.apply_rules("我的邮箱是abc@qq.com吧你记一下", year) == "我的邮箱是[邮箱]吧你记一下"
    assert sanitize.apply_rules("看这个https://t.cn/abc好不好", year) == "看这个[链接]好不好"
    assert sanitize.apply_rules("电话138 1234 5678或+86-13812345678", year) == "电话[手机号]或[手机号]"
    assert sanitize.apply_rules("座机0571-88886666 卡号6222 0212 3456 7890 123", year) == "座机[电话] 卡号[卡号]"
    assert sanitize.apply_rules("加我微信号：abc_12345呀 车牌浙A12345", year) == "加我微信号[账号]呀 车牌[车牌]"
    # 2026-10-30 跟着平移; 10/30 与比分分不开, 不改 (人工复核兜底)
    assert sanitize.apply_rules("2026-10-30见 10/30也行 比分3/12", year) == "2026-09-25见 10/30也行 比分3/12"
    assert sanitize.shift_text_dates("2月30号", year) == "2月30号"  # 无效日期原样保留
    moved = datetime.fromisoformat(sanitize.shift_timestamp("2026-10-30T11:21:45+00:00"))
    assert moved.weekday() == datetime(2026, 10, 30).weekday() and moved.day == 25


async def test_sanitize_model_pass_keeps_structure():
    case = {"id": "rr-1", "now": "2026-10-30T11:21:45+00:00", "received_at": "2026-10-30T11:21:31+00:00",
            "history_rows": [{"id": "h0", "role": "user", "content": "小美今天去西湖了", "createdAt": "2026-10-30T11:00:00+00:00"}],
            "message": "嗯嗯", "judge_user_text": "小美说嗯嗯", "previous_ai_turns": [["西湖好玩吗"]],
            "production_reply": ["好玩就好"]}
    texts = sanitize.rule_pass(case)
    assert texts == ["小美今天去西湖了", "嗯嗯", "小美说嗯嗯", "西湖好玩吗", "好玩就好"]

    async def complete(prompt):
        rows = json.loads(prompt.rsplit("\n\n", 1)[1])
        return json.dumps({"rows": [{"i": r["i"], "text": r["text"].replace("小美", "[人名A]")} for r in rows]})

    out = sanitize.with_texts(case, await sanitize.model_pass(texts, complete))
    assert out["history_rows"][0]["content"] == "[人名A]今天去西湖了"
    assert out["judge_user_text"] == "[人名A]说嗯嗯" and out["previous_ai_turns"] == [["西湖好玩吗"]]
    assert out["history_rows"][0]["createdAt"].startswith("2026-09-25")
    assert out["sanitization"]["review_status"] == "pending_human_review"

    async def drops_a_row(prompt):
        return json.dumps({"rows": [{"i": 0, "text": "x"}]})

    with pytest.raises(ValueError):
        await sanitize.model_pass(texts, drops_a_row)


def _run_scores(cases, runs):
    """runs: {run: {case: [repeated…]}}; 回应全为 True, 其余字段补齐。"""
    rows = []
    for run, per_case in runs.items():
        for case, values in per_case.items():
            rows += [{"arm": run, "case": case, "sample": i, "repeated": v, "verbatim": False, "responded": True,
                      "chars": 10, "bubble_count": 2, "seconds": 1.0, "cost_cny": 0.001} for i, v in enumerate(values)]
    return rows


def _aa_fixture():
    cases = [{"id": f"c{i}", "user": f"u{i % 4}", "kind": "positive" if i < 8 else "negative", "tags": []}
             for i in range(10)]
    # 同一份源码的 4 次回放: 每次每例 5 个采样, 整体均值在 0.4 与 0.2 之间来回跳
    patterns = {"r1": [1, 0, 0, 1, 0], "r2": [1, 0, 0, 0, 0], "r3": [0, 1, 1, 0, 0], "r4": [0, 0, 0, 0, 1]}
    runs = {r: {c["id"]: [bool(x) for x in p] for c in cases} for r, p in patterns.items()}
    return cases, _run_scores(cases, runs)


def test_calibrate_from_whole_runs():
    cases = [{"id": f"c{i}", "user": f"u{i % 4}", "kind": "positive" if i < 40 else "negative", "tags": []}
             for i in range(44)]
    # 同一份源码的 4 次回放, 每次每例 5 个采样; 只有 c0..c3 在各次之间不同, 整体均值 0.01 / 0.005 / 0.01 / 0.005
    def run(hits):
        return {c["id"]: [c["id"] in hits] + [False] * 4 for c in cases}
    runs = {"r1": run({"c0", "c1"}), "r2": run({"c2"}), "r3": run({"c1", "c3"}), "r4": run({"c0"})}
    cal = compare.calibrate(cases, _run_scores(cases, runs), [["r1", "r2", "r3", "r4"]], runs_per_arm=4)
    rep = cal["positive_repeated"]
    assert rep["run_means"] == {"r1": 0.01, "r2": 0.005, "r3": 0.01, "r4": 0.005} and rep["df"] == 3
    assert rep["run_sd"] == pytest.approx(0.0029, abs=1e-4) and rep["cases"] == 40
    # 预期半宽: 用例层 (1.96) 与超出采样噪声的整次回放层 (t, 自由度 3) 方差相加; 4 次回放时除以 2
    extra = math.sqrt(max(2 * rep["run_sd"] ** 2 - rep["se_sampling_one_run_pair"] ** 2, 0))
    half = math.hypot(1.96 * rep["se_case_one_run_pair"], (compare.t975(3) if extra else 1.96) * extra) / 2
    assert rep["expected_half_width"] == pytest.approx(half, abs=1e-4)
    assert rep["margin"] == compare._ceil_half_point(2 * half)
    assert cal["all_responded"]["run_sd"] == 0.0 and cal["all_responded"]["cases"] == 44
    with pytest.raises(ValueError):
        compare.calibrate(cases, _run_scores(cases, runs), [["r1"]], runs_per_arm=4)  # 一组至少两次回放
    with pytest.raises(ValueError):
        compare.calibrate(cases, _run_scores(cases, runs), [], runs_per_arm=4)
    doc = compare.calibration_document(cal, {"runs_per_arm": 4})
    assert doc["metrics"] is cal and doc["rule"]["factor"] == 2 and doc["runs_per_arm"] == 4


def test_calibrate_rejects_noise_too_large():
    cases, scores = _aa_fixture()  # 4 次回放整体均值在 0.2 与 0.4 之间来回跳
    with pytest.raises(ValueError, match="too large"):
        compare.calibrate(cases, scores, [["r1", "r2", "r3", "r4"]], runs_per_arm=4)


CAL = {"runs_per_arm": 4, "samples_per_run": 5,
       "metrics": {"positive_repeated": {"margin": 0.06, "run_sd": 0.02, "df": 4},
                   "all_responded": {"margin": 0.015, "run_sd": 0.003, "df": 4}}}


def _report(rep_ci, resp_ci, runs=4, samples=20, run_sd=None):
    return {"paired_diff": {"positive_repeated": {"diff": 0.0, "ci95": rep_ci},
                            "all_responded": {"diff": 0.0, "ci95": resp_ci}},
            "runs": {"a": runs, "b": runs},
            "calibration_run_sd": run_sd if run_sd is not None else {"positive_repeated": 0.02, "all_responded": 0.003},
            "coverage": {"all": {"ratio": 1.0}, "positive": {"ratio": 1.0}, "unequal_samples": [],
                         "min_samples": {"a": samples, "b": samples}}}


def test_t_quantiles_round_down_in_degrees_of_freedom():
    assert compare.t975(4) == 2.776 and compare.t975(11) == 2.228 and compare.t975(40) == 1.96
    with pytest.raises(ValueError):
        compare.t975(0)


def test_gate_uses_and_validates_calibration():
    gated = compare.gate(_report([-0.02, 0.055], [-0.01, 0.01]), CAL)
    assert gated["passed"] and gated["calibrated"] and gated["thresholds"]["samples_per_case"] == 20
    assert not compare.gate(_report([0.0, 0.065], [-0.01, 0.01]), CAL)["checks"]["repetition_not_worse"]
    assert not compare.gate(_report([-0.02, 0.0], [-0.02, 0.01]), CAL)["checks"]["responded_not_worse"]
    # 回放次数或采样不够 / 比较时没用这份标定: 界值不成立
    assert not compare.gate(_report([-0.02, 0.0], [-0.01, 0.01], runs=2), CAL)["checks"]["enough_runs"]
    assert not compare.gate(_report([-0.02, 0.0], [-0.01, 0.01], samples=5), CAL)["checks"]["enough_samples"]
    assert not compare.gate(_report([-0.02, 0.0], [-0.01, 0.01], run_sd={}), CAL)["checks"]["same_calibration"]
    # 没有标定: 先验界值, 明确标出未标定
    assert compare.gate(_report([-0.02, 0.0], [-0.01, 0.01]))["calibrated"] is False
    # 标定内容有问题直接报错, 不悄悄退回先验值
    good = {"margin": 0.01, "run_sd": 0.0, "df": 4}
    for bad in ({}, {"metrics": {}}, {"metrics": {"positive_repeated": good}},
                {"metrics": {"positive_repeated": {**good, "margin": 0.2}, "all_responded": good}},
                {"metrics": {"positive_repeated": {**good, "df": 0}, "all_responded": good}},
                {"metrics": {"positive_repeated": {"margin": 0.05}, "all_responded": good}}):
        with pytest.raises(ValueError):
            compare.gate(_report([-0.02, 0.0], [-0.01, 0.01]), bad)


def test_compare_adds_run_noise_beyond_sampling():
    cases, scores = _aa_fixture()
    plain = compare.compare(cases, scores, "r1+r2", "r3+r4")["paired_diff"]["positive_repeated"]
    run_sd = 0.3  # 远大于逐例采样噪声: 区间应由整次回放层主导
    cal = {"metrics": {"positive_repeated": {"margin": 0.06, "run_sd": run_sd, "df": 4},
                       "all_responded": {"margin": 0.015, "run_sd": 0.0, "df": 4}}}
    noisy = compare.compare(cases, scores, "r1+r2", "r3+r4", cal)
    widened = noisy["paired_diff"]["positive_repeated"]
    run_se = run_sd * math.sqrt(1 / 2 + 1 / 2)
    case_se = max(plain["se_sampling"], (plain["ci95_bootstrap"][1] - plain["ci95_bootstrap"][0]) / 3.92)
    half = math.hypot(1.96 * case_se, 2.776 * math.sqrt(run_se ** 2 - widened["se_sampling"] ** 2))
    assert widened["se_runs"] == pytest.approx(run_se, abs=1e-4)
    assert widened["ci95"][1] - widened["diff"] == pytest.approx(half, abs=2e-4)
    assert noisy["calibration_run_sd"] == {"positive_repeated": 0.3, "all_responded": 0.0}
    # 整次回放的波动比采样噪声小时不再加宽
    small = {"metrics": {m: {**v, "run_sd": 0.0} for m, v in cal["metrics"].items()}}
    assert compare.compare(cases, scores, "r1+r2", "r3+r4", small)["paired_diff"]["positive_repeated"]["ci95"] == \
        plain["ci95"]


def test_committed_calibration_matches_its_inputs():
    repo = json.loads(compare.DEFAULT_CALIBRATION.read_text())
    assert repo["runs_per_arm"] >= 4 and repo["samples_per_run"] == 5 and repo["rule"]["factor"] == compare.AA_FACTOR
    for metric, entry in repo["metrics"].items():
        groups = entry["aa_groups"]
        ss = sum(sum((entry["run_means"][r] - sum(entry["run_means"][x] for x in g) / len(g)) ** 2 for r in g)
                 for g in groups)
        assert entry["df"] == sum(len(g) - 1 for g in groups)
        assert math.sqrt(ss / entry["df"]) == pytest.approx(entry["run_sd"], abs=2e-4)
        extra = math.sqrt(max(2 * entry["run_sd"] ** 2 - entry["se_sampling_one_run_pair"] ** 2, 0))
        half = math.hypot(1.96 * entry["se_case_one_run_pair"],
                          (compare.t975(entry["df"]) if extra else 1.96) * extra) / math.sqrt(repo["runs_per_arm"])
        assert half == pytest.approx(entry["expected_half_width"], abs=2e-4)
        assert entry["margin"] == compare._ceil_half_point(2 * entry["expected_half_width"]) <= compare.MAX_MARGIN
