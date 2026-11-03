"""CHAT01.10.2 重复表述回放评测的纯函数部分: 用例构建、打分口径、配对比较、脱敏规则。"""
from __future__ import annotations

import json
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
    assert diff["ci95"] == [-0.5, -0.5]
    assert report["arms"]["new"]["unjudged_repetition"] == 1
    assert report["arms"]["old"]["positive_repeated"] == 0.75
    assert report["arms"]["new"]["by_tag_repeated"]["ack"]["cases"] == 4
    assert report["coverage"]["positive"] == {"cases": 6, "paired": 6, "ratio": 1.0}
    assert report["coverage"]["unequal_samples"] == ["c0"]  # new 多了一个 (没评到的) 采样
    gated = compare.gate(report)
    assert gated["checks"] == {"repetition_not_worse": True, "responded_not_worse": True,
                               "coverage": True, "equal_samples": False}
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
