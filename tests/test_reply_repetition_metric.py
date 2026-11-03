"""CHAT01.10.1 主回复重复表述指标: 分轮、候选、评判汇总与按日聚合 (真实 SQL)。"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from app.services.effect import daily_facts, repetition
from tests.test_beta_daily_facts_postgres import (  # noqa: F401
    DAY, _at, _conversation, _now, _user, facts_refresh_redis, pg,
)

MAIN = {"response_diagnostics": {"reply_path": "main_llm"}}


def _row(i, role, content, at, **meta):
    return {"id": f"m{i}", "conversation_id": "c", "user_id": "u", "role": role, "content": content,
            "created_at": at, "metadata": meta.get("metadata"), "trace_id": meta.get("trace_id"),
            "reply_path": meta.get("reply_path"), "proactive": meta.get("proactive", False)}


T0 = datetime(2027, 3, 12, 2, 0)


def _sec(s):
    return T0 + timedelta(seconds=s)


def test_turns_split_by_trace_and_skip_system_lines():
    rows = [
        _row(1, "user", "转塘是哪里啊", _sec(0)),
        _row(2, "user", "不过听起来是看日落的好地方", _sec(4)),
        _row(3, "assistant", "是杭州西湖区的一个街道呀", _sec(13), trace_id="t1", reply_path="main_llm"),
        _row(4, "assistant", "有好多好看的日落点", _sec(13), reply_path="main_llm"),
        # 上一条还没回完又回了一轮: 带新 trace_id, 分成新的一轮
        _row(5, "assistant", "对呀 好多地方看日落都绝了", _sec(20), trace_id="t2", reply_path="main_llm"),
        _row(6, "assistant", "你们已退出共听《X》", _sec(21), metadata={"kind": "music_status"}),
        _row(7, "user", "是呀", _sec(60)),
        _row(8, "assistant", "对啊 好多地方看日落都绝了", _sec(70), trace_id="t3", reply_path="main_llm"),
    ]
    turns = repetition.group_turns(rows)
    assert [(t.role, [r["id"] for r in t.rows]) for t in turns] == [
        ("user", ["m1", "m2"]), ("assistant", ["m3", "m4"]), ("assistant", ["m5"]),
        ("user", ["m7"]), ("assistant", ["m8"])]
    scored = repetition.eligible_turns(turns, T0, T0 + timedelta(days=1))
    assert [(s.turn.rows[0]["id"], s.situation) for s in scored] == [("m5", "overlap"), ("m8", "ack")]
    # 第一轮主回复之前没有 AI 发言, 不进分母
    assert all(s.turn.rows[0]["id"] != "m3" for s in scored)


def test_previous_window_and_short_bubbles():
    rows = [
        _row(1, "assistant", "早上好呀今天天气不错", _sec(0), trace_id="a", reply_path="main_llm"),
        _row(2, "user", "嗯", _sec(25 * 3600)),
        _row(3, "assistant", "早上好呀今天天气不错", _sec(25 * 3600 + 10), trace_id="b", reply_path="main_llm"),
        _row(4, "user", "好", _sec(25 * 3600 + 60)),
        _row(5, "assistant", "哈哈", _sec(25 * 3600 + 70), trace_id="c", reply_path="main_llm"),
    ]
    scored = repetition.eligible_turns(repetition.group_turns(rows), T0, T0 + timedelta(days=3))
    # 第 3 行离上一次 AI 发言超过 24 小时: 不算; 第 5 行算, 但"哈哈"太短不参与比较
    assert [s.turn.rows[0]["id"] for s in scored] == ["m5"]
    assert scored[0].bubbles == []


def _vec(text: str) -> list[float]:
    """假向量: 相邻两字的哈希计数 (跨进程稳定); 原样相同的句子余弦为 1。"""
    key = repetition.text_key(text)
    v = [0.0] * 64
    for i in range(len(key) - 1):
        v[hashlib.md5(key[i:i + 2].encode()).digest()[0] % 64] += 1
    return v if any(v) else [1.0] + [0.0] * 63


async def _embed(texts):
    return [_vec(t) for t in texts]


async def _judge(prompt):
    bubble = prompt.split("【要判断的那条气泡】\n", 1)[1].split("\n", 1)[0]
    matched = prompt.split("【AI 之前说过的那句】（和要判断的气泡最像的一句）\n", 1)[1].split("\n", 1)[0]
    if "坏" in bubble:
        raise RuntimeError("judge down")
    return "verbatim" if repetition.text_key(bubble) == repetition.text_key(matched) else "not_repeat"


def test_candidates_and_turn_label():
    rows = [
        _row(1, "assistant", "我刚还在跟导演走位呢", _sec(0), trace_id="a", reply_path="main_llm"),
        _row(2, "user", "现在又没啥朋友", _sec(30)),
        _row(3, "assistant", "那也太孤单了吧", _sec(40), trace_id="b", reply_path="main_llm"),
        _row(4, "assistant", "我刚还在跟导演走位呢", _sec(40), reply_path="main_llm"),
    ]
    score = repetition.eligible_turns(repetition.group_turns(rows), T0, T0 + timedelta(days=1))[0]
    vectors = {t: _vec(t) for t in ["我刚还在跟导演走位呢", "那也太孤单了吧"]}
    repetition.find_candidates(score, vectors)
    assert [c.bubble for c in score.candidates] == ["我刚还在跟导演走位呢"]
    assert math.isclose(score.max_similarity, 1.0)
    assert repetition.turn_label(score) == "failed"  # 有候选但还没评判: 不能当作不重复
    score.labels = ["not_repeat", "verbatim"]
    assert repetition.turn_label(score) == "verbatim"
    assert "我刚还在跟导演走位呢" in repetition.judge_prompt(score, score.candidates[0])


def test_reply_turn_id_groups_sub_intent_replies():
    rows = [
        _row(1, "assistant", "早上好呀今天天气不错", _sec(0), trace_id="a", reply_path="main_llm"),
        _row(2, "user", "今天去哪玩 顺便帮我记下明天开会", _sec(20)),
        _row(3, "assistant", "可以去西湖边走走", _sec(30), trace_id="b", reply_path="main_llm",
             metadata={"achievement_turn_id": "u2"}),
        # 多意图的子回复晚 8 秒发出, 带同一个 trace_id 和回应对象: 仍是同一次回复
        _row(4, "assistant", "开会的事我记下啦", _sec(38), trace_id="b", metadata={"achievement_turn_id": "u2"}),
        _row(5, "user", "好", _sec(40)),
        _row(6, "user", "对了", _sec(41)),
        # 两次回复挨得很近, 但回应的用户消息不同: 分开
        _row(7, "assistant", "嗯嗯", _sec(45), trace_id="c", reply_path="main_llm",
             metadata={"achievement_turn_id": "u5"}),
        _row(8, "assistant", "你说", _sec(46), trace_id="c", metadata={"achievement_turn_id": "u6"}),
        # 紧跟着的主动消息不是这次回复的一部分
        _row(9, "assistant", "对了今天降温", _sec(47), proactive=True),
    ]
    turns = repetition.group_turns(rows)
    assert [[r["id"] for r in t.rows] for t in turns] == [
        ["m1"], ["m2"], ["m3", "m4"], ["m5", "m6"], ["m7"], ["m8"], ["m9"]]


def test_turn_label_never_reads_unresolved_as_not_repeated():
    score = repetition.TurnScore(repetition.Turn("assistant", []), "other", "", [], ["甲乙丙丁"])
    score.candidates = [repetition.Candidate("甲乙丙丁", 0.7, "x"), repetition.Candidate("戊己庚辛", 0.8, "y")]
    score.labels = ["not_repeat"]  # 另一条被上限挡掉没评
    assert repetition.turn_label(score) == "failed"
    score.labels = ["not_repeat", "unjudgeable"]
    assert repetition.turn_label(score) == "unjudgeable"
    score.labels = ["failed", "paraphrase"]  # 有一条判为重复就足以定论
    assert repetition.turn_label(score) == "paraphrase"
    score.labels = ["legit", "not_repeat"]
    assert repetition.turn_label(score) == "legit"
    score.unembedded = True
    assert repetition.turn_label(score) == "unjudgeable"


async def _msg(pg, conv, role, at, content, metadata=None):
    ident = uuid4().hex
    await pg.db.execute_raw(
        "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)"
        " VALUES ($1, $2, $3, $4, $5::jsonb, $6::timestamp)",
        ident, conv, role, content, json.dumps(metadata) if metadata is not None else None, at.isoformat())
    return ident


async def test_score_aggregate_review_and_api(pg, facts_refresh_redis, monkeypatch):
    from app.api.admin import stats
    from app.services.auth import create_jwt

    external = await _user(pg)
    admin = await _user(pg, role="admin")
    conv, _ = await _conversation(pg, external)
    await _msg(pg, conv, "user", _at(DAY, 10), "转塘是哪里啊")
    await _msg(pg, conv, "assistant", _at(DAY, 10, 0, 10), "我刚还在跟导演走位呢", {**MAIN, "trace_id": "t1"})
    await _msg(pg, conv, "user", _at(DAY, 10, 1), "现在又没啥朋友")
    repeated_turn = await _msg(pg, conv, "assistant", _at(DAY, 10, 1, 10), "那也太孤单了吧",
                               {**MAIN, "trace_id": "t2"})
    await _msg(pg, conv, "assistant", _at(DAY, 10, 1, 12), "我刚还在跟导演走位呢", MAIN)
    await _msg(pg, conv, "user", _at(DAY, 10, 2), "是呀")
    await _msg(pg, conv, "assistant", _at(DAY, 10, 2, 10), "坏掉的评判这句会失败吗", {**MAIN, "trace_id": "t3"})
    await _msg(pg, conv, "user", _at(DAY, 10, 3), "嗯嗯")
    await _msg(pg, conv, "assistant", _at(DAY, 10, 3, 10), "那你下班早点休息呀", {**MAIN, "trace_id": "t4"})
    now = _now(DAY + timedelta(days=1), 4)
    try:
        # 还没打分: 指标不出行 (页面显示未运行, 不是 0)
        await daily_facts.compute_day(DAY, now=now)
        await pg.db.execute_raw("DELETE FROM reply_repetition_runs WHERE local_date=$1::date", str(DAY))
        await daily_facts.compute_day(DAY, now=now)
        rows = await pg.db.query_raw(
            "SELECT 1 FROM metrics_daily WHERE local_date=$1::date AND metric='reply_repetition_rate'", str(DAY))
        assert rows == []
        # 按情形拆的线也是 "未运行", 不能因为当天别的统计算过就补成 0/0 "无样本"
        monkeypatch.setattr("app.services.runtime.tasks.fire_background", lambda coro: coro.close())
        unrun = await daily_facts.read_series(DAY, DAY, cohort="all", metrics=["reply_repetition_rate"],
                                              dimension="situation", now=now)
        assert {p["state"] for p in unrun["series"]["reply_repetition_rate"]} == {"not_computed"}

        result = await repetition.score_day(DAY, now=now, embedder=_embed, judge=_judge)
        # 共享测试库里同一天可能有别的用例的数据: 只核对本会话的行
        assert result["status"] == "complete" and result["failed"] == 0
        scores = await pg.db.query_raw(
            "SELECT turn_message_id, situation, label, repeated FROM reply_repetition_scores"
            " WHERE local_date=$1::date AND conversation_id=$2 ORDER BY occurred_at", str(DAY), conv)
        assert [(s["situation"], s["label"], s["repeated"]) for s in scores] == [
            ("other", "verbatim", True), ("ack", "none", False), ("ack", "none", False)]
        assert scores[0]["turn_message_id"] == repeated_turn

        await daily_facts.compute_day(DAY, now=now)
        metric = {(r["dimension_value"]): r for r in await pg.db.query_raw(
            "SELECT dimension_value, numerator, denominator, settled FROM metrics_daily"
            " WHERE local_date=$1::date AND metric='reply_repetition_rate' AND cohort='all'", str(DAY))}
        truth = {r["situation"]: r for r in await pg.db.query_raw(
            "SELECT COALESCE(situation, '') AS situation, COUNT(*)::int AS n,"
            " COUNT(*) FILTER (WHERE repeated)::int AS rep FROM reply_repetition_scores"
            " WHERE local_date=$1::date GROUP BY GROUPING SETS ((), (situation))", str(DAY))}
        assert metric[""]["settled"] and metric[""]["denominator"] == truth[""]["n"]
        assert metric[""]["numerator"] == truth[""]["rep"] >= 1
        assert metric["ack"]["denominator"] == truth["ack"]["n"] >= 2
        review = await pg.db.query_raw(
            "SELECT source_id FROM stats_review_items WHERE local_date=$1::date AND kind='reply_repetition'",
            str(DAY))
        assert repeated_turn in [r["source_id"] for r in review]

        monkeypatch.setattr(stats, "db", pg.db)
        from app.services.effect import definitions
        monkeypatch.setattr(definitions, "local_today", lambda now=None: DAY + timedelta(days=1))
        monkeypatch.setattr(stats, "local_today", definitions.local_today)
        monkeypatch.setattr("app.services.runtime.tasks.fire_background", lambda coro: coro.close())
        app = FastAPI()
        app.include_router(stats.router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            client.headers["Authorization"] = "Bearer " + create_jwt(admin, "admin")
            params = {"days": 1, "end": str(DAY), "cohort": "all"}
            ok = await client.get("/admin-api/stats/daily-metrics",
                                  params={**params, "metrics": "reply_repetition_rate", "dimension": "situation"})
            assert ok.status_code == 200, ok.text[:300]
            assert "导演" not in ok.text
            items = await client.get("/admin-api/stats/review-items", params={**params, "kind": "reply_repetition"})
            assert items.status_code == 200
            assert repeated_turn in [i["id"] for i in items.json()["items"]]
            assert "导演" not in items.text
            bad = await client.get("/admin-api/stats/daily-metrics",
                                   params={**params, "metrics": "user_messages", "dimension": "situation"})
            assert bad.status_code == 422

        # 评判失败: 当天记为不完整, 指标不定型
        async def broken(prompt):
            raise RuntimeError("judge down")

        partial = await repetition.score_day(DAY, now=now, embedder=_embed, judge=broken)
        assert partial["status"] == "partial" and partial["failed"] >= 1
        await daily_facts.compute_day(DAY, now=now)
        settled = await pg.db.query_raw(
            "SELECT settled FROM metrics_daily WHERE local_date=$1::date AND metric='reply_repetition_rate'"
            " AND cohort='external' AND dimension=''", str(DAY))
        assert settled == [{"settled": False}]
        assert await repetition.days_needing_scores(DAY, DAY) == [DAY]
        # 连着失败到重试上限: 不再重试, 没评到的轮次不进分母, 当天定型
        for _ in range(repetition.MAX_ATTEMPTS - 1):
            await repetition.score_day(DAY, now=now, embedder=_embed, judge=broken)
        assert await repetition.days_needing_scores(DAY, DAY) == []
        await daily_facts.compute_day(DAY, now=now)
        overall = await pg.db.query_raw(
            "SELECT denominator, settled FROM metrics_daily WHERE local_date=$1::date"
            " AND metric='reply_repetition_rate' AND cohort='all' AND dimension=''", str(DAY))
        failed_turns = await pg.db.query_raw(
            "SELECT COUNT(*)::int AS n FROM reply_repetition_scores WHERE local_date=$1::date AND label='failed'",
            str(DAY))
        assert failed_turns[0]["n"] >= 1 and overall[0]["settled"]
        assert overall[0]["denominator"] == truth[""]["n"] - failed_turns[0]["n"]

        # 内容审核拒收: 归为 unjudgeable, 不算失败, 当天照常完整
        async def rejecting(prompt):
            raise repetition.Unjudgeable()

        rejected = await repetition.score_day(DAY, now=now, embedder=_embed, judge=rejecting)
        assert rejected["status"] == "complete" and rejected["unjudgeable"] >= 1 and rejected["failed"] == 0

        # 向量服务拒收一条: 整批失败后逐条重来, 只有那条没有向量, 它所在的轮次不下结论
        async def picky(texts):
            if "那你下班早点休息呀" in texts:
                raise repetition.Unjudgeable()
            return await _embed(texts)

        await repetition.score_day(DAY, now=now, embedder=picky, judge=_judge)
        labels = await pg.db.query_raw(
            "SELECT label FROM reply_repetition_scores WHERE conversation_id=$1 ORDER BY occurred_at", conv)
        assert [r["label"] for r in labels] == ["verbatim", "none", "unjudgeable"]
        runs = await pg.db.query_raw("SELECT attempts FROM reply_repetition_runs WHERE local_date=$1::date", str(DAY))
        assert runs == [{"attempts": 1}]

        # 超出当天上限: 截断也记为不完整, 不能把没评到的当作不重复; 结果确定, 不再重试
        closed = []

        async def close():
            closed.append(True)

        monkeypatch.setattr(repetition, "dashscope_clients", lambda: (_embed, _judge, close))
        monkeypatch.setattr(repetition, "MAX_TURNS_PER_DAY", 1)
        monkeypatch.setattr(repetition, "MAX_JUDGED_PER_DAY", 0)
        capped = await repetition.score_day(DAY, now=now)
        assert capped["status"] == "partial" and capped["truncated"] and capped["judged"] == 0
        assert closed == [True] and await repetition.days_needing_scores(DAY, DAY) == []
    finally:
        await pg.db.execute_raw("DELETE FROM reply_repetition_scores WHERE local_date=$1::date", str(DAY))
        await pg.db.execute_raw("DELETE FROM reply_repetition_runs WHERE local_date=$1::date", str(DAY))
        await pg.db.execute_raw("DELETE FROM stats_review_items WHERE local_date=$1::date", str(DAY))


def test_candidates_skip_bubbles_without_vectors_or_history():
    rows = [
        _row(1, "assistant", "哈哈", _sec(0), trace_id="a", reply_path="main_llm"),
        _row(2, "user", "嗯嗯", _sec(30)),
        _row(3, "assistant", "那也太孤单了吧", _sec(40), trace_id="b", reply_path="main_llm"),
    ]
    score = repetition.eligible_turns(repetition.group_turns(rows), T0, T0 + timedelta(days=1))[0]
    # 之前只有太短的气泡: 没有可比的, 不出候选也不出相似度
    repetition.find_candidates(score, {"那也太孤单了吧": _vec("那也太孤单了吧")})
    assert score.candidates == [] and score.max_similarity is None


async def test_dashscope_clients_record_usage_and_reject_bad_labels(monkeypatch):
    import openai

    from app.services.llm import usage_tracker

    replies = iter(['{"label": "paraphrase"}', '{"label": "failed"}', "{}", "inspect", "timeout"])
    closed = []

    class _Fake:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 60
            self.embeddings = self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            if "input" in kwargs:
                assert kwargs["model"] == repetition.EMBED_MODEL
                if kwargs["input"] == ["拒收"]:
                    raise RuntimeError("Error code: 400 - data_inspection_failed")
                if kwargs["input"] == ["网络"]:
                    raise RuntimeError("connection reset")
                return type("R", (), {"data": [type("D", (), {"embedding": [0.1]})() for _ in kwargs["input"]],
                                      "usage": type("U", (), {"total_tokens": 7})()})()
            assert kwargs["model"] == repetition.JUDGE_MODEL and kwargs["temperature"] == 0
            content = next(replies)
            if content == "inspect":
                raise RuntimeError("data_inspection_failed: Input data may contain inappropriate content.")
            if content == "timeout":
                raise RuntimeError("Request timed out.")
            message = type("M", (), {"content": content})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()],
                                  "usage": type("U", (), {"prompt_tokens": 30, "completion_tokens": 5})()})()

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(openai, "AsyncOpenAI", _Fake)
    embed, judge, close = repetition.dashscope_clients()
    token = usage_tracker.start_session()
    try:
        assert await embed(["a", "b"]) == [[0.1], [0.1]]
        assert await judge("p") == "paraphrase"
        # 评判给出内部占位标签或缺标签: 当作失败, 不当作"不重复"
        for _ in range(2):
            with pytest.raises(ValueError):
                await judge("p")
        # 内容审核拒收: 重试也一样, 单独归类; 其他错误照常往上抛 (可重试)
        with pytest.raises(repetition.Unjudgeable):
            await judge("p")
        with pytest.raises(RuntimeError):
            await judge("p")
        with pytest.raises(repetition.Unjudgeable):
            await embed(["拒收"])
        with pytest.raises(RuntimeError):
            await embed(["网络"])
    finally:
        usage = usage_tracker.flush_session(token)
    assert usage["tokens_by_model"]["dashscope/text-embedding-v4"]["input"] == 7
    assert usage["tokens_by_model"]["dashscope/qwen3.5-plus"]["output"] == 15
    await close()
    assert closed == [True]


async def test_load_rows_parses_text_columns_and_caps_rows(monkeypatch):
    class _Client:
        async def query_raw(self, sql, *args):
            assert args[3] == 3  # 多取一行用于判断截断
            return [{"conversation_id": "c", "created_at": "2027-03-12T02:00:00+00:00",
                     "metadata": '{"kind": "x"}'} for _ in range(3)]

    monkeypatch.setattr(repetition, "MAX_ROWS_PER_DAY", 2)
    by_conv, truncated = await repetition._load_rows(T0, T0 + timedelta(days=1), _Client())
    assert truncated and len(by_conv["c"]) == 2
    assert by_conv["c"][0]["created_at"] == T0 and by_conv["c"][0]["metadata"] == {"kind": "x"}


async def test_run_nightly_heals_old_days_and_reports_failures(monkeypatch):
    from datetime import date

    today = date(2027, 3, 20)
    monkeypatch.setattr(repetition.defs, "local_today", lambda now=None: today)

    async def needing(start, end, client=None):
        assert (start, end) == (date(2027, 3, 13), date(2027, 3, 19))
        return [date(2027, 3, 14), date(2027, 3, 15)]

    async def score(day, now=None):
        if day == date(2027, 3, 15):
            raise RuntimeError("db down")
        return {"day": day.isoformat(), "status": "complete"}

    computed = []

    async def compute_days(days, now=None):
        computed.append(days)
        return {"computed": [d.isoformat() for d in days], "failed": []}

    monkeypatch.setattr(repetition, "days_needing_scores", needing)
    monkeypatch.setattr(repetition, "score_day", score)
    monkeypatch.setattr(daily_facts, "compute_days", compute_days)
    with pytest.raises(RuntimeError, match="2027-03-15"):
        await repetition.run_nightly()
    # 打分成功的日子都立刻重算 (不等 04:40 那次); 失败的那天不算
    assert computed == [[date(2027, 3, 19), date(2027, 3, 14)]]

    async def needing_none(start, end, client=None):
        return []

    monkeypatch.setattr(repetition, "days_needing_scores", needing_none)
    result = await repetition.run_nightly()
    assert [r["day"] for r in result["days"]] == ["2027-03-19"] and result["recomputed"] == ["2027-03-19"]


async def test_scheduler_reports_scoring_failure(monkeypatch):
    from jobs import scheduler

    async def run_job(name, ttl, body, **kwargs):
        assert name == "reply_repetition_scoring"
        await body()

    async def broken(**kwargs):
        raise RuntimeError("judge down")

    failures = []
    monkeypatch.setattr(scheduler, "_run_distributed_job", run_job)
    monkeypatch.setattr(scheduler, "_job_failed", lambda label, exc: failures.append(label))
    monkeypatch.setattr(repetition, "run_nightly", broken)
    await scheduler._run_reply_repetition_scoring()
    assert failures == ["Reply repetition scoring"]
