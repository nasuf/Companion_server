"""《主动聊天机制（新增）》A/B 双模式端到端验证 —— 真 Postgres + 真 Redis, 仅桩掉 LLM.

单测把 db / redis 都 mock 掉了, 验证不到 SQL 本身 (CAS、jsonb 合并、INSERT…WHERE
NOT EXISTS 守卫、判定窗的 claim/推进)。这个脚本在一套**隔离的本地库**上把完整
链路跑一遍: 用户发消息 → AI 回复收尾 arm → 时间推进 → 真实 scan → 判定 → B 追问
/ A 模式 → 用户回来 → 聊天侧读承接 cue → 真实 build_system_prompt。

只桩三类东西: LLM 调用 (脚本化输出)、TTS / 推送通知、链接卡联网搜索。

用法 (绝不能指向生产/共享库, 脚本会拒绝非 localhost 且库名不是 companion_e2e 的连接):

    docker run -d --name e2e-pg -e POSTGRES_USER=e2e -e POSTGRES_PASSWORD=e2e \
        -e POSTGRES_DB=companion_e2e -p 55432:5432 pgvector/pgvector:pg16
    docker run -d --name e2e-redis -p 56379:6379 redis:7-alpine
    docker exec e2e-pg psql -U e2e -d companion_e2e \
        -c "CREATE SCHEMA IF NOT EXISTS extensions; CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;"
    # 按真实 migration 历史建库 (migrate diff 生成的 DDL 缺列默认值, 与生产不一致)
    DATABASE_URL=postgresql://e2e:e2e@localhost:55432/companion_e2e \
    DIRECT_DATABASE_URL=postgresql://e2e:e2e@localhost:55432/companion_e2e \
        .venv/bin/prisma migrate deploy
    E2E_DATABASE_URL=postgresql://e2e:e2e@localhost:55432/companion_e2e \
    E2E_REDIS_URL=redis://localhost:56379/0 \
        .venv/bin/python scripts/e2e_proactive_ab_mode.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

# ── 环境隔离: 必须在 import app 之前 ──────────────────────────────────────
_DB = os.environ.get("E2E_DATABASE_URL", "")
_REDIS = os.environ.get("E2E_REDIS_URL", "")
for _url, _name in ((_DB, "E2E_DATABASE_URL"), (_REDIS, "E2E_REDIS_URL")):
    if urlparse(_url).hostname not in ("localhost", "127.0.0.1"):
        sys.exit(f"{_name} 必须指向本地隔离实例 (localhost), 当前: {_url!r}")
os.environ.update(
    DATABASE_URL=_DB,
    DIRECT_DATABASE_URL=_DB,
    REDIS_URL=_REDIS,
    APP_ENV="development",
    TRACE_BACKEND="off",
    LANGSMITH_TRACING="false",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import AsyncMock, patch  # noqa: E402

from app.db import connect_db, db, disconnect_db  # noqa: E402
from app.redis_client import get_redis  # noqa: E402
from app.services.interaction import topic_continuity as tc  # noqa: E402
from app.services.proactive.state import ARM_REASON_REPLY  # noqa: E402
from app.services.runtime import tasks as bg_tasks  # noqa: E402

UTC = timezone.utc
_failures: list[str] = []
_passes = 0


def check(cond: bool, label: str) -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  ✓ {label}")
    else:
        _failures.append(label)
        print(f"  ✗ {label}")


# ── LLM 脚本 ───────────────────────────────────────────────────────────

class Script:
    """按调用顺序返回预设输出, 并记录 prompt (用于断言模板确实从 registry 取到)."""

    def __init__(self):
        self.judge: deque = deque()
        self.followup: deque = deque()
        self.a_mode: deque = deque()
        self.prompts: list[tuple[str, str]] = []
        self.on_a_mode_call = None  # 模拟"生成期间用户回来"

    async def judge_json(self, _model, prompt):
        self.prompts.append(("judge", prompt))
        return self.judge.popleft() if self.judge else {"status": "finished"}

    async def followup_text(self, _model, prompt):
        self.prompts.append(("followup", prompt))
        return self.followup.popleft() if self.followup else "SKIP"

    async def a_mode_text(self, _model, prompt):
        self.prompts.append(("a_mode", prompt))
        if self.on_a_mode_call:
            await self.on_a_mode_call()
        return self.a_mode.popleft() if self.a_mode else "最近怎么样呀"

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.prompts if k == kind)


SCRIPT = Script()


# ── 夹具: 用户 / agent / workspace / 会话 ─────────────────────────────────

class World:
    def __init__(self, tag: str):
        self.tag = tag
        self.user_id = str(uuid.uuid4())
        self.agent_id = str(uuid.uuid4())
        self.ws_id = str(uuid.uuid4())
        self.conv_id = str(uuid.uuid4())

    async def create(self):
        now = _naive(datetime.now(UTC))
        await db.execute_raw(
            'INSERT INTO users (id, username, updated_at) VALUES ($1, $2, $3::timestamp)',
            self.user_id, f"e2e-{self.tag}-{self.user_id[:6]}", now,
        )
        await db.execute_raw(
            "INSERT INTO ai_agents (id, name, user_id, updated_at, mbti) "
            "VALUES ($1, '小伴', $2, $3::timestamp, $4::jsonb)",
            self.agent_id, self.user_id, now,
            json.dumps({"EI": 78, "NS": 70, "TF": 30, "JP": 40, "type": "ENFP"}),
        )
        await db.execute_raw(
            "INSERT INTO chat_workspaces (id, user_id, agent_id, status, updated_at) "
            "VALUES ($1, $2, $3, 'active', $4::timestamp)",
            self.ws_id, self.user_id, self.agent_id, now,
        )
        await db.execute_raw(
            "INSERT INTO conversations (id, user_id, agent_id, workspace_id, updated_at) "
            "VALUES ($1, $2, $3, $4, $5::timestamp)",
            self.conv_id, self.user_id, self.agent_id, self.ws_id, now,
        )
        return self

    # 消息 -------------------------------------------------------------
    async def _insert(self, role: str, content: str, metadata: dict | None = None) -> str:
        mid = str(uuid.uuid4())
        await db.execute_raw(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) "
            "VALUES ($1, $2, $3, $4, $5::jsonb, $6::timestamp)",
            mid, self.conv_id, role, content, json.dumps(metadata or {}),
            _naive(datetime.now(UTC)),
        )
        return mid

    async def user_says(self, text: str) -> str:
        from app.services.proactive.state import mark_user_replied_for_conversation

        mid = await self._insert("user", text)
        await mark_user_replied_for_conversation(self.conv_id)  # = ws._persist_user_message
        return mid

    async def ai_replies(
        self, text: str, *, reason: str = ARM_REASON_REPLY, turn_ids: list[str] | None = None,
    ) -> str:
        from app.services.chat.turn_lifecycle import finish_assistant_turn

        mid = await self._insert("assistant", text)
        if turn_ids is None:
            last_user = await db.message.find_first(
                where={"conversationId": self.conv_id, "role": "user"},
                order={"createdAt": "desc"},
            )
            turn_ids = [last_user.id] if last_user else []
        await finish_assistant_turn(
            conversation_id=self.conv_id, agent_id=self.agent_id, user_id=self.user_id,
            workspace_id=self.ws_id, proactive_reason=reason, turn_message_ids=turn_ids,
        )
        await drain()
        return mid

    # 时间推进: 把这个世界里所有时间戳一起往回拨 -------------------------------
    async def advance(self, minutes: float) -> None:
        delta = f"{minutes} minutes"
        await db.execute_raw(
            f"UPDATE messages SET created_at = created_at - interval '{delta}' "
            "WHERE conversation_id = $1", self.conv_id,
        )
        await db.execute_raw(
            f"""
            UPDATE proactive_states SET
                t0_at = t0_at - interval '{delta}',
                window_due_at = window_due_at - interval '{delta}',
                response_deadline_at = response_deadline_at - interval '{delta}',
                last_assistant_reply_at = last_assistant_reply_at - interval '{delta}',
                last_user_reply_at = last_user_reply_at - interval '{delta}',
                last_proactive_at = last_proactive_at - interval '{delta}',
                last_attempt_at = last_attempt_at - interval '{delta}'
            WHERE workspace_id = $1
            """,
            self.ws_id,
        )
        redis = await get_redis()
        key = f"topic_continuity:{self.conv_id}"
        raw = await redis.hgetall(key)
        shifted = {}
        for field in ("last_user_at", "followup_sent_at", "judged_at"):
            if raw.get(field):
                ts = datetime.fromisoformat(raw[field]) - timedelta(minutes=minutes)
                shifted[field] = ts.isoformat()
        if shifted:
            await redis.hset(key, mapping=shifted)

    # 读状态 -----------------------------------------------------------
    async def state(self):
        from app.services.proactive.state import get_proactive_state_by_workspace

        return await get_proactive_state_by_workspace(self.ws_id)

    async def messages(self, role: str | None = None) -> list[dict]:
        rows = await db.query_raw(
            "SELECT id, role, content, metadata, created_at FROM messages "
            "WHERE conversation_id = $1 ORDER BY created_at ASC",
            self.conv_id,
        )
        return [r for r in rows if role is None or r["role"] == role]

    async def events(self) -> list[str]:
        rows = await db.query_raw(
            "SELECT event_type FROM proactive_event_logs WHERE workspace_id = $1 "
            "ORDER BY created_at ASC", self.ws_id,
        )
        return [r["event_type"] for r in rows]

    async def continuity(self):
        return await tc.load_continuity(self.conv_id)

    async def cue(self):
        """聊天侧: 以真实上一条 AI 消息 + 真实 Redis 状态解析承接 cue."""
        from app.services.chat.orchestrator import _resolve_topic_cue

        previous = await db.message.find_first(
            where={"conversationId": self.conv_id, "role": "assistant"},
            order={"createdAt": "desc"},
        )
        last_user = await db.message.find_first(
            where={"conversationId": self.conv_id, "role": "user"},
            order={"createdAt": "desc"},
        )
        return await _resolve_topic_cue(
            asyncio.ensure_future(tc.load_continuity(self.conv_id)),
            previous_assistant=previous,
            replied_at=last_user.createdAt if last_user else None,
            offering_turn=False,
            patience_low=False,
            response_diagnostics={},
        )

    async def system_prompt(self, gap_seconds: float) -> str:
        from app.services.chat.prompt_builder import build_system_prompt

        agent = await db.aiagent.find_unique(where={"id": self.agent_id})
        return await build_system_prompt(
            agent=agent, memory_relevance="weak", reengagement_gap_seconds=gap_seconds,
            topic_continuation=await self.cue(),
        )


def _naive(ts: datetime) -> str:
    return ts.astimezone(UTC).replace(tzinfo=None).isoformat()


async def drain() -> None:
    for _ in range(20):
        pending = [t for t in list(bg_tasks._inflight) if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)


async def scan() -> None:
    from app.services.proactive.orchestrator import scan_proactive_states

    await scan_proactive_states()
    await drain()


# ── 场景 ───────────────────────────────────────────────────────────────

async def scenario_unfinished_then_b_once():
    print("\n[S1] 话题未完结 → 5 分钟后 B 追问 → 同会话只追问一次")
    w = await World("s1").create()
    await w.user_says("我周末想出去玩")
    ai_msg = await w.ai_replies("好呀，你想去哪儿？海边还是山里？")

    st = await w.state()
    check(st.status == "running" and st.current_window_index == 0, "AI 回复后 arm 到判定窗 (window 0)")
    check(
        abs((st.window_due_at - st.t0_at).total_seconds() - 300) < 2,
        "判定窗到期 = AI 最后一句 + 5 分钟",
    )

    await scan()
    check((await w.state()).current_window_index == 0, "不满 5 分钟不判定")

    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "周末去哪玩"})
    SCRIPT.followup.append("要不我先说，我想去海边吹吹风~")
    await w.advance(5)
    await scan()

    assistant_msgs = await w.messages("assistant")
    b = assistant_msgs[-1]
    meta = b["metadata"] if isinstance(b["metadata"], dict) else json.loads(b["metadata"])
    check(len(assistant_msgs) == 2 and meta.get("trigger_type") == "followup_unfinished", "B 追问已发出")
    check(meta.get("proactive") is True, "B 带 proactive 标记 (计入全局冷却)")
    judge_prompt = next(p for k, p in SCRIPT.prompts if k == "judge")
    check("AI: 好呀，你想去哪儿？" in judge_prompt and "unfinished" in judge_prompt,
          "判定 prompt 来自 registry 且带最近对话")
    fu_prompt = next(p for k, p in SCRIPT.prompts if k == "followup")
    check("【通用回复规则】" in fu_prompt and "周末去哪玩" in fu_prompt, "B prompt 带回复前置 + 未完结点")
    check("活泼外向、脑洞大、感性细腻" in fu_prompt, "B prompt 带 MBTI 推出的人设 (不再恒为温和友善)")

    st = await w.state()
    check(st.status == "running" and st.current_window_index == 1, "B 发出后进入 A 模式 window 1")
    check(abs((st.t0_at - datetime.now(UTC)).total_seconds()) < 5, "A 模式时钟从 B 这句重新起算")
    cont = await w.continuity()
    check(cont.followup_sent_at is not None, "本会话 B 名额已用")
    check(cont.anchor_message_id == ai_msg and cont.verdict.unfinished, "判定结论锚定在被判的 AI 消息")
    events = await w.events()
    check("topic_judged" in events and "followup_sent" in events, "事件日志: topic_judged + followup_sent")

    # 用户 20 分钟后回 B: 回复的是追问本身 → 不带承接话术
    await w.advance(20)
    await w.user_says("海边！我超想看海")
    check((await w.state()).status == "idle", "用户回来 → 状态 idle")
    cue = await w.cue()
    check(cue is not None and cue.template_key is None, "回 B 追问: 压掉重逢短档, 不注入承接段")

    # 同会话再次停顿 → 判定仍跑 (供被动承接), 但只能走 A
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "去哪片海"})
    await w.ai_replies("那你想去近一点的还是远一点的？")
    await w.advance(5)
    before = SCRIPT.count("followup")
    await scan()
    check(SCRIPT.count("followup") == before, "同会话第二次未完结不再生成 B")
    check(len(await w.messages("assistant")) == 3, "同会话没有第二条 B")
    st = await w.state()
    check(st.current_window_index == 1 and st.status == "running", "强制走 A 模式 (window 1)")


async def scenario_finished_goes_a_and_passive_suppression():
    print("\n[S2] 话题已完结 → A 模式; 用户隔 45 分钟回来不带承接")
    w = await World("s2").create()
    await w.user_says("今天好累呀")
    await w.ai_replies("辛苦啦，早点休息，泡个热水澡会舒服很多")
    SCRIPT.judge.append({"status": "finished", "reason": "natural_end", "pending_topic": ""})
    await w.advance(5)
    await scan()
    st = await w.state()
    check(st.current_window_index == 1 and len(await w.messages("assistant")) == 1, "完结 → 不追问, 进 window 1")
    check(
        timedelta(minutes=30) <= st.window_due_at - st.t0_at <= timedelta(hours=1),
        "window 1 仍按 AI 最后一句的 t0 算 (30min-1h)",
    )
    await w.advance(45)
    await w.user_says("我洗完澡了")
    prompt = await w.system_prompt(gap_seconds=50 * 60)
    check("## 重逢感知" not in prompt and "## 话题接续" not in prompt,
          "已完结话题: 45 分钟后回来既无重逢寒暄也无承接段")


async def scenario_passive_return_and_jump():
    print("\n[S3] 未完结但 B 被门槛拦下 → 用户 15 分钟后回来: 回归承接; 8 分钟: 仅跳话题过渡")
    for gap, expect_key, label in (
        (15, tc.CUE_RETURN_KEY, "隔 15 分钟: 回归承接段"),
        (8, tc.CUE_JUMP_KEY, "隔 8 分钟: 仅跳话题过渡段"),
    ):
        w = await World(f"s3-{gap}").create()
        await w.user_says("我明天有个面试，有点紧张")
        await w.ai_replies("是什么岗位的面试呀？")
        SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "面试岗位"})
        with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: False):
            await w.advance(5)
            await scan()
        check(len(await w.messages("assistant")) == 1, f"{label}: 夜间门槛拦下 B")
        await w.advance(gap - 5)
        await w.user_says("对了你吃饭了吗")
        cue = await w.cue()
        check(cue is not None and cue.template_key == expect_key, label)
        prompt = await w.system_prompt(gap_seconds=gap * 60)
        check("## 话题接续" in prompt and "面试岗位" in prompt, f"{label}: 主 prompt 注入且带未完结点")
        check("## 重逢感知" not in prompt, f"{label}: 不叠重逢感知")


async def scenario_short_circuit_and_boundary_arming():
    print("\n[S4] 短路回复也 arm (修复: 之前停在 idle); 边界回复不 arm")
    from app.services.chat import multi_intent, orchestrator

    w = await World("s4").create()
    await w.user_says("哈哈哈")

    async def _save(conversation_id, replies, **_kwargs):
        text = replies[0]["text"] if isinstance(replies[0], dict) else replies[0]
        await w._insert("assistant", text)

    await multi_intent.short_circuit_reply("😄", w.conv_id, w.agent_id, w.user_id, _save)
    await drain()
    st = await w.state()
    check(st is not None and st.status == "running" and st.current_window_index == 0,
          "语气词表情短路后 arm 到判定窗")
    check((st.metadata or {}).get("reason") == "short_circuit", "arm 原因记为 short_circuit")

    w2 = await World("s4b").create()
    await w2.user_says("你真烦")
    await orchestrator._boundary_short_circuit_reply("……我不太想说话了", w2.conv_id, w2.agent_id, w2.user_id)
    await drain()
    st2 = await w2.state()
    check(st2 is None or st2.status == "idle", "边界系统回复后不 arm (AI 生气时不追人)")


async def _noop(*_a, **_k):
    return None


async def scenario_metadata_survives_chat():
    print("\n[S5] 记忆冷却跨多轮聊天存活 (修复: 之前每次 AI 回复都被冲掉)")
    from app.services.proactive.state import get_proactive_state_by_workspace, mark_proactive_sent

    w = await World("s5").create()
    await w.user_says("在吗")
    await w.ai_replies("在的呀")
    st = await get_proactive_state_by_workspace(w.ws_id)
    await mark_proactive_sent(
        st, trigger_type="memory_proactive", message="x", assistant_message_id=None,
        extra_metadata={"memory_cooldown": {"mem-1": 50}},
    )
    await w.user_says("我回来啦")
    await w.ai_replies("欢迎回来~")
    st = await w.state()
    check((st.metadata or {}).get("memory_cooldown") == {"mem-1": 50}, "memory_cooldown 在新一轮 arm 后仍在")
    check((st.metadata or {}).get("reason") == ARM_REASON_REPLY, "reason 被新值覆盖")


async def scenario_user_returns_during_generation():
    print("\n[S6] 生成期间用户回来 → 不插入 + 状态保持 idle (修复: 之前照发并覆盖状态)")
    w = await World("s6").create()
    await w.user_says("你喜欢什么电影")
    await w.ai_replies("我最近在看老电影，你呢？喜欢什么类型")
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "喜欢的电影类型"})

    async def _user_interrupts(_model, prompt):
        SCRIPT.prompts.append(("followup", prompt))
        await w.user_says("喜欢悬疑的！")  # 用户恰好在追问生成时回来
        return "我猜你喜欢悬疑片？"

    with patch("app.services.proactive.followup.invoke_text", _user_interrupts):
        await w.advance(5)
        await scan()
    msgs = await w.messages()
    check([m["role"] for m in msgs] == ["user", "assistant", "user"], "B 没有插到用户新消息后面")
    st = await w.state()
    check(st.status == "idle", "CAS 落空: 状态保持用户回来时的 idle")
    check(st.current_window_index is None and st.window_due_at is None, "没有被推进回 running 窗口")

    # 过期 arm: 回复 msg1 期间用户又发了 msg2 (且 msg2 那轮是边界回复, 刻意不 arm)
    # → msg1 回复的收尾 arm 必须跳过, 不能把 msg2 置的 idle 覆盖回判定窗
    w3 = await World("s6c").create()
    msg1 = await w3.user_says("你觉得我该换工作吗")
    await w3.user_says("算了你闭嘴")  # 生成期间到达的新消息
    await w3.ai_replies("我觉得可以先想想你最在意什么？", turn_ids=[msg1])
    st3 = await w3.state()
    check(st3 is None or st3.status == "idle", "过期 arm 被 SQL 守卫跳过 (不覆盖新消息的 idle)")

    # A 模式同理
    w2 = await World("s6b").create()
    await w2.user_says("晚点聊")
    await w2.ai_replies("好哒")
    await w2.advance(45)
    await db.execute_raw(
        "UPDATE proactive_states SET current_window_index = 1, window_due_at = now() - interval '1 second' "
        "WHERE workspace_id = $1", w2.ws_id,
    )

    async def _interrupt():
        SCRIPT.on_a_mode_call = None
        await w2.user_says("我回来了")

    SCRIPT.on_a_mode_call = _interrupt
    # 固定触发类型: 随机抽到「记忆主动」而库里没有记忆时本次直接取消, 走不到生成
    with (
        patch("app.services.proactive.orchestrator.should_hit_window", lambda *_a, **_k: (True, 1.0)),
        patch("app.services.proactive.orchestrator.select_trigger_type", lambda: "silence_wakeup"),
        patch("app.services.proactive.sender.select_topic_source", lambda *_a: "greeting"),
        patch("app.services.proactive.gates.is_in_active_hours", lambda _now: True),
    ):
        await scan()
    check(SCRIPT.on_a_mode_call is None, "A 模式确实走到了生成 (模拟的用户回归已触发)")
    SCRIPT.on_a_mode_call = None
    roles = [m["role"] for m in await w2.messages()]
    check(roles == ["user", "assistant", "user"], "A 模式生成期间用户回来: 不插入")
    check((await w2.state()).status == "idle", "A 模式: 状态保持 idle")


async def scenario_farewell_and_sessions():
    print("\n[S7] 告别按规则判完结; 告别 / 长间隔 / A 开场后开启新会话")
    from app.services.proactive.state import ARM_REASON_FAREWELL

    w = await World("s7").create()
    await w.user_says("我去睡啦")
    await w.ai_replies("晚安~做个好梦", reason=ARM_REASON_FAREWELL)
    judged_before = SCRIPT.count("judge")
    await w.advance(5)
    await scan()
    check(SCRIPT.count("judge") == judged_before, "告别不调判定 LLM")
    cont = await w.continuity()
    check(cont.verdict and cont.verdict.reason == "farewell" and cont.session_closed, "判完结 + 会话已关闭")

    # 同会话里用过 B, 隔 3h+ 回来 → 新会话, B 名额恢复
    w2 = await World("s7b").create()
    await w2.user_says("你猜我今天遇到谁了")
    await w2.ai_replies("谁呀谁呀？")
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "遇到了谁"})
    SCRIPT.followup.append("快说快说，我好奇死了")
    with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: True):
        await w2.advance(5)
        await scan()
    check((await w2.continuity()).followup_available is False, "B 已用")
    await w2.advance(200)
    await w2.user_says("我回来啦，遇到了初中同学")
    check((await w2.continuity()).followup_available is True, "隔 3h+ 回来 → 新会话, B 名额恢复")


async def scenario_gates():
    print("\n[S8] B 门槛: 全局冷却 / 耐心低")
    w = await World("s8").create()
    await w.user_says("周末干嘛好呢")
    await w.ai_replies("你想宅家还是出门？")
    # 10 分钟前刚发过一条提醒 (主动消息) → 冷却中
    await w._insert("assistant", "记得喝水哦", {"proactive": True, "trigger_type": "reminder"})
    await db.execute_raw(
        "UPDATE messages SET created_at = created_at - interval '10 minutes' "
        "WHERE conversation_id = $1 AND content = '记得喝水哦'", w.conv_id,
    )
    await db.execute_raw(
        "UPDATE messages SET created_at = created_at + interval '1 second' "
        "WHERE conversation_id = $1 AND content = '你想宅家还是出门？'", w.conv_id,
    )
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "周末安排"})
    with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: True):
        await w.advance(5)
        await scan()
    events = await db.query_raw(
        "SELECT payload FROM proactive_event_logs WHERE workspace_id = $1 AND event_type = 'followup_skipped'",
        w.ws_id,
    )
    reasons = [(e["payload"] if isinstance(e["payload"], dict) else json.loads(e["payload"]))["reason"] for e in events]
    check(reasons == ["cooldown"], f"冷却期内不追问 (reasons={reasons})")

    # 15 分钟内有提醒要响 → B 让路 (提醒豁免冷却, 否则会一分钟内连收两条)
    w3 = await World("s8c").create()
    await w3.user_says("周末干嘛好呢")
    await w3.ai_replies("你想宅家还是出门？")
    await db.execute_raw(
        "INSERT INTO time_triggers (id, ai_agent_id, user_id, trigger_time, action_type, is_active) "
        "VALUES ($1, $2, $3, $4::timestamp, 'reminder', TRUE)",
        str(uuid.uuid4()), w3.agent_id, w3.user_id,
        _naive(datetime.now(UTC) + timedelta(minutes=10)),
    )
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "周末安排"})
    await w3.advance(5)
    await scan()
    check(len(await w3.messages("assistant")) == 1, "15 分钟内有提醒要响 → 不追问")

    # 系统确认类问句 (删除确认 / 提醒要时间) → 不调判定、不追问, 照常进 A
    from app.services.proactive.state import ARM_REASON_SYSTEM

    w4 = await World("s8d").create()
    await w4.user_says("把我说过住苏州那条删了")
    await w4.ai_replies("找到这条：「我住在苏州」，回我「对」就删掉哦", reason=ARM_REASON_SYSTEM)
    judged = SCRIPT.count("judge")
    await w4.advance(5)
    await scan()
    st4 = await w4.state()
    check(SCRIPT.count("judge") == judged and len(await w4.messages("assistant")) == 1,
          "系统确认类问句: 不判定、不追问")
    check(st4.current_window_index == 1 and st4.status == "running", "系统确认后照常进 A 模式")

    w2 = await World("s8b").create()
    redis = await get_redis()
    await redis.set(f"patience:{w2.agent_id}:{w2.user_id}", "40")
    await w2.user_says("随便吧")
    await w2.ai_replies("那你到底想怎么样嘛？")
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "想怎么样"})
    with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: True):
        await w2.advance(5)
        await scan()
    check(len(await w2.messages("assistant")) == 1, "耐心 <70 不追问")


async def scenario_timed_memory_a_mode():
    print("\n[S9] A 模式新来源: 带时间戳的历史记忆 → 发送 → 会话关闭")
    from app.services.proactive.context import _load_timed_user_memories
    from app.services.proactive.sender import generate_and_send_proactive

    w = await World("s9").create()
    now = datetime.now(UTC)
    rows = [
        ("周五要去面试产品经理", now - timedelta(days=2), now - timedelta(days=4), "生活", "计划"),  # 事先说的
        ("昨天面试完了感觉挺好", now - timedelta(days=1, hours=2), now - timedelta(hours=20), "生活", "日常"),  # 事后讲述
        ("提醒我喝水", now - timedelta(days=1), None, "生活", "提醒"),
        ("最近在学吉他", None, now - timedelta(days=3), "偏好", "兴趣"),  # 偏好事实, 不问"后来怎样"
        ("这周在搬家", None, now - timedelta(days=3), "生活", "日常"),
        ("刚说的事", None, now - timedelta(hours=1), "生活", "日常"),  # 半天内说的不算"想起来"
    ]
    for content, occur, said, main, sub in rows:
        await db.execute_raw(
            "INSERT INTO memories_user (id, user_id, workspace_id, content, main_category, sub_category, "
            "occur_time, statement_time, updated_at) VALUES ($1,$2,$3,$4,$5,$6,$7::timestamp,$8::timestamp,$9::timestamp)",
            str(uuid.uuid4()), w.user_id, w.ws_id, content, main, sub,
            _naive(occur) if occur else None, _naive(said) if said else None, _naive(now),
        )
    texts, _ids = await _load_timed_user_memories(user_id=w.user_id, workspace_id=w.ws_id, exclude_memory_ids=set())
    check(
        len(texts) == 2 and texts[0].startswith("[前天的事") and "面试" in texts[0]
        and texts[1] == "[3天前聊到] 这周在搬家",
        f"真库筛选正确 (事件优先; 排除提醒/偏好/刚说的/事后讲述): {texts}",
    )

    await w.user_says("在忙")
    await w.ai_replies("好的去吧")
    st = await w.state()
    await db.execute_raw("UPDATE proactive_states SET status='processing' WHERE id=$1", st.id)
    st = await w.state()
    SCRIPT.a_mode.append("前天你说要去面试，怎么样啦？")
    with (
        patch("app.services.proactive.sender.determine_proactive_stage", AsyncMock(return_value="warming")),
        patch("app.services.proactive.sender.select_topic_source", lambda *_a: "timed_user"),
    ):
        sent = await generate_and_send_proactive(st, trigger_type="memory_proactive")
    check(sent is True, "A 模式发送成功")
    a_prompt = [p for k, p in SCRIPT.prompts if k == "a_mode"][-1]
    check("前天的事" in a_prompt and "问问后来怎么样了" in a_prompt, "走 proactive.memory_timed 模板并带时间")
    check((await w.state()).status == "waiting_user", "A 发出 → waiting_user")
    check((await w.continuity()).session_closed is True, "A 是新开场 → 会话关闭")


class _RoutedFakeLLM:
    """聊天主路径用的假模型: 主回复给一句正常回复, 其余小模型调用返回空 JSON
    (各调用方都有解析失败兜底 —— 这本身也是在验证兜底链路)。"""

    main_prompts: list[str] = []

    @classmethod
    def build(cls):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class Fake(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "e2e-fake"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                text = "\n".join(str(getattr(m, "content", m)) for m in messages)
                if "## 回复要求" in text:
                    cls.main_prompts.append(text)
                    reply = "吃了吃了，刚吃完一碗面[EMO:高兴/50]"
                else:
                    reply = "{}"
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        return Fake()


async def scenario_real_chat_turn():
    print("\n[S10] 真实聊天主路径: 读判定 → 主 prompt 注入话题接续 → 回复收尾 arm 判定窗")
    from langchain_core.embeddings import Embeddings

    from app.services.chat.orchestrator import stream_chat_response
    from app.services.llm import models

    class _Emb(Embeddings):
        def embed_documents(self, texts):
            return [[0.01] * 1024 for _ in texts]

        def embed_query(self, text):
            return [0.01] * 1024

    fake = _RoutedFakeLLM.build()
    w = await World("s10").create()
    await w.user_says("我明天有个面试，有点紧张")
    await w.ai_replies("别紧张！是什么岗位的面试呀？")
    SCRIPT.judge.append({"status": "unfinished", "reason": "ai_question", "pending_topic": "面试岗位"})
    with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: False):
        await w.advance(5)
        await scan()
    await w.advance(10)
    user_msg = await w.user_says("对了你吃饭了吗")
    agent = await db.aiagent.find_unique(where={"id": w.agent_id})

    events = []
    with (
        patch.object(models, "_build_chat_model", lambda _k: fake),
        patch.object(models, "_build_utility_model", lambda _k: fake),
        patch.object(models, "_build_fallback_chat_model", lambda _k: fake),
        patch("app.services.memory.storage.embedding.get_embedding_model", lambda: _Emb()),
    ):
        async for evt in stream_chat_response(
            conversation_id=w.conv_id,
            user_message="对了你吃饭了吗",
            agent=agent,
            user_id=w.user_id,
            reply_context={
                "received_at": datetime.now(UTC).isoformat(),
                "turn_message_ids": [user_msg],
            },
            save_user_message=False,
            user_message_id=user_msg,
            delivered_from_queue=True,
        ):
            events.append(evt)
        await drain()

    kinds = [e.get("event") for e in events]
    check("reply" in kinds and kinds[-1] == "done", f"主路径正常出回复 (events={kinds})")
    check(bool(_RoutedFakeLLM.main_prompts), "走了主 prompt (话题接续强制跳过 tier)")
    main = _RoutedFakeLLM.main_prompts[-1] if _RoutedFakeLLM.main_prompts else ""
    check("## 话题接续" in main and "面试岗位" in main and "过了约15 分钟" in main,
          "主 prompt 注入回归承接段 (未完结 + 15 分钟 + 未用 B)")
    check("## 重逢感知" not in main, "主 prompt 不叠重逢感知")
    if "## 话题接续" in main:
        section = main.split("## 话题接续", 1)[1].split("\n## ", 1)[0].strip()
        print("    └ 注入的话题接续段:\n      " + section.replace("\n", "\n      "))
    st = await w.state()
    check(st.status == "running" and st.current_window_index == 0, "回复收尾后重新 arm 判定窗")
    check((st.metadata or {}).get("reason") == ARM_REASON_REPLY, "arm 原因 = 普通回复")
    replies = [m for m in await w.messages("assistant")]
    check(any("吃了" in m["content"] for m in replies), "AI 回复已落库")


async def scenario_prompt_sync():
    print("\n[S0] 启动同步: 新增 prompt key 已入库")
    from app.services.prompting.store import ensure_prompt_templates, get_prompt_text

    await ensure_prompt_templates()
    for key in (
        "proactive.topic_completion_judge", "proactive.followup_unfinished",
        "proactive.memory_timed", "chat.topic_continuation_return", "chat.topic_continuation_jump",
    ):
        row = await db.prompttemplate.find_unique(where={"key": key})
        text = await get_prompt_text(key)
        check(row is not None and len(text) > 20, f"{key} 已同步且可取")


async def main() -> int:
    await connect_db()
    rows = await db.query_raw("SELECT current_database() AS db")
    if rows[0]["db"] != "companion_e2e":
        print(f"拒绝运行: 连到了 {rows[0]['db']!r}")
        return 2
    # 不清 Redis: 所有 key 都以本次运行生成的 uuid 命名。库里只让上次运行残留的状态行
    # 停摆 (已确认是 companion_e2e), 否则真实 scan 会先处理它们, 打乱脚本化的 LLM 输出。
    await db.execute_raw(
        "UPDATE proactive_states SET status = 'idle', window_due_at = NULL, "
        "response_deadline_at = NULL"
    )

    patches = [
        patch("app.services.proactive.followup.invoke_json", SCRIPT.judge_json),
        patch("app.services.proactive.followup.invoke_text", SCRIPT.followup_text),
        patch("app.services.proactive.followup.get_utility_model", lambda: None),
        patch("app.services.proactive.followup.get_chat_model", lambda: None),
        patch("app.services.proactive.sender.invoke_text", SCRIPT.a_mode_text),
        patch("app.services.proactive.sender.get_chat_model", lambda: None),
        patch("app.services.proactive.context.invoke_json", AsyncMock(return_value={"ids": []})),
        patch("app.services.proactive.context.get_utility_model", lambda: None),
        patch("app.services.proactive.sender._bg_proactive_ai_memory", _noop),
        patch("app.services.proactive.sender._should_use_music_source", lambda *_a: False),
        patch("app.services.speech_output.policy.should_generate_voice", AsyncMock(return_value=False)),
        patch("app.services.notifications.service.notify_agent_message_created", _noop),
        patch(
            "app.services.chat_links.maybe_prepare_proactive_link_recommendation",
            AsyncMock(return_value=(None, "e2e_disabled")),
        ),
    ]
    for p in patches:
        p.start()
    try:
        with patch("app.services.proactive.gates.is_in_active_hours", lambda _now: True):
            await scenario_prompt_sync()
            await scenario_unfinished_then_b_once()
            await scenario_finished_goes_a_and_passive_suppression()
            await scenario_short_circuit_and_boundary_arming()
            await scenario_metadata_survives_chat()
            await scenario_user_returns_during_generation()
            await scenario_farewell_and_sessions()
            await scenario_gates()
            await scenario_timed_memory_a_mode()
        await scenario_passive_return_and_jump()
        await scenario_real_chat_turn()
    finally:
        for p in patches:
            p.stop()
        await drain()
        await disconnect_db()

    print(f"\n{_passes} passed, {len(_failures)} failed")
    for label in _failures:
        print(f"  FAILED: {label}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
