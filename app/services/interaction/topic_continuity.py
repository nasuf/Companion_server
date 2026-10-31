"""话题连续性状态 —— 主动交流侧写, 聊天侧读 (《主动聊天机制（新增）》A/B 双模式).

每个 conversation 一份 Redis hash, 只记三件事:

1. 最近一次「话题完结判定」的结论: AI 说完最后一句满 5 分钟用户没回时,
   proactive/followup.py 判一次, 结论绑定到被判的那条 AI 消息 (anchor)。
2. 本会话的 B 模式 (话题未完结·温柔追问) 是否已经用过 —— 单会话仅一次。
3. 会话边界。以下任一情况, 下一条用户消息开启新会话 (B 名额恢复):
   - 用户距上一条自己的消息 ≥ SESSION_GAP_SECONDS (与话题栈重置 / 重逢摘要同线)
   - AI 发出了 A 模式新开场 (旧话题已翻篇)
   - 本轮是告别 (晚安 / 去忙了)

聊天侧用 `resolve_cue` 决定被动回复要不要先说「回归承接短句」/「跳话题过渡句」
(chat/topic_continuation.py 生成)。
判定结论只对它 anchor 的那条 AI 消息之后的第一条用户回复有效 —— AI 之后又
说了话 (比如 B 追问), anchor 就对不上, 结论自然作废, 不需要显式消费。

Redis 挂时: 读返回 None (聊天侧不注入、B 不发), 写静默失败 —— 这里的每个
功能都是锦上添花, 宁可少做也不能打扰主流程或重复追问。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.redis_client import get_redis
from app.services.proactive import tuning
from app.services.prompting.utils import is_skip_output
from app.services.time_domain.timestamps import parse_time

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 会话边界 & 承接上限。≥3h 的重逢由「重逢感知 + 上次聊到」接管 (且 3h 前的
# 逐字历史已被裁掉), 那时再"接着刚才的话题"反而不自然。
SESSION_GAP_SECONDS = 3 * 3600
_TTL_SECONDS = 7 * 86400

# B 模式追问消息的 trigger_type (metadata.trigger_type), 聊天侧据此认出"上一句是追问"
FOLLOWUP_TRIGGER_TYPE = "followup_unfinished"

VerdictStatus = Literal["finished", "unfinished"]


@dataclass(frozen=True)
class TopicVerdict:
    status: VerdictStatus
    # llm (小模型判定) / farewell (告别按规则判完结)
    reason: str

    @property
    def unfinished(self) -> bool:
        return self.status == "unfinished"


@dataclass(frozen=True)
class ContinuityState:
    verdict: TopicVerdict | None
    anchor_message_id: str | None
    # 本次会话最近一次 B 追问的时间
    followup_sent_at: datetime | None
    last_user_at: datetime | None
    session_closed: bool
    # 本次会话已追问次数 (旧数据只有 followup_sent_at → 按 1 次算)
    followup_count: int = 0

    @property
    def followups_used(self) -> int:
        return max(self.followup_count, 1 if self.followup_sent_at is not None else 0)

    @property
    def followup_used(self) -> bool:
        """本次会话追问过 —— 规格: 回归承接话术要求「从未触发过 B」."""
        return self.followups_used > 0

    @property
    def followup_available(self) -> bool:
        """本次会话还能再追问 (单会话上限默认 1, 后台可调; 0 = 关闭 B 追问)."""
        return self.followups_used < int(tuning.value("followup_max_per_session"))


@dataclass(frozen=True)
class ContinuationCue:
    """被动回复的话题接续决策。

    unfinished=False: 上一轮已聊完 / 对方在回 B 追问 —— 什么都不补, 但要压掉重逢
    感知短档 (spec: 已完结话题无论间隔多久都不带承接话术)。
    unfinished=True: 用户换了无关新话题时补一句过渡; return_line 再决定要不要先说
    一句承接时间差的短句 (隔 >10min 且本会话没追问过)。
    """

    unfinished: bool
    return_line: bool
    gap_seconds: float


# 兜底过滤: 提示词已明令禁止, LLM 仍写出来时宁可不发。两类分开:
#   催促 / 抱怨 / 质问 (等你好久、怎么才回、消失很久…): 任何时候都不能说
#   查岗 (在吗、人呢、忙完了吗、是不是去忙了…): 只在对方还没回来时 (B 追问) 算冒犯;
#   对方已经回来了 (回归承接), 一句"忙完啦？"恰恰是自然的时间差承接
# 自然追问不能误杀: "后来那个人呢""现在吗""然后呢？后来你去哪儿了" 都得放行 ——
# 所以"在吗"要求前面是句首/标点/你/宝, "人呢"要求句首或标点, "去哪了/睡着了"
# 只在整句就是它时才算。
_REPROACH_PATTERN = re.compile(
    r"怎么(?:不|没)(?:回|理|说话|动静)|不理我|没回我|理理我|回我一下"
    r"|^\W*你?(?:去哪(?:了|啦|儿了)|睡着了)[吗呀啊？?！!~～。]*$"
    r"|消失了|消失很久|怎么才回|等你(?:回|好久)"
)
_CHECK_IN_PATTERN = re.compile(
    r"(?:^|[，。！？!?,.~～…\s]|你|宝)(?:还)?(?:在吗|在不在|在嘛)"
    r"|(?:^|[，。！？!?,.~～…\s])人呢"
    r"|还在忙|是不是(?:还)?(?:在|去)?忙|忙完了?[吗没]"
    r"|打扰(?:到)?你|不好意思打扰"
)


def clean_single_line(
    text: str | None,
    *,
    min_len: int = 2,
    user_present: bool = False,
) -> str | None:
    """主动 / 承接类"只输出一句"的生成结果清洗; 不该发时返回 None.

    通用回复规则允许 "||" 分条, 这里要的是一句: 拼起来而不是只留前半句。
    user_present: 对方此刻就在 (刚发来消息) —— 查岗口吻不再算冒犯。
    """
    if not text or is_skip_output(text):
        return None
    segments = [seg.strip() for seg in text.split("||") if seg.strip()]
    line = "，".join(segments).strip().strip("\"'“”「」")
    if len(line) < min_len:
        return None
    if _REPROACH_PATTERN.search(line) or (not user_present and _CHECK_IN_PATTERN.search(line)):
        logger.info(f"[CONTINUITY] dropped needy phrasing: {line[:30]!r}")
        return None
    return line


def is_dialogue_noise(metadata: dict | None) -> bool:
    """游戏播报 / 收礼小灰条不是"聊天内容", 不进任何对话上下文."""
    metadata = metadata if isinstance(metadata, dict) else {}
    kind = str(metadata.get("kind") or "")
    return kind.startswith("game") or bool(metadata.get("offering_received"))


def render_dialogue(
    entries: Iterable[tuple[str, str, datetime | None]],
    *,
    max_chars: int = 120,
) -> str:
    """(role, text, created_at) → "[MM-DD HH:MM] 用户/AI: 内容" 行 (UTC+8, 与主聊天历史同格式).

    主动交流 / 被动承接的各个提示词都以这种带时间戳的对话上下文为输入。
    """
    from app.services.time_domain.clock import _TZ

    lines = []
    for role, text, created_at in entries:
        # Database naive timestamps mean UTC, never the host's local timezone.
        # Reuse the chat clock contract before rendering proactive context.
        instant = parse_time(created_at)
        stamp = ""
        if instant is not None:
            try:
                stamp = f"[{instant.astimezone(_TZ).strftime('%m-%d %H:%M')}] "
            except (ValueError, OverflowError):
                stamp = "[时间未知] "
        elif created_at is not None:
            stamp = "[时间未知] "
        speaker = "AI" if role == "assistant" else "用户"
        lines.append(f"{stamp}{speaker}: {text[:max_chars]}")
    return "\n".join(lines)


def _key(conversation_id: str) -> str:
    return f"topic_continuity:{conversation_id}"


def _parse_dt(raw: str | None) -> datetime | None:
    return parse_time(raw)


def _now(now: datetime | None) -> datetime:
    ts = now or datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _state_from_hash(raw: dict[str, str]) -> ContinuityState:
    status = raw.get("verdict_status")
    verdict = (
        TopicVerdict(status=status, reason=raw.get("verdict_reason") or "")  # type: ignore[arg-type]
        if status in ("finished", "unfinished")
        else None
    )
    return ContinuityState(
        verdict=verdict,
        anchor_message_id=raw.get("anchor_id") or None,
        followup_sent_at=_parse_dt(raw.get("followup_sent_at")),
        last_user_at=_parse_dt(raw.get("last_user_at")),
        session_closed=raw.get("session_closed") == "1",
        followup_count=_parse_int(raw.get("followup_count")),
    )


def _parse_int(value: str | None) -> int:
    try:
        return max(0, int(value or 0))
    except ValueError:
        return 0


async def load_continuity(conversation_id: str | None) -> ContinuityState | None:
    """读状态; Redis 不可用返回 None (调用方按"不知道"处理, 不是"空状态")."""
    if not conversation_id:
        return None
    try:
        redis = await get_redis()
        raw = await redis.hgetall(_key(conversation_id))
    except Exception as e:
        logger.warning(f"[CONTINUITY] load failed conv={conversation_id[:8]}: {e}")
        return None
    return _state_from_hash(raw or {})


async def _hset(conversation_id: str, mapping: dict[str, str]) -> bool:
    try:
        redis = await get_redis()
        key = _key(conversation_id)
        pipe = redis.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, _TTL_SECONDS)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"[CONTINUITY] write failed conv={conversation_id[:8]}: {e}")
        return False
    return True


async def record_verdict(
    conversation_id: str,
    verdict: TopicVerdict,
    *,
    anchor_message_id: str,
    now: datetime | None = None,
) -> None:
    await _hset(conversation_id, {
        "verdict_status": verdict.status,
        "verdict_reason": verdict.reason,
        "anchor_id": anchor_message_id,
        "judged_at": _now(now).isoformat(),
    })


async def reserve_followup(conversation_id: str, *, now: datetime | None = None) -> bool:
    """发 B 之前先占名额: 写不进去就不发 —— 宁可少追问一次, 不能记账失败导致多追问."""
    try:
        redis = await get_redis()
        key = _key(conversation_id)
        pipe = redis.pipeline()
        pipe.hincrby(key, "followup_count", 1)
        pipe.hset(key, mapping={"followup_sent_at": _now(now).isoformat()})
        pipe.expire(key, _TTL_SECONDS)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"[CONTINUITY] reserve followup failed conv={conversation_id[:8]}: {e}")
        return False
    return True


async def release_followup(conversation_id: str) -> None:
    """占了名额但最终没发出去 (用户恰好回来 / 生成失败): 还回去."""
    try:
        redis = await get_redis()
        key = _key(conversation_id)
        if await redis.hincrby(key, "followup_count", -1) <= 0:
            await redis.hdel(key, "followup_count", "followup_sent_at")
    except Exception as e:
        logger.warning(f"[CONTINUITY] release followup failed conv={conversation_id[:8]}: {e}")


async def close_session(conversation_id: str | None) -> None:
    """A 模式新开场 / 告别后调用: 下一条用户消息开新会话。"""
    if conversation_id:
        await _hset(conversation_id, {"session_closed": "1"})


# 只重置 B 名额。判定结论不随会话清掉 —— 它靠 anchor 自证有效期, 而「晚安」后
# 隔两小时回来这种跨会话的第一句, 恰恰要靠"上一轮已完结"压掉重逢寒暄。
_SESSION_FIELDS = ("followup_sent_at", "followup_count", "session_closed")


def starts_new_session(state: ContinuityState, now: datetime) -> bool:
    if state.session_closed or state.last_user_at is None:
        return True
    return (now - state.last_user_at).total_seconds() >= SESSION_GAP_SECONDS


async def note_user_message(conversation_id: str | None, *, now: datetime | None = None) -> None:
    """每条用户消息落库时调用 (proactive.state.mark_user_replied_for_conversation).

    跨会话时恢复 B 名额。两个并发的用户消息同时判定"新会话"是幂等的。
    """
    if not conversation_id:
        return
    now_ts = _now(now)
    state = await load_continuity(conversation_id)
    if state is None:
        return
    try:
        redis = await get_redis()
        key = _key(conversation_id)
        pipe = redis.pipeline()
        if starts_new_session(state, now_ts):
            pipe.hdel(key, *_SESSION_FIELDS)
        pipe.hset(key, mapping={"last_user_at": now_ts.isoformat()})
        pipe.expire(key, _TTL_SECONDS)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"[CONTINUITY] note user message failed conv={conversation_id[:8]}: {e}")


def resolve_cue(
    state: ContinuityState | None,
    *,
    previous_assistant_id: str | None,
    gap_seconds: float | None,
    previous_assistant_is_followup: bool = False,
) -> ContinuationCue | None:
    """spec「被动回复高级承接机制」的决策表 (纯函数).

    | 上一轮判定 | 间隔        | 本会话已用 B | 结果                              |
    |-----------|-------------|-------------|-----------------------------------|
    | 无 / 过期  | -           | -           | None (维持原重逢感知逻辑)            |
    | 任意       | ≥3h         | -           | None (交给重逢感知/上次聊到)         |
    | 已完结     | <3h         | -           | 什么都不补, 压掉重逢短档              |
    | 未完结     | 10min-3h    | 否          | 回归承接短句 (+ 跳话题时过渡句)       |
    | 未完结     | 10min-3h    | 是          | 仅跳话题时过渡句                     |
    | 未完结     | <10min      | -           | 仅跳话题时过渡句                     |
    | AI 上一句就是 B 追问 | <3h  | 是          | 什么都不补, 压掉重逢短档              |

    "是否跳了话题"要看用户这条消息, 由 chat/topic_continuation.py 另行判定。
    """
    return explain_cue(
        state,
        previous_assistant_id=previous_assistant_id,
        gap_seconds=gap_seconds,
        previous_assistant_is_followup=previous_assistant_is_followup,
    )[0]


def explain_cue(
    state: ContinuityState | None,
    *,
    previous_assistant_id: str | None,
    gap_seconds: float | None,
    previous_assistant_is_followup: bool = False,
) -> tuple[ContinuationCue | None, str]:
    """`resolve_cue` plus why: the reason labels the cue, or explains its absence.

    Reasons: followup_reply / finished / unfinished when a cue applies;
    gap_over_session / state_unavailable / no_verdict / anchor_mismatch otherwise.
    """
    if gap_seconds is None or gap_seconds >= SESSION_GAP_SECONDS:
        return None, "gap_over_session"
    if previous_assistant_is_followup:
        # 对方在回 B 追问: 追问本身已经把旧话题接住了, spec 要求不再带承接话术
        return ContinuationCue(unfinished=False, return_line=False, gap_seconds=gap_seconds), "followup_reply"
    if state is None:
        return None, "state_unavailable"
    if state.verdict is None:
        return None, "no_verdict"
    if not previous_assistant_id or state.anchor_message_id != previous_assistant_id:
        return None, "anchor_mismatch"
    if not state.verdict.unfinished:
        return ContinuationCue(unfinished=False, return_line=False, gap_seconds=gap_seconds), "finished"
    return ContinuationCue(
        unfinished=True,
        return_line=(
            gap_seconds > float(tuning.value("return_line_min_gap_minutes")) * 60
            and not state.followup_used
        ),
        gap_seconds=gap_seconds,
    ), "unfinished"
