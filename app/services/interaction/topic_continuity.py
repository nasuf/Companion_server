"""话题连续性状态 —— 主动交流侧写, 聊天侧读 (《主动聊天机制（新增）》A/B 双模式).

每个 conversation 一份 Redis hash, 只记三件事:

1. 最近一次「话题完结判定」的结论: AI 说完最后一句满 5 分钟用户没回时,
   proactive/followup.py 判一次, 结论绑定到被判的那条 AI 消息 (anchor)。
2. 本会话的 B 模式 (话题未完结·温柔追问) 是否已经用过 —— 单会话仅一次。
3. 会话边界。以下任一情况, 下一条用户消息开启新会话 (B 名额恢复):
   - 用户距上一条自己的消息 ≥ SESSION_GAP_SECONDS (与话题栈重置 / 重逢摘要同线)
   - AI 发出了 A 模式新开场 (旧话题已翻篇)
   - 本轮是告别 (晚安 / 去忙了)

聊天侧用 `resolve_cue` 决定被动回复要不要带「回归承接」或「跳话题过渡」。
判定结论只对它 anchor 的那条 AI 消息之后的第一条用户回复有效 —— AI 之后又
说了话 (比如 B 追问), anchor 就对不上, 结论自然作废, 不需要显式消费。

Redis 挂时: 读返回 None (聊天侧不注入、B 不发), 写静默失败 —— 这里的每个
功能都是锦上添花, 宁可少做也不能打扰主流程或重复追问。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from app.redis_client import get_redis

logger = logging.getLogger(__name__)

UTC = timezone.utc

# 会话边界 & 承接上限。≥3h 的重逢由「重逢感知 + 上次聊到」接管 (且 3h 前的
# 逐字历史已被裁掉), 那时再"接着刚才的话题"反而不自然。
SESSION_GAP_SECONDS = 3 * 3600
# spec: 用户隔 >10 分钟回来才带「回归承接短句」
RETURN_CUE_MIN_GAP_SECONDS = 10 * 60
_TTL_SECONDS = 7 * 86400

CUE_RETURN_KEY = "chat.topic_continuation_return"
CUE_JUMP_KEY = "chat.topic_continuation_jump"
# B 模式追问消息的 trigger_type (metadata.trigger_type), 聊天侧据此认出"上一句是追问"
FOLLOWUP_TRIGGER_TYPE = "followup_unfinished"

VerdictStatus = Literal["finished", "unfinished"]


@dataclass(frozen=True)
class TopicVerdict:
    status: VerdictStatus
    # ai_question / user_story / interrupted / natural_end / farewell / crisis ...
    reason: str
    # 未完结时, 还没聊完的点 (≤15 字); 已完结为空
    pending_topic: str = ""

    @property
    def unfinished(self) -> bool:
        return self.status == "unfinished"


@dataclass(frozen=True)
class ContinuityState:
    verdict: TopicVerdict | None
    anchor_message_id: str | None
    followup_sent_at: datetime | None
    last_user_at: datetime | None
    session_closed: bool

    @property
    def followup_available(self) -> bool:
        return self.followup_sent_at is None


@dataclass(frozen=True)
class ContinuationCue:
    """被动回复的话题接续指引。

    template_key=None 表示「上一轮已聊完」: 不注入任何接续段, 但要压掉重逢感知
    短档 (spec: 已完结话题无论间隔多久都不带承接话术)。
    """

    template_key: str | None
    pending_topic: str
    gap_seconds: float


def _key(conversation_id: str) -> str:
    return f"topic_continuity:{conversation_id}"


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _now(now: datetime | None) -> datetime:
    ts = now or datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _state_from_hash(raw: dict[str, str]) -> ContinuityState:
    status = raw.get("verdict_status")
    verdict = (
        TopicVerdict(
            status=status,  # type: ignore[arg-type]
            reason=raw.get("verdict_reason") or "",
            pending_topic=raw.get("pending_topic") or "",
        )
        if status in ("finished", "unfinished")
        else None
    )
    return ContinuityState(
        verdict=verdict,
        anchor_message_id=raw.get("anchor_id") or None,
        followup_sent_at=_parse_dt(raw.get("followup_sent_at")),
        last_user_at=_parse_dt(raw.get("last_user_at")),
        session_closed=raw.get("session_closed") == "1",
    )


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
        "pending_topic": verdict.pending_topic,
        "anchor_id": anchor_message_id,
        "judged_at": _now(now).isoformat(),
    })


async def reserve_followup(conversation_id: str, *, now: datetime | None = None) -> bool:
    """发 B 之前先占名额: 写不进去就不发 —— 宁可少追问一次, 不能记账失败导致追问两次."""
    return await _hset(conversation_id, {"followup_sent_at": _now(now).isoformat()})


async def release_followup(conversation_id: str) -> None:
    """占了名额但最终没发出去 (用户恰好回来 / 生成失败): 还回去."""
    try:
        redis = await get_redis()
        await redis.hdel(_key(conversation_id), "followup_sent_at")
    except Exception as e:
        logger.warning(f"[CONTINUITY] release followup failed conv={conversation_id[:8]}: {e}")


async def close_session(conversation_id: str | None) -> None:
    """A 模式新开场 / 告别后调用: 下一条用户消息开新会话。"""
    if conversation_id:
        await _hset(conversation_id, {"session_closed": "1"})


# 只重置 B 名额。判定结论不随会话清掉 —— 它靠 anchor 自证有效期, 而「晚安」后
# 隔两小时回来这种跨会话的第一句, 恰恰要靠"上一轮已完结"压掉重逢寒暄。
_SESSION_FIELDS = ("followup_sent_at", "session_closed")


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

    | 上一轮判定 | 间隔        | 本会话已用 B | 结果                      |
    |-----------|-------------|-------------|---------------------------|
    | 无 / 过期  | -           | -           | None (维持原重逢感知逻辑)    |
    | 任意       | ≥3h         | -           | None (交给重逢感知/上次聊到) |
    | 已完结     | <3h         | -           | 不注入, 压掉重逢短档         |
    | 未完结     | 10min-3h    | 否          | 回归承接 + 跳话题时轻过渡    |
    | 未完结     | 10min-3h    | 是          | 仅跳话题时轻过渡             |
    | 未完结     | <10min      | -           | 仅跳话题时轻过渡             |
    | AI 上一句就是 B 追问 | <3h  | 是          | 不注入, 压掉重逢短档         |

    "是否跳了话题"交给主回复 LLM 按段内条件自己判断 —— 它本来就看得到完整
    历史, 不值得为此多一次分类调用。
    """
    if gap_seconds is None or gap_seconds >= SESSION_GAP_SECONDS:
        return None
    if previous_assistant_is_followup:
        # 对方在回 B 追问: 追问本身已经把旧话题接住了, spec 要求不再带承接话术
        return ContinuationCue(template_key=None, pending_topic="", gap_seconds=gap_seconds)
    if state is None or state.verdict is None:
        return None
    if not previous_assistant_id or state.anchor_message_id != previous_assistant_id:
        return None
    verdict = state.verdict
    if not verdict.unfinished:
        return ContinuationCue(template_key=None, pending_topic="", gap_seconds=gap_seconds)
    if gap_seconds > RETURN_CUE_MIN_GAP_SECONDS and state.followup_available:
        key = CUE_RETURN_KEY
    else:
        key = CUE_JUMP_KEY
    return ContinuationCue(
        template_key=key,
        pending_topic=verdict.pending_topic or "刚才那件事",
        gap_seconds=gap_seconds,
    )
