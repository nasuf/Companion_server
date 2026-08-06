"""开局就走时说什么 —— 或者干脆不说.

## 这是社交场景, 不是游戏场景

实测 768 局终局里 **401 局 (52%) 步数 ≤2**, 342 局是 0 步。也就是说"点开又关"
不是边缘情况, 而是最常见的行为。

这些局没有棋可复盘, 但**有事可说** —— 事情不在棋盘上, 在"你刚点开就走了"本身。
真朋友会顺口提一句"才动两下就撤啦", 而不是宣布输赢 (系统判了用户负, 但他的体感是
根本没玩), 更不是每次都念同一句。

## 为什么先决定"要不要说话"

相邻两次开局就走有 **42% 发生在 2 分钟内** (158/380)。两分钟里点开关掉六次的人
显然是在翻界面, 而朋友看着你翻, 不会说六次话 —— 会说一次, 然后闭嘴等你。

所以这里先过一道冷却: 窗口内第一次给个轻反应, 之后保持安静。安静是真的不发消息
—— `companion_reply` 只在 Flutter 里被解析成字段而从未渲染, web 完全不用, 所以
回复只通过聊天消息到达用户, 空回复就是干净的沉默。

## 为什么这一句**不**调 LLM

先试过调, 结论是不值得。同一组 7 个场景 (0-4 步、2-31 秒、今天第 1 到第 12 次、
在忙/在睡/无状态), 小模型 qwen3.5-flash 和主模型豆包 character **都只产出 2 种
不同的话**, 其中 5 条一字不差地是「咦，不玩了？」。

换过四版 prompt 也没救回来:
- 给例句 → 模型照抄例句
- 给反例 (别说"不玩了吗") → 照抄反例, 更糟
- 硬要求"必须用上步数/秒数/第几次" → 仍然收敛
- 改成"说你自己在等他落子" → 仍然收敛

原因不在措辞, 在任务性质: 一局 0 步 2 秒的空局可说的东西太少, 而「不玩了」就是
中文里这个处境最自然的说法, 任何模型都会收敛到它。既然模型说的跟写死的一样,
这次调用就是纯成本 —— 而这类局占终局的 52%。

所以改成按素材分档挑句子。分化来自**处境不同**而不是模型创造力: 一步没走跟走了
两步不一样, 今天第一次跟第十二次不一样。同档内轮换措辞, 避免连着重复。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 冷却窗口。实测 42% 的开局就走发生在前一次的 2 分钟内, 60% 在 10 分钟内 ——
# 取 10 分钟能挡掉大半个"翻界面"的连击, 又不会让隔了一阵的那次也失声。
REACTION_COOLDOWN_SECONDS = 600

# "今天第几次"的计数窗口。用 24h 滚动而不是自然日 —— 跨零点重置会让深夜连着点开的
# 那几次看起来像"今天第一次"。
_COUNT_TTL_SECONDS = 86400

# 按处境分档的说法。分化靠"这局到底什么样"而不是靠模型创造力 (见 docstring)。
#
# 都不提输赢: 系统判了用户负, 但用户的体感是根本没玩, 恭喜或认输都莫名其妙。
# 都不催他回来玩, 也不追问为什么走。
_LINES: dict[str, tuple[str, ...]] = {
    # 一步没走就关 —— 棋盘刚摆开
    "untouched": (
        "咦，一子没落就收了。",
        "棋盘刚摆开又收啦。",
        "我这边刚坐下，你就走了。",
        "行，那这盘先不摆了。",
    ),
    # 走了一两步 —— 我这边还在等
    "barely": (
        "才动两下就撤啦？",
        "我还在等你下一手呢。",
        "刚有点意思就停了。",
        "这就收摊了，那下次接着。",
    ),
    # 今天已经反复这样 —— 熟人之间可以调侃
    "repeated": (
        "你今天已经点开关了好几回了。",
        "又摆又收，是拿不定主意玩哪个吧。",
        "这盘也就摆了一下，跟刚才一样。",
        "我这边棋盘都摆熟了。",
    ),
}
_REPEAT_THRESHOLD = 3


async def check_reaction(conversation_id: str | None) -> tuple[bool, int]:
    """该不该出声, 以及今天这是第几次点开又关.

    冷却用 Redis SET NX: 窗口内第一次返回 True, 之后 False。

    次数单独计 (24h 过期), 因为它是**素材**而不只是门控 —— 实测同一会话一天里最多
    点开又关 56 次, 而"今天第一次"和"今天第七次"该说的话完全不同。

    Redis 不可用时返回 (True, 1) —— 宁可多说一句也不要整个功能静默失效 (静默失效
    的表现是"AI 对游戏毫无反应", 排查起来比多一条消息麻烦得多)。
    """
    if not conversation_id:
        return False, 0
    try:
        from app.redis_client import get_redis

        r = await get_redis()
        count_key = f"game:quick_exit:count:{conversation_id}"
        pipe = r.pipeline()
        pipe.set(
            f"game:quick_exit:{conversation_id}",
            "1",
            ex=REACTION_COOLDOWN_SECONDS,
            nx=True,
        )
        # 先 SET NX 建种再 INCR, 而**不是** INCR 后 EXPIRE: 后者每次都刷新 TTL,
        # 窗口就变成了"距上次点开 24 小时"而不是"距第一次 24 小时" —— 每天点开一次
        # 的用户计数会跨天无限累积, 说出"今天已经第 30 次"而那是几周攒的。
        # INCR 不会清掉已有 TTL, 所以种下的过期时间会一直有效。
        pipe.set(count_key, 0, ex=_COUNT_TTL_SECONDS, nx=True)
        pipe.incr(count_key)
        acquired, _seeded, count = await pipe.execute()
        return bool(acquired), int(count or 1)
    except Exception as e:
        logger.warning("quick-exit cooldown unavailable, reacting anyway: %s", e)
        return True, 1


def _bucket(action_count: int, repeat: int) -> str:
    """这局属于哪种处境。重复优先 —— 反复摆又收比走了几步更值得说。"""
    if repeat >= _REPEAT_THRESHOLD:
        return "repeated"
    return "untouched" if action_count <= 0 else "barely"


def quick_exit_line(*, action_count: int, repeat: int) -> str:
    """开局就走时说的那一句.

    同档内按 repeat 轮换而不是随机取 —— 随机会连着撞同一句 (实测 7 次里 5 次
    撞在第一句上就是这么来的), 轮换能保证相邻两次不同。
    """
    lines = _LINES[_bucket(action_count, repeat)]
    return lines[max(0, repeat - 1) % len(lines)]
