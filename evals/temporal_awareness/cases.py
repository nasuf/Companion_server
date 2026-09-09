"""时间感知回复评测的用例库。

背景 (P4, 2026-09): 系统已有相当完整的时间感知机制 —— 历史消息带 [MM-DD HH:MM]
前缀、重逢感知段 (三档 gap 文本)、话题栈重置、AI 作息状态。但从没有**回复层**的
度量: 隔了两天回来 AI 会不会当无缝续聊? 凌晨会不会说"早上好"? 5 分钟的间隔会不会
硬凹"好久不见"? 过期的旧话题会不会被无端接上? temporal_recall 测的是检索召回,
不是回复是否真的用对了时间。

这是"先有度量再改体感"里的度量 —— P1/P3 改完拿它量提升。

每个用例控制三个时间旋钮 (run_eval.py 注入):
  gap_seconds  当前消息距上一条的间隔 → 驱动重逢感知段 + 历史时间戳
  now_hour     当前本地时刻 (UTC+8 的小时) → 驱动 time_context + 作息状态
  history      间隔前的对话 (决定"旧话题"是什么)

判定分确定性 (must_not_contain: 时段问候词/无缝续聊词/过度重逢词) + LLM judge
(重逢是否得体/有无时间幻觉/有没有无端复活旧话题) 两层, 见 judge.py / standard.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemporalCase:
    id: str
    group: str  # reunion | time_of_day | no_hallucination | stale_topic
    message: str
    gap_seconds: float
    now_hour: int = 15  # UTC+8 当前小时 (默认下午)
    history: tuple[tuple[str, str], ...] = ()
    # 确定性红线: 回复出现其中任一词即判失败 (那句话对这个时间情境是硬伤)
    must_not_contain: tuple[str, ...] = ()
    note: str = ""


_H = 3600
_D = 86400

# 常用历史片段
_MIDTRIP = (
    ("user", "国庆想去趟成都"),
    ("assistant", "成都好呀，你想去几天？"),
    ("user", "大概四五天吧，还没定"),
)
_MIDMOOD = (
    ("assistant", "那你今晚打算怎么放松？"),
    ("user", "可能看个电影"),
)


CASES: tuple[TemporalCase, ...] = (
    # ── reunion: 间隔大小 → 重逢语义该不该出现、出现多重 ──
    TemporalCase(
        "reunion-5min", "reunion", "对了刚说到哪了", gap_seconds=5 * 60,
        now_hour=15, history=_MIDTRIP,
        must_not_contain=("好久不见", "好几天", "这几天没", "这段时间"),
        note="5分钟不是重逢, 硬凹'好久不见'是幻觉"),
    TemporalCase(
        "reunion-45min", "reunion", "回来啦", gap_seconds=45 * 60,
        now_hour=16, history=_MIDTRIP,
        must_not_contain=("好久不见", "好几天"),
        note="45分钟: 轻轻带'刚忙完'可以, 不该大张旗鼓重逢"),
    TemporalCase(
        "reunion-5h", "reunion", "在吗", gap_seconds=5 * _H,
        now_hour=20, history=_MIDTRIP,
        note="5小时: 该自然回应'回来了', 不该直接续成都话题"),
    TemporalCase(
        "reunion-2day", "reunion", "最近咋样", gap_seconds=2 * _D,
        now_hour=15, history=_MIDTRIP,
        note="2天: 有重逢感, 不该接着聊成都行程细节"),
    TemporalCase(
        "reunion-8day", "reunion", "我回来了", gap_seconds=8 * _D,
        now_hour=15, history=_MIDMOOD,
        note="8天: 明显重逢, 惦记/问近况, 别责怪消失"),

    # ── time_of_day: 当前时刻决定问候/语气 ──
    TemporalCase(
        "tod-2am", "time_of_day", "睡不着", gap_seconds=6 * _H,
        now_hour=2, history=(),
        must_not_contain=("早上好", "早安", "上午", "中午好", "下午"),
        note="凌晨2点: 该有夜里的关心, 绝不能说'早上好'"),
    TemporalCase(
        "tod-morning", "time_of_day", "早", gap_seconds=9 * _H,
        now_hour=7, history=(("user", "先睡了晚安"), ("assistant", "晚安好梦")),
        must_not_contain=("晚上好", "下午好", "深夜"),
        note="隔夜早上7点: '早'合理, 不能说'晚上好'"),
    TemporalCase(
        "tod-latenight-work", "time_of_day", "还在忙", gap_seconds=30 * 60,
        now_hour=1, history=(),
        must_not_contain=("早上好", "上午", "下午好"),
        note="凌晨1点: 时段感知, 不能用白天问候"),

    # ── no_hallucination: 不虚构流逝时间 / 不无缝续接 ──
    TemporalCase(
        "hall-seamless-3day", "no_hallucination", "在不在", gap_seconds=3 * _D,
        now_hour=11, history=_MIDTRIP,
        must_not_contain=("刚才我们", "刚说到", "接着刚才", "继续刚才", "刚聊到"),
        note="3天后不能装作'刚才还在聊'"),
    TemporalCase(
        "hall-overclaim-3min", "no_hallucination", "嗯嗯你继续", gap_seconds=3 * 60,
        now_hour=14, history=_MIDTRIP,
        must_not_contain=("好久不见", "这么久", "好几天", "这段时间", "终于"),
        note="3分钟不能过度宣称间隔"),

    # ── stale_topic: 大间隔后不无端复活旧话题 ──
    TemporalCase(
        "stale-newmood", "stale_topic", "今天好累啊", gap_seconds=2 * _D,
        now_hour=21, history=_MIDTRIP,
        note="隔2天用户说累: 该接'累', 不该无端追问成都行程"),
)

GROUPS: tuple[str, ...] = ("reunion", "time_of_day", "no_hallucination", "stale_topic")
