"""时间推理召回评测集 (v1).

为什么单独建这一套: LoCoMo (ACL 2024) 在 32 轮长会话上实测, LLM 的**时间推理**落后
人类 73%, 而整体只落后 56% —— 时间是长期记忆里最弱的一维。2026 年 SOTA 的 Chronos
把提升的 58.9% 归因于"结构化事件日历", 即让每条记忆锚定可查询的时间区间, 检索时按
时间过滤而不只是按相似度。

我们手上已经有 statement_time / occur_time 双字段和 search_by_time_range, 但生产
数据显示: 用户记忆里只有 13% 填了 occur_time, 而近 30 天只有 2% 的消息含显式时间
表达 (时间检索通路因此基本不触发)。这套用例就是要量出这个差距有多大。

题型对齐 LoCoMo 的时间推理分类:

    point       某个时间点发生了什么 ("上周三我说要做什么")
    range       某段时间内发生了什么 ("这个月我提过哪些计划")
    order       先后关系 ("换工作是在搬家之前还是之后")
    duration    持续时长 ("我学吉他多久了")
    update      同一事实的时间演进 ("我现在住哪" —— 需要最新那条而非最早那条)
    relative    相对当下 ("我最近一次去健身是什么时候")

每类都同时给出"只靠语义相似度能不能命中"的判断 —— 这是关键: 如果一道题不查时间也
能靠语义答对, 它就测不出时间能力。标 `needs_time=True` 的才是真正的时间题。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

# 所有相对时间以此为基准, 保证用例可复现 (不随运行日期漂移)。
NOW = datetime(2026, 7, 29, 20, 0)


def _days_ago(n: int) -> datetime:
    return NOW - timedelta(days=n)


@dataclass(frozen=True)
class TemporalSeed:
    """一条带时间锚点的种子记忆.

    occur_time 是事件发生时刻, statement_time 是用户说这句话的时刻 —— 两者可以差
    很远 ("我上个月去了西藏" 说于今天, 事件在一个月前), 而正是这个差值让时间推理
    变得必要。
    """

    id: str
    text: str
    main: str
    sub: str
    occur_time: datetime | None
    statement_time: datetime
    source: str = "user"
    importance: float = 0.6
    level: int = 2


@dataclass(frozen=True)
class TemporalCase:
    id: str
    kind: str          # point | range | order | duration | update | relative
    query: str
    expect_hit: tuple[str, ...]
    # 只靠语义相似度能否命中。False 的题才真正在考时间能力 —— 语义能答对的题,
    # 就算全对也说明不了系统有时间推理。
    needs_time: bool = True
    note: str = ""


# ── 种子: 一个用户三个月的生活片段 ────────────────────────────────────────
#
# 刻意安排了几组"语义相近但时间不同"的记忆 (三次面试、两次搬家意向), 因为纯向量
# 检索对这类完全无能为力 —— 它们的相似度几乎一样, 只有时间能区分。

SEED_BANK: tuple[TemporalSeed, ...] = (
    # ── 三次面试: 语义几乎相同, 只有时间不同 ──
    TemporalSeed(
        "interview_1", "去一家做教育软件的公司面试了，感觉一般",
        "生活", "工作", _days_ago(75), _days_ago(75),
    ),
    TemporalSeed(
        "interview_2", "又去面试了，这次是家做医疗器械的，聊得还不错",
        "生活", "工作", _days_ago(40), _days_ago(40),
    ),
    TemporalSeed(
        "interview_3", "上周去面了一家游戏公司，对方当场给了口头 offer",
        "生活", "工作", _days_ago(7), _days_ago(5),
    ),

    # ── 居住地演进: 同一属性的时间更新 ──
    TemporalSeed(
        "live_old", "我住在苏州工业园区，通勤挺方便",
        "身份", "居住地", _days_ago(90), _days_ago(90),
        importance=0.86, level=1,
    ),
    TemporalSeed(
        "move_plan", "在考虑搬到上海，还没定",
        "生活", "计划", _days_ago(30), _days_ago(30),
    ),
    TemporalSeed(
        "live_new", "已经搬到上海了，住在杨浦",
        "身份", "居住地", _days_ago(10), _days_ago(10),
        importance=0.86, level=1,
    ),

    # ── 持续性事件: 有起点, 用于算时长 ──
    TemporalSeed(
        "guitar_start", "报了个吉他班，从今天开始学",
        "生活", "爱好", _days_ago(120), _days_ago(120),
    ),
    TemporalSeed(
        "guitar_now", "吉他还在坚持练，最近在啃扫弦",
        "生活", "爱好", _days_ago(3), _days_ago(3),
    ),

    # ── 健身: 用于"最近一次" ──
    TemporalSeed(
        "gym_1", "今天去健身房练了腿", "生活", "运动", _days_ago(45), _days_ago(45),
    ),
    TemporalSeed(
        "gym_2", "又去健身了，练的背", "生活", "运动", _days_ago(20), _days_ago(20),
    ),
    TemporalSeed(
        "gym_3", "昨天去健身房了，这次练胸", "生活", "运动", _days_ago(2), _days_ago(1),
    ),

    # ── 无时间锚点的对照组: 检验系统会不会把它们错当成有时间的 ──
    TemporalSeed(
        "like_coffee", "我特别喜欢喝手冲咖啡", "偏好边界", "饮食", None, _days_ago(60),
        importance=0.86, level=1,
    ),
    TemporalSeed(
        "fear_height", "我有点恐高，坐缆车都紧张", "情绪", "恐惧", None, _days_ago(55),
        importance=0.86, level=1,
    ),
)


# ── 用例 ──────────────────────────────────────────────────────────────────

CASES: tuple[TemporalCase, ...] = (
    # point: 某个时间点
    TemporalCase(
        "point_last_week_interview", "point",
        "我上周面试的是哪家公司？",
        ("interview_3",),
        note="三次面试语义几乎相同, 只有 occur_time 能区分。纯向量必然三条都召回",
    ),
    TemporalCase(
        "point_two_months_ago", "point",
        "两个多月前我去面的那家是做什么的？",
        ("interview_1",),
    ),

    # range: 某段时间内
    TemporalCase(
        "range_this_month_gym", "range",
        "我这个月去过几次健身房？",
        ("gym_3",),
        note="需要按时间窗过滤; 语义检索会把三次都拉出来, 答案就错了",
    ),
    TemporalCase(
        "range_recent_interviews", "range",
        "最近一个月我面试过几家？",
        ("interview_3",),
    ),

    # order: 先后关系
    TemporalCase(
        "order_move_vs_interview", "order",
        "我是先搬的家还是先拿到 offer 的？",
        ("live_new", "interview_3"),
        note="要同时取两条并比较 occur_time —— 多跳时间推理",
    ),

    # duration: 时长
    TemporalCase(
        "duration_guitar", "duration",
        "我学吉他多久了？",
        ("guitar_start",),
        note="需要 occur_time 与当下作差; 只召回 guitar_now 答不出时长",
    ),

    # update: 同一事实的演进
    TemporalCase(
        "update_where_live", "update",
        "我现在住在哪儿？",
        ("live_new",),
        needs_time=True,
        note="live_old 与 live_new 都是「居住地」L1, 语义相似度接近。"
             "取错就是把已经搬走的地址当成现住址 —— 生产上真出过这类矛盾",
    ),

    # relative: 相对当下
    TemporalCase(
        "relative_last_gym", "relative",
        "我最近一次去健身是什么时候？",
        ("gym_3",),
    ),

    # 对照组: 不需要时间也能答对的题
    TemporalCase(
        "control_coffee", "point",
        "我喜欢喝什么？",
        ("like_coffee",),
        needs_time=False,
        note="纯语义题。它的作用是确认检索本身没坏 —— 如果连这个都错, "
             "那时间题的失败就不能归因于时间能力",
    ),
    TemporalCase(
        "control_fear", "point",
        "我怕什么？",
        ("fear_height",),
        needs_time=False,
    ),
)
