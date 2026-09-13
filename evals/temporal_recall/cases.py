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
    # 另一种无 occur_time 的 L1: 稳定职业身份. 用它给 recency_stable_l1_coffee
    # 补一道不同领域的重复实验 (recency_stable_l1_job), 避免"就是咖啡这一道特例"的可能.
    TemporalSeed(
        "profession_designer", "我是产品设计师，在一家做工业软件的公司",
        "身份", "职业/与经济", None, _days_ago(180),
        importance=0.90, level=1,
    ),
    # 一条新鲜但完全无关的事件: 用来测"P3 boost 会不会把新鲜噪音顶上来".
    # 语义跟大多数求近查询都不搭, 但 occur_time=1d 让它拿到接近满 boost.
    TemporalSeed(
        "unrelated_fresh", "刚点了顿麻辣香锅外卖，加了鹌鹑蛋",
        "生活", "日常", _days_ago(1), _days_ago(1),
    ),
    # 一条相似度会很高但事件比较老的餐饮偏好, 用来测 P3 是否会打压 semantic winner.
    TemporalSeed(
        "old_coffee_event", "上次跟同事去了家精品咖啡店，喝了瑰夏，好喝到爆",
        "生活", "日常", _days_ago(50), _days_ago(50),
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

    # ── 求近对抗组: 查询字面含求近词但正确答案不是"最新的" ─────────────────
    # 这组是 P3 权重扫描的"上限探测器": 权重加大时若这些题从对变错, 就是过头的
    # 副作用。加权重前必须看到它们仍对; 加完仍对 = 加得起, 加完变错 = 越界。
    # 前 3 道是 2026-09-13 那次拆解加的; 后 7 道 (2026-09-13 后半) 补齐, 让"扫描
    # 未见回退" 从弱结论变成"覆盖了 10 种不同失败模式仍未见回退"的可信结论。

    # (a) 求近但正确答案就是最新那次 —— P3 应该帮忙, 不能反打错
    TemporalCase(
        "recency_but_want_old_gym", "range",
        "我最近一次去健身房练的是哪个部位？",
        ("gym_3",),
        needs_time=True,
        note="求近词'最近一次', 正确答案就是最新那次 → P3 应该帮忙, 不能打错",
    ),
    TemporalCase(
        "recency_multi_return_full_history", "range",
        "我最近去健身房都在练哪些部位？",
        ("gym_3", "gym_2", "gym_1"),
        needs_time=True,
        note="求近但要 3 条: P3 权重过大会只留最新那条, 挤掉 gym_1/gym_2",
    ),

    # (b) 无 occur_time 的 L1 偏好/身份 vs 有 occur_time 的新鲜事件
    TemporalCase(
        "recency_stable_l1_coffee", "point",
        "我最近喝咖啡还是喝茶多？",
        ("like_coffee",),
        needs_time=False,
        note="求近词'最近', 但正确答案是无 occur_time 的 L1 偏好. P3 机制上不主动"
             "保护它 (occ=None → 无 boost), 但对手 (近期无关事件如 gym_3) 会拿到接近"
             "满值 boost 挤下去 —— 拿这道题量'非对称 boost 引发的漂移'",
    ),
    TemporalCase(
        "recency_stable_l1_job", "point",
        "你现在做的是什么工作？",
        ("profession_designer",),
        needs_time=False,
        note="求近词'现在', 但答案是无 occur_time 的 L1 身份 (职业). 跟 _coffee 同型,"
             "换一个话题重跑防特例: 若两道都被 P3 挤下去, 就是系统性倾向"
             "'新鲜事件 > 稳定身份'",
    ),

    # (c) semantic winner 相似度显著高, 但更旧 —— P3 是否会硬把更新但更弱的顶上去
    TemporalCase(
        "recency_semantic_dominates", "point",
        "我最近对咖啡有什么新的感受？",
        # 语义: old_coffee_event 相似度最高 (咖啡+具体感受); like_coffee 次之.
        # 时间: old_coffee_event 是 50 天前, unrelated_fresh 才 1 天前, 但完全离题.
        # 期望排序仍应是 old_coffee_event 领先.
        ("old_coffee_event",),
        needs_time=True,
        note="P3 权重过高时会把 unrelated_fresh (新但离题) 挤到 old_coffee_event"
             "(旧但正中主题) 之前, 检验'相似度悬崖'能不能挡住新鲜噪音",
    ),

    # (d) 求近词命中但期望多目标全都很老 (P3 全部小 boost 之和不该盖过 fresh 噪音)
    TemporalCase(
        "recency_older_series_all_wanted", "range",
        "最近几次面试情况都怎样？",
        ("interview_1", "interview_2", "interview_3"),
        needs_time=True,
        note="求近词命中, 但期望是全部 3 次面试 —— P3 会重奖 interview_3, 若权重"
             "过大, 排到 top-3 的第 3 位可能被 unrelated_fresh / gym_3 之类挤掉",
    ),

    # (e) 求近词命中但正确答案就是 3 条 fresh 事件里的"中间那次"
    TemporalCase(
        "recency_pick_middle", "point",
        "我最近倒数第二次去健身练的是什么？",
        ("gym_2",),
        needs_time=True,
        note="'倒数第二次'触发求近但答案是 gym_2 (中间那次, 20 天前). P3 会把 gym_3"
             "顶到 top-1, 这道必错 —— 但真正的问题是: 加权重会不会更错. 保留这道即"
             "留一个'P3 帮不了但也别更糟'的红线",
    ),

    # (f) 求近词命中但语义指向明确的旧标记事件 (offer + 上周 = 面试)
    TemporalCase(
        "recency_specific_named_event", "point",
        "最近拿到 offer 那家公司是做什么的？",
        ("interview_3",),
        needs_time=True,
        note="求近 + 语义强指向 (拿到 offer). P3 应帮忙但不该被过强的语义盖过",
    ),

    # (g) 全都是老事件时 P3 应"哑火"(occ 都很小, boost 都很微弱)
    TemporalCase(
        "recency_all_old_no_effect", "range",
        "我最近去过几家公司面试？",
        ("interview_3",),
        needs_time=True,
        note="面试都是几十天前的, P3 的 boost 应该都很微弱 (无差别衰减). 这道跟"
             "range_recent_interviews 是同一 query 不同措辞, 用来看权重扫描的稳定性",
    ),
)
