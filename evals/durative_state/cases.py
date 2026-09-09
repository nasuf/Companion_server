"""持续性用户状态(durative state)评测用例库。

背景 (P2): 系统把"我这周去三亚出差四天"当作一个 occur_time 点事件存下, 但没有
**状态有效窗**的概念 —— 状态在窗口内是"进行中"、窗口过了就"已结束"。真人聊天里
这个区别很关键: 出差第 2 天问"三亚玩得咋样"很自然; 出差结束一周后还问"你还在
三亚吗"就出戏。

temporal_awareness 已覆盖"大间隔后别无端复活旧话题" (stale_topic), 但只测了
**别把过去当现在**这一个方向。这套补的是**有效窗内外的双向判定**:
  active  —— 状态窗还开着 → 可以当"进行中"聊, 不该说成已结束("出差回来啦")
  expired —— 状态窗已关 → 不该当"还在进行"("还在出差吧？"), 该当过去或不提

模型拿不到任何显式 valid_until, 只有历史里那句带时长的陈述 + 时间戳 + 当前时刻,
要自己推 "说于 D 前 + 持续 N 天 → 现在在窗内还是窗外"。这正是 Temporal Blindness
(arXiv:2510.23853) 说 LLM 最不擅长的时长推算。基线会告诉我们: 豆包能不能从上下文
自己推出来 (像它处理重逢那样), 还是真需要把状态窗显式注入 prompt。

每个用例控制 (run_eval.py 注入):
  stated_hours_ago  状态陈述发生在多少小时前 (决定它落在历史时间轴的位置)
  now_hour          当前本地时刻
  message           当前这句用户消息 (中性, 不重新提起状态)
两层判定: 确定性红线 (露骨的方向性硬伤词) + LLM judge (有效窗内外处理是否正确)。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DurativeCase:
    id: str
    kind: str  # active | expired —— 现在相对状态窗的位置 (judge 的 ground truth)
    state_line: str          # 用户当初陈述状态的原话 (带时长线索)
    state_ack: str           # AI 当时的回应 (让历史像真对话)
    duration_hint: str       # 人读的时长描述 (给 judge 当 ground truth, 非注入)
    stated_hours_ago: float  # 陈述距今多少小时
    message: str             # 当前用户消息 (中性, 不重提状态)
    now_hour: int = 15
    # 确定性红线: 出现即方向性硬伤 (仅放能靠字面稳抓的; 细腻处交给 judge)
    must_not_contain: tuple[str, ...] = ()
    note: str = ""


CASES: tuple[DurativeCase, ...] = (
    # ── active: 状态窗还开着, 不该说成"已结束/回来了" ──
    DurativeCase(
        "active-trip-day2", "active",
        "我明天开始去三亚出差，待四天", "出差呀，路上注意安全～",
        duration_hint="四天的出差, 现在是第 2 天 (窗内)",
        stated_hours_ago=24, now_hour=15,
        message="累死了，一天到晚连轴转",
        must_not_contain=("出差回来", "回来啦", "回来了吗", "结束了吧"),
        note="出差第2天喊累: 该顺着'出差辛苦', 不该当成已回来"),
    DurativeCase(
        "active-exam-week", "active",
        "这周都在考试，考完就解放了", "加油！考完带你好好放松",
        duration_hint="一周的考试周, 现在第 3 天 (窗内)",
        stated_hours_ago=3 * 24, now_hour=9,
        message="早，又要早起了",
        must_not_contain=("考完了吧", "考完啦", "都结束了"),
        note="考试周第3天: 状态仍进行, 不该当成考完了"),
    DurativeCase(
        "active-sick", "active",
        "感冒了，医生说得歇一周", "多喝热水好好休息，别硬扛",
        duration_hint="一周的病假, 现在第 2 天 (窗内)",
        stated_hours_ago=30, now_hour=20,
        message="今天没什么胃口",
        must_not_contain=("病好了吧", "康复了", "好利索了"),
        note="生病第2天没胃口: 该关心还没好, 不该当作痊愈"),

    # ── expired: 状态窗已关, 不该当"还在进行中" ──
    DurativeCase(
        "expired-trip", "expired",
        "我后天去三亚出差，就待三天", "好呀，玩得开心点～",
        duration_hint="三天的出差, 说于 8 天前 → 早已结束 (窗外)",
        stated_hours_ago=8 * 24, now_hour=15,
        message="终于闲下来了，想找部剧看",
        must_not_contain=("还在三亚", "出差还顺利吗", "还在出差", "三亚待得怎么样"),
        note="出差3天早结束: 不该问'还在三亚吗'"),
    DurativeCase(
        "expired-exam", "expired",
        "这两天期中考，考完请你吃饭", "没问题，等你好消息",
        duration_hint="两天的考试, 说于 6 天前 → 早已结束 (窗外)",
        stated_hours_ago=6 * 24, now_hour=18,
        message="最近想报个健身房",
        must_not_contain=("还在考试", "考试加油", "考得怎么样今天", "还在复习吧"),
        note="两天期中考6天前: 不该当成还在考"),
    DurativeCase(
        "expired-visit", "expired",
        "爸妈来我这住三天，周末走", "陪陪他们，难得聚聚",
        duration_hint="三天的探访, 说于 9 天前 → 早已结束 (窗外)",
        stated_hours_ago=9 * 24, now_hour=21,
        message="一个人在家有点无聊",
        must_not_contain=("爸妈还在吗", "还陪着爸妈", "陪爸妈呢", "他们还没走"),
        note="爸妈住3天9天前: 用户说一个人在家, 不该问爸妈还在不在"),
    DurativeCase(
        "expired-biztrip-long", "expired",
        "下周出差一趟，大概去五天", "五天不短呢，照顾好自己",
        duration_hint="五天的出差, 说于 15 天前 → 早已结束 (窗外)",
        stated_hours_ago=15 * 24, now_hour=11,
        message="今天天气总算凉快了",
        must_not_contain=("还在出差", "出差顺利吗", "还没回来吧", "出差怎么样"),
        note="五天出差15天前: 早回来了, 不该问出差"),
)

KINDS: tuple[str, ...] = ("active", "expired")
