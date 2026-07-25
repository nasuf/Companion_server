"""Reply-register eval case bank (v1).

Measures **语域** — whether she sounds like a friend or like a service —
rather than factual correctness, which the memory_recall eval already covers.
Thresholds and their justification live in ``standard.py``.

Case shapes are grounded in published corpora rather than in our own database,
because most conversations in production were produced by the team testing the
product; fitting the bank to them would fit test behaviour, not users.

    真实中文 IM     5.64 字/行, 1.84 行/次   (即時通訊文字訊息對話特性)
    WhatsApp        1-2 词 33% | 3-5 词 34% | >20 词 <4%
    LCCC 微博对话   单轮 6.79 词/句, 多轮 8.32 词/句

The one thing our production data does corroborate is the input shape: users
average 7.9 characters per message, which lines up with the IM references. So
cases are written short and elliptical, meaningful only against the preceding
turns — a bank of well-formed standalone questions would measure a
distribution neither the research nor the product ever sees.

Three groups, one failure mode each:

- ``fact``     百科腔 — answering a factual question by emptying a drawer of
               facts with nothing of her own attached. Deliberately
               over-sampled: a scan of production found exactly one genuine
               encyclopedia question, so this is a tail risk rather than a
               live epidemic — and it is the tail the product owner asked
               about. That one real failure is kept verbatim as a golden case,
               following the benchmark practice of building from observed
               production failures rather than synthetic prompts alone.
- ``chitchat`` 过度展开 — meeting a three-character message with a paragraph,
               or wandering off what was actually said.
- ``emotion``  客服腔 — leading with advice, analysis or interrogation when
               both the product rule and the ESConv strategy ordering put
               acknowledgement first.

Grow the bank when a register bug ships: add the failing (history, message,
group) here first, then fix.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterCase:
    id: str
    group: str  # "fact" | "chitchat" | "emotion"
    message: str
    # (role, content) — role is "user" | "assistant", oldest first.
    history: tuple[tuple[str, str], ...] = ()
    note: str = ""


# ── fact: 外部事实提问 → 百科腔风险 ────────────────────────────────────────
# 混了三种子形态: 冷门专名 (联网判定曾漏判的那类) / 常识 / 带安全约束的健康问题.

FACT_CASES: tuple[RegisterCase, ...] = (
    RegisterCase(
        id="fact_yongle_palace",
        group="fact",
        # 逐字保留生产原句 (含错字与两问合一) — golden case 改写就失去了意义.
        message="你知道运城永乐宫建于哪一年 有那怎样的历史故事吗？",
        history=(
            ("user", "国庆想去山西玩"),
            ("assistant", "山西好啊"),
            ("assistant", "你想去哪几个地方"),
        ),
        note="生产真实失败: 回了「它是元代1247年动工的哦👌」— 纯数字, 且漏答历史故事",
    ),
    RegisterCase(
        id="fact_yongle_palace_short",
        group="fact",
        message="永乐宫哪年建的",
        history=(("user", "国庆想去山西"),),
        note="同一问题的 IM 常见短形态 — 对照 golden case 看长度是否影响语域",
    ),
    RegisterCase(
        id="fact_everest_height",
        group="fact",
        message="珠峰多高来着",
        history=(
            ("user", "刚看了个纪录片"),
            ("assistant", "啥纪录片呀"),
            ("user", "登山的"),
        ),
    ),
    RegisterCase(
        id="fact_hongloumeng_author",
        group="fact",
        message="红楼梦谁写的",
        history=(("user", "我妈让我读点名著"),),
    ),
    RegisterCase(
        id="fact_light_year",
        group="fact",
        message="一光年多远啊",
        history=(("user", "刚看到个星空的图 好好看"),),
    ),
    RegisterCase(
        id="fact_sky_blue",
        group="fact",
        message="天为啥是蓝的",
        history=(("user", "今天天气特别好"),),
    ),
    RegisterCase(
        id="fact_cat_water",
        group="fact",
        message="猫为什么怕水",
        history=(("user", "我家猫今天洗澡差点把我挠死"),),
        note="判定 prompt 里明确列为「不需要联网」的常识",
    ),
    RegisterCase(
        id="fact_caffeine",
        group="fact",
        message="咖啡因多久代谢完",
        history=(
            ("user", "下午喝了杯美式"),
            ("assistant", "这个点喝晚上不困吗"),
        ),
    ),
    RegisterCase(
        id="fact_qin_dynasty",
        group="fact",
        message="秦朝一共多少年",
        history=(("user", "在看大秦赋"),),
    ),
    RegisterCase(
        id="fact_terracotta",
        group="fact",
        message="兵马俑在哪个市来着",
        history=(("user", "想去西安"),),
    ),
    RegisterCase(
        id="fact_worldcup",
        group="fact",
        message="世界杯几年一次",
        history=(("user", "最近老看到世界杯的新闻"),),
    ),
    RegisterCase(
        id="fact_titanic",
        group="fact",
        message="泰坦尼克哪年沉的",
        history=(("user", "重看了一遍泰坦尼克号"),),
    ),
    RegisterCase(
        id="fact_panda_color",
        group="fact",
        message="熊猫为啥是黑白的",
        history=(("user", "看熊猫直播看了一下午"),),
    ),
    RegisterCase(
        id="fact_bluetooth_name",
        group="fact",
        message="蓝牙为啥叫蓝牙",
        history=(("user", "耳机又断连了 烦"),),
    ),
    RegisterCase(
        id="fact_jin_to_gram",
        group="fact",
        message="一斤是多少克",
        history=(("user", "在按菜谱做饭"),),
    ),
    RegisterCase(
        id="fact_river_length",
        group="fact",
        message="长江和黄河哪个长",
        history=(("user", "陪我侄子写作业呢"),),
    ),
    RegisterCase(
        id="fact_apple_founder",
        group="fact",
        message="苹果创始人是谁",
        history=(("user", "换了个新手机"),),
    ),
    RegisterCase(
        id="fact_relativity",
        group="fact",
        message="相对论讲的啥",
        history=(("user", "刷到个科普视频没看懂"),),
    ),
    RegisterCase(
        id="fact_social_insurance",
        group="fact",
        message="五险一金哪五险",
        history=(
            ("user", "新公司hr让我填表"),
            ("assistant", "入职手续挺多的吧"),
        ),
    ),
    RegisterCase(
        id="fact_three_body",
        group="fact",
        message="三体讲的什么啊",
        history=(("user", "同事都在聊三体"),),
    ),
    RegisterCase(
        id="fact_gpu_cpu",
        group="fact",
        message="显卡和cpu啥区别",
        history=(("user", "想配台电脑"),),
    ),
    RegisterCase(
        id="fact_ibuprofen",
        group="fact",
        message="布洛芬和扑热息痛啥区别",
        history=(("user", "有点头疼"),),
        note="健康类 — 另有「不诊断不保证效果」的安全规则, 观察是否退化成说明书",
    ),
    RegisterCase(
        id="fact_mortgage_rate",
        group="fact",
        message="房贷利率怎么算的",
        history=(("user", "在看房子"),),
    ),
)


# ── chitchat: 极短省略句 → 过度展开风险 ────────────────────────────────────

CHITCHAT_CASES: tuple[RegisterCase, ...] = (
    RegisterCase(
        id="chat_ack_en",
        group="chitchat",
        message="嗯",
        history=(
            ("user", "今天加班到十点"),
            ("assistant", "这么晚"), ("assistant", "你早点休息吧"),
        ),
    ),
    RegisterCase(
        id="chat_shi_a",
        group="chitchat",
        message="是啊",
        history=(
            ("user", "周末又要下雨"),
            ("assistant", "最近雨是真多"), ("assistant", "哪都去不了"),
        ),
    ),
    RegisterCase(
        id="chat_what_is_that",
        group="chitchat",
        message="那是啥",
        history=(
            ("user", "今天干嘛了"),
            ("assistant", "整理了一下午的落叶纹理"), ("assistant", "手都酸了"),
        ),
    ),
    RegisterCase(
        id="chat_whatever",
        group="chitchat",
        message="都可以",
        history=(
            ("user", "晚上吃啥好"),
            ("assistant", "火锅还是烧烤"), ("assistant", "你想吃哪个"),
        ),
    ),
    RegisterCase(
        id="chat_not_yet_want",
        group="chitchat",
        message="还没有 想看",
        history=(("assistant", "最近那部新片你看了没"),),
    ),
    RegisterCase(
        id="chat_lying_down",
        group="chitchat",
        message="躺着了",
        history=(("assistant", "这会儿在干嘛呢"),),
    ),
    RegisterCase(
        id="chat_bored",
        group="chitchat",
        message="我就是有点无聊",
        history=(
            ("user", "在吗"),
            ("assistant", "在的"), ("assistant", "怎么啦"),
        ),
    ),
    RegisterCase(
        id="chat_how_you_know",
        group="chitchat",
        message="哈哈你咋知道",
        history=(
            ("user", "刚点了外卖"),
            ("assistant", "又是那家麻辣烫吧"),
        ),
    ),
    RegisterCase(
        id="chat_forget_it",
        group="chitchat",
        message="那就算了",
        history=(
            ("user", "周末去爬山不"),
            ("assistant", "我周末排满了诶"),
        ),
    ),
    RegisterCase(
        id="chat_tried_useless",
        group="chitchat",
        message="试过 没啥用",
        history=(
            ("user", "最近老睡不好"),
            ("assistant", "睡前别看手机试试"),
        ),
    ),
    RegisterCase(
        id="chat_and_college",
        group="chitchat",
        message="大学呢",
        history=(
            ("user", "你高中在哪上的"),
            ("assistant", "在老家那边念的"), ("assistant", "离家挺近"),
        ),
    ),
    RegisterCase(
        id="chat_not_quite",
        group="chitchat",
        message="不太对吧",
        history=(
            ("user", "你觉得我是i人还是e人"),
            ("assistant", "感觉你挺e的"), ("assistant", "挺爱说话"),
        ),
    ),
    RegisterCase(
        id="chat_then_what",
        group="chitchat",
        message="然后呢",
        history=(("assistant", "今天路上遇到只三花"), ("assistant", "一直跟着我走"),),
    ),
    RegisterCase(
        id="chat_you_there",
        group="chitchat",
        message="在吗",
        history=(("assistant", "那你早点睡"), ("assistant", "晚安"),),
    ),
    RegisterCase(
        id="chat_eaten_yet",
        group="chitchat",
        message="吃了吗",
        history=(),
    ),
    RegisterCase(
        id="chat_what_doing",
        group="chitchat",
        message="你干嘛呢",
        history=(),
    ),
    RegisterCase(
        id="chat_sleepy",
        group="chitchat",
        message="困了",
        history=(
            ("user", "今天好累"),
            ("assistant", "咋了"), ("assistant", "干啥了这么累"),
            ("user", "搬了一天东西"),
        ),
    ),
    RegisterCase(
        id="chat_just_home",
        group="chitchat",
        message="刚到家",
        history=(("assistant", "下班了没"),),
    ),
    RegisterCase(
        id="chat_weekend_plan",
        group="chitchat",
        message="周末有安排没",
        history=(),
    ),
    RegisterCase(
        id="chat_raining",
        group="chitchat",
        message="下雨了",
        history=(("assistant", "今天外面天气咋样"),),
    ),
    RegisterCase(
        id="chat_me_too",
        group="chitchat",
        message="我也是",
        history=(("assistant", "我最近特别懒"), ("assistant", "啥都不想干"),),
    ),
    RegisterCase(
        id="chat_really",
        group="chitchat",
        message="真的假的",
        history=(("assistant", "我今天走路差点摔了"), ("assistant", "被自己鞋带绊的"),),
    ),
)


# ── emotion: 情绪倾诉 → 客服腔风险 ─────────────────────────────────────────
# 刻意避开危机语义 (「撑不住」「不想活」等) — 危机路径会短路, 会污染语域测量.

EMOTION_CASES: tuple[RegisterCase, ...] = (
    RegisterCase(
        id="emo_annoyed",
        group="emotion",
        message="我好烦",
        history=(),
    ),
    RegisterCase(
        id="emo_tired",
        group="emotion",
        message="很累",
        history=(("assistant", "今天过得咋样"),),
    ),
    RegisterCase(
        id="emo_unhappy",
        group="emotion",
        message="我不开心",
        history=(),
    ),
    RegisterCase(
        id="emo_more_annoyed",
        group="emotion",
        message="更烦了",
        history=(
            ("user", "我好烦"),
            ("assistant", "咋啦"), ("assistant", "谁惹你了"),
        ),
    ),
    RegisterCase(
        id="emo_insomnia",
        group="emotion",
        message="现在经常失眠",
        history=(("assistant", "最近睡得好吗"),),
    ),
    RegisterCase(
        id="emo_off_work_tired",
        group="emotion",
        message="才下班 好累",
        history=(),
    ),
    RegisterCase(
        id="emo_hate_new_boss",
        group="emotion",
        message="换领导了 很讨厌",
        history=(("assistant", "最近工作还顺利吗"),),
    ),
    RegisterCase(
        id="emo_tomorrow_torture",
        group="emotion",
        message="明天还要继续被折磨",
        history=(
            ("user", "今天开了一天会"),
            ("assistant", "一天会开下来人都木了吧"),
        ),
    ),
    RegisterCase(
        id="emo_sad_leaving",
        group="emotion",
        message="我难过去了",
        history=(
            ("user", "养了三年的花死了"),
            ("assistant", "啊"), ("assistant", "怎么突然就死了"),
        ),
    ),
    RegisterCase(
        id="emo_playful_angry",
        group="emotion",
        message="嘿嘿 生气啦",
        history=(("assistant", "你今天怎么话这么少"),),
        note="口是心非的撒娇 — 不该被当成真愤怒来处理",
    ),
    RegisterCase(
        id="emo_not_relaxed",
        group="emotion",
        message="我不轻松",
        history=(("assistant", "听起来你最近还挺轻松的"),),
    ),
    RegisterCase(
        id="emo_scolded",
        group="emotion",
        message="被领导骂了",
        history=(),
    ),
    RegisterCase(
        id="emo_failed_exam",
        group="emotion",
        message="考砸了",
        history=(("assistant", "今天考试咋样"),),
    ),
    RegisterCase(
        id="emo_fight_partner",
        group="emotion",
        message="跟对象吵架了",
        history=(),
    ),
    RegisterCase(
        id="emo_low_mood",
        group="emotion",
        message="心情很低落",
        history=(),
    ),
    RegisterCase(
        id="emo_useless",
        group="emotion",
        message="感觉自己很没用",
        history=(
            ("user", "又被退回来改了"),
            ("assistant", "改了几版了都"),
        ),
    ),
    RegisterCase(
        id="emo_pressure",
        group="emotion",
        message="压力好大",
        history=(),
    ),
    RegisterCase(
        id="emo_cant_sleep",
        group="emotion",
        message="睡不着",
        history=(),
    ),
    RegisterCase(
        id="emo_homesick",
        group="emotion",
        message="有点想家",
        history=(("assistant", "一个人在外面待着还习惯吗"),),
    ),
    RegisterCase(
        id="emo_promoted",
        group="emotion",
        message="我今天升职了",
        history=(),
        note="正向情绪 — 规则要求第一句表达高兴或好奇, 再追问",
    ),
    RegisterCase(
        id="emo_passed_interview",
        group="emotion",
        message="面试过了",
        history=(("user", "今天去面试了"), ("assistant", "紧张不"), ("assistant", "结果啥时候出"),),
    ),
    RegisterCase(
        id="emo_happiest_in_long",
        group="emotion",
        message="好久没这么开心过了",
        history=(("user", "今天见了很久没见的朋友"),),
    ),
)


ALL_CASES: tuple[RegisterCase, ...] = FACT_CASES + CHITCHAT_CASES + EMOTION_CASES

GROUPS: tuple[str, ...] = ("fact", "chitchat", "emotion")
