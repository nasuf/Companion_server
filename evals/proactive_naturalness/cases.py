"""主动交流·自然度评测用例库.

背景 (2026-09): 主动交流的热点+卡片链路是本周新上的功能 (ba6dd42), 生产已开
enabled=True/probability=1.0, 但实测拿到的 trending 内容是名人八卦 / 网站 UI
残余 / API tracker JSON, 豆包读到几乎全部忽略回 "最近咋样"; 而卡片按 topic 独立
搜, 跟消息完全没关系. 目标群体 (20+ 一线城市) 的真人朋友互动是**多源的**:
用户兴趣 / AI 自己兴趣 / 纯社交谈资, 三种都合法, 但都要求"消息-来源-卡"三者
有可见的耦合. 眼前形态既没多源分化, 也没耦合. 本 eval 就是这些性质的度量.

## 三档话题源 (对齐 V3 设计)

  user_interest_match  勾用户: "你不是说过 X 吗, 刚看到 Y"
  ai_persona_match     分享自己: "我最近迷上 X" / "我在 X, 你听过吗"
  socially_hot         公共谈资: "刷到一个热搜" / "你听说了吗那个 X"

## 判定维度 (LLM judge, 见 judge.py)

  naturalness         这句话像真人朋友主动发的吗? (核心)
  source_fit          消息里的表达跟号称的话题源风格匹配吗?
  persona_match       跟 agent 人设一致吗?
  mentions_card       (仅当 card 挂了) 消息里明确提到卡里的内容了吗?
  advertorial_feel    有没有"AI 在推广告"的负面观感? (关键反指标, want False)

## 与其它 eval 的关系

  temporal_awareness  测时间感知回复 (被动响应用户)
  persona_drift       测长对话人设一致性
  proactive_naturalness  <-- 本 eval, 测主动发送时的朋友感
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendingCandidate:
    """一条候选热点/内容, mock 传给 message LLM 的 trending section."""
    title: str
    snippet: str
    url: str = ""
    platform: str = ""  # 微博 / 小红书 / B站 / 知乎 / 头条 (真实内容源)


@dataclass(frozen=True)
class ProactiveCase:
    """一条主动交流用例. 案例喂给 runner, runner 生成消息 + card, judge 判自然度."""
    id: str
    source_kind: str  # user_interest_match | ai_persona_match | socially_hot | none

    # Agent 人设片段 (给 judge 判 persona_match 用; runner 用真 agent 覆盖也可)
    agent_name: str
    agent_persona_brief: str   # e.g. "小菁, ENFP, 皮具制作师, 独立音乐爱好者"

    # 用户上下文 (给 user_interest 档必需, 其它档只作背景)
    user_portrait: str  # e.g. "25岁上海白领, 喜欢摄影/露营"

    # 场景 (决定用哪个 proactive prompt)
    trigger_type: str = "silence_wakeup"
    stage: str = "warming"

    # trending 候选池 (mock, 3-5 条真实文本的内容). None=不注入 trending.
    trending_candidates: tuple[TrendingCandidate, ...] = ()

    # 期望是否出卡 (V3 卡片二次概率决定; 这里作为 case 的 ground truth 标注)
    should_emit_card: bool = False

    note: str = ""


# ── 内容源: 真实感样例 (跟真实"该拿到什么"接近) ──

_USER_PHOTOGRAPHY = TrendingCandidate(
    title="手机也能拍出电影感夜景 - 3 个参数就够",
    snippet="ISO 拉到 800 以下, 快门 1/30, 白平衡手动. 关键是找光源反射面...",
    url="https://www.bilibili.com/video/BV1abc", platform="B站",
)
_USER_HIKING = TrendingCandidate(
    title="国庆想去 但不想人挤人 - 华东小众徒步 5 条",
    snippet="莫干山北面山脊线, 全程 4 小时, 周末不到 30 人. 装备清单...",
    url="https://xiaohongshu.com/explore/abc123", platform="小红书",
)
_AI_LEATHER = TrendingCandidate(
    title="意大利植鞣革保养 - 干燥季必看",
    snippet="马油 + 白蜡, 一年两次. 千万别用鞋油. 手工皮包会越用越有光泽...",
    url="https://xiaohongshu.com/explore/def456", platform="小红书",
)
_AI_INDIE_MUSIC = TrendingCandidate(
    title="腰乐队新专辑发布 - 十年沉淀",
    snippet="录音室专辑, 保留了他们标志性的诗性叙事, 编曲更内敛...",
    url="https://music.163.com/album?id=99", platform="网易云",
)
_HOT_IPHONE = TrendingCandidate(
    title="iPhone 17 首销破纪录 果粉半夜排队",
    snippet="Pro Max 定价上调 200, 但首销依旧秒空. 天猫官旗店 5 秒售罄...",
    url="https://weibo.com/1234567/abc", platform="微博",
)
_HOT_DRAMA = TrendingCandidate(
    title="《漫长的季节 2》定档 - 悬疑迷坐不住了",
    snippet="导演辛爽携原班人马回归, 依旧是 90 年代东北底色...",
    url="https://www.zhihu.com/question/12345", platform="知乎",
)
_HOT_WEATHER = TrendingCandidate(
    title="全国降温 20 度 断崖式入秋",
    snippet="华北华东周末最低 8 度, 华南下周入秋. 建议翻出秋裤...",
    url="https://weibo.com/2/def", platform="微博",
)

# 反例: 名人八卦 (当前生产 tavily 抓到的典型垃圾), 用作 V0 baseline 输入
_JUNK_CELEB_1 = TrendingCandidate(
    title="小七龄童去世 热度 259w",
    snippet="XX 家人透露, 因病去世, 享年 63 岁. 生前最后动态...",
    url="", platform="微博",
)
_JUNK_CELEB_2 = TrendingCandidate(
    title="韩安冉出车祸 车头凹陷",
    snippet="据经纪人朋友圈, 事发凌晨, 无生命危险但需静养...",
    url="", platform="微博",
)
_JUNK_UI = TrendingCandidate(
    title="微博热搜榜 - 今日热榜聚合",
    snippet="科技 娱乐 社区 购物 财经 开发 简报 AI 更多 报刊 设计 校务...",
    url="", platform="微博",
)


# ── 12 个用例, 三档均衡 + 1 组 V0 baseline (垃圾内容) ──

CASES: tuple[ProactiveCase, ...] = (
    # ── user_interest_match: 用户兴趣命中, 消息应勾用户 ──
    ProactiveCase(
        "user-photography-tips", "user_interest_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="25岁上海白领, 喜欢摄影和露营",
        trending_candidates=(_USER_PHOTOGRAPHY,),
        should_emit_card=True,
        note="用户提过摄影, 命中. 期望消息勾住摄影 + 提到那个视频",
    ),
    ProactiveCase(
        "user-hiking-plan", "user_interest_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="26岁北京程序员, 户外爱好者, 常去徒步",
        trending_candidates=(_USER_HIKING,),
        should_emit_card=True,
        note="露营/徒步命中. 期望勾'国庆想不想去人少的地方' + 卡",
    ),
    ProactiveCase(
        "user-only-no-card", "user_interest_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="24岁广州设计师, 喜欢咖啡",
        trending_candidates=(),  # 没找到匹配内容
        should_emit_card=False,
        note="用户兴趣就是咖啡但今天没搜到相关热点. 应回落到关切开场, 不出卡",
    ),

    # ── ai_persona_match: AI 自己兴趣命中 ──
    ProactiveCase(
        "ai-leather-craft", "ai_persona_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 苏州皮具制作师",
        user_portrait="24岁上海HR, 无明显兴趣标签",
        trending_candidates=(_AI_LEATHER,),
        should_emit_card=True,
        note="皮具是 AI 自己的活儿, 分享保养技巧. 期望第一人称视角",
    ),
    ProactiveCase(
        "ai-indie-music", "ai_persona_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 独立音乐爱好者",
        user_portrait="27岁深圳产品经理, 平时听流行乐",
        trending_candidates=(_AI_INDIE_MUSIC,),
        should_emit_card=True,
        note="AI 迷腰乐队, 用户不一定, 但分享是自然的. 期望 '我最近在听...' 语气",
    ),
    ProactiveCase(
        "ai-no-match", "ai_persona_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="22岁武汉学生",
        trending_candidates=(_HOT_IPHONE,),  # AI 不 particular 关心
        should_emit_card=False,
        note="AI 档但没有真兴趣素材. 应换档 or 弱化, 不该硬凹 '我最近在关注 iPhone'",
    ),

    # ── socially_hot: 纯社交谈资, 有质量 ──
    ProactiveCase(
        "hot-iphone-release", "socially_hot",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="25岁上海白领",
        trending_candidates=(_HOT_IPHONE,),
        should_emit_card=False,  # 纯谈资默认不带卡
        note="iPhone 首销. 大家都在聊, 期望 '你换 17 了吗' / '刷到 iPhone 那个了吗' 感觉",
    ),
    ProactiveCase(
        "hot-drama-release", "socially_hot",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="28岁北京公务员",
        trending_candidates=(_HOT_DRAMA,),
        should_emit_card=True,  # 剧类的可以带 zhihu 链接讨论区
        note="爆剧续集. 期望 '看到第二季定档了吗' + 讨论卡",
    ),
    ProactiveCase(
        "hot-weather-drop", "socially_hot",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="26岁杭州白领",
        trending_candidates=(_HOT_WEATHER,),
        should_emit_card=False,
        note="降温. 典型日常谈资, 期望 '你那边冷不冷' 而不是 '刷到降温热搜'",
    ),

    # ── V0 baseline: 现状生产会拿到的垃圾内容 (八卦 / UI 残余) ──
    ProactiveCase(
        "v0-junk-celeb-death", "socially_hot",  # 声称是社交谈资, 但内容恶劣
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="25岁上海白领",
        trending_candidates=(_JUNK_CELEB_1, _JUNK_CELEB_2, _JUNK_UI),
        should_emit_card=False,
        note="现状 baseline: tavily 抓到的垃圾. 陪伴 agent 主动提名人死讯 = 灾难. "
             "**期望模型拒绝使用**, 消息不带这些内容, judge 应给 naturalness=True 但"
             "source_fit=True (因为它'正确地'没用垃圾). 拒绝率是关键指标.",
    ),

    # ── 极端 case: 什么 trending 都没抓到 (常见, tavily 也会返空) ──
    ProactiveCase(
        "no-trending-fallback", "none",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="25岁上海白领",
        trending_candidates=(),
        should_emit_card=False,
        note="baseline 对照: 完全不注入 trending 的现有形态. 生成'最近咋样'这种也算"
             "naturalness=True, 但不该硬套 hotness 表达",
    ),

    # ── 极端 case: 消息 LLM 拿到多条候选, 应挑一条最能勾用户的 ──
    ProactiveCase(
        "user-multi-candidate", "user_interest_match",
        agent_name="小菁", agent_persona_brief="小菁, ENFP, 皮具制作师",
        user_portrait="25岁上海白领, 喜欢摄影和露营",
        trending_candidates=(_USER_PHOTOGRAPHY, _USER_HIKING, _HOT_IPHONE),
        should_emit_card=True,
        note="3 条候选里 2 条命中用户兴趣, 1 条无关. 期望 LLM 从 2 条命中里挑, 不选 iPhone",
    ),
)

SOURCE_KINDS: tuple[str, ...] = (
    "user_interest_match", "ai_persona_match", "socially_hot", "none",
)
