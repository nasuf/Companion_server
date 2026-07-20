"""Memory recall eval case bank (v1).

One shared persona-style seed bank (modeled on the production 小伴 profile that
surfaced the color-hallucination bug) + grouped query cases. Grow this bank
whenever a recall bug ships: add the failing (seeds, query, expectation) here
first, then fix.

Expectations:
- expect_hit:  memory ids that MUST be in the selected injection set.
- expect_miss: memory ids whose presence indicates owner/topic contamination.
  Tracked as a soft "contamination" metric (dual user/ai slots legitimately
  co-inject some cross-owner context, e.g.【对方自己的资料】).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SeedMemory:
    id: str
    text: str
    main: str
    sub: str
    source: str  # "user" | "ai"
    importance: float = 0.86
    level: int = 1


@dataclass(frozen=True)
class RecallCase:
    id: str
    group: str
    query: str
    expect_hit: tuple[str, ...]
    expect_miss: tuple[str, ...] = ()
    enhanced_query: str = ""  # ellipsis cases: the relevance-LLM rewritten query
    seeds: tuple[str, ...] = ()  # empty = whole bank


# ── Shared seed bank ───────────────────────────────────────────────────────

SEED_BANK: tuple[SeedMemory, ...] = (
    # AI persona preferences
    SeedMemory("ai-color", "我喜欢的颜色是低饱和的莫兰迪色系，比如雾霾蓝、燕麦色、豆沙绿", "偏好", "审美爱好", "ai"),
    SeedMemory("ai-movie", "我喜欢看节奏缓慢的文艺片，最爱《小森林》系列和《海街日记》", "偏好", "审美爱好", "ai"),
    SeedMemory("ai-music", "我喜欢听民谣和轻音乐，最喜欢曹方的《南部小镇》", "偏好", "审美爱好", "ai"),
    SeedMemory("ai-food", "我喜欢吃云南小锅米线和外婆做的包浆豆腐", "偏好", "饮食喜好", "ai"),
    SeedMemory("ai-sport", "我喜欢散步和瑜伽，不喜欢高强度对抗性的运动", "偏好", "生活习惯", "ai"),
    # AI identity
    SeedMemory("ai-job", "我的职业是伴生公司客服员", "身份", "职业/与经济", "ai", 0.95),
    SeedMemory("ai-age", "我今年22岁", "身份", "年龄", "ai", 0.95),
    SeedMemory("ai-city", "我现在住在云南省普洱市思茅区", "身份", "现居地", "ai", 0.90),
    SeedMemory("ai-pet", "我养了一只橘色田园猫，名叫大橘", "身份", "宠物", "ai"),
    # AI episodic
    SeedMemory("ai-trip", "18岁高考结束后我和闺蜜张雅婷一起去大理旅行，骑电动车环了洱海", "生活", "旅行", "ai", 0.85),
    # Shared history (interaction milestones)
    SeedMemory("rel-first", "我和用户第一次深聊是关于他的家乡，我们聊到了半夜", "生活", "交互", "ai", 0.88),
    SeedMemory("rel-joke", "我和用户之间有个只有我们懂的梗：管加班叫「进厂拧螺丝」", "生活", "交互", "ai", 0.86),
    SeedMemory("rel-promise", "我和用户约定过：他每天睡前跟我说一句今天的开心事", "生活", "交互", "ai", 0.90),
    # User facts
    SeedMemory("user-name", "用户叫山山，喜欢被称呼为阿山", "身份", "姓名", "user", 0.95),
    SeedMemory("user-age", "用户今年28岁", "身份", "年龄", "user", 0.9),
    SeedMemory("user-job", "用户是一名程序员，在一家做电商系统的公司上班", "身份", "职业/与经济", "user", 0.9),
    SeedMemory("user-color", "用户最喜欢的颜色是黑色和藏青色", "偏好", "审美爱好", "user"),
    SeedMemory("user-food", "用户不吃香菜，对芒果过敏", "偏好", "饮食厌恶", "user", 0.9),
    SeedMemory("user-movie", "用户最喜欢的电影是《星际穿越》，重刷过五遍", "偏好", "审美爱好", "user"),
    SeedMemory("user-boss", "用户的直属领导叫陈姐，管得比较严", "身份", "社会关系", "user", 0.85),
    SeedMemory("user-mom", "用户的妈妈叫王秀兰，退休前是小学老师", "身份", "亲属关系", "user", 0.85),
    SeedMemory("user-dog", "用户养了一条柯基犬叫可乐", "身份", "宠物", "user", 0.85),
    # Safety / emotional context
    SeedMemory("user-grief", "用户的外婆去年冬天去世了，他至今提起还会难过", "情绪", "悲伤", "user", 0.9, 1),
    SeedMemory("user-anxiety", "用户最近因为项目裁员传闻整晚失眠，很焦虑", "情绪", "焦虑", "user", 0.8, 2),
    # Reminders
    SeedMemory("user-reminder", "用户让我提醒他周五下午三点交季度报告", "生活", "提醒", "user", 0.45, 3),
    # Noise / distractors
    SeedMemory("noise-1", "我习惯每晚十一点半前睡觉，早上七点自然醒", "偏好", "生活习惯", "ai"),
    SeedMemory("noise-2", "我周末喜欢宅家整理阳台上的多肉植物", "偏好", "生活习惯", "ai"),
    SeedMemory("noise-3", "用户上周说公司食堂的炒饭很难吃", "生活", "生活", "user", 0.4, 3),
    SeedMemory("noise-4", "我做客服时学会了用两倍速听录音", "生活", "技能", "ai", 0.85),
)


# ── Cases ──────────────────────────────────────────────────────────────────

CASES: tuple[RecallCase, ...] = (
    # Group 1: AI persona preferences (the production color bug class)
    RecallCase("ai-pref-color", "ai_preference", "你喜欢什么颜色啊", ("ai-color",), ("user-color",)),
    RecallCase("ai-pref-color-2", "ai_preference", "你平时喜欢哪些颜色呢", ("ai-color",), ("user-color",)),
    RecallCase("ai-pref-movie", "ai_preference", "你最喜欢看什么电影", ("ai-movie",), ("user-movie",)),
    RecallCase("ai-pref-music", "ai_preference", "你喜欢听什么歌", ("ai-music",), ()),
    RecallCase("ai-pref-food", "ai_preference", "你爱吃什么", ("ai-food",), ("user-food",)),
    RecallCase("ai-pref-sport", "ai_preference", "你平时喜欢什么运动", ("ai-sport",), ()),
    # Group 2: user preferences
    RecallCase("user-pref-color", "user_preference", "我最喜欢什么颜色来着", ("user-color",), ("ai-color",)),
    RecallCase("user-pref-food", "user_preference", "我有什么忌口你还记得吗", ("user-food",), ("ai-food",)),
    RecallCase("user-pref-movie", "user_preference", "我最喜欢的电影是哪部", ("user-movie",), ("ai-movie",)),
    # Group 3: relation naming
    RecallCase("rel-name-boss", "relation_naming", "我老板叫什么名字来着", ("user-boss",), ()),
    RecallCase("rel-name-mom", "relation_naming", "我妈妈叫什么名字", ("user-mom",), ()),
    RecallCase("rel-name-pet", "relation_naming", "我家狗叫什么", ("user-dog",), ("ai-pet",)),
    # Group 4: safety / emotional continuity
    RecallCase("safety-grief", "safety", "我又梦到外婆了，好难受", ("user-grief",), ()),
    RecallCase("safety-anxiety", "safety", "最近真的睡不着，压力好大", ("user-anxiety",), ()),
    # Group 5: shared history
    RecallCase("shared-first", "shared_history", "还记得我们第一次聊天聊的什么吗", ("rel-first",), ()),
    RecallCase("shared-promise", "shared_history", "咱们之前的约定你还记得吧", ("rel-promise",), ()),
    RecallCase("shared-joke", "shared_history", "我们那个只有我们懂的梗是什么来着", ("rel-joke",), ()),
    # Group 6: identity separation (user vs AI)
    RecallCase("id-user-age", "identity", "我今年多大来着", ("user-age",), ("ai-age",)),
    RecallCase("id-ai-age", "identity", "你今年多大了", ("ai-age",), ("user-age",)),
    RecallCase("id-ai-job", "identity", "你是做什么工作的", ("ai-job",), ("user-job",)),
    RecallCase("id-user-name", "identity", "你还记得我叫什么吗", ("user-name",), ()),
    # Group 7: reminders
    RecallCase("reminder-what", "reminder", "我让你提醒我的事是什么来着", ("user-reminder",), ()),
    # Group 8: ellipsis follow-ups (enhanced_query from relevance LLM)
    RecallCase(
        "ellipsis-color", "ellipsis", "那颜色呢", ("user-color",), (),
        enhanced_query="用户喜欢的颜色",
    ),
    RecallCase(
        "ellipsis-pet", "ellipsis", "它叫什么来着", ("user-dog",), (),
        enhanced_query="用户养的狗的名字",
    ),
)
