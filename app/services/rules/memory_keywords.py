"""Central keyword catalog for memory gating and retrieval ranking."""

from __future__ import annotations

import re


MEMORY_EMOTION_WORDS = {
    "开心", "难过", "伤心", "焦虑", "害怕", "生气", "愤怒", "感动",
    "孤独", "压力", "兴奋", "失望", "后悔", "感恩", "紧张", "无聊",
    "幸福", "痛苦", "委屈", "满足", "期待", "惊讶", "厌恶", "嫉妒",
    "喜欢", "讨厌", "爱", "恨", "想念", "思念", "担心", "不安",
}
MEMORY_EMOTION_WORDS_EN = {
    "happy", "sad", "upset", "angry", "anxious", "afraid", "excited",
    "lonely", "stressed", "disappointed", "grateful", "worried", "love",
    "hate", "miss", "nervous",
}
MEMORY_TIME_WORDS = {
    "昨天", "今天", "明天", "上周", "下周", "去年", "今年", "明年",
    "小时候", "以前", "之前", "最近", "刚才", "刚刚", "上个月", "下个月",
    "周末", "春节", "暑假", "寒假", "毕业", "当时", "那时候",
}
MEMORY_TIME_WORDS_EN = {
    "yesterday", "today", "tomorrow", "last week", "next week", "last year",
    "this year", "next year", "recently", "before", "just now", "weekend",
    "childhood", "vacation", "christmas", "spring festival",
}
MEMORY_FACT_WORDS = {
    "是", "叫", "在", "住", "岁", "工作", "学习", "上学", "毕业",
    "喜欢", "讨厌", "不喜欢", "养", "有", "买了", "去了", "来了",
    "家", "公司", "学校", "初中", "高中", "中学", "大学", "专业", "职业",
    "学历", "教育背景",
}
MEMORY_FACT_WORDS_EN = {
    "am", "live", "work", "study", "major", "job", "age", "from",
    "born", "birthday", "like", "love", "hate", "prefer", "married",
    "single", "family", "mom", "dad", "wife", "husband", "boyfriend",
    "girlfriend", "pet", "dog", "cat",
}
FIRST_PERSON_TERMS = {"我", "咱", "俺", "自己"}
FIRST_PERSON_TERMS_EN = {"i", "i'm", "im", "my", "me", "mine"}

SELF_DISCLOSURE_PATTERNS = [
    re.compile(r"我(是|叫|在|住|有|喜欢|讨厌|想|觉得|认为|打算|准备|希望)"),
    re.compile(r"我(的|们)(家|妈|爸|朋友|同事|老板|同学|女朋友|男朋友|老公|老婆)"),
]
SELF_DISCLOSURE_PATTERNS_EN = [
    re.compile(r"\bi\s*(?:am|'m)\s+\w+"),
    re.compile(r"\bmy\s+(?:name|job|work|major|family|mom|dad|wife|husband|boyfriend|girlfriend|dog|cat)\b"),
    re.compile(r"\bi\s+(?:live|work|study|like|love|hate|prefer|feel|want|plan|grew up)\b"),
]
CORE_PROFILE_PATTERNS = [
    re.compile(r"我\d{1,2}岁"),
    re.compile(r"我(在|住在|来自|老家在|工作|上班|读|学|养了|有)(.+)"),
    re.compile(r"(我是|我叫)(.+)"),
    re.compile(r"\b(?:i am|i'm|im)\s+\d{1,2}\b"),
    re.compile(r"\b(?:i am|i'm|im)\s+(?:a|an)\s+\w+"),
    re.compile(r"\bi\s+(?:live|work|study)\s+(?:in|at)\b"),
    re.compile(r"\bmy\s+(?:name|birthday|job|major|family)\s+"),
]

MEMORY_FILLER_WORDS = {
    "嗯", "哦", "好", "对", "啊", "哈", "呢", "吧", "呀", "噢", "唔",
    "ok", "嗯嗯", "哦哦", "好的", "好吧", "行", "是", "对对",
    "哈哈", "嘿", "嘻嘻", "呵呵", "hihi", "yeah", "yep", "nope",
    "mhm", "hmm", "haha", "lol",
}

SAFETY_QUERY_KEYWORDS: tuple[str, ...] = (
    "想死", "不想活", "活不下去", "活着没意思", "活着没意义",
    "轻生", "自杀", "自残", "自伤", "跳楼", "跳河", "跳桥", "跳轨",
    "结束生命", "结束自己", "了结自己", "消失算了", "撑不住",
)
DISTRESS_QUERY_KEYWORDS: tuple[str, ...] = (
    "难过", "委屈", "崩溃", "压力", "焦虑", "抑郁", "孤独",
    "想哭", "哭", "绝望", "痛苦", "撑不住", "心情不好", "很累",
    "低落", "沮丧", "受不了", "空落落", "空唠唠", "心里空", "有点空",
)
EMOTIONAL_SAFETY_SUBCATEGORIES: tuple[str, ...] = (
    "悲伤", "恐惧", "焦虑", "失望", "孤独",
)
RECALL_HINT_KEYWORDS: tuple[str, ...] = (
    "还记得", "记不记得", "记得", "以前", "之前", "去年", "前年",
    "上次", "那次", "那件事", "当时", "那时候", "很久", "小时候",
    "过去", "曾经", "后来呢", "然后呢",
)
CATEGORY_QUERY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "情绪": SAFETY_QUERY_KEYWORDS + DISTRESS_QUERY_KEYWORDS + (
        "开心", "高兴", "生气", "害怕", "恐惧", "失望", "遗憾",
    ),
    "生活": (
        "工作", "上班", "老板", "同事", "学校", "考试", "旅行", "搬家",
        "住院", "出院", "手术", "生病", "健康", "宠物", "生活",
    ),
    "身份": (
        "名字", "多大", "几岁", "年龄", "生日", "家人", "妈妈", "爸爸", "父母",
        "妻子", "丈夫", "女朋友", "男朋友", "职业", "住哪", "哪里人",
        "学历", "教育背景", "学校", "初中", "高中", "中学", "大学", "专业", "毕业",
    ),
    "偏好": (
        "喜欢", "不喜欢", "讨厌", "爱吃", "不吃", "偏好", "雷区",
        "习惯", "口味", "审美",
    ),
    "思维": (
        "想法", "观点", "价值观", "人生", "目标", "理想", "信仰",
        "怎么看", "觉得",
    ),
}
LEXICAL_MEMORY_KEYWORDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        kw
        for kws in CATEGORY_QUERY_KEYWORDS.values()
        for kw in kws
        if len(kw) >= 2
    )
)
