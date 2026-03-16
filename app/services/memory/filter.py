"""记忆消息过滤器。

7规则加权准入，决定是否对消息进行记忆提取。
通过条件 = 2+条规则命中 或 1条权重≥2的规则命中。
"""

from __future__ import annotations

import re

# 情感词库
_EMOTION_WORDS = {
    "开心", "难过", "伤心", "焦虑", "害怕", "生气", "愤怒", "感动",
    "孤独", "压力", "兴奋", "失望", "后悔", "感恩", "紧张", "无聊",
    "幸福", "痛苦", "委屈", "满足", "期待", "惊讶", "厌恶", "嫉妒",
    "喜欢", "讨厌", "爱", "恨", "想念", "思念", "担心", "不安",
}

# 时间词库
_TIME_WORDS = {
    "昨天", "今天", "明天", "上周", "下周", "去年", "今年", "明年",
    "小时候", "以前", "之前", "最近", "刚才", "刚刚", "上个月", "下个月",
    "周末", "春节", "暑假", "寒假", "毕业", "当时", "那时候",
}

# 事实词库（表示陈述事实信息）
_FACT_WORDS = {
    "是", "叫", "在", "住", "岁", "工作", "学习", "上学", "毕业",
    "喜欢", "讨厌", "不喜欢", "养", "有", "买了", "去了", "来了",
    "家", "公司", "学校", "大学", "城市", "专业", "职业",
}

# 第一人称词
_FIRST_PERSON = {"我", "咱", "俺", "自己"}

# "我…"句式模式
_SELF_DISCLOSURE_PATTERNS = [
    re.compile(r"我(是|叫|在|住|有|喜欢|讨厌|想|觉得|认为|打算|准备|希望)"),
    re.compile(r"我(的|们)(家|妈|爸|朋友|同事|老板|同学|女朋友|男朋友|老公|老婆)"),
]


def should_extract_memory(message: str) -> bool:
    """判断消息是否值得进行记忆提取。

    7规则加权:
    - 长度≥5汉字 (w=1)
    - 含第一人称 (w=1)
    - 含情感词 (w=2)
    - 含时间词 (w=1)
    - 含事实词 (w=1)
    - "我…"自我暴露句式 (w=2)
    - 消息长度≥30 (w=1, 长消息可能有信息量)

    通过条件: 总权重≥2
    """
    if not message or not message.strip():
        return False

    # 统计汉字数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', message))
    total_weight = 0

    # Rule 1: 长度≥5汉字 (w=1)
    if chinese_chars >= 5:
        total_weight += 1

    # Rule 2: 第一人称 (w=1)
    if any(p in message for p in _FIRST_PERSON):
        total_weight += 1

    # Rule 3: 情感词 (w=2)
    if any(w in message for w in _EMOTION_WORDS):
        total_weight += 2

    # Rule 4: 时间词 (w=1)
    if any(w in message for w in _TIME_WORDS):
        total_weight += 1

    # Rule 5: 事实词 (w=1)
    if any(w in message for w in _FACT_WORDS):
        total_weight += 1

    # Rule 6: "我…"自我暴露句式 (w=2)
    if any(p.search(message) for p in _SELF_DISCLOSURE_PATTERNS):
        total_weight += 2

    # Rule 7: 长消息 (w=1)
    if chinese_chars >= 30:
        total_weight += 1

    return total_weight >= 2
