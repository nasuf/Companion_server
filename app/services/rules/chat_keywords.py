"""Central keyword catalog for chat control-flow rules.

Keep business logic in chat/memory modules; keep deterministic word lists here
with explicit purpose labels. Broad semantic understanding should remain in
LLM prompts/classifiers, while these rules cover safety, fallback, write
actions, and cheap high-confidence gates.
"""

from __future__ import annotations

import re

from app.services.rules.keyword_policy import KeywordRuleSet, RulePurpose


CONVERSATION_END = KeywordRuleSet(
    name="conversation_end",
    purpose=RulePurpose.LLM_FALLBACK,
    terms=(
        "拜拜", "88", "886", "先忙了", "睡了", "睡觉了", "回头聊",
        "不说了", "下次聊", "告辞", "再见", "晚安", "拜", "byebye",
        "bye", "先走了", "下次再聊", "改天聊", "先这样", "我先走了",
        "不聊了", "去忙了",
    ),
    description="Farewell intent fallback when unified intent LLM is unavailable.",
)
CONVERSATION_END_KEYWORDS = list(CONVERSATION_END.terms)

PROMISE_KEYWORDS = (
    "我以后不会了", "我保证", "我发誓", "不会再这样", "再也不会",
    "保证不再", "我承诺", "绝对不会", "我答应你",
)

SCHEDULE_ADJUST_KEYWORDS = (
    "你能不能晚点睡", "晚点睡", "早点睡", "早点休息", "别睡了",
    "能不能抽空", "陪我聊", "别忙了", "你先别忙", "能不能陪我",
    "你今天早点", "今天晚点",
)

CURRENT_STATE_FAST_PHRASES = frozenset({
    "你在干嘛",
    "你在干嘛呢",
    "在干嘛",
    "在干嘛呢",
    "干嘛呢",
    "你干嘛呢",
    "忙啥",
    "你忙啥",
    "在忙啥",
    "你在忙啥",
    "在忙什么",
    "你在忙什么",
    "现在忙吗",
    "你现在忙吗",
    "你现在在干嘛",
    "你现在在干嘛呢",
    "你现在做什么",
    "你现在在做什么",
    "你在做什么",
    "你在做啥",
    "你现在干啥",
    "你现在干啥呢",
})
CURRENT_STATE_TIME_BLOCKERS = (
    "明天", "后天", "大后天", "昨天", "前天", "今晚", "晚上", "下午",
    "上午", "中午", "凌晨", "周末", "星期", "礼拜", "几点", "什么时候",
    "等会", "待会", "一会", "以后", "之前", "刚才",
)
CURRENT_STATE_HISTORY_BLOCKERS = (
    "刚才", "刚刚", "刚说", "刚说的", "之前说", "前面说", "上一句", "上句",
)
CURRENT_STATE_EXPLICIT_PHRASES = frozenset({
    "忙吗",
    "不忙吗",
    "你忙吗",
    "你不忙吗",
    "有空吗",
    "你有空吗",
    "现在有空吗",
    "你现在有空吗",
    "最近怎么样",
    "你最近怎么样",
    "你怎么样",
    "你还好吗",
    "你开心吗",
    "你心情怎么样",
})
CURRENT_STATE_SUBJECT_TERMS = ("你", "你现在", "你最近", "你今天", "你那边")
CURRENT_STATE_PREDICATE_TERMS = (
    "干嘛", "做什么", "做啥", "忙", "有空", "怎么样", "还好吗",
    "心情", "感觉", "开心", "难过", "状态",
)

L3_EXPLICIT_OLD_RE = re.compile(
    r"("
    r"更早|更久|久远|很久以前|很久之前|好久以前|好久之前|"
    r"很早以前|多年前|几年前|"
    r"半年前|[六七八九十]个?月前|[6-9]个?月前|[1-9]\d+个?月前|"
    r"去年|前年|大前年|"
    r"第一次|初次|刚认识|刚见面|"
    r"小时候|童年|小学|初中|高中|大学(时候|那会|时期)?"
    r")"
)
L3_RECALL_CUE_RE = re.compile(
    r"(记得|想起|回忆|回想|说过|提过|聊过|告诉过|讲过|记不记得)"
)

CRISIS_KEYWORDS = (
    "跳楼", "跳河", "跳桥", "跳轨", "跳海",
    "自杀", "自残", "自伤", "轻生",
    "想死", "我去死", "去死算了", "不想活", "活不下去",
    "活着没意思", "活着没意义", "活够了",
    "结束生命", "结束自己", "了结自己", "了结我自己",
    "上吊", "割腕", "吃药自尽",
    "不想存在", "消失算了", "消失就好",
    "跟这个世界说再见", "和这个世界说再见", "向这个世界说再见",
    "跟世界说再见", "和世界说再见", "向世界说再见",
    "告别这个世界", "告别世界", "离开这个世界", "离开世界",
    "对这个世界的最后一次", "在这个世界的最后一次",
)
CRISIS_RELEASE_KEYWORDS = (
    "我安全", "安全了", "现在安全",
    "不想死了", "不会自杀", "不会自残",
    "刚才是气话", "只是气话", "刚才是开玩笑",
)
CRISIS_SEMANTIC_HINTS = (
    "再见", "永别", "告别", "最后一次", "世界", "离开", "消失",
    "撑不住", "受不了", "没意义", "没意思", "解脱", "放弃",
    "不想继续", "别管我", "算了", "下辈子",
)
CRISIS_SEMANTIC_DIRECT_PHRASES = (
    "说再见", "最后一次发泄", "最后一次告别", "让我走",
    "不想继续了", "不想撑了", "撑不下去了",
)
CRISIS_CARE_ASSISTANT_MARKERS = (
    "你现在安全吗",
    "有没有伤害自己",
    "伤害自己的冲动",
    "我还在看着你刚才",
    "没翻过去",
    "我不会跳过",
)
CRISIS_CHECK_ANNOYED_TERMS = (
    "无聊的问题", "问这么多", "别问", "不要问", "烦不烦",
    "烦死", "审问", "查户口",
)

MEMORY_FACT_RECALL_CUES = (
    "还记得", "记得", "记不记得", "记得吗", "记得嘛",
    "我跟你说过", "我和你说过", "告诉过你", "跟你讲过",
)

RECURRENCE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("每年", "yearly"), ("每月", "monthly"), ("每周", "weekly"),
    ("每星期", "weekly"), ("每天", "daily"), ("每日", "daily"),
    ("每晚", "daily"), ("每早", "daily"), ("每个月", "monthly"),
    ("每一年", "yearly"), ("每一月", "monthly"), ("每一周", "weekly"),
    ("每一天", "daily"),
)
REMINDER_ACTION_CUES = (
    "提醒", "叫我", "喊我", "闹钟", "到时候", "盯着", "催我",
)
RECORD_MEMORY_CUES = (
    "记住", "记一下", "记下来", "记着", "帮我记", "替我记",
)
SELF_NOTE_CUES = (
    "我想记下来", "我想记一下", "我想写下来", "我能贴在备忘录",
    "贴在备忘录", "备忘录里的话", "备忘录里的一句话",
)
REMINDER_CONTENT_CUES = (
    "提醒内容", "提醒文案", "内容就写", "内容写成", "文案写成",
    "改成那句", "还是那句",
)
TIME_OR_EVENT_CUES = (
    "明天", "后天", "大后天", "今晚", "今天", "下周", "下星期", "下个月",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日", "周天",
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日", "星期天",
    "每周", "每月", "每天", "每日", "每年", "点", "号", "月",
)
HIGH_CONFIDENCE_CANCEL_KEYWORDS = (
    "别提醒", "不用提醒", "取消提醒", "撤销提醒", "撤回提醒",
    "不记了", "别记了", "不用记", "别再记",
)
LOW_CONFIDENCE_CANCEL_KEYWORDS = (
    "算了", "不用了",
    "我吃过了", "我做过了", "我喝过了",
    "已经做了", "已经吃了", "已经喝了", "已经办了",
)
CANCEL_NEG_TOKENS = ("取消", "删掉", "删除", "撤销", "不要", "别")
UNDO_CANCEL_KEYWORDS = (
    "撤回", "撤销", "恢复", "复原", "我反悔", "刚才取消错了", "撤回刚才",
    "恢复刚才", "把刚才的", "撤回提醒", "恢复提醒",
)
CANCEL_CONFIRM_KEYWORDS = {
    "对", "对的", "是", "是的", "好", "好的", "嗯", "嗯嗯",
    "确认", "撤", "撤掉", "撤了", "ok", "yes",
}
CANCEL_DENY_KEYWORDS = {
    "不是", "不", "不对", "算了", "不要", "不用", "保留",
    "继续", "no", "别动",
}
CANCEL_CHOICE_ALL_KEYWORDS = {"全部", "都", "全删", "all"}

ENHANCED_QUERY_FIRST_HINTS = (
    "他呢", "她呢", "它呢", "这个呢", "那个呢", "这件事", "那件事",
    "那次", "上次", "当时", "后来呢", "然后呢", "颜色呢", "名字呢",
    "情况呢", "怎么样了", "怎样了", "怎么了", "咋样了",
)
FAST_WEAK_WORDS = {
    "嗯", "嗯嗯", "哦", "哦哦", "好", "好的", "行", "行吧", "好吧",
    "ok", "okay", "收到", "知道了", "可以", "当然",
    "哈哈", "哈哈哈", "呵呵", "嘻嘻", "嘿嘿", "hh", "hhh", "666",
    "哇", "啊", "啊啊", "额", "呃", "唔", "喔", "噢",
    "是", "是的", "对", "对的", "对对",
    "谢谢", "感谢",
    "早", "早上好", "晚安", "你好", "hello", "hi", "嗨",
    "了", "吧", "呢", "吗", "呀",
}
FAST_WEAK_REPEAT_CHARS = set("嗯哦喔噢啊哈呵嘻嘿呃额唔哇吧呀呢吗啦了")
FAST_WEAK_NOISE_RE = re.compile(r"[\s.,!?。，！？…~～、]+")
FAST_WEAK_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0E\uFE0F\u200D]+")
FAST_WEAK_PROTECTED_HINTS = (
    "记得", "忘了", "忘记", "以前", "之前", "上次", "那次", "当时",
    "去年", "前年", "小时候",
    "妈妈", "母亲", "爸爸", "父亲", "家人", "老婆", "妻子", "老公", "丈夫",
    "女朋友", "男朋友", "前任",
    "名字", "年龄", "生日", "工作", "职业", "学校", "公司",
    "喜欢", "不喜欢", "讨厌", "过敏", "手术", "住院", "出院",
    "活不下去", "想死", "自杀", "轻生",
)

RELATIONAL_COMPLAINT_KEYWORDS = (
    "怎么不理我", "不理我", "不回我", "不想理我", "你在忙吗", "你还在吗",
    "你是不是不想理我", "是不是不想聊", "是不是烦我", "怎么才回",
)
DISTRESS_KEYWORDS = (
    "不好", "难受", "烦", "委屈", "崩溃", "糟糕", "不开心", "很累", "想哭",
    "好难过", "撑不住", "心情不好",
)
