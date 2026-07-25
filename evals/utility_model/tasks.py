"""小模型基准的任务与标注集.

选小模型不能只看单价. 它在热路径上做的是**判定**: 这句话记不记、要不要查记忆、
是什么意图、要不要联网. 判错的代价是漏记一条记忆、漏召一次上下文、走错一个
意图分支 —— 这些都比省下来的那点 token 费贵得多. 所以先测准确率, 再谈价格.

标注依据是提示词自己写的规则, 不是我的偏好: 这里问的是"换个模型还能不能听懂
我们的指令", 所以 ground truth 就是指令声明的应有输出.

四个任务的选取标准是: 输出离散可判、在热路径上每条消息都跑、判错有明确代价.
自由文本类任务 (回复生成/摘要) 不在这里 —— 那类质量归 reply_register 评测管.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    message: str
    expected: str
    context: str = ""
    note: str = ""


@dataclass(frozen=True)
class Task:
    key: str                              # registry prompt key
    name: str
    params: Callable[[Case], dict[str, Any]]
    parse: Callable[[str], str | None]    # 原始输出 → 归一化答案 (None = 解析失败)
    cases: tuple[Case, ...]
    json_mode: bool = False
    labels: tuple[str, ...] = field(default_factory=tuple)


# ── 1. 记忆预筛 (memory.judgement_user) — 每条用户消息都跑 ─────────────────

def _parse_memorize(raw: str) -> str | None:
    t = (raw or "").strip()
    if "不记" in t:
        return "不记"
    if "记" in t:
        return "记"
    return None


_JUDGEMENT_CASES = (
    Case("我下周三要去上海出差", "记", note="具体计划"),
    Case("我最讨厌吃香菜", "记", note="稳定偏好"),
    Case("我是做产品经理的", "记", note="身份事实"),
    Case("我妈妈最近住院了", "记", note="重要生活事件"),
    Case("我养了只叫豆豆的猫", "记", note="稳定事实"),
    Case("我大学在武汉读的", "记", note="身份事实"),
    Case("下个月我要结婚了", "记", note="重大事件"),
    Case("我对花生过敏", "记", note="安全相关事实"),
    Case("嗯", "不记", note="应答词"),
    Case("哈哈哈", "不记", note="语气"),
    Case("在吗", "不记", note="招呼"),
    Case("好的", "不记", note="应答词"),
    Case("你说得对", "不记", note="一次性回应"),
    Case("今天天气不错", "不记", note="一次性闲聊"),
    Case("是啊", "不记", note="应答词"),
    Case("晚安", "不记", note="寒暄"),
)

TASK_MEMORY_JUDGEMENT = Task(
    key="memory.judgement_user",
    name="记忆预筛",
    params=lambda c: {"message": c.message},
    parse=_parse_memorize,
    cases=_JUDGEMENT_CASES,
    labels=("记", "不记"),
)


# ── 2. 记忆相关度 (memory.relevance) — 决定要不要查长期记忆 ────────────────

def _parse_relevance(raw: str) -> str | None:
    t = (raw or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            level = str(json.loads(m.group(0)).get("level", "")).strip()
            if level in ("强", "中", "弱"):
                return level
        except json.JSONDecodeError:
            pass
    for level in ("强", "中", "弱"):
        if f'"{level}"' in t or f"「{level}」" in t:
            return level
    return None


_RELEVANCE_CASES = (
    # 强: 不查记忆只能编造
    Case("你多大了", "强", note="AI 身份事实, prompt 明确点名不能判弱"),
    Case("你叫什么名字", "强", note="AI 身份事实"),
    Case("我上次说想买的那个是啥来着", "强", context="用户: 我最近想买个相机\nAI: 想拍什么呀"),
    Case("你还记得我在哪上班吗", "强", note="用户身份事实"),
    Case("我生日是哪天你记得不", "强"),
    Case("你大学学的什么专业", "强", note="AI 稳定资料"),
    # 中: 不查也能聊, 查了更贴合
    Case("我最近老失眠", "中", note="prompt: 情绪可能与反复状态有关, 至少判中"),
    Case("你喜欢猫吗", "中", note="询问 AI 对某对象的偏好"),
    Case("我今天特别焦虑", "中", note="情绪类"),
    Case("你看过这个电影没", "中", context="用户: 昨天看了《八仙》"),
    Case("我又跟我妈吵架了", "中", note="关系委屈 + 可能反复"),
    # 弱: 任何角色都能立刻回
    Case("哈哈", "弱", note="纯回应"),
    Case("嗯嗯", "弱", note="纯回应"),
    Case("早上好呀", "弱", note="纯招呼"),
    Case("是吗", "弱", context="AI: 今天下雨了", note="对刚说过的话的简单回应"),
    Case("人为什么会做梦", "弱", note="抽象闲聊, 不依赖个人信息"),
)

TASK_MEMORY_RELEVANCE = Task(
    key="memory.relevance",
    name="记忆相关度",
    params=lambda c: {"message": c.message, "context": c.context or "（无）"},
    parse=_parse_relevance,
    cases=_RELEVANCE_CASES,
    json_mode=True,
    labels=("强", "中", "弱"),
)


# ── 3. 意图识别 (intent.unified) — 决定走哪条回复分支 ──────────────────────

_INTENT_LABELS = (
    "危机求助", "终结意图", "计划查询", "作息调整", "询问当前状态",
    "道歉承诺", "删除", "调用久远记忆", "记录请求", "日常交流",
)


def _parse_intent(raw: str) -> str | None:
    t = (raw or "").strip()
    # 多标签时取第一个 —— 主意图决定分支.
    for label in sorted(_INTENT_LABELS, key=len, reverse=True):
        if label in t:
            return label
    return None


_INTENT_CASES = (
    Case("提醒我明天八点吃药", "记录请求"),
    Case("下周二我面试", "记录请求"),
    Case("算了别提醒了", "记录请求", note="取消也归记录请求"),
    # 「你在干嘛」同时符合"询问当前状态"和"计划查询"两条定义, 但提示词把它
    # 逐字列进了计划查询的例子里 —— 以指令自己的举例为准. (我最初标成"询问
    # 当前状态", 5 个候选里有 3 个按指令答对却被判错, 排名因此失真.)
    Case("你在干嘛呢", "计划查询", note="提示词例子里逐字列在计划查询下"),
    Case("你现在忙吗", "计划查询", note="查询 AI 可用性"),
    Case("你现在心情怎么样", "询问当前状态", note="问的是当下情绪, 不是可用性"),
    Case("不聊了 我去睡了", "终结意图"),
    Case("对不起 刚才是我不好", "道歉承诺"),
    Case("把我说过喜欢咖啡这事忘了吧", "删除"),
    Case("你还记得很久以前我跟你说过的那件事吗", "调用久远记忆"),
    Case("今天上班好累", "日常交流"),
    Case("我最近心情不太好", "日常交流", note="主语是我, 自陈情绪 → 日常交流"),
    Case("你觉得我该辞职吗", "日常交流"),
    Case("周末一起去爬山吗", "日常交流", note="邀请不是计划查询"),
    Case("我不想活了", "危机求助", note="优先级最高"),
)

TASK_INTENT = Task(
    key="intent.unified",
    name="意图识别",
    params=lambda c: {"user_message": c.message, "context": c.context or "（无）"},
    parse=_parse_intent,
    cases=_INTENT_CASES,
    labels=_INTENT_LABELS,
)


# ── 4. 联网判定 (chat.web_search_decision) ────────────────────────────────
# 这组用例来自 2026-07-25 修复联网门控时的实测集, 已在生产小模型上验证过 12/12.

def _parse_web_search(raw: str) -> str | None:
    t = (raw or "").strip()
    if "不需要" in t or "无需" in t:
        return "不需要联网"
    if "需要联网" in t or t.startswith("需要"):
        return "需要联网"
    return None


_WEB_SEARCH_CASES = (
    Case("你知道运城永乐宫建于哪一年 有那怎样的历史故事吗？", "不需要联网", note="静态百科"),
    Case("珠穆朗玛峰有多高", "不需要联网", note="静态百科"),
    Case("红楼梦是谁写的", "不需要联网", note="静态百科"),
    Case("永乐宫在哪个省？", "不需要联网", note="静态百科"),
    Case("你知道最近有谁获得了菲尔兹奖吗", "需要联网", note="近期事件"),
    Case("最近的电影你有喜欢看的吗", "需要联网", note="近期动态"),
    Case("今天北京天气好吗", "需要联网", note="随时间变化"),
    Case("还有啥推荐的电影吗，马上要上映的？", "需要联网", note="时新推荐"),
    Case("对的，你看过没", "需要联网", context="用户: 你听说过《八仙》吗\nAI: 好像有印象",
         note="外部作品"),
    Case("运城永乐宫值得去吗，最近开放吗", "需要联网", note="时效性"),
    Case("我今天好累啊，被领导骂了", "不需要联网", note="情绪倾诉"),
    Case("你还记得我上次说的那家店吗", "不需要联网", note="查记忆不查网"),
)

TASK_WEB_SEARCH = Task(
    key="chat.web_search_decision",
    name="联网判定",
    params=lambda c: {"message": c.message, "context": c.context or "(无)"},
    parse=_parse_web_search,
    cases=_WEB_SEARCH_CASES,
    labels=("需要联网", "不需要联网"),
)


ALL_TASKS: tuple[Task, ...] = (
    TASK_MEMORY_JUDGEMENT,
    TASK_MEMORY_RELEVANCE,
    TASK_INTENT,
    TASK_WEB_SEARCH,
)
