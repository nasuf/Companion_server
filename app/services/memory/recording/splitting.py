"""把"一条记忆装了多个事实"的长文本拆开.

起因 (2026-07-29 生产排查): 检索时有 8% 的 AI 记忆因超过单条 token 上限被整条跳过
—— 它们躺在库里, 但**任何检索都不会注入它们**。追下去发现两种完全不同的形态, 必须
分别处理:

    多事实拼接   "我的工作是在线咨询：…；用户情绪安抚：…；BUG 复现：…"
                 来自 agent 创建期把 profile 的一个字段整段转成一条记忆。
                 它本来就该是几条独立记忆 —— 拆开既解决超长, 也让每条能被单独
                 检索到 ("BUG 复现" 这件事现在根本搜不出来)。

    单段叙事     "与大橘的首次相遇：在那个下着冬雨的黄昏, 她在公司楼下…"
                 一个完整的故事, 279 字。按**句子边界**拆, 每段补回标题前缀。

2026-07-30 修订: 单段叙事原先是"不拆", 理由是"从中间切开会得到两个都不成立的
片段"。实测 63 段存量长记忆后反转了这个决定 —— 真正不能做的是切在句子中间或切进
引号里, 而不是拆本身。按句号切并把 "标题：" 补给每一段, 出来的两条各自读得通
(「大橘怎么捡回来的」/「大橘什么性格」), 都能被单独检索到。而不拆的代价是整条
100% 检索不到, 严格更差。

所以两阶段都做, 且顺序固定: 先按「；」拆多事实, 每段仍超限的再按句子拆。
"""

from __future__ import annotations

import re

# 只认全角分号。逗号/句号在中文叙事里到处都是, 按它们切会把单段故事切碎;
# 而「；」在这批数据里恰好只出现在"多事实拼接"那一种形态里 (实测: 145 条超长记忆中
# 含「；」的 22 条全是职业描述, 其余 123 条单段叙事一个「；」都没有)。
_SEGMENT_SEP = "；"

# 切出来太短的段不单独成条 —— "略懂皮毛" 这种半句话单独存着既检索不到也没有信息。
# 20 字是按实测定的: 切分后各段中位 85 字, p90 119, 没有正常段落低于这个数。
MIN_SEGMENT_CHARS = 20

# 段数上限。一条记忆切出十几条会让这个 agent 的某个话题在检索里过度膨胀,
# 挤掉其他类目。超过就保持原样, 交给人工审视。
MAX_SEGMENTS = 8

# "我的工作是XXX：…" 这类前缀在拆分后要保留给每一段, 否则第二段之后就失去了主语。
_LEAD_PATTERN = re.compile(r"^(我的[\u4e00-\u9fff]{1,6}是)")

# "入职第一天的“手抖”：第一天正式接听电话…" 这类标题式开头。句级拆分时补给每一段,
# 否则第二段变成没头没尾的半个故事。限长 24 字是为了不把整个第一句当成标题。
_TITLE_PATTERN = re.compile(r"^([^：:；。！？]{2,24}[：:])")

_SENT_ENDERS = "。！？"
_QUOTE_OPEN = "“「『（(《【"
_QUOTE_CLOSE = "”」』）)》】"


def _sentences(text: str) -> list[str]:
    """按句子切, 但**不切进引号内部**.

    生产样本里踩过这个坑: 「阿姨安慰她：“小姑娘, 别急, 我不赶时间。”」按句号裸切会
    在引号里断开, 前一段引号不闭合、后一段以一个孤零零的 ” 开头, 两条都读不通。
    这里跟踪引号深度, 只在深度为 0 时断句; 句末紧跟的右引号一并收进本句。

    引号不配对时 (源数据里常见) 深度回不到 0, 整段作为一句返回 —— 退化成"拆不动",
    跟修改前的行为一致, 不会产出坏片段。
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in _QUOTE_OPEN:
            depth += 1
        elif ch in _QUOTE_CLOSE:
            depth = max(0, depth - 1)
            # 「…我不赶时间。”」—— 句号在引号内, 真正的句末是右引号之后。
            if depth == 0 and len(buf) >= 2 and buf[-2] in _SENT_ENDERS:
                out.append("".join(buf).strip())
                buf = []
        elif ch in _SENT_ENDERS and depth == 0:
            while i + 1 < n and text[i + 1] in _QUOTE_CLOSE:
                i += 1
                buf.append(text[i])
            out.append("".join(buf).strip())
            buf = []
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def should_split(text: str) -> bool:
    """判断这条记忆是不是"多事实拼接"."""
    if not text or _SEGMENT_SEP not in text:
        return False
    return len(_raw_segments(text)) >= 2


def _raw_segments(text: str) -> list[str]:
    return [s.strip() for s in text.split(_SEGMENT_SEP) if s.strip()]


def split_multi_fact(text: str) -> list[str]:
    """把多事实拼接拆成多条; 不该拆的原样返回单元素列表.

    拆分后每段都补上原文的引导语 ("我的工作是…"), 否则从第二段起就没有主语, 单独
    检索出来时读者不知道这是在说什么 —— 而记忆恰恰是被单独检索出来用的。
    """
    if not should_split(text):
        # 没有「；」不代表不用管: 单段叙事超限时按句子拆 (见模块 docstring 的
        # 2026-07-30 修订)。未超限的会原样返回, 短文本走这条路等价于不动。
        return _ensure_within_limit(text, "")

    segments = _raw_segments(text)
    if len(segments) > MAX_SEGMENTS:
        return [text]

    lead_match = _LEAD_PATTERN.match(segments[0])
    lead = lead_match.group(1) if lead_match else ""

    out: list[str] = []
    merged_tail = ""
    for i, seg in enumerate(segments):
        piece = seg.rstrip("。；;，, ")
        if not piece:
            continue
        # 过短的段并到上一条, 而不是丢掉 —— 它可能是上一句的补充说明。
        if len(piece) < MIN_SEGMENT_CHARS:
            if out:
                out[-1] = f"{out[-1]}；{piece}"
            else:
                merged_tail = piece
            continue
        if merged_tail:
            piece = f"{merged_tail}；{piece}"
            merged_tail = ""
        # 第一段自带引导语; 后续段补上, 免得失去主语。
        if i > 0 and lead and not piece.startswith(lead):
            piece = f"{lead}{piece}"
        out.append(f"{piece}。" if not piece.endswith("。") else piece)

    if merged_tail and out:
        out[-1] = f"{out[-1]}；{merged_tail}"
    # 拆不出两条以上就没有拆的意义, 保持原样。
    if len(out) < 2:
        return [text]
    return [p for piece in out for p in _ensure_within_limit(piece, lead)]


def unsplittable_oversized(text: str) -> list[str]:
    """拆完仍然超限的片段 —— 这些存进去检索时会被整条跳过.

    存在的理由: 后台上传 profile 文档建模板时走 profile_override, 完全绕过 LLM,
    所以生成侧收紧字数的约束对它无效。管理员写多长就是多长, 而他看不到"这条以后
    永远检索不到"。要在导入时就告诉他, 就得先按转换时的同一套规则预演一遍 ——
    否则会把本来能拆好的多事实字段也报成问题。
    """
    from app.services.memory.retrieval.context_selector import (
        MAX_MEMORY_TOKENS_PER_ITEM,
        estimate_tokens,
    )

    if not text or not text.strip():
        return []
    return [
        piece
        for piece in split_multi_fact(text)
        if estimate_tokens(piece) > MAX_MEMORY_TOKENS_PER_ITEM
    ]


def _ensure_within_limit(piece: str, lead: str) -> list[str]:
    """超限的段按句子边界拆开, 每段补回前缀.

    `lead` 是调用方已知的主语前缀 ("我的工作是"); 传空时自动从 "标题：" 里取。两者
    都没有就不补 —— 无前缀的叙事 (「大橘是林昕刚工作时…」) 本身各句都带主语。

    仍然拆不动 (整段就是一句) 时原样返回。硬切会切在句子中间, 那比超限更糟。
    """
    from app.services.memory.retrieval.context_selector import (
        MAX_MEMORY_TOKENS_PER_ITEM,
        estimate_tokens,
    )

    if estimate_tokens(piece) <= MAX_MEMORY_TOKENS_PER_ITEM:
        return [piece]

    if not lead:
        title = _TITLE_PATTERN.match(piece)
        lead = title.group(1) if title else ""

    sentences = _sentences(piece)
    if len(sentences) < 2:
        return [piece]

    out: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = f"{buf}{sent}"
        if buf and estimate_tokens(candidate) > MAX_MEMORY_TOKENS_PER_ITEM:
            out.append(buf)
            # 续段补引导语, 否则第二段起失去主语。
            buf = sent if not lead or sent.startswith(lead) else f"{lead}{sent}"
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out or [piece]
