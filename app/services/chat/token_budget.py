"""Per-section input measurement and the web-search 32k cap for the main reply.

Ark prices a Responses web-search request by the *sum* of its inferences: the
11-07 11:50 turn billed 33204 input tokens and the console put all 33294 of
that hour's tokens into the 32k-128K tier, although each inference alone was
far below 32k. A search request is two inferences over the same text plus the
results: over 206 production search turns (10-29..11-25) billed input was at
most 2 x calibrated text + 8333. Trimming history until 2 x text + 9000 fits
32768 would have touched 24 of them, cutting at most 916 tokens of history.
"""

from __future__ import annotations

import re

from app.services.llm import token_count
from app.services.memory.retrieval.context_selector import estimate_tokens

SEARCH_INPUT_CAP = 32768
SEARCH_RESULT_RESERVE = 9000
_MAX_TRIM_STEPS = 6
_SECTION = re.compile(r"(?m)^## (.+)\n")
CATEGORIES = {
    "persona": ("回复要求", "核心规则", "反幻觉硬约束", "你的身份", "对话一致性", "表达习惯参考",
                "你的作息", "你的心情", "当前情绪", "情绪状态提醒", "你的隐性状态约束"),
    "memory": ("你记得的事情", "久远记忆（L3）", "相关时间记忆", "事件时间线", "时间计算结果",
               "上次聊到", "用户画像", "你推荐过的活动"),
    "tools": ("联网结果使用", "刚聊过的作品"),
}
_CATEGORY_OF = {title: name for name, titles in CATEGORIES.items() for title in titles}


def measure(chat_messages: list[dict], model: str | None) -> dict:
    """Calibrated tokens per system section, history and the current message."""
    sections: dict[str, int] = {}
    categories = dict.fromkeys((*CATEGORIES, "other_system"), 0)
    history = current = 0
    last_user = max((i for i, m in enumerate(chat_messages) if m.get("role") == "user"), default=None)
    for index, message in enumerate(chat_messages):
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if message.get("role") != "system":
            tokens = token_count.count_text(text, model)
            if index == last_user:
                current += tokens
            else:
                history += tokens
            continue
        heads = list(_SECTION.finditer(text))
        spans = [("（无标题）", 0, heads[0].start() if heads else len(text))] if not heads or heads[0].start() else []
        # Titles come from "## " lines; injected text can carry one too, so keep them short.
        spans += [(h.group(1)[:40], h.start(), heads[i + 1].start() if i + 1 < len(heads) else len(text))
                  for i, h in enumerate(heads)]
        for title, start, end in spans:
            tokens = token_count.count_text(text[start:end], model)
            if tokens:
                sections[title] = sections.get(title, 0) + tokens
                categories[_CATEGORY_OF.get(title, "other_system")] += tokens
    rate = token_count.rate(model)
    return {
        "model": model,
        "calibrated": rate.calibrated,
        "input_tokens": token_count.count_messages(chat_messages, model),
        # The old heuristic over the same messages, so admin can compare both with billing.
        "heuristic_input_tokens": sum(estimate_tokens(m["content"]) for m in chat_messages
                                      if isinstance(m.get("content"), str)),
        "system_tokens": sum(sections.values()),
        "history_tokens": history,
        "current_tokens": current,
        "categories": categories,
        "sections": sections,
        "basis": "llm.token_count calibrated on production billing; 1000+ token calls within ±5% (doubao)",
    }


def search_projection(chat_messages: list[dict], model: str | None) -> int:
    from app.services.chat.prompt_builder import fold_trailing_system

    return 2 * token_count.count_messages(fold_trailing_system(chat_messages), model) + SEARCH_RESULT_RESERVE


def fit_search_cap(build, budget: int, model: str | None) -> tuple[list[dict], dict]:
    """Rebuild with a smaller history budget until the search request fits 32k.

    `build(budget)` returns chat messages. History goes first; the system prompt
    and the current message are never cut. Each heuristic history token is at
    least one billed token here (two inferences of >=0.5 calibrated each), so
    lowering the budget by the excess converges in a step or two.
    """
    messages = build(budget)
    projected = first = search_projection(messages, model)
    steps = 0
    while projected > SEARCH_INPUT_CAP and budget > 0 and steps < _MAX_TRIM_STEPS:
        budget = max(0, budget - (projected - SEARCH_INPUT_CAP) - 50)
        messages = build(budget)
        projected = search_projection(messages, model)
        steps += 1
    return messages, {
        "cap": SEARCH_INPUT_CAP,
        "result_reserve": SEARCH_RESULT_RESERVE,
        "projected_input_tokens": projected,
        "projected_before_trim": first,
        "trimmed": steps > 0,
        "history_budget": budget,
        "over_cap": projected > SEARCH_INPUT_CAP,
    }
