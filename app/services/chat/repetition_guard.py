"""Phrases the AI has already repeated across its recent replies.

The main reply sees its own last few replies in the history and tends to copy
them: on 2026-10-15 it ended four replies in a row with "你今天还忙吗", until
the user said it sounded like an AI. Phrases that already appear in two or more
of the recent replies are listed back to the model so it can avoid a third.
"""

from __future__ import annotations

import re

_RECENT_REPLIES = 6
_MIN_REPLIES = 2
_MAX_PHRASES = 3
_MIN_PHRASE_CHARS = 4
_CLAUSE_SPLIT = re.compile(r"\|\||[。！？!?，,；;~～…\s]+")
_NON_WORD = re.compile(r"[\W_]+")


def _recent_ai_replies(messages: list[dict]) -> list[str]:
    replies: list[list[str]] = []
    previous_role = None
    for message in messages:
        role = message.get("role")
        text = str(message.get("content") or "").strip()
        if role == "assistant" and text:
            if previous_role == "assistant" and replies:
                replies[-1].append(text)
            else:
                replies.append([text])
        if text:
            previous_role = role
    return [" ".join(parts) for parts in replies[-_RECENT_REPLIES:]]


# 只是附和一句: 是呀 / 对啊 / 嗯嗯 / 好的 / 哈哈哈 / 确实 (带问号的 "啊？" "是吧？" 是没听懂, 不算)
_ACK = re.compile(
    r"^(?:是|对|嗯|哦|噢|喔|好|行|啊|哈|嘿|嘻|哇|确实|没错|可以|当然|就是|也是|ok|okay)+"
    r"(?:呀|啊|的|吧|呢|哦|嘛|哒|滴|耶)*$",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"[?？]|吗|么|啥|什么|哪|几|谁|怎么|来着")
# 对方让你再说一遍时, 复述是对的
_ASKED_TO_REPEAT = re.compile(r"再说一遍|再说一次|刚才|刚刚|说啥|什么意思|啥意思|没看清|没听清|来着")
# 删掉之后剩下的开头是连接词 / 只剩问句, 读起来就断了, 不删
_DANGLING_START = re.compile(r"^(?:而且|所以|然后|但是|不过|还有|并且|因为|就是)")
# 一条气泡和上一轮某一条的重合度 (按相邻两字计) 到这个比例就算复述; 太短的气泡不判
_RESTATE_OVERLAP = 0.6
_RESTATE_MIN_CHARS = 8
_PREVIOUS_MAX_CHARS = 120


def _key(text: str) -> str:
    return _NON_WORD.sub("", text or "")


def is_acknowledgement(message: str) -> bool:
    """对方只是附和一句 (是呀/嗯嗯/好的/哈哈), 没有新内容, 也不是在问。"""
    key = _key(message)
    return 0 < len(key) <= 6 and not _QUESTION.search(message or "") and bool(_ACK.match(key))


def _created(message: dict):
    from app.services.chat.message_utils import _parse_message_created_at

    return _parse_message_created_at(message.get("createdAt"))


def previous_ai_turn(messages: list[dict], current_turn_ids: set[str]) -> tuple[list[str], bool]:
    """紧挨着本轮之前的那次 AI 回复 (各条气泡), 以及它是否在本轮用户消息之后才发出。

    任何用户行 (包括没有文字的卡片) 都是边界; 隔了 10 分钟以上的 AI 消息不算同一轮
    (与 message_utils.previous_assistant_turn 同一口径)。
    """
    from app.services.chat.message_utils import _same_assistant_turn

    ids = current_turn_ids or set()
    rows: list[dict] = []
    for message in reversed([m for m in messages if m.get("id") not in ids]):
        if message.get("role") != "assistant":
            break
        if not str(message.get("content") or "").strip():
            continue
        if rows and not _same_assistant_turn(message, rows[-1]):
            break
        rows.append(message)
    rows.reverse()
    current_at = [t for m in messages if m.get("id") in ids and (t := _created(m)) is not None]
    last_at = _created(rows[-1]) if rows else None
    after = bool(last_at and current_at and last_at > min(current_at))
    return [str(m.get("content")).strip() for m in rows], after


def previous_reply_situation(messages: list[dict], current_turn_ids: set[str],
                             user_message: str) -> tuple[str, str] | None:
    """("overlap" | "ack", 上一轮原话): 本轮容易把上一轮再说一遍的两种情形。"""
    bubbles, after = previous_ai_turn(messages, current_turn_ids)
    if not bubbles:
        return None
    previous = quote_previous(bubbles)
    if after:
        return "overlap", previous
    if is_acknowledgement(user_message):
        return "ack", previous
    return None


def quote_previous(bubbles: list[str]) -> str:
    """提示里引用的上一轮原话; 太长时留离这句最近的部分。"""
    text = "，".join(bubbles)
    return text if len(text) <= _PREVIOUS_MAX_CHARS else "…" + text[-_PREVIOUS_MAX_CHARS:]


def _bigrams(text: str) -> set[str]:
    key = _key(text)
    return {key[i:i + 2] for i in range(len(key) - 1)}


def restated_overlap(reply: str, previous: list[str]) -> float:
    """这条气泡和上一轮里最像的那一条的重合度; 太短的气泡记 0。"""
    grams = _bigrams(reply)
    if len(_key(reply)) < _RESTATE_MIN_CHARS or not grams:
        return 0.0
    return max((len(grams & _bigrams(p)) / len(grams) for p in previous), default=0.0)


def drop_restated_bubbles(replies: list[str], previous: list[str],
                          user_message: str = "") -> tuple[list[str], int, bool]:
    """去掉和上一轮基本是同一句话的气泡: (保留的, 去掉几条, 是否全是复述)。

    全都是复述时不动 (不能发空回复), 只报告出来; 删完剩下的读不通 (开头是连接词、
    只剩问句) 也不动。对方在问 / 让你再说一遍时不判。
    """
    if not replies or not previous or _ASKED_TO_REPEAT.search(user_message or "") \
            or _QUESTION.search(user_message or ""):
        return replies, 0, False
    # 整轮复述按整段比 (短气泡单独看都不够长, 合起来就是上一轮原话)
    whole = restated_overlap("".join(replies), ["".join(previous)])
    kept = [r for r in replies if restated_overlap(r, previous) < _RESTATE_OVERLAP]
    if whole >= _RESTATE_OVERLAP or not kept:
        return replies, 0, True
    if len(kept) == len(replies):
        return replies, 0, False
    if _DANGLING_START.match(kept[0].strip()) or all(_QUESTION.search(r) for r in kept):
        return replies, 0, False
    return kept, len(replies) - len(kept), False


async def restated_retry_note(previous: list[str]) -> str | None:
    """重写时追加的提示 (chat.previous_reply_retry); 后台停用时返回 None。"""
    from app.services.chat.prompt_builder import _get_optional_prompt, _render_section

    template = await _get_optional_prompt("chat.previous_reply_retry")
    if template is None:
        return None
    return _render_section(template, {"previous": quote_previous(previous)})


def find_repeated_ai_phrases(messages: list[dict]) -> list[str]:
    """Clauses found in at least two of the AI's last few replies, most repeated first."""
    seen_in: dict[str, set[int]] = {}
    display: dict[str, str] = {}
    for index, reply in enumerate(_recent_ai_replies(messages)):
        for clause in _CLAUSE_SPLIT.split(reply):
            key = _NON_WORD.sub("", clause)
            # Laughter and filler ("哈哈哈哈", "嗯嗯嗯嗯") repeat naturally.
            if len(key) < _MIN_PHRASE_CHARS or len(set(key)) <= 2:
                continue
            seen_in.setdefault(key, set()).add(index)
            display[key] = clause.strip()
    repeated = [key for key, replies in seen_in.items() if len(replies) >= _MIN_REPLIES]
    repeated.sort(key=lambda key: (-len(seen_in[key]), -max(seen_in[key])))
    return [display[key] for key in repeated[:_MAX_PHRASES]]
