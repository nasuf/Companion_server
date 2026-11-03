"""CHAT01.10.2 原始窗口 → 回放用例。

一条用例 = 生产里一个主回复轮次: 当时这次回复回应的那几条用户消息 + 之前 6 轮。回放时把
窗口原样交给真实编排器, 让模型重新回一次, 再按 10.1 的口径判断有没有复述之前 3 轮说过的话。

"这次回复回应的是哪几条" 以落库的 achievement_turn_id 为准 (生产投递时的 turn_message_ids):
用户连发两句、第二句在上一轮生成期间才到时, 上一轮已经接过第一句, 这一轮只回应第二句 ——
把两句都当成本轮会让模型把第一句再答一遍, 正好是要测的那种复述。老消息没有这个字段时退回到
"最后一段连续的用户消息"。

用例不带账号、会话、消息 ID: 用例编号按顺序生成 (rr-001…), 消息 ID 换成 h0、h1…, 用户换成按
出现顺序编号的 u01… (只用于按用户聚集算置信区间)。回查用的映射表单独存在证据目录, 和原始
窗口一起到期删除; 映射表删了以后, 用例就连不回生产数据。
"""
from __future__ import annotations

from datetime import datetime

from app.services.effect.repetition import PREVIOUS_TURNS, PREVIOUS_WINDOW, group_turns

WINDOW_TURNS = 7  # 用户这一句所在的轮次 + 之前 6 轮
DELAY_SECTION_SECONDS = 20  # 与 prompt_builder.DELAY_EXPLANATION_MIN_SECONDS 一致: 到这个延迟才带「回复时机说明」
KEEP_META = ("proactive", "trigger_type", "kind", "component_card", "offering_received")
TURN_ID_PREFIX = "user-turn:"


def _parse(ts) -> datetime:
    value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return value.replace(tzinfo=None) if value.tzinfo else value


def _rows(raw_rows: list[dict]) -> list[dict]:
    """导出的行 → group_turns 能用的形状 (时间转成 naive UTC, trace_id 提出来)。"""
    out = []
    for r in raw_rows:
        meta = r.get("metadata") or {}
        out.append({**r, "created_at": _parse(r["created_at"]), "trace_id": meta.get("trace_id"),
                    "proactive": bool(meta.get("proactive")),
                    "reply_path": (meta.get("response_diagnostics") or {}).get("reply_path")})
    return out


def answered_rows(turn, user_turn, rows: list[dict]) -> list[dict]:
    """这次回复回应的那几条用户消息 (按时间顺序)。

    有 achievement_turn_id 就只认它: 它点名的消息不在导出窗口里时返回空 (用例丢弃),
    不退回 "最后一段用户消息" —— 那正是会把上一轮接过的话再塞进本轮的错误口径。
    """
    turn_id = (turn.rows[0].get("metadata") or {}).get("achievement_turn_id") or ""
    if turn_id.startswith(TURN_ID_PREFIX):
        ids = set(turn_id[len(TURN_ID_PREFIX):].split(","))
        return [r for r in rows if r["id"] in ids and r["role"] == "user"]
    return list(user_turn.rows)


def build_case(window: dict, number: int, user_key: str) -> tuple[dict, dict] | None:
    """一个导出窗口 → (用例, 映射)。找不到对应的回复轮次或前面没有用户消息时返回 None。"""
    score = window["score"]
    rows = _rows(window["rows"])
    turns = group_turns(rows)
    index = next((i for i, t in enumerate(turns)
                  if t.role == "assistant" and t.rows[0]["id"] == score["turn_message_id"]), None)
    if index is None:
        return None
    turn = turns[index]
    user_index = next((i for i in range(index - 1, -1, -1) if turns[i].role == "user"), None)
    if user_index is None:
        return None
    user_turn = turns[user_index]
    current = answered_rows(turn, user_turn, rows)
    if not current:
        return None
    previous_ai = [t for t in turns[:index] if t.role == "assistant"
                   and turn.start - t.end <= PREVIOUS_WINDOW][-PREVIOUS_TURNS:]
    ids: dict[str, str] = {}

    def alias(real: str) -> str:
        return ids.setdefault(real, f"h{len(ids)}")

    history_rows = []
    for t in turns[max(0, index - WINDOW_TURNS):index]:
        for r in t.rows:
            meta = {k: r["metadata"][k] for k in KEEP_META if k in (r.get("metadata") or {})}
            history_rows.append({"id": alias(r["id"]), "role": r["role"], "content": r["content"],
                                 "createdAt": r["created_at"].isoformat() + "+00:00", "metadata": meta})
    if any(r["id"] not in ids for r in current):  # 回应的消息不在窗口里: 复现不出当时的上下文
        return None
    diag = (turn.rows[0].get("metadata") or {}).get("response_diagnostics") or {}
    delay = (turn.start - current[-1]["created_at"]).total_seconds()
    tags = [score["situation"]]
    if score["label"] in ("verbatim", "paraphrase", "legit"):
        tags.append(score["label"])
    if delay >= DELAY_SECTION_SECONDS:
        tags.append("delay_section")
    if (diag.get("topic_continuation") or {}).get("status") == "applied":
        tags.append("topic_continuation")
    if any((r.get("metadata") or {}).get("trigger_type") == "followup_unfinished" for t in previous_ai for r in t.rows):
        tags.append("after_followup")
    if len(current) < len(user_turn.rows):
        tags.append("partial_user_turn")
    case = {
        "id": f"rr-{number:03d}",
        "kind": "positive" if score["label"] in ("verbatim", "paraphrase") else "negative",
        "production_label": score["label"],
        "situation": score["situation"],
        "tags": tags,
        "user": user_key,
        "now": turn.start.isoformat() + "+00:00",
        "received_at": current[-1]["created_at"].isoformat() + "+00:00",
        # 生产里同一轮连发的几条在聚合时直接拼接 (aggregation 用 "".join)
        "message": "".join(r["content"] for r in current),
        "current_ids": [alias(r["id"]) for r in current],
        "history_rows": history_rows,
        # 评判用的口径与 10.1 eligible_turns 一致: 最后一段用户消息以空格相连, 之前各轮气泡不过滤
        "judge_user_text": " ".join(user_turn.bubbles),
        "previous_ai_turns": [list(t.bubbles) for t in previous_ai],
        "production_reply": turn.bubbles,
    }
    mapping = {"case": case["id"], "turn_message_id": score["turn_message_id"],
               "conversation_id": score["conversation_id"], "user_id": score["user_id"],
               "ids": {v: k for k, v in ids.items()}}
    return case, mapping


def build_cases(export: dict) -> tuple[list[dict], list[dict], list[str]]:
    """整份导出 → (用例, 映射表, 丢弃的轮次)。用户按首次出现编号。"""
    users: dict[str, str] = {}
    cases, mappings, dropped = [], [], []
    for window in export["windows"]:
        user = users.setdefault(window["score"]["user_id"], f"u{len(users) + 1:02d}")
        built = build_case(window, len(cases) + 1, user)
        if built is None:
            dropped.append(window["score"]["turn_message_id"])
            continue
        cases.append(built[0])
        mappings.append(built[1])
    return cases, mappings, dropped
