"""主动交流·自然度 LLM 评审.

判定五维: naturalness / source_fit / persona_match / mentions_card / advertorial_feel.
评审建议非 ark 厂商 (chat=豆包, judge 换 dashscope:qwen-plus 避免自评偏好).
"""

from __future__ import annotations

import json


JUDGE_PROMPT = """你在评估一个 AI 陪伴角色**主动**发给用户的消息 (加可选的分享卡片) 像不像真人朋友.

【场景】
- AI 名字/人设: {agent_persona}
- 用户画像: {user_portrait}
- 触发场景: {trigger_type}
- 号称的话题来源: {source_kind}
  - user_interest_match = 从用户兴趣里挑话题, 期望表达是"勾用户"式 ("你不是说过X吗")
  - ai_persona_match    = 从 AI 自己兴趣挑, 期望表达是"我最近..."/"我在..."
  - socially_hot        = 纯社交谈资 (大家都在聊), 期望表达是"你听说了吗X"/"刷到个X"
  - none                = 无外部内容, 期望是普通问候/关切
- 提供给 AI 参考的候选内容 (它可能用也可能不用):
{trending_bullets}

【AI 实际发出】
- 消息: {message}
- 附带卡片: {card_desc}

【判定 JSON】
{{
  "naturalness": true/false,        // 这句像 20+ 一线上班族朋友群里会发的一句吗? 生硬/客服腔/新闻播报腔=false
  "source_fit": true/false,         // 消息表达跟号称的话题来源匹配吗? user档硬凹AI兴趣=false, 谈资档说得像新闻主持人=false
  "persona_match": true/false,      // 跟 AI 人设一致吗? (说话方式 / 兴趣领域)
  "mentions_card": true/false,      // 若有卡: 消息里提到了卡里的东西吗? (无卡:填 true, 空跳过)
  "advertorial_feel": true/false,   // 有"AI 在推广告"的感觉吗? (关键反指标, 想要 false)
  "used_junk_content": true/false,  // AI 有没有引用了明显不适合朋友闲聊的候选内容 (八卦/死讯/纯 UI 残余)? 想要 false
  "reason": "一句话说明最关键的判断"
}}

只看主动消息本身的自然度, 不评价文法/长度. 只输出 JSON."""


def _fmt_bullets(candidates: list[dict]) -> str:
    if not candidates:
        return "  (无候选内容, AI 只靠上下文自己发)"
    lines = []
    for c in candidates:
        title = c.get("title", "")
        snippet = (c.get("snippet") or "")[:100]
        plat = c.get("platform", "")
        lines.append(f"  - [{plat}] {title}: {snippet}")
    return "\n".join(lines)


def _fmt_card(card: dict | None) -> str:
    if not card:
        return "(无卡)"
    return f"[{card.get('platform', '?')}] {card.get('title', '')[:60]}"


def build_prompt(
    *, agent_persona: str, user_portrait: str, trigger_type: str,
    source_kind: str, trending_candidates: list[dict],
    message: str, card: dict | None,
) -> str:
    return JUDGE_PROMPT.format(
        agent_persona=agent_persona,
        user_portrait=user_portrait,
        trigger_type=trigger_type,
        source_kind=source_kind,
        trending_bullets=_fmt_bullets(trending_candidates),
        message=message,
        card_desc=_fmt_card(card),
    )


_BOOL_KEYS = (
    "naturalness", "source_fit", "persona_match",
    "mentions_card", "advertorial_feel", "used_junk_content",
)


def parse_verdict(raw: str) -> dict | None:
    t = (raw or "").strip()
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    try:
        d = json.loads(t)
    except Exception:
        return None
    out = {}
    for k in _BOOL_KEYS:
        v = d.get(k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip().lower() in ("true", "是", "yes", "1")
        else:
            return None
    out["reason"] = str(d.get("reason") or "")[:150]
    return out
