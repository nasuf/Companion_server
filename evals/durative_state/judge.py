"""持续性状态回复的 LLM 评审。

确定性红线只抓最露骨的方向硬伤 (过期还问"还在出差吗")。窗内外处理是否得体要
语义判断, 且必须把**状态窗的 ground truth** 喂给评审 —— 评审自己也不该去算时长,
只按我们给定的"现在在窗内/窗外"判 reply 的处理方向对不对。评审建议非 ark 厂商。
"""

from __future__ import annotations

import json


JUDGE_PROMPT = """你在评估一个 AI 陪伴角色，对用户一个**有时限的状态**处理得对不对。

【状态事实（这是你判断的依据，不用自己算时间）】
- 用户当初说过：{state_line}
- 这个状态的真实时限：{duration_hint}
- 现在相对这个状态窗：{window_desc}
- 那之后用户现在新说了：{message}
- AI 的回复：{reply}

【要判断的】这个状态现在是「{window_word}」。看 AI 回复对它的处理方向对不对。
- 如果现在是「进行中」：AI 若提到这个状态，应把它当**正在发生**（如出差还没结束、病还没好）。把它说成**已经结束/已回来/已痊愈**就是方向错。
- 如果现在是「已结束」：AI 若提到这个状态，应把它当**已经过去**。把它当成**还在进行**（问"还在出差吗""还在考试吧"）就是方向错。
- AI **完全不提**这个状态、只回应用户新说的事，永远算对（不提不扣分）。

输出 JSON：
{{
  "state_ok": true/false,      // 对状态窗的处理方向对吗？不提=true；提了且方向对=true；提了但方向错=false
  "mentioned": true/false,     // AI 回复里提到/暗示了这个状态吗？
  "reason": "一句话说明关键判断"
}}

只看对这个时限状态的处理方向，不评价其他好坏。只输出 JSON。"""


def build_prompt(*, state_line: str, duration_hint: str, kind: str,
                 message: str, reply: str) -> str:
    active = kind == "active"
    return JUDGE_PROMPT.format(
        state_line=state_line,
        duration_hint=duration_hint,
        window_desc=("状态窗还开着（现在正处于这个状态里）" if active
                     else "状态窗已经关了（这个状态早该结束了）"),
        window_word=("进行中" if active else "已结束"),
        message=message,
        reply=reply,
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
    for k in ("state_ok", "mentioned"):
        v = d.get(k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip().lower() in ("true", "是", "yes", "1")
        else:
            return None
    out["reason"] = str(d.get("reason") or "")[:120]
    return out
