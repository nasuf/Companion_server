"""时间感知回复的 LLM 评审。

确定性红线 (must_not_contain) 只能抓最露骨的失败 (凌晨说"早上好")。得体与否、
有没有微妙的时间幻觉、有没有无端复活旧话题, 这些要语义判断。仿 reply_register/
judge.py: 评审模型最好来自另一个厂商 (LLM-as-judge 自偏好是已知问题)。
"""

from __future__ import annotations

import json


def gap_text(seconds: float) -> str:
    m = int(seconds // 60)
    if m < 60:
        return f"约 {m} 分钟"
    h = int(seconds // 3600)
    if h < 24:
        return f"约 {h} 小时"
    return f"约 {int(seconds // 86400)} 天"


JUDGE_PROMPT = """你在评估一个 AI 陪伴角色的回复**对时间的感知是否自然**, 像不像真人。

【客观事实】
- 距离用户上一条消息, 实际过去了：{gap}   ← 这是"用户离开了多久"(间隔)
- 现在是本地时间 {hour} 点（0-23 时制）   ← 这是"现在几点"(时刻)
- 间隔前的对话：
{history}
- 用户现在说：{message}
- AI 的回复：{reply}

【两个维度必须分清, 别混】
- 间隔 = 用户离开了多久。间隔尺度参考: 15分钟内≈几乎没离开(继续聊就好, 提"间隔"反而怪); 15分钟-2小时=短暂离开; 2-12小时=离开了一阵; 超过1天=久别。
- 时刻 = 现在几点。AI 说"这么晚""深夜了""这么早"是在评论**当前时刻**(时刻维度), **绝不能**当成对间隔的处理, 也**绝不是**时间幻觉——凌晨1点说"这么晚还在忙"是对的。

输出 JSON（只依据时间处理, 不评价其他方面好坏）：
{{
  "gap_ok": true/false,      // 对"用户离开了{gap}"处理得体吗？久别(>1天)要自然承认、别硬接旧话题；几分钟的间隔不该硬凹"好久不见"；短暂离开轻描淡写即可。只看间隔, 不看时刻。
  "hallucination": true/false, // 有没有把**间隔时长**说错？把久别当"刚才还在聊"、把几分钟夸成"好几天"、编一个明显错误的时长=true。对当前时刻的正确评论(这么晚/这么早)不算幻觉。
  "tod_ok": true/false,      // 贴合 {hour} 点吗？不能用错时段的问候(凌晨/深夜说"早上好""下午好"就是错; 早上说"晚上好"也错)；深夜可有夜的意识但不强求。
  "stale_topic": true/false, // 间隔较大时, AI 有没有**无端**把用户没重新提起的旧话题翻出来接着聊？(用户已转到新话题却被拉回旧话题=true)
  "reason": "一句话说明最关键的判断"
}}

只输出 JSON。回复内容普通、话少都不扣分。"""


def build_prompt(*, gap_seconds: float, hour: int, history: str, message: str, reply: str) -> str:
    return JUDGE_PROMPT.format(
        gap=gap_text(gap_seconds), hour=hour,
        history=history or "（无——这是这段时间的第一句）",
        message=message, reply=reply,
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
    for k in ("gap_ok", "hallucination", "tod_ok", "stale_topic"):
        v = d.get(k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip().lower() in ("true", "是", "yes", "1")
        else:
            return None
    out["reason"] = str(d.get("reason") or "")[:120]
    return out
