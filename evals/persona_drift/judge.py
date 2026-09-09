"""人设语气漂移的 LLM 评审。

身份事实 (名字/职业/城市) 用确定性红线兜; 语气/性格是否还跟开场一致要语义判断。
比较深层对话后的风格探针回复 vs 开场时的, 看人设 voice 有没有被长上下文磨平成
通用助手腔。评审模型建议换非 ark 厂商 (chat 是豆包)。
"""

from __future__ import annotations

import json

JUDGE_PROMPT = """你在评估一个 AI 陪伴角色在长对话后**人设语气有没有漂移**。

这个角色的人设基调：{persona}

同一个开放问题「{question}」，在两个时间点的回答：
- 对话刚开始时：{early}
- 聊了很久之后：{late}

判断「聊了很久之后」那条，相对开场，人设语气是否还稳（输出 JSON）：
{{
  "voice_consistent": true/false,  // 语气/性格还是同一个人吗？变得明显更平淡、更客服腔、更"助手味"、丢了开场那种鲜活个性 = false
  "persona_match": true/false,     // 「聊了很久之后」那条还贴合上面的人设基调吗？
  "reason": "一句话说明最关键的差别"
}}

只看人设语气一致性，不评价内容对错。只输出 JSON。"""


def build_prompt(*, persona: str, question: str, early: str, late: str) -> str:
    return JUDGE_PROMPT.format(persona=persona, question=question, early=early, late=late)


def parse_verdict(raw: str) -> dict | None:
    t = (raw or "").strip()
    if "{" in t and "}" in t:
        t = t[t.index("{"): t.rindex("}") + 1]
    try:
        d = json.loads(t)
    except Exception:
        return None
    out = {}
    for k in ("voice_consistent", "persona_match"):
        v = d.get(k)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip().lower() in ("true", "是", "yes", "1")
        else:
            return None
    out["reason"] = str(d.get("reason") or "")[:120]
    return out
