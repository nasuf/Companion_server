"""Regression: chat.response_instruction 必须禁"括号内动作/神态/心理描写"旁白.

生产 trace (2026-05-02 trace 019ded17): 用户发 "😱😘", AI 回:
  "（看到那个😘表情，噗嗤笑出来）你这是在恐怖片里突然撒糖啊..."
对话上下文也充斥 "（微微一怔，语气放缓）" "（摇头笑了笑，眼神坦荡）" 这种网文/RP
旁白片段. 真实微信聊天没人写这些 — LLM 把"有人格 + 自然" 误解成"小说写法".

修: RESPONSE_INSTRUCTION_PROMPT 加一般原则规则 (不 lift 具体 trace phrase, 防
跟之前 RECORD_REQUEST / contradiction prompt 同样 overfit 过具体反例).
"""

from __future__ import annotations


def test_response_instruction_forbids_parenthetical_action_narration():
    """禁止旁白式动作/神态/心理描写 — 用一般原则措辞."""
    from app.services.prompting.defaults import RESPONSE_INSTRUCTION_PROMPT
    p = RESPONSE_INSTRUCTION_PROMPT

    # 必须显式说"聊天不是小说/角色扮演"
    assert "小说" in p or "角色扮演" in p, (
        "必须说明聊天不是小说/RP, 否则 LLM 把'有人格' 当成可以写旁白"
    )
    # 必须显式禁止动作/神态/心理类旁白
    assert "动作" in p, "必须显式禁动作旁白"
    assert "神态" in p or "表情" in p, "必须显式禁神态/表情旁白"
    assert "心理" in p or "心理活动" in p, "必须显式禁心理活动旁白"
    # 必须 cover 多种写法 (括号 + 星号), 防 LLM 换 markdown 写法绕过
    assert "括号" in p, "必须禁括号写法 (主流 RP 写法)"


def test_response_instruction_no_specific_trace_phrases():
    """守门: 不允许 lift 生产 trace 的具体反例 phrase 进 prompt — 这是过拟合.

    维护者: 加新规则用一般原则 (动作/神态/心理 + 多种写法), 不写具体例子
    ('噗嗤笑出来'/'微微一怔'). 同 contradiction prompt overfit 教训."""
    from app.services.prompting.defaults import RESPONSE_INSTRUCTION_PROMPT
    p = RESPONSE_INSTRUCTION_PROMPT

    case_phrases = ("噗嗤", "微微一怔", "摇头笑了笑", "眼神坦荡", "语气放缓")
    for phrase in case_phrases:
        assert phrase not in p, (
            f"response_instruction 含生产 trace phrase '{phrase}' — 应该用"
            f"抽象表述 (动作/神态/心理 旁白) 替代具体反例"
        )
