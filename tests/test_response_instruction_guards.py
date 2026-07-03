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


def test_response_instruction_no_forced_reply_count():
    """C1 拟人度契约: 不允许回到"分{n}条消息回复"强制装配模式.

    条数应由 LLM 按内容/情绪自然决定 (最多三条), 硬上限由
    split_and_validate_replies 的 MAX_REPLY_COUNT 保证, 不靠 prompt 强制."""
    from app.services.prompting.defaults import (
        L3_MEMORY_REPLY_PROMPT,
        MEDIUM_MEMORY_REPLY_PROMPT,
        RESPONSE_INSTRUCTION_PROMPT,
        STRONG_MEMORY_REPLY_PROMPT,
        WEAK_MEMORY_REPLY_PROMPT,
    )

    for p in (
        RESPONSE_INSTRUCTION_PROMPT,
        WEAK_MEMORY_REPLY_PROMPT,
        MEDIUM_MEMORY_REPLY_PROMPT,
        STRONG_MEMORY_REPLY_PROMPT,
        L3_MEMORY_REPLY_PROMPT,
    ):
        assert "分{n}条" not in p, "不允许恢复强制条数指令"
        assert "||" in p, "仍需说明 || 分隔符 (下游 split 依赖该约定)"


def test_response_instruction_forbids_timestamp_prefix_mimicry():
    """B1 配套: 历史消息带 [MM-DD HH:MM] 前缀后, 必须显式告知 LLM 输出不带前缀/时间戳."""
    from app.services.prompting.defaults import RESPONSE_INSTRUCTION_PROMPT

    assert "时间戳" in RESPONSE_INSTRUCTION_PROMPT or "前缀" in RESPONSE_INSTRUCTION_PROMPT
