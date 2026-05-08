"""Phase: split LLM 整层删除验收.

历史: 主 LLM 输出后再调小模型拆 N 句, 引入 2 个 bug:
1. lstrip 字符集贪心 → "12月8号" 首字符被误删 (生产 trace 019e0116)
2. prompt 含"适当扩写" → LLM 看 🎂 编"祝你生日快乐" (5 月语境错)

修复: 删 split LLM 整层, 主 LLM 在 chat.response_instruction 已被指令
"分N条||分隔", 信主 LLM 输出按 ||/\\n\\n 切.
"""

from __future__ import annotations


def test_split_reply_to_n_sentences_removed():
    """split_reply_to_n_sentences 函数已从 intent_replies 移除."""
    from app.services.chat import intent_replies

    assert not hasattr(intent_replies, "split_reply_to_n_sentences"), (
        "split_reply_to_n_sentences 已删除 — 主 LLM 直接按 || 输出"
    )


def test_reply_split_prompts_removed_from_registry():
    """reply.split_2 / reply.split_3 已从 prompt registry 移除."""
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    assert "reply.split_2" not in PROMPT_DEFINITION_MAP
    assert "reply.split_3" not in PROMPT_DEFINITION_MAP


def test_reply_split_prompt_constants_removed():
    """REPLY_SPLIT_2_PROMPT / REPLY_SPLIT_3_PROMPT 模块属性不存在."""
    from app.services.prompting import defaults

    assert not hasattr(defaults, "REPLY_SPLIT_2_PROMPT")
    assert not hasattr(defaults, "REPLY_SPLIT_3_PROMPT")


def test_split_replies_signature_no_split_llm_fn():
    """_split_replies 不再接收 split_llm_fn 参数 (已移除整层)."""
    import inspect
    from app.services.chat.reply_generate import _split_replies

    sig = inspect.signature(_split_replies)
    assert "split_llm_fn" not in sig.parameters


def test_generate_reply_signature_no_split_llm_fn():
    """generate_reply 不再接收 split_llm_fn 参数."""
    import inspect
    from app.services.chat.reply_generate import generate_reply

    sig = inspect.signature(generate_reply)
    assert "split_llm_fn" not in sig.parameters


def test_main_llm_output_with_double_pipe_splits_into_two():
    """主 LLM 给"句1||句2" → split_and_validate_replies 切成 2 条 (不调任何额外 LLM)."""
    from app.services.chat.orchestrator import split_and_validate_replies

    raw = "今天天气真好||我想去散步"
    result = split_and_validate_replies(raw, max_count=3, max_per_reply=60, max_total=150)

    assert result == ["今天天气真好", "我想去散步"]


def test_main_llm_single_sentence_stays_single():
    """主 LLM 给单句 (无 ||) → 保持单条, 不再硬拆/扩写.

    历史 bug 复现: 主 LLM 给"12月8号，射手座，没错吧？😌🎂"
    旧路径会调 split_2 LLM 强拆, LLM 编"祝你生日快乐" (5 月语境错).
    新路径直接单条返回, 内容 100% 来自主 LLM.
    """
    from app.services.chat.orchestrator import split_and_validate_replies

    raw = "12月8号，射手座，没错吧？😌🎂"
    result = split_and_validate_replies(raw, max_count=3, max_per_reply=60, max_total=150)

    # 单条, 内容跟主 LLM 输出一致
    assert len(result) == 1
    assert result[0] == "12月8号，射手座，没错吧？😌🎂"


def test_main_llm_double_newline_splits_too():
    """主 LLM 用 \\n\\n 分段 → 也按多条切 (split_and_validate_replies 兼容)."""
    from app.services.chat.orchestrator import split_and_validate_replies

    raw = "你好啊\n\n最近怎么样"
    result = split_and_validate_replies(raw, max_count=3, max_per_reply=60, max_total=150)

    assert result == ["你好啊", "最近怎么样"]


def test_split_replies_bounds_overlong_single_reply_without_split_llm():
    """主 LLM 给 1 条但很长 → 本地按标点切短, 不再调用 split LLM 扩写。"""
    import asyncio
    from app.services.chat.reply_generate import _split_replies
    from app.services.chat.orchestrator import (
        split_and_validate_replies, truncate_at_sentence,
    )

    long_reply = "这是一段很长的回复。" * 20  # ~200 chars
    result, source = asyncio.run(_split_replies(
        long_reply,
        max_reply_count=3,
        max_per_reply=60,
        max_total=150,
        truncate_fn=lambda t, n: truncate_at_sentence(t, n),
        pipe_fallback_fn=split_and_validate_replies,
    ))

    assert 1 <= len(result) <= 3
    assert all(len(part) <= 60 for part in result)
    assert "".join(result).startswith("这是一段很长的回复。")
    assert source == "main_split"
