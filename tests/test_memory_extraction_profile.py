"""Regression: extract_memories must use memory_extract profile (long timeout +
streaming idle_timeout protection), not default chat_extract.

生产 trace 复现 (2026-04-29 019dd808, conv 690d80fc):
  用户一次性发画像 dump (血型/体重/体型/穿搭/忌口/亲戚...20+ 条事实) 长 ~250 字.
  LLM 输出 1500-2500 tokens JSON, qwen-plus 50-80 tok/s → 25-50s 接近
  chat_extract profile 45s 总 timeout. asyncio.wait_for cancel 时 langchain
  callback 跳过 → langsmith trace 永留 status=pending → extract_memories try/
  except 吃异常返空 dict → 用户 20+ 条事实**全部 silent 丢失**, DB 一条没存.

修: extract_memories 改用 memory_extract profile (streaming + idle_timeout 60s
+ 总 180s + ollama fallback). idle_timeout 保护持续吐字场景, fallback 保 LLM
临时挂时本地兜底.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_extract_memories_uses_memory_extract_profile():
    """extract_memories 调 invoke_json 时必须传 profile="memory_extract" (字符串名).

    ⚠️ 关键: profile 必须是 **字符串名字**, 不是 get_profile(...) 拿到的 CallProfile
    对象. 因为 invoke_json → _invoke_with_resilience → _provider_and_profile 内部
    自己会调 get_profile(override) 解析, 误传 CallProfile 对象会变成
    get_profile(get_profile("memory_extract")) → KeyError(CallProfile(...)) →
    extract_memories 的 try/except 吃掉返空 dict → DB 0 条. 生产 trace 019e0004
    (2026-05-07) 全双侧 user/ai 同时 silent failure.
    """
    from app.services.memory.recording.extraction import extract_memories
    from app.services.llm.resilience import get_profile

    captured_profile = {"value": None}

    async def fake_invoke_json(model, prompt, **kwargs):
        captured_profile["value"] = kwargs.get("profile")
        return {"memories": [], "entities": [], "preferences": [], "topics": []}

    with (
        patch(
            "app.services.memory.recording.extraction.invoke_json",
            side_effect=fake_invoke_json,
        ),
        patch(
            "app.services.memory.recording.extraction.get_chat_model",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.memory.recording.extraction.get_prompt_text",
            new_callable=AsyncMock,
            return_value=(
                "test prompt {new_conversation} {context_conversation} "
                "{current_time} {taxonomy_list}"
            ),
        ),
    ):
        await extract_memories("user: 测试", side="user")

    profile_arg = captured_profile["value"]
    # 必须是字符串名字, 不是 CallProfile 对象 (regression guard for 019e0004)
    assert profile_arg == "memory_extract", (
        f"extract_memories must pass profile='memory_extract' string, got {profile_arg!r}. "
        "传 CallProfile 对象会导致 invoke_json 内部嵌套 get_profile 调用 → "
        "KeyError(CallProfile) → silent failure"
    )

    # 独立从 registry 拿 profile, 验证它的 streaming + idle_timeout 防护属性
    # (确保 registry 注册的 memory_extract 真有这些保护)
    profile = get_profile("memory_extract")
    assert profile.stream_mode is True, "memory_extract must use streaming for idle_timeout protection"
    assert profile.idle_timeout_s is not None and profile.idle_timeout_s >= 30
    assert profile.timeout_s >= 120, f"timeout_s={profile.timeout_s} 太短 (chat_extract=45 已是 bug 边界)"
    assert profile.allow_ollama_fallback is True, "extraction 失败应允许 ollama 兜底"


@pytest.mark.asyncio
async def test_extract_memories_handles_timeout_silently_returns_empty():
    """LLM timeout 时返回空 dict 而非抛异常 (post_process pipeline 不能被炸掉).

    Trade-off: 静默 fallback 让用户感知不到, 但避免 cascading failure. 通过
    EVT_LLM_FAIL 日志在 axiom 留痕方便排查.
    """
    import asyncio
    from app.services.memory.recording.extraction import extract_memories

    async def hang_then_timeout(*args, **kwargs):
        raise asyncio.TimeoutError("LLM timeout")

    with (
        patch(
            "app.services.memory.recording.extraction.invoke_json",
            side_effect=hang_then_timeout,
        ),
        patch(
            "app.services.memory.recording.extraction.get_chat_model",
            return_value=MagicMock(),
        ),
        patch(
            "app.services.memory.recording.extraction.get_prompt_text",
            new_callable=AsyncMock,
            return_value=(
                "test prompt {new_conversation} {context_conversation} "
                "{current_time} {taxonomy_list}"
            ),
        ),
    ):
        result = await extract_memories("user: 测试", side="user")

    # Silent fallback: 不抛异常, 不破坏 post_process pipeline. 但带
    # _extraction_error 标记, 让上层 pipeline 抛 MemoryExtractionError → 水位线
    # 不推进 → 后续 batch 重试 (Phase 3 修复: 防瞬时失败永久漏录).
    assert result == {
        "memories": [], "entities": [], "preferences": [], "topics": [],
        "_extraction_error": True,
    }
