"""ensure_prompt_templates 启动同步策略测试.

核心语义: 代码 defaults.py 的 default_text 改了 → DB.content 一并覆盖,
用户 UI 定制作废; default 未变 → 保留 UI 定制.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_definition(
    key: str = "test.key",
    default_text: str = "new default v2",
    stage: str = "聊天",
    category: str = "回复",
    title: str = "测试 prompt",
    description: str = "for tests",
):
    return SimpleNamespace(
        key=key,
        stage=stage,
        category=category,
        title=title,
        description=description,
        default_text=default_text,
    )


def _make_existing(
    key: str = "test.key",
    row_id: str = "row-1",
    content: str = "user custom",
    defaultContent: str = "old default v1",
    stage: str = "聊天",
    category: str = "回复",
    title: str = "测试 prompt",
    description: str = "for tests",
):
    return SimpleNamespace(
        id=row_id,
        key=key,
        content=content,
        defaultContent=defaultContent,
        stage=stage,
        category=category,
        title=title,
        description=description,
        isEnabled=True,
        canaryConfig=None,
        updatedAt=SimpleNamespace(isoformat=lambda: "2026-05-19T00:00:00+00:00"),
    )


@pytest.fixture
def prompt_store_mocks(fake_aggregation_redis):
    """安装 store.py 内 db / get_redis / PROMPT_DEFINITIONS 所需 mock.

    返回一个 (mock_db, fake_redis, set_definitions) helper; 测试用
    set_definitions([def1, def2, ...]) 指定要注入的 PROMPT_DEFINITIONS.

    fake_aggregation_redis fixture 由 conftest 提供, 带 pipeline 支持
    (set/delete/execute), 复用避免造轮子.
    """
    fake_redis = fake_aggregation_redis
    mock_db = MagicMock()
    mock_db.prompttemplate = MagicMock()
    mock_db.prompttemplate.find_many = AsyncMock(return_value=[])
    mock_db.prompttemplate.create = AsyncMock(return_value=SimpleNamespace(id="new-id"))
    mock_db.prompttemplate.update = AsyncMock()
    mock_db.prompttemplate.delete_many = AsyncMock()
    mock_db.prompttemplateversion = MagicMock()
    mock_db.prompttemplateversion.create = AsyncMock()
    mock_db.prompttemplateversion.delete_many = AsyncMock()
    mock_db.query_raw = AsyncMock(return_value=[])

    definitions_holder: list = []

    def set_definitions(defs):
        definitions_holder[:] = defs

    with patch("app.services.prompting.store.db", mock_db), \
         patch("app.services.prompting.store.get_redis", AsyncMock(return_value=fake_redis)), \
         patch("app.services.prompting.store.PROMPT_DEFINITIONS", definitions_holder):
        yield mock_db, fake_redis, set_definitions


@pytest.mark.asyncio
async def test_new_key_creates_fresh(prompt_store_mocks):
    """DB 无此 key 时走 create 分支, content 初始化为 defaultContent."""
    from app.services.prompting.store import ensure_prompt_templates

    mock_db, fake_redis, set_defs = prompt_store_mocks
    definition = _make_definition(key="new.key", default_text="fresh text")
    set_defs([definition])

    await ensure_prompt_templates()

    create_kwargs = mock_db.prompttemplate.create.call_args.kwargs
    data = create_kwargs["data"]
    assert data["key"] == "new.key"
    assert data["content"] == "fresh text"
    assert data["defaultContent"] == "fresh text"
    # bootstrap 版本
    ver = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
    assert ver["source"] == "default"
    assert ver["changeType"] == "bootstrap"
    # Redis 缓存到新值
    assert fake_redis.strings["prompt_template:new.key"] == "fresh text"


@pytest.mark.asyncio
async def test_default_unchanged_preserves_user_edit(prompt_store_mocks):
    """existing.defaultContent == definition.default_text → 保留 UI 定制."""
    from app.services.prompting.store import ensure_prompt_templates

    mock_db, fake_redis, set_defs = prompt_store_mocks
    existing = _make_existing(content="user custom", defaultContent="same default")
    definition = _make_definition(default_text="same default")  # 未变
    mock_db.prompttemplate.find_many = AsyncMock(return_value=[existing])
    mock_db.query_raw = AsyncMock(
        return_value=[{"prompt_id": "row-1"}]  # 已有版本, 不补 bootstrap
    )
    set_defs([definition])

    await ensure_prompt_templates()

    # 无 update (metadata 也没变)
    mock_db.prompttemplate.update.assert_not_called()
    # 无新版本写入
    mock_db.prompttemplateversion.create.assert_not_called()
    # Redis 缓存用户定制
    assert fake_redis.strings["prompt_template:test.key"] == "user custom"


@pytest.mark.asyncio
async def test_default_changed_overrides_user_edit(prompt_store_mocks):
    """existing.defaultContent != definition.default_text → content 被覆盖,
    写 code_sync 版本记录, Redis 缓存新 default."""
    from app.services.prompting.store import ensure_prompt_templates

    mock_db, fake_redis, set_defs = prompt_store_mocks
    existing = _make_existing(content="user custom", defaultContent="old v1")
    definition = _make_definition(default_text="new v2")
    mock_db.prompttemplate.find_many = AsyncMock(return_value=[existing])
    mock_db.query_raw = AsyncMock(return_value=[{"prompt_id": "row-1"}])
    set_defs([definition])

    await ensure_prompt_templates()

    # update 被调, 新 content + defaultContent 都是 v2
    update_data = mock_db.prompttemplate.update.call_args.kwargs["data"]
    assert update_data["content"] == "new v2"
    assert update_data["defaultContent"] == "new v2"
    # code_sync 版本记录
    ver_data = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
    assert ver_data["source"] == "default"
    assert ver_data["changeType"] == "code_sync"
    assert ver_data["content"] == "new v2"
    # Redis 缓存到新 default
    assert fake_redis.strings["prompt_template:test.key"] == "new v2"


@pytest.mark.asyncio
async def test_metadata_only_change_preserves_content(prompt_store_mocks):
    """default 未变, 仅 title/description 改 → content 保留, metadata 更新."""
    from app.services.prompting.store import ensure_prompt_templates

    mock_db, fake_redis, set_defs = prompt_store_mocks
    existing = _make_existing(
        content="user custom", defaultContent="same default",
        title="旧标题", description="旧描述",
    )
    definition = _make_definition(
        default_text="same default", title="新标题", description="新描述",
    )
    mock_db.prompttemplate.find_many = AsyncMock(return_value=[existing])
    mock_db.query_raw = AsyncMock(return_value=[{"prompt_id": "row-1"}])
    set_defs([definition])

    await ensure_prompt_templates()

    # update 调一次, 但没有 content / defaultContent 字段 (不覆盖)
    update_data = mock_db.prompttemplate.update.call_args.kwargs["data"]
    assert update_data["title"] == "新标题"
    assert update_data["description"] == "新描述"
    assert "content" not in update_data
    assert "defaultContent" not in update_data
    # 无 version 写入 (未 code_sync)
    mock_db.prompttemplateversion.create.assert_not_called()
    # Redis 保留用户 content
    assert fake_redis.strings["prompt_template:test.key"] == "user custom"


@pytest.mark.asyncio
async def test_bootstrap_prompt_version_omits_null_eval_result(prompt_store_mocks):
    from app.services.prompting import store

    mock_db, _fake_redis, _set_defs = prompt_store_mocks

    await store._create_prompt_version(
        prompt_id="prompt-1",
        prompt_key="test.key",
        content="default prompt",
        source="default",
        change_type="bootstrap",
    )

    ver_data = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
    assert "evalResult" not in ver_data


@pytest.mark.asyncio
async def test_prompt_update_version_attaches_eval_result(prompt_store_mocks):
    from app.services.prompting import store

    mock_db, _fake_redis, _set_defs = prompt_store_mocks
    definition = _make_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    mock_db.prompttemplateversion.create = AsyncMock(return_value=SimpleNamespace(id="ver-1"))
    mock_db.prompttemplateversion.update = AsyncMock()

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}), \
         patch("app.services.prompting.store._prompt_eval_result", return_value={"ok": True}):
        await store._persist_prompt_update(
            "test.key",
            "new prompt",
            source="redis",
            change_type="manual_save",
        )
        # eval 快照已移出保存关键路径 (后台任务回填), 等其跑完再断言.
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            await asyncio.gather(*pending)

    ver_data = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
    assert ver_data["changeType"] == "manual_save"
    assert "evalResult" not in ver_data  # 版本行先落库保证持久性, 不带 eval
    update_kwargs = mock_db.prompttemplateversion.update.call_args.kwargs
    assert update_kwargs["where"] == {"id": "ver-1"}
    assert update_kwargs["data"]["evalResult"] is not None  # 后台回填


@pytest.mark.asyncio
async def test_prompt_update_rejects_missing_required_placeholders(prompt_store_mocks):
    from app.services.prompting.store import update_prompt_text

    _mock_db, _fake_redis, _set_defs = prompt_store_mocks
    definition = _make_definition(
        key="test.key",
        default_text="请根据 {message} 和 {context} 输出",
    )

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        with pytest.raises(ValueError, match=r"\{context\}"):
            await update_prompt_text("test.key", "请根据 {message} 输出")


def test_missing_required_placeholders_allows_cosmetic_removal():
    """装饰性占位符 ({max_per}/{total}/{n}/…) 可被管理员合法删除, 不算缺失;
    数据类占位符 ({message}/{context}/…) 删除仍算缺失。"""
    from app.services.prompting.store import _missing_required_placeholders

    default = "每条不超过{max_per}字，总共{total}字，分{n}条；根据{message}和{context}回复"
    # 只去掉装饰性占位符 → 不算缺失
    assert _missing_required_placeholders(
        default, "根据{message}和{context}回复，最多3条每条15字",
    ) == []
    # 去掉数据类 {context} → 仍算缺失
    assert _missing_required_placeholders(
        default, "根据{message}回复",
    ) == ["context"]


@pytest.mark.asyncio
async def test_prompt_canary_agent_rollout_overrides_runtime_text(prompt_store_mocks):
    from app.services.prompting.store import (
        get_prompt_text_for_context,
        update_prompt_canary_config,
    )

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="active", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    fake_redis.strings["prompt_template:test.key"] = "active"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}), \
         patch("app.services.prompting.store._prompt_eval_result", return_value={"ok": True}):
        config = await update_prompt_canary_config(
            "test.key",
            is_enabled=True,
            mode="agents",
            content="canary text",
            agent_ids=["agent-2", "agent-1", "agent-1"],
            rollout_percent=0,
        )
        matched = await get_prompt_text_for_context("test.key", agent_id="agent-1", user_id="u1")
        unmatched = await get_prompt_text_for_context("test.key", agent_id="agent-9", user_id="u1")

    assert config["agent_ids"] == ["agent-1", "agent-2"]
    assert json.loads(fake_redis.strings["prompt_canary:test.key"])["content"] == "canary text"
    assert str(matched) == "canary text"
    assert getattr(matched, "prompt_variant") == "canary"
    assert str(unmatched) == "active"


@pytest.mark.asyncio
async def test_prompt_runtime_context_applies_canary_to_plain_get(prompt_store_mocks):
    from app.services.prompting.store import (
        get_prompt_text,
        reset_prompt_runtime_context,
        set_prompt_runtime_context,
        update_prompt_canary_config,
    )

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="active", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    fake_redis.strings["prompt_template:test.key"] = "active"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}), \
         patch("app.services.prompting.store._prompt_eval_result", return_value={"ok": True}):
        await update_prompt_canary_config(
            "test.key",
            is_enabled=True,
            mode="agents",
            content="canary text",
            agent_ids=["agent-1"],
            rollout_percent=0,
        )
        token = set_prompt_runtime_context(agent_id="agent-1", user_id="u1")
        try:
            selected = await get_prompt_text("test.key")
        finally:
            reset_prompt_runtime_context(token)
        active = await get_prompt_text("test.key")

    assert str(selected) == "canary text"
    assert getattr(selected, "prompt_variant") == "canary"
    assert str(active) == "active"


@pytest.mark.asyncio
async def test_prompt_canary_rejects_missing_required_placeholders(prompt_store_mocks):
    from app.services.prompting.store import update_prompt_canary_config

    mock_db, _fake_redis, _set_defs = prompt_store_mocks
    definition = _make_definition(
        key="test.key",
        default_text="请根据 {message} 输出 {name}",
    )
    existing = _make_existing(key="test.key", content="active", defaultContent=definition.default_text)
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}), \
         patch("app.services.prompting.store._prompt_eval_result", return_value={"ok": True}):
        with pytest.raises(ValueError, match=r"\{name\}"):
            await update_prompt_canary_config(
                "test.key",
                is_enabled=True,
                mode="agents",
                content="请根据 {message} 输出",
                agent_ids=["agent-1"],
                rollout_percent=0,
            )

    mock_db.prompttemplate.update.assert_not_called()


def _make_dc_definition(key: str = "test.key", default_text: str = "default"):
    """真实 PromptDefinition dataclass — set_prompt_enabled/update_prompt_text 走 asdict()."""
    from app.services.prompting.registry import PromptDefinition

    return PromptDefinition(
        key=key,
        title="测试 prompt",
        stage="聊天",
        category="回复",
        description="for tests",
        default_text=default_text,
    )


# ─────────────────────────────────────────────────────────────────
# enable/disable 端到端 + 版本管理加固
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_enabled_local_cache():
    from app.services.prompting import store

    store._enabled_local_cache.clear()
    yield
    store._enabled_local_cache.clear()


@pytest.mark.asyncio
async def test_set_prompt_enabled_false_blocks_get_prompt_text(prompt_store_mocks):
    """停用后 get_prompt_text 抛 PromptDisabledError → 该模板从运行时彻底消失."""
    from app.services.prompting.store import (
        PromptDisabledError,
        get_prompt_text,
        set_prompt_enabled,
    )

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="active", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    mock_db.prompttemplateversion.create = AsyncMock(return_value=SimpleNamespace(id="ver-1"))
    fake_redis.strings["prompt_template:test.key"] = "active"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        result = await set_prompt_enabled("test.key", False)
        assert result["is_enabled"] is False
        # Redis enabled 缓存写入
        assert fake_redis.strings["prompt_enabled:test.key"] == "0"
        # 审计版本记录
        ver_data = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
        assert ver_data["changeType"] == "disable"

        with pytest.raises(PromptDisabledError):
            await get_prompt_text("test.key")

        # 重新启用后恢复可读
        await set_prompt_enabled("test.key", True)
        text = await get_prompt_text("test.key")
        assert str(text) == "active"


@pytest.mark.asyncio
async def test_render_prompt_returns_none_for_disabled(prompt_store_mocks):
    """render_prompt 捕获 PromptDisabledError → None (调用方 fallback 语义)."""
    from app.services.prompting.store import set_prompt_enabled
    from app.services.prompting.utils import render_prompt

    mock_db, _fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="hello {name}")
    existing = _make_existing(key="test.key", content="hello {name}", defaultContent="hello {name}")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    mock_db.prompttemplateversion.create = AsyncMock(return_value=SimpleNamespace(id="ver-1"))

    invoked = []

    async def _invoke(prompt: str):
        invoked.append(prompt)
        return "reply"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        await set_prompt_enabled("test.key", False)
        result = await render_prompt("test.key", {"name": "A"}, _invoke)

    assert result is None
    assert invoked == []  # LLM 调用彻底跳过


@pytest.mark.asyncio
async def test_update_prompt_text_dedup_skips_duplicate_version(prompt_store_mocks):
    """内容未变 → 不落新版本, 不写 DB (防版本表被重复保存刷爆)."""
    from app.services.prompting.store import update_prompt_text

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="same text", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    fake_redis.strings["prompt_template:test.key"] = "same text"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        result = await update_prompt_text("test.key", "same text")

    assert result["content"] == "same text"
    mock_db.prompttemplate.update.assert_not_called()
    mock_db.prompttemplateversion.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_prompt_text_optimistic_lock_conflict(prompt_store_mocks):
    """expected_updated_at 与 DB 不一致 → PromptUpdateConflictError (API 409)."""
    from app.services.prompting.store import PromptUpdateConflictError, update_prompt_text

    mock_db, _fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="server text", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        with pytest.raises(PromptUpdateConflictError):
            await update_prompt_text(
                "test.key",
                "my new text",
                expected_updated_at="2020-01-01T00:00:00+00:00",  # 过期快照
            )

    mock_db.prompttemplate.update.assert_not_called()
    mock_db.prompttemplateversion.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_prompt_text_rolls_back_redis_on_db_failure(prompt_store_mocks):
    """DB 落库失败 → Redis 回滚旧值, 保存报错 (不静默分叉)."""
    from app.services.prompting.store import update_prompt_text

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="old text", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(side_effect=RuntimeError("db down"))
    fake_redis.strings["prompt_template:test.key"] = "old text"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}):
        with pytest.raises(RuntimeError, match="db down"):
            await update_prompt_text("test.key", "new text")

    # Redis 回滚到编辑前值
    assert fake_redis.strings["prompt_template:test.key"] == "old text"
    mock_db.prompttemplateversion.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_prompt_text_persists_synchronously(prompt_store_mocks):
    """保存路径同步落库: 返回时 DB update + 版本记录已完成 (无 crash 丢失窗口)."""
    from app.services.prompting.store import update_prompt_text

    mock_db, fake_redis, _set_defs = prompt_store_mocks
    definition = _make_dc_definition(key="test.key", default_text="default")
    existing = _make_existing(key="test.key", content="old text", defaultContent="default")
    mock_db.prompttemplate.find_unique = AsyncMock(return_value=existing)
    mock_db.prompttemplate.update = AsyncMock(return_value=existing)
    mock_db.prompttemplateversion.create = AsyncMock(return_value=SimpleNamespace(id="ver-1"))
    mock_db.prompttemplateversion.update = AsyncMock()
    fake_redis.strings["prompt_template:test.key"] = "old text"

    with patch("app.services.prompting.store.PROMPT_DEFINITION_MAP", {"test.key": definition}), \
         patch("app.services.prompting.store._prompt_eval_result", return_value={"ok": True}):
        result = await update_prompt_text("test.key", "new text")
        # 返回即已持久化 (同步), 不依赖后台 task
        mock_db.prompttemplate.update.assert_called()
        ver_data = mock_db.prompttemplateversion.create.call_args.kwargs["data"]
        assert ver_data["changeType"] == "manual_save"
        assert ver_data["content"] == "new text"
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            await asyncio.gather(*pending)

    assert result["content"] == "new text"
    assert fake_redis.strings["prompt_template:test.key"] == "new text"
