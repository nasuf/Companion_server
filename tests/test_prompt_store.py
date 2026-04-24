"""ensure_prompt_templates 启动同步策略测试.

核心语义: 代码 defaults.py 的 default_text 改了 → DB.content 一并覆盖,
用户 UI 定制作废; default 未变 → 保留 UI 定制.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeRedisPipeline:
    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def set(self, key, value):
        self.ops.append(("set", key, value))

    def delete(self, key):
        self.ops.append(("delete", key))

    async def execute(self):
        return None


class _FakeRedis:
    def __init__(self) -> None:
        self.pipeline_ref = _FakeRedisPipeline()

    def pipeline(self):
        return self.pipeline_ref


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
    )


@pytest.fixture
def prompt_store_mocks():
    """安装 store.py 内 db / get_redis / PROMPT_DEFINITIONS 所需 mock.

    返回一个 (mock_db, fake_redis, set_definitions) helper; 测试用
    set_definitions([def1, def2, ...]) 指定要注入的 PROMPT_DEFINITIONS.
    """
    fake_redis = _FakeRedis()
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
    assert ("set", "prompt_template:new.key", "fresh text") in fake_redis.pipeline_ref.ops


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
    assert ("set", "prompt_template:test.key", "user custom") in fake_redis.pipeline_ref.ops


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
    assert ("set", "prompt_template:test.key", "new v2") in fake_redis.pipeline_ref.ops


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
    assert ("set", "prompt_template:test.key", "user custom") in fake_redis.pipeline_ref.ops
