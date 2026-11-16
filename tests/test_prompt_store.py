"""Prompt rendering and API guards; transaction behavior uses real PG/Redis tests."""
from datetime import datetime, timezone
from types import SimpleNamespace as Row
from unittest.mock import AsyncMock
import pytest
from app.services.prompting import store

@pytest.mark.parametrize('content',['',' ','\n\t'])
async def test_empty_save_is_rejected_without_any_write(content):
    with pytest.raises(ValueError):
        await store.update_prompt_text('memory.relevance',content)

@pytest.mark.parametrize('operation', ['get_prompt_text','is_prompt_enabled','reset_prompt_text'])
async def test_unknown_key_is_rejected(operation):
    with pytest.raises(KeyError):
        await getattr(store,operation)('does.not.exist')

@pytest.mark.parametrize('expected',['2026-01-01T00:00:00Z','2026-01-01T08:00:00+08:00'])
def test_timestamp_guard_compares_instants_not_format(expected):
    store._check_expected(Row(revision=4,updatedAt=datetime(2026,1,1,tzinfo=timezone.utc)),expected,4)

def test_timestamp_guard_rejects_invalid_input():
    with pytest.raises(ValueError): store._check_expected(Row(revision=1),'invalid')

@pytest.mark.parametrize('revision',[0,3,5])
def test_revision_guard_rejects_stale_and_missing_rows(revision):
    with pytest.raises(store.PromptUpdateConflictError): store._check_expected(Row(revision=4),expected_revision=revision)

async def test_live_disable_overrides_bound_content(monkeypatch):
    key='memory.relevance'
    monkeypatch.setattr(store,'db',Row(prompttemplate=Row(find_unique=AsyncMock(return_value=Row(isEnabled=False)))))
    token=store._prompt_snapshot.set({key:('old captured content',True)})
    try:
        with pytest.raises(store.PromptDisabledError): await store.get_prompt_text(key)
    finally: store._prompt_snapshot.reset(token)

async def test_bound_disabled_prompt_cannot_be_enabled_mid_turn(monkeypatch):
    monkeypatch.setattr(store,'db',Row(prompttemplate=Row(find_unique=AsyncMock(return_value=Row(isEnabled=True)))))
    token=store._prompt_snapshot.set({'memory.relevance':('old',False)})
    try: assert await store.is_prompt_enabled('memory.relevance') is False
    finally: store._prompt_snapshot.reset(token)

async def test_snapshot_keeps_source_provenance_and_content(monkeypatch):
    monkeypatch.setattr(store,'db',Row(prompttemplate=Row(find_unique=AsyncMock(return_value=Row(isEnabled=True,content='new')))))
    token=store._prompt_snapshot.set({'memory.relevance':('captured',True)})
    try:
        text=await store.get_prompt_text('memory.relevance')
        assert text=='captured'
        assert str(text)=='captured'
    finally: store._prompt_snapshot.reset(token)

@pytest.mark.parametrize('content',['回复 {1,2,4} 条','输出 {title, body}','少了一半 {name','多了 }'])
async def test_save_rejects_templates_that_cannot_render(content):
    with pytest.raises(ValueError):
        await store.update_prompt_text('memory.relevance',content)

@pytest.mark.parametrize('content',['你好 {name}','JSON 示例 {{"a": 1}}','{n:.2f} {x!r}','没有占位符'])
def test_valid_templates_pass_syntax_check(content):
    store.check_template_syntax(content)

def test_every_code_default_passes_syntax_check():
    from app.services.prompting.registry import PROMPT_DEFINITIONS
    for definition in PROMPT_DEFINITIONS:
        store.check_template_syntax(definition.default_text)

def test_raw_text_prompts_may_keep_literal_braces():
    # chat.system_base 原样拼进 prompt, 不走 format; 字面大括号是合法内容
    store.check_template_syntax('示例 {"a": 1} 和 {1,2}', 'chat.system_base')
    with pytest.raises(ValueError):
        store.check_template_syntax('示例 {1,2}', 'memory.relevance')


# Q01.01: 变量提取与 PromptDefinition 新字段
def test_extract_vars_from_template():
    from app.services.prompting.store import _extract_vars
    # 基本提取
    assert _extract_vars("你好 {name}，你的年龄是 {age}") == ["name", "age"]
    # 去重、保序
    assert _extract_vars("{x} {y} {x}") == ["x", "y"]
    # 转义 {{ }} 不提取
    assert _extract_vars("示例 {{json}} 和 {real}") == ["real"]
    # 无占位符
    assert _extract_vars("纯文本") == []


def test_response_includes_vars_from_default_text():
    from app.services.prompting.store import _extract_vars, _response
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    # chat.temporal_check 有 {date}/{evidence}/{question}/{reply}
    defn = PROMPT_DEFINITION_MAP["chat.temporal_check"]
    response = _response(defn, None)
    assert isinstance(response["required_vars"], list)
    assert "date" in response["required_vars"]
    assert "question" in response["required_vars"]
    assert response["optional_vars"] == []


def test_response_uses_registry_vars_when_declared():
    from app.services.prompting.store import _response
    from app.services.prompting.registry import PromptDefinition
    defn = PromptDefinition(
        key="test.fake", title="T", stage="S", category="C",
        description="D", default_text="hello {name} {extra}",
        required_vars=("name",), optional_vars=("extra",),
    )
    response = _response(defn, None)
    assert response["required_vars"] == ["name"]
    assert response["optional_vars"] == ["extra"]


def test_every_definition_has_valid_var_contract():
    """registry 里所有 prompt 的 required_vars + optional_vars 子集等于 default_text 占位符集合（允许 registry 比实际少，不允许多声明不存在的变量）."""
    from app.services.prompting.registry import PROMPT_DEFINITIONS
    from app.services.prompting.store import _extract_vars
    for defn in PROMPT_DEFINITIONS:
        actual = set(_extract_vars(defn.default_text))
        declared = set(defn.required_vars) | set(defn.optional_vars)
        extra = declared - actual
        assert not extra, (
            f"{defn.key}: registry 声明了不存在于 default_text 的变量 {extra}"
        )
