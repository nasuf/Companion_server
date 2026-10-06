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
