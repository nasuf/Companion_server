"""The receipt retains exactly the document text consumed by the parser."""
import hashlib
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.services.agent_template import knowledge as kn
from app.services.agent_template.knowledge_import import (
    parse_knowledge_document, parse_knowledge_document_snapshot, validate_knowledge_snapshot,
)
from app.services.memory.input_evidence import prepare_input_origin
from tests.knowledge_input_test_support import mock_append


def test_snapshot_preserves_parser_behavior_and_actual_decoded_input():
    text = "公司介绍\r\n公司名称：伴生\r\n业务：陪伴服务\r\n"
    raw = text.encode("utf-8-sig")
    items, origin = parse_knowledge_document_snapshot(raw, filename="company.txt")
    assert items == parse_knowledge_document(raw, filename="company.txt")
    validate_knowledge_snapshot(items, origin)
    payload = json.loads(origin.payload)["input"]
    assert payload["upload_sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["upload_size_bytes"] == len(raw)
    assert "公司名称：伴生" in payload["text"]
    assert "filename" not in payload
    assert payload["items"][0]["summary"] == items[0].summary


@pytest.mark.parametrize("mutation", ["kind", "parser", "items", "text", "empty", "overlimit"])
def test_snapshot_rejects_mismatched_batch(mutation):
    items, origin = parse_knowledge_document_snapshot("知识\n内容：测试".encode())
    payload = json.loads(origin.payload)["input"]
    if mutation == "kind":
        origin = prepare_input_origin(kind="user_edit", payload=payload)
    elif mutation == "parser":
        payload["parser_version"] = "future"
        origin = prepare_input_origin(kind="knowledge_document", payload=payload)
    elif mutation == "items":
        items = [replace(items[0], summary="wrong")]
    elif mutation == "text":
        payload["text"] = "知识\n内容：不同"
        origin = prepare_input_origin(kind="knowledge_document", payload=payload)
    elif mutation == "empty":
        items = []
    else:
        items *= 201
    with pytest.raises(ValueError):
        validate_knowledge_snapshot(items, origin)


@pytest.mark.asyncio
async def test_duplicate_upload_has_no_embedding_origin_or_new_memory(monkeypatch):
    text = "知识\n内容：短的"
    parsed, _ = parse_knowledge_document_snapshot(text.encode())
    items, origin, create, _ = mock_append(monkeypatch, text, existing=[parsed[0].summary])
    result = await kn.append_knowledge_to_template(
        template_agent_id="agent", template_user_id="user", items=items, input_origin=origin,
    )
    assert result["stored"] == 0 and result["skipped_duplicates"] == 1
    create.assert_not_awaited()
    kn.generate_embedding.assert_not_awaited()
    kn.persist_input_origin.assert_not_awaited()
    kn.after_memory_changelog.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_workspace_rejects_after_embedding_before_writes(monkeypatch):
    items, origin, create, tx = mock_append(monkeypatch, "知识\n内容：短的")
    tx.query_raw = AsyncMock(side_effect=[[], [], []])
    with pytest.raises(ValueError, match="scope_changed"):
        await kn.append_knowledge_to_template(
            template_agent_id="agent", template_user_id="user", items=items, input_origin=origin,
        )
    create.assert_not_awaited()
    kn.persist_input_origin.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_duplicate_requires_retry_instead_of_missing_vector(monkeypatch):
    text = "知识\n内容：短的"
    parsed, _ = parse_knowledge_document_snapshot(text.encode())
    items, origin, create, tx = mock_append(monkeypatch, text, existing=[parsed[0].summary])
    tx.query_raw = AsyncMock(side_effect=[[], [], [{"id": "ws"}], []])
    with pytest.raises(ValueError, match="retry"):
        await kn.append_knowledge_to_template(
            template_agent_id="agent", template_user_id="user", items=items, input_origin=origin,
        )
    create.assert_not_awaited()
    kn.persist_input_origin.assert_not_awaited()
