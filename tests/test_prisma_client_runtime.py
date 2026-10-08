"""Generated-client resource and real PostgreSQL relation regressions.

Depth-expanded TypedDicts formerly consumed >1.4 GiB on import per worker.
The resource check runs in a fresh interpreter; the query checks use an owned
synthetic user, including a relation chain deeper than the old depth setting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from pydantic import ValidationError
from prisma import Json, Prisma
from prisma.models import User


def test_generated_client_fits_a_single_worker_import_budget():
    if sys.platform != "linux":
        pytest.skip("The worker resource contract uses Linux /proc")
    probe = """
import json, resource
from pathlib import Path
from datetime import datetime, timezone
from prisma.models import AiMemory
from prisma._compat import model_parse
now = datetime.now(timezone.utc).isoformat()
row = dict(id='synthetic-memory', userId='synthetic-user', level=2,
    content='Isolated generated-client test', importance=0.6, mentionCount=0,
    isArchived=False, createdAt=now, updatedAt=now)
records = [model_parse(AiMemory, dict(row, id=f'synthetic-{i}'))
           for i in range(25742)]
assert len(records) == 25742
print(json.dumps({'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'generated_type_bytes': Path(__import__('prisma.types', fromlist=['']).__file__).stat().st_size}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=90, check=True,
    )
    metrics = json.loads(result.stdout)
    assert metrics["peak_rss_kib"] < 1024 * 1024, metrics
    assert metrics["generated_type_bytes"] < 8 * 1024 * 1024, metrics


@pytest_asyncio.fixture
async def generated_client_scope():
    url = os.getenv("PROACTIVE_E2E_DATABASE_URL")
    if not url:
        pytest.skip("An isolated migrated PostgreSQL database is required")
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    assert parsed.path == "/companion_proactive_e2e"
    client = Prisma(datasource={"url": url})
    await client.connect()
    user = None
    try:
        marker = "generated-client-" + uuid.uuid4().hex
        user = await client.user.create(data={
            "username": marker,
            "authIdentities": {"create": [{
                "provider": "synthetic-client-regression",
                "providerAccountId": marker,
                "rawProfile": Json({"synthetic": True, "label": "你好"}),
            }]},
        })
        yield client, user
    finally:
        try:
            if user is not None:
                await client.user.delete(where={"id": user.id})
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_deep_relation_query_and_serialization(generated_client_scope):
    client, owner = generated_client_scope
    include: dict = {"authIdentities": True}
    for _ in range(3):
        include = {"authIdentities": {"include": {"user": {"include": include}}}}
    result = await client.user.find_unique(where={"id": owner.id}, include=include)
    assert result is not None
    payload = result.model_dump(mode="json")
    for _ in range(3):
        assert payload["id"] == owner.id
        assert len(payload["authIdentities"]) == 1
        identity = payload["authIdentities"][0]
        assert identity["rawProfile"] == {"synthetic": True, "label": "你好"}
        payload = identity["user"]
    assert payload["id"] == owner.id
    assert payload["authIdentities"][0]["userId"] == owner.id


@pytest.mark.asyncio
async def test_scalar_update_defaults_nulls_and_datetime(generated_client_scope):
    client, owner = generated_client_scope
    updated = await client.user.update(
        where={"id": owner.id}, data={"displayName": "Synthetic name"},
    )
    assert updated is not None
    assert updated.displayName == "Synthetic name"
    assert updated.avatarKey is None
    assert updated.status == "active"
    assert updated.executionGeneration
    assert isinstance(updated.createdAt, datetime)
    assert updated.createdAt.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert updated.createdAt == owner.createdAt
    assert updated.updatedAt >= owner.updatedAt


@pytest.mark.asyncio
async def test_empty_and_unrequested_relations_preserve_wire_shape(generated_client_scope):
    client, owner = generated_client_scope
    result = await client.user.find_unique(
        where={"id": owner.id}, include={"agents": True},
    )
    assert result is not None
    assert result.agents == []
    assert result.authIdentities is None
    assert result.model_dump(mode="json")["agents"] == []


@pytest.mark.asyncio
async def test_recursive_models_keep_required_field_validation(generated_client_scope):
    _, owner = generated_client_scope
    payload = owner.model_dump(mode="json")
    payload.pop("username")
    with pytest.raises(ValidationError):
        User.model_validate(payload)
