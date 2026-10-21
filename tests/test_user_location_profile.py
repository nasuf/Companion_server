"""Profile-location acknowledgements must reflect Prisma raw SQL mapping rows."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from app.api.public import users
from app.api.jwt_auth import require_user


@pytest.mark.parametrize('permission', ['whileInUse', 'always'])
async def test_saved_location_http_response_preserves_raw_sql_fields(monkeypatch, permission):
    updated = datetime(2026, 10, 21, 6, 31, tzinfo=UTC)
    row = {
        'location_latitude': 0.0,
        'location_longitude': 0.0,
        'location_city': '测试市',
        'location_region': '测试区',
        'location_country': '测试国家',
        'location_permission_status': permission,
        'location_updated_at': updated,
    }
    query = AsyncMock(return_value=[row])
    monkeypatch.setattr(users, 'db', SimpleNamespace(query_raw=query))
    app = FastAPI()
    app.include_router(users.router)
    app.dependency_overrides[require_user] = lambda: {'sub': 'authenticated-user'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.put('/users/me/location', json={
            'latitude': 0, 'longitude': 0, 'city': '测试市', 'region': '测试区',
            'country': '测试国家', 'permission_status': permission,
        })
    assert response.status_code == 200
    assert response.json() == {
        'has_location': True, 'latitude': 0.0, 'longitude': 0.0,
        'city': '测试市', 'region': '测试区', 'country': '测试国家',
        'permission_status': permission, 'updated_at': updated.isoformat(),
    }
    assert query.await_args.args[1] == 'authenticated-user'


@pytest.mark.parametrize('latitude,longitude,permission', [
    (None, None, 'whileInUse'), (1.0, None, 'always'),
    (None, 2.0, 'always'), (1.0, 2.0, 'denied'),
    (1.0, 2.0, 'deniedForever'), (1.0, 2.0, 'service_disabled'),
    (1.0, 2.0, None),
])
def test_missing_coordinates_or_permission_never_report_success(latitude, longitude, permission):
    result = users._location_response({
        'location_latitude': latitude, 'location_longitude': longitude,
        'location_permission_status': permission,
    })
    assert result.has_location is False
    assert result.latitude == latitude
    assert result.longitude == longitude
    assert result.permission_status == permission
