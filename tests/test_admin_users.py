from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from app.api.admin.users import _serialize_wechat_identity


def test_serialize_wechat_identity_exposes_admin_profile_fields():
    identity = SimpleNamespace(
        provider="wechat",
        providerAccountId="union-1",
        openid="open-1",
        unionid="union-1",
        scope="snsapi_userinfo",
        rawProfile={
            "nickname": "七七",
            "headimgurl": "https://wx.example/avatar.png",
            "sex": 2,
            "province": "Guangdong",
            "city": "Shenzhen",
            "country": "CN",
            "privilege": ["tester"],
        },
        lastLoginAt=datetime(2026, 6, 1, 8, 30, tzinfo=UTC),
        createdAt=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
        updatedAt=datetime(2026, 6, 1, 8, 31, tzinfo=UTC),
    )

    payload = _serialize_wechat_identity(identity)

    assert payload == {
        "provider": "wechat",
        "provider_account_id": "union-1",
        "openid": "open-1",
        "unionid": "union-1",
        "scope": "snsapi_userinfo",
        "nickname": "七七",
        "avatar_url": "https://wx.example/avatar.png",
        "sex": 2,
        "province": "Guangdong",
        "city": "Shenzhen",
        "country": "CN",
        "privilege": ["tester"],
        "last_login_at": "2026-06-01 08:30:00+00:00",
        "created_at": "2026-05-20 10:00:00+00:00",
        "updated_at": "2026-06-01 08:31:00+00:00",
    }
