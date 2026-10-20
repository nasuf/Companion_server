"""Real PostgreSQL + HTTP: exact mobile copy, read-only inspection and scope fences."""

import json
import io
from PIL import Image
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from prisma import Prisma

from app.api.admin.activities import router
from app.api.public.offline import router as public_router
from app.services import game_points
from app.services.auth import create_jwt
from app.services.offline import (
    activity_admin as svc,
    gift_admin_repository,
    repository as mobile,
)
from app.services.offline import activity_media_storage
from app.services.chat_media import storage as chat_storage


@pytest.fixture
async def scenario(monkeypatch, tmp_path):
    url = os.getenv("PROACTIVE_E2E_DATABASE_URL")
    if not url:
        pytest.skip("isolated PostgreSQL URL required")
    assert (
        urlsplit(url).hostname in {"127.0.0.1", "localhost"}
        and urlsplit(url).path == "/companion_proactive_e2e"
    )
    database = Prisma(datasource={"url": url}, http={"trust_env": False})
    await database.connect()
    monkeypatch.setattr('app.db.db', database)
    for module in (svc, mobile, gift_admin_repository, game_points):
        monkeypatch.setattr(module, "db", database)
    monkeypatch.setattr(activity_media_storage, "_MEDIA_DIR", tmp_path)
    monkeypatch.setattr(chat_storage, "_MEDIA_DIR", tmp_path)
    (
        u,
        b,
        agent,
        other,
        wa,
        wb,
        c,
        cb,
        deleted,
        activity,
        foreign,
        condition,
        media,
        message,
        attachment,
    ) = [uuid4().hex for _ in range(15)]
    now = datetime.now(UTC).replace(tzinfo=None)
    for owner, ai, ws, conv in [(u, agent, wa, c), (b, other, wb, cb)]:
        await database.execute_raw(
            "INSERT INTO users(id,username,display_name,updated_at) VALUES($1,$2,$3,$4::timestamp)",
            owner,
            "admin-activity-" + owner,
            "用户" + owner[:4],
            now,
        )
        await database.execute_raw(
            "INSERT INTO ai_agents(id,user_id,name,mbti,updated_at) VALUES($1,$2,'小伴','{}'::jsonb,$3::timestamp)",
            ai,
            owner,
            now,
        )
        await database.execute_raw(
            "INSERT INTO chat_workspaces(id,user_id,agent_id,status,updated_at) VALUES($1,$2,$3,'active',$4::timestamp)",
            ws,
            owner,
            ai,
            now,
        )
        await database.execute_raw(
            "INSERT INTO conversations(id,user_id,agent_id,workspace_id,updated_at) VALUES($1,$2,$3,$4,$5::timestamp)",
            conv,
            owner,
            ai,
            ws,
            now,
        )
    await database.execute_raw(
        "INSERT INTO conversations(id,user_id,agent_id,workspace_id,is_deleted,updated_at) VALUES($1,$2,$3,$4,TRUE,$5::timestamp)",
        deleted,
        u,
        agent,
        wa,
        now,
    )
    summary = "值得了解的花艺小店，先看看项目再决定。"
    description = "第一段真实推荐文案。\n\n第二段是实际地点与玩法。"
    note = "原始推荐寄语：你喜欢探店，可以先了解这里的作品。\n不用赶时间。"
    for ident, owner, ai, ws, conv, title in [
        (activity, u, agent, wa, c, "春树·花艺"),
        (foreign, b, other, wb, cb, "另一用户的活动"),
    ]:
        await database.execute_raw(
            """INSERT INTO offline_activity_recommendations(id,user_id,agent_id,workspace_id,conversation_id,status,title,summary,description,
            image_urls,discovery_metadata,reached,arrival_confirmed_at,conditions_ready_at,updated_at)
            VALUES($1,$2,$3,$4,$5,'accepted',$6,$7,$8,$9::jsonb,$10::jsonb,TRUE,$11::timestamp,$11::timestamp,$11::timestamp)""",
            ident,
            owner,
            ai,
            ws,
            conv,
            title,
            summary,
            description,
            json.dumps(
                [
                    "https://example.com/1.jpg",
                    "https://example.com/2.jpg",
                    "https://example.com/3.jpg",
                ]
            ),
            json.dumps(
                {"recommendation_message": note, "private_token": "DO_NOT_LEAK"}
            ),
            now - timedelta(hours=1),
        )
    await database.execute_raw(
        "INSERT INTO offline_shooting_conditions(id,recommendation_id,short_name,criteria,category,triggered) VALUES($1,$2,'窗边植物','画面主体清晰','植物',TRUE)",
        condition,
        activity,
    )
    await database.execute_raw(
        "UPDATE offline_activity_recommendations SET focus_condition_id=$1 WHERE id=$2",
        condition,
        activity,
    )
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (70, 190, 170)).save(image, "PNG")
    png = image.getvalue()
    key = f"{u}_image_test.png"
    (tmp_path / key).write_bytes(png)
    await database.execute_raw(
        "INSERT INTO messages(id,conversation_id,role,content,metadata) VALUES($1,$2,'assistant','活动卡片',$3::jsonb)",
        message,
        c,
        json.dumps(
            {
                "source_id": activity,
                "real_world_type": "activity",
                "trace_id": "PRIVATE_TRACE_ID",
                "private_token": "DO_NOT_LEAK",
                "component_card": {
                    "version": 1,
                    "type": "offline_activity",
                    "title": "发送时的卡片标题",
                    "subtitle": "待回应 · 花店",
                    "body": "发送时的原始摘要",
                    "footer": "点击查看活动详情",
                    "payload": {
                        "activity_id": activity,
                        "image_urls": [f"/offline/media/{key}"],
                    },
                },
            }
        ),
    )
    await database.execute_raw(
        "INSERT INTO offline_activity_media(id,recommendation_id,user_id,kind,mime,size,storage_key,url,source_message_id) VALUES($1,$2,$3,'image','image/png',15,$4,$5,$6)",
        media,
        activity,
        u,
        key,
        f"/offline/media/{key}",
        message,
    )
    await database.execute_raw(
        "INSERT INTO chat_message_attachments(id,user_id,conversation_id,message_id,kind,mime,size,storage_key,url,updated_at) VALUES($1,$2,$3,$4,'image','image/png',15,$5,$6,$7::timestamp)",
        attachment,
        u,
        c,
        message,
        f"{u}_chat.png",
        f"/chat/media/{u}_chat.png",
        now,
    )
    (tmp_path / f"{u}_chat.png").write_bytes(png)
    await database.execute_raw(
        "INSERT INTO offline_thought_fragments(id,recommendation_id,tier,text,condition_id,source_message_id) VALUES($1,$2,'rare','已经发给用户的片刻感想',$3,$4)",
        uuid4().hex,
        activity,
        condition,
        message,
    )
    app = FastAPI()
    app.include_router(router)
    app.include_router(public_router)
    await database.user.update(where={'id':u}, data={'role':'admin'})
    token = create_jwt(u, "admin")
    headers = {"Authorization": "Bearer " + token}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as client:
        try:
            yield SimpleNamespace(
                db=database,
                client=client,
                user=u,
                bob=b,
                agent=agent,
                other=other,
                workspace=wa,
                foreign_workspace=wb,
                conversation=c,
                bob_conversation=cb,
                deleted=deleted,
                activity=activity,
                foreign=foreign,
                condition=condition,
                media=media,
                message=message,
                attachment=attachment,
                now=now,
                tmp=tmp_path,
                png=png,
            )
        finally:
            await database.execute_raw(
                "DELETE FROM messages WHERE conversation_id=ANY($1::text[])",
                [c, cb, deleted],
            )
            await database.execute_raw(
                "DELETE FROM offline_activity_recommendations WHERE user_id=ANY($1::text[])",
                [u, b],
            )
            await database.execute_raw(
                "DELETE FROM conversations WHERE user_id=ANY($1::text[])", [u, b]
            )
            await database.execute_raw(
                "DELETE FROM chat_workspaces WHERE user_id=ANY($1::text[])", [u, b]
            )
            await database.execute_raw(
                "DELETE FROM ai_agents WHERE user_id=ANY($1::text[])", [u, b]
            )
            await database.execute_raw(
                "DELETE FROM users WHERE id=ANY($1::text[])", [u, b]
            )
            await database.disconnect()


def path(s, suffix="", **params):
    return f"/admin-api/activities/{s.activity}{suffix}", {"user_id": s.user, **params}


async def test_exact_mobile_copy_and_read_only_detail_with_hidden_tasks(scenario):
    s = scenario
    before = (
        await s.db.query_raw(
            "SELECT * FROM offline_activity_recommendations WHERE id=$1", s.activity
        )
    )[0]
    r = await s.client.get(*path(s)[:1], params=path(s)[1])
    assert r.status_code == 200, r.text
    public = await s.client.get(f"/offline/activities/{s.activity}")
    assert public.status_code == 200 and r.json()["activity"] == public.json()
    assert r.json()["stage"] == "in_progress" and r.json()["counts"] == {
        "tasks": 1,
        "media": 1,
        "fragments": 1,
    }
    assert (
        r.json()["task_generation"] == "ready"
        and len(r.json()["activity"]["image_urls"]) == 3
    )
    assert "DO_NOT_LEAK" not in r.text and "discovery_metadata" not in r.text
    after = (
        await s.db.query_raw(
            "SELECT * FROM offline_activity_recommendations WHERE id=$1", s.activity
        )
    )[0]
    assert after == before
    if os.getenv("ACTIVITY_ADMIN_WEB_PROJECT"):
        from tests.activity_admin_browser_support import check_browser

        await check_browser(s, public.json(), os.environ["ACTIVITY_ADMIN_WEB_PROJECT"])


async def test_user_search_workspaces_filters_paging_and_scope(scenario):
    s = scenario
    search = await s.client.get("/admin-api/activities/users", params={"q": s.user})
    assert [u["user_id"] for u in search.json()] == [s.user]
    context = await s.client.get(f"/admin-api/activities/users/{s.user}")
    assert context.status_code == 200 and [
        w["id"] for w in context.json()["workspaces"]
    ] == [s.workspace]
    params = {"user_id": s.user, "limit": 1}
    rows = await s.client.get("/admin-api/activities", params=params)
    assert rows.status_code == 200, rows.text
    assert rows.json()["total"] == 1 and rows.json()["items"][0]["id"] == s.activity
    for f in ["pending", "planned", "completed", "closed"]:
        assert (
            await s.client.get("/admin-api/activities", params={**params, "status": f})
        ).json()["total"] == 0
    assert (
        await s.client.get(
            "/admin-api/activities", params={**params, "status": "in_progress"}
        )
    ).json()["total"] == 1
    assert (
        await s.client.get("/admin-api/activities", params={**params, "offset": 1})
    ).json()["items"] == []
    assert (
        await s.client.get("/admin-api/activities", params={**params, "q": "%"})
    ).json()["total"] == 0
    assert (
        await s.client.get(
            "/admin-api/activities", params={**params, "q": "' OR 1=1 --"}
        )
    ).json()["total"] == 0
    assert (
        await s.client.get(
            "/admin-api/activities",
            params={**params, "workspace_id": s.foreign_workspace},
        )
    ).status_code == 404
    assert (
        await s.client.get(
            *path(s, user_id=s.bob)[:1], params=path(s, user_id=s.bob)[1]
        )
    ).status_code == 404


@pytest.mark.parametrize(
    "status,reached,stage",
    [
        ("pending", False, "pending"),
        ("accepted", False, "planned"),
        ("accepted", True, "in_progress"),
        ("completed", True, "completed"),
        ("ignored", False, "closed"),
        ("cancelled", False, "closed"),
        ("expired", False, "closed"),
    ],
)
async def test_all_users_list_preserves_status_filters_owners_and_paging(
    scenario, status, reached, stage
):
    s = scenario
    marker = "global-scope-" + s.activity
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET title=$1,status=$2,reached=$3 WHERE id=ANY($4::text[])",
        marker,
        status,
        reached,
        [s.activity, s.foreign],
    )
    params = {"q": marker, "status": stage, "limit": 1}
    first = await s.client.get("/admin-api/activities", params=params)
    assert first.status_code == 200, first.text
    data = first.json()
    assert data["total"] == 2 and len(data["items"]) == 1 and data["sampled_at"]
    second = (
        await s.client.get("/admin-api/activities", params={**params, "offset": 1})
    ).json()
    rows = data["items"] + second["items"]
    assert {r["user_id"] for r in rows} == {s.user, s.bob}
    for row in rows:
        assert row["username"] == "admin-activity-" + row["user_id"]
        assert row["nickname"] == "用户" + row["user_id"][:4]
        detail = await s.client.get(
            "/admin-api/activities/" + row["id"], params={"user_id": row["user_id"]}
        )
        assert detail.status_code == 200 and detail.json()["user_id"] == row["user_id"]
    own = (
        await s.client.get(
            "/admin-api/activities", params={**params, "user_id": s.user}
        )
    ).json()
    assert own["total"] == 1 and own["items"][0]["id"] == s.activity
    other_stage = "pending" if stage != "pending" else "planned"
    assert (
        await s.client.get(
            "/admin-api/activities", params={**params, "status": other_stage}
        )
    ).json()["total"] == 0
    assert (
        await s.client.get("/admin-api/activities", params={**params, "status": "all"})
    ).json()["total"] == 2
    assert (
        await s.client.get(
            "/admin-api/activities", params={**params, "q": marker + "%"}
        )
    ).json()["total"] == 0


async def test_global_list_owner_nickname_uses_latest_wechat_profile_only(scenario):
    s = scenario
    for offset, nickname in [(-1, "旧昵称"), (0, "当前微信昵称")]:
        await s.db.execute_raw(
            "INSERT INTO auth_identities(id,user_id,provider,provider_account_id,raw_profile,updated_at) VALUES($1,$2,'wechat',$3,$4::jsonb,$5::timestamp)",
            uuid4().hex,
            s.user,
            uuid4().hex,
            json.dumps({"nickname": nickname, "private_token": "DO_NOT_LEAK"}),
            s.now + timedelta(hours=offset),
        )
    response = await s.client.get("/admin-api/activities", params={"q": s.activity})
    assert response.status_code == 200
    assert response.json()["items"][0]["nickname"] == "当前微信昵称"
    assert "DO_NOT_LEAK" not in response.text and "raw_profile" not in response.text


async def test_all_users_list_is_admin_only_and_workspace_requires_owner(scenario):
    s = scenario
    url = "/admin-api/activities"
    assert (await s.client.get(url, headers={"Authorization": ""})).status_code == 401
    assert (
        await s.client.get(
            url, headers={"Authorization": "Bearer " + create_jwt(s.bob, "user")}
        )
    ).status_code == 403
    assert (
        await s.client.get(url, params={"workspace_id": s.workspace})
    ).status_code == 422
    assert (await s.client.get(url + "/" + s.activity)).status_code == 422


async def test_tasks_fragments_media_and_snapshot_preserve_real_states_and_text(
    scenario,
):
    s = scenario

    async def fetch(section):
        url, params = path(s, "/" + section)
        r = await s.client.get(url, params=params)
        assert r.status_code == 200, r.text
        return r.json()

    tasks = await fetch("tasks")
    assert tasks["items"][0]["state"] == "triggered" and tasks["items"][0]["focused"]
    fragments = await fetch("fragments")
    assert fragments["items"][0]["text"] == "已经发给用户的片刻感想"
    media = await fetch("media")
    asset = await s.client.get(media["items"][0]["url"])
    assert (
        asset.status_code == 200
        and asset.content == s.png
        and "no-store" in asset.headers["cache-control"]
    )
    messages = await fetch("messages")
    row = messages["items"][0]
    assert (
        row["component_card"]["body"] == "发送时的原始摘要"
        and row["has_trace"]
        and row["directly_linked"]
    )
    assert "PRIVATE_TRACE_ID" not in json.dumps(
        messages
    ) and "DO_NOT_LEAK" not in json.dumps(messages)
    image = await s.client.get(row["component_card"]["image_urls"][0])
    assert image.status_code == 200
    chat = await s.client.get(row["attachments"][0]["url"])
    assert chat.content == s.png


async def test_chats_fence_deleted_other_users_workspaces_and_label_time_context(
    scenario,
):
    s = scenario
    for conv, text, when, meta in [
        (s.conversation, "旅途中普通聊天", s.now, {}),
        (s.conversation, "到达前的无关聊天", s.now - timedelta(hours=2), {}),
        (
            s.bob_conversation,
            "跨用户伪造关联",
            s.now,
            {"real_world_type": "activity", "source_id": s.activity},
        ),
        (
            s.deleted,
            "已删除会话",
            s.now,
            {"real_world_type": "activity", "source_id": s.activity},
        ),
        (
            s.conversation,
            "另一活动推送",
            s.now,
            {"real_world_type": "activity", "source_id": s.foreign},
        ),
    ]:
        await s.db.execute_raw(
            "INSERT INTO messages(id,conversation_id,role,content,created_at,metadata) VALUES($1,$2,'user',$3,$4::timestamp,$5::jsonb)",
            uuid4().hex,
            conv,
            text,
            when,
            json.dumps(meta),
        )
    url, params = path(s, "/messages")
    rows = (await s.client.get(url, params=params)).json()
    assert rows["total"] == 1
    context = (await s.client.get(url, params={**params, "mode": "journey"})).json()
    assert context["total"] == 2 and any(
        r["content"] == "旅途中普通聊天" and not r["directly_linked"]
        for r in context["items"]
    )
    for text in ["跨用户伪造关联", "已删除会话", "到达前的无关聊天", "另一活动推送"]:
        assert text not in json.dumps(context, ensure_ascii=False)


@pytest.mark.parametrize(
    "suffix", ["", "/tasks", "/media", "/fragments", "/messages", "/images/0"]
)
async def test_admin_only_and_requested_owner_required(scenario, suffix):
    s = scenario
    url, params = path(s, suffix)
    assert (
        await s.client.get(url, params=params, headers={"Authorization": ""})
    ).status_code == 401
    assert (
        await s.client.get(
            url,
            params=params,
            headers={"Authorization": "Bearer " + create_jwt(s.bob, "user")},
        )
    ).status_code == 403
    assert (await s.client.get(url, params={"user_id": s.bob})).status_code == 404


@pytest.mark.parametrize(
    "params",
    [{"limit": 51}, {"offset": 100001}, {"status": "unknown"}, {"user_id": ""}],
)
async def test_list_query_bounds(scenario, params):
    s = scenario
    r = await s.client.get(
        "/admin-api/activities", params={"user_id": s.user, **params}
    )
    assert r.status_code == 422


async def test_private_media_uses_owner_check_even_for_admin_and_read_errors_are_not_empty(
    scenario,
):
    s = scenario
    key = f"{s.bob}_private.png"
    (s.tmp / key).write_bytes(b"foreign-secret")
    await s.db.execute_raw(
        "UPDATE offline_activity_media SET url=$1,storage_key=$2 WHERE id=$3",
        f"/offline/media/{key}",
        key,
        s.media,
    )
    r = await s.client.get(
        f"/admin-api/activities/{s.activity}/media/{s.media}/content",
        params={"user_id": s.user},
    )
    assert r.status_code == 403 and b"foreign-secret" not in r.content
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET conditions_ready_at=NULL WHERE id=$1",
        s.activity,
    )
    assert (await s.client.get(*path(s)[:1], params=path(s)[1])).json()[
        "task_generation"
    ] == "partial"


async def test_archived_feedback_is_stored_text_and_no_read_triggers_generation(
    scenario,
):
    s = scenario
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET status='completed',travel_note='属于本次旅途的已存故事',completed_at=CURRENT_TIMESTAMP,archived_at=CURRENT_TIMESTAMP WHERE id=$1",
        s.activity,
    )
    await s.db.execute_raw(
        "INSERT INTO offline_activity_feedback(id,recommendation_id,user_id,kind,text) VALUES($1,$2,$3,'completion','用户留下的原始反馈')",
        uuid4().hex,
        s.activity,
        s.user,
    )
    url, params = path(s)
    r = await s.client.get(url, params=params)
    assert r.json()["travel_note"] == "属于本次旅途的已存故事"
    assert r.json()["activity"]["completion_feedback"]["text"] == "用户留下的原始反馈"
    await s.db.execute_raw(
        "UPDATE offline_shooting_conditions SET triggered=FALSE WHERE id=$1",
        s.condition,
    )
    tasks = await s.client.get(url + "/tasks", params=params)
    assert tasks.json()["items"][0]["state"] == "untriggered"
    await s.db.execute_raw(
        "DELETE FROM offline_shooting_conditions WHERE id=$1", s.condition
    )
    assert (await s.client.get(url, params=params)).json()[
        "task_generation"
    ] == "ready_empty"
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET status='accepted',conditions_ready_at=NULL WHERE id=$1",
        s.activity,
    )
    assert (await s.client.get(url, params=params)).json()[
        "task_generation"
    ] == "pending"
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET reached=FALSE WHERE id=$1",
        s.activity,
    )
    assert (await s.client.get(url, params=params)).json()[
        "task_generation"
    ] == "not_generated"
    rows = await s.client.get(
        "/admin-api/activities", params={"user_id": s.user, "status": "planned"}
    )
    assert rows.json()["total"] == 1


@pytest.mark.parametrize("status", ["ignored", "cancelled", "expired"])
async def test_closed_filter_uses_real_business_status(scenario, status):
    s = scenario
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET status=$1 WHERE id=$2",
        status,
        s.activity,
    )
    r = await s.client.get(
        "/admin-api/activities", params={"user_id": s.user, "status": "closed"}
    )
    assert r.json()["items"][0]["stage"] == status


async def test_unavailable_user_conversation_asset_and_malformed_snapshot_are_explicit(
    scenario,
):
    s = scenario
    url, params = path(s)
    assert (
        await s.client.get("/admin-api/activities/users/missing")
    ).status_code == 404
    for suffix in [
        "/media/missing/content",
        "/images/-1",
        "/images/99",
        f"/messages/{s.message}/attachments/missing",
        f"/messages/{s.message}/images/99",
        "/messages/missing/images/0",
    ]:
        assert (await s.client.get(url + suffix, params=params)).status_code == 404
    key = f"{s.user}_image_test.png"
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET image_urls=$1::jsonb WHERE id=$2",
        json.dumps([f"/offline/media/{key}"]),
        s.activity,
    )
    assert (await s.client.get(url + "/images/0", params=params)).content == s.png
    await s.db.execute_raw(
        "UPDATE messages SET metadata=$1::jsonb WHERE id=$2",
        json.dumps(
            {
                "source_id": s.activity,
                "real_world_type": "activity",
                "component_card": {"type": "offline_activity", "payload": "bad"},
            }
        ),
        s.message,
    )
    r = await s.client.get(url + "/messages", params=params)
    assert r.json()["items"][0]["component_card"] is None
    await s.db.execute_raw(
        "UPDATE conversations SET is_deleted=TRUE WHERE id=$1", s.conversation
    )
    assert (await s.client.get(url + "/messages", params=params)).json()["items"] == []
    assert (await s.client.get(url, params=params)).json()[
        "conversation_available"
    ] is False


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "javascript:alert(1)",
        "http://[",
        "https://user:password@example.com/image.jpg",
    ],
)
def test_invalid_or_credential_bearing_public_image_urls_never_become_fetches(bad):
    assert svc._media_url(bad, "/admin-api/activities/safe") is None


@pytest.mark.parametrize(
    "bad", ["https://example.com/a.png", "http://[", "/unexpected/key"]
)
def test_private_gateway_never_redirects_or_fetches_an_external_url(bad):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        svc._serve(bad, None, "user")
    assert error.value.status_code == 404


async def test_legacy_null_workspace_reads_only_original_owned_conversation(scenario):
    s = scenario
    await s.db.execute_raw(
        "UPDATE offline_activity_recommendations SET workspace_id=NULL WHERE id=$1",
        s.activity,
    )
    url, params = path(s, "/messages")
    result = await s.client.get(url, params=params)
    assert result.status_code == 200 and result.json()["total"] == 1
    assert (await s.client.get(*path(s)[:1], params=path(s)[1])).json()[
        "conversation_available"
    ] is True
