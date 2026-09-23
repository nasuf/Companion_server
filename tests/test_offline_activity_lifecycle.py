"""P2 到达/预言/归档服务编排单测（mock 仓储/发射，覆盖 spec §3.3/§4.5/§4.7）。"""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.services.offline import activity_service

# 上海人广附近；~200m 内的到达点与 ~8km 外的远点。
_PLACE_LAT, _PLACE_LNG = 31.2304, 121.4737
_NEAR_LAT, _NEAR_LNG = 31.2312, 121.4737  # ~89m
_FAR_LAT, _FAR_LNG = 31.30, 121.4737       # ~7.7km


def _row(**over):
    base = {
        "id": "a1", "user_id": "u1", "workspace_id": "w1", "agent_id": "ag1",
        "conversation_id": "c1", "status": "accepted", "title": "植物园",
        "summary": "", "description": "", "reached": False,
        "place_lat": _PLACE_LAT, "place_lng": _PLACE_LNG,
        "created_at": "2026-09-19T00:00:00+00:00",
        "updated_at": "2026-09-19T00:00:00+00:00",
    }
    base.update(over)
    return base


def _patch_common(monkeypatch):
    monkeypatch.setattr(
        activity_service.repo, "resolve_user_context",
        AsyncMock(return_value={"conversation_id": "c1", "agent_id": "ag1", "workspace_id": "w1"}),
    )
    monkeypatch.setattr(activity_service, "insert_user_activity_card", AsyncMock())
    monkeypatch.setattr(activity_service, "emit_assistant", AsyncMock())
    monkeypatch.setattr(activity_service, "emit_activity_card", AsyncMock())
    monkeypatch.setattr(activity_service, "remember_user_event", lambda **_: None)
    monkeypatch.setattr(
        activity_service.repo,
        "find_other_reached_activity",
        AsyncMock(return_value=None),
    )
    # 拍摄条件生成在后台 fire——测试里吞掉，避免真实执行/未 await 警告。
    monkeypatch.setattr(
        activity_service, "fire_background",
        lambda coro: coro.close() if hasattr(coro, "close") else None,
    )


async def test_arrive_success_within_radius(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=_row()))
    mark = AsyncMock(return_value=_row(reached=True))
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)

    result = await activity_service.arrive_activity("u1", "a1", lat=_NEAR_LAT, lng=_NEAR_LNG)

    assert result.reached is True
    mark.assert_awaited_once()
    activity_service.insert_user_activity_card.assert_awaited_once()
    activity_service.emit_assistant.assert_awaited_once()  # 拍照引导


async def test_arrive_rejects_when_another_activity_is_already_reached(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo,
        "get_activity",
        AsyncMock(return_value=_row()),
    )
    monkeypatch.setattr(
        activity_service.repo,
        "find_other_reached_activity",
        AsyncMock(return_value=_row(id="other", reached=True)),
    )
    mark = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)

    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity("u1", "a1")

    assert exc.value.status_code == 409
    mark.assert_not_awaited()


async def test_arrive_rejected_too_far(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=_row()))
    mark = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)

    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity("u1", "a1", lat=_FAR_LAT, lng=_FAR_LNG)

    assert exc.value.status_code == 422
    assert exc.value.detail["reason"] == "too_far"
    mark.assert_not_awaited()  # 失败不改状态（spec §4.5）


async def test_arrive_honor_system_without_coords(monkeypatch):
    # 无客户端坐标：即便地点有经纬度也放行（允许用户直接点击到达）。
    _patch_common(monkeypatch)
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=_row()))
    mark = AsyncMock(return_value=_row(reached=True))
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)

    result = await activity_service.arrive_activity("u1", "a1")  # 不传坐标

    assert result.reached is True
    mark.assert_awaited_once()
    assert mark.await_args.kwargs["lat"] is None


async def test_arrive_idempotent_when_reached(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity", AsyncMock(return_value=_row(reached=True))
    )
    mark = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "mark_arrived", mark)

    result = await activity_service.arrive_activity("u1", "a1", lat=_NEAR_LAT, lng=_NEAR_LNG)

    assert result.reached is True
    mark.assert_not_awaited()


async def test_arrive_rejects_non_accepted(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity", AsyncMock(return_value=_row(status="completed"))
    )
    with pytest.raises(HTTPException) as exc:
        await activity_service.arrive_activity("u1", "a1", lat=_NEAR_LAT, lng=_NEAR_LNG)
    assert exc.value.status_code == 409


async def test_prophecy_draw_once(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(activity_service.repo, "get_activity", AsyncMock(return_value=_row()))
    monkeypatch.setattr(
        activity_service.repo, "set_prophecy",
        AsyncMock(return_value=_row(prophecy_text="路上会有一阵刚好合适的风。")),
    )
    result = await activity_service.draw_prophecy("u1", "a1")
    assert result.prophecy_text


async def test_prophecy_idempotent_when_already_drawn(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity",
        AsyncMock(return_value=_row(prophecy_text="已经抽过的预言")),
    )
    set_p = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "set_prophecy", set_p)
    result = await activity_service.draw_prophecy("u1", "a1")
    assert result.prophecy_text == "已经抽过的预言"
    set_p.assert_not_awaited()


async def test_prophecy_rejected_after_reached(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity", AsyncMock(return_value=_row(reached=True))
    )
    with pytest.raises(HTTPException) as exc:
        await activity_service.draw_prophecy("u1", "a1")
    assert exc.value.status_code == 409


async def test_archive_success(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity", AsyncMock(return_value=_row(reached=True))
    )
    monkeypatch.setattr(
        activity_service.repo, "mark_archived",
        AsyncMock(return_value=_row(status="completed", reached=True)),
    )
    monkeypatch.setattr(
        activity_service, "_with_completion_feedback",
        AsyncMock(side_effect=lambda a: a),
    )
    result = await activity_service.archive_activity("u1", "a1")
    assert result.status == "completed"
    activity_service.emit_activity_card.assert_awaited_once()  # 档案卡


async def test_archive_idempotent_when_completed(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        activity_service.repo, "get_activity",
        AsyncMock(return_value=_row(status="completed")),
    )
    mark = AsyncMock()
    monkeypatch.setattr(activity_service.repo, "mark_archived", mark)
    monkeypatch.setattr(
        activity_service, "_with_completion_feedback",
        AsyncMock(side_effect=lambda a: a),
    )
    result = await activity_service.archive_activity("u1", "a1")
    assert result.status == "completed"
    mark.assert_not_awaited()
