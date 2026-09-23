"""聊天媒体 → 识图捕获回归。

锁死 bug: ws 传入的 attachment 其 vision_summary 尚未回填（ensure_vision_summaries
只写库 + 造新对象），on_user_chat_media 必须按 message_id 重新拉已落库的附件，
否则 photo_description 为空、识图被跳过（media 存为 material 但 miss_count 不动）。
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

import pytest

from app.services.offline import chat_capture


def _img(att_id: str, vision):
    return types.SimpleNamespace(
        id=att_id, kind="image", storage_key="k" + att_id, url="u" + att_id,
        mime="image/jpeg", size=1, width=1, height=1, vision_summary=vision,
    )


class _FakeTraceCM:
    async def __aenter__(self):
        return types.SimpleNamespace(safe_trace_id=None)

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
class TestOnUserChatMedia:
    def _patch_common(self, monkeypatch, recog, *, fresh):
        monkeypatch.setattr(
            chat_capture, "is_activity_enabled", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            chat_capture, "_active_reached_activity",
            AsyncMock(return_value={"id": "a1", "user_id": "u1", "workspace_id": "w1"}),
        )
        monkeypatch.setattr(
            chat_capture.repo, "resolve_user_context",
            AsyncMock(return_value={"conversation_id": "c1", "agent_id": "ag1"}),
        )
        monkeypatch.setattr(
            chat_capture.repo, "create_captured_media",
            AsyncMock(return_value="media1"),
        )
        monkeypatch.setattr(
            chat_capture.chat_media_repo, "find_attachments_for_message",
            AsyncMock(return_value=fresh),
        )
        monkeypatch.setattr(
            chat_capture, "offline_trace", lambda *a, **k: _FakeTraceCM()
        )
        monkeypatch.setattr(chat_capture.recognition, "recognize_on_photo", recog)

    async def test_uses_fresh_vision_when_passed_attachment_lacks_it(self, monkeypatch):
        # ws 传入的附件 vision 为空（bug 场景）；DB 已落库的 fresh 附件带描述。
        stale = _img("m1", None)
        fresh = _img("m1", "画面是宽阔的水面和青山")
        recog = AsyncMock(return_value=None)
        self._patch_common(monkeypatch, recog, fresh=[fresh])

        await chat_capture.on_user_chat_media(
            user_id="u1", workspace_id="w1", message_id="msg1", attachments=[stale]
        )
        recog.assert_awaited_once()
        # 关键: 识图拿到的是 fresh vision，不是空的 stale。
        assert "水面" in recog.await_args.kwargs["photo_description"]

    async def test_falls_back_to_passed_attachment_when_refetch_empty(self, monkeypatch):
        # 重新拉取拿不到（异常/空）时，回退到传入附件自身的 vision。
        passed = _img("m2", "石砖步道，两侧石墙")
        recog = AsyncMock(return_value=None)
        self._patch_common(monkeypatch, recog, fresh=[])

        await chat_capture.on_user_chat_media(
            user_id="u1", workspace_id="w1", message_id="msg2", attachments=[passed]
        )
        recog.assert_awaited_once()
        assert "石砖" in recog.await_args.kwargs["photo_description"]

    async def test_no_recognition_without_active_activity(self, monkeypatch):
        monkeypatch.setattr(
            chat_capture, "is_activity_enabled", AsyncMock(return_value=True)
        )
        monkeypatch.setattr(
            chat_capture, "_active_reached_activity", AsyncMock(return_value=None)
        )
        recog = AsyncMock()
        monkeypatch.setattr(chat_capture.recognition, "recognize_on_photo", recog)
        await chat_capture.on_user_chat_media(
            user_id="u1", workspace_id="w1", message_id="msg3",
            attachments=[_img("m3", "desc")],
        )
        recog.assert_not_awaited()

    async def test_only_last_image_may_emit_a_miss_hint(self, monkeypatch):
        images = [_img("m1", "第一张"), _img("m2", "第二张"), _img("m3", "第三张")]
        recog = AsyncMock(return_value=None)
        self._patch_common(monkeypatch, recog, fresh=images)

        await chat_capture.on_user_chat_media(
            user_id="u1",
            workspace_id="w1",
            message_id="msg-multi",
            attachments=images,
        )

        assert recog.await_count == 3
        flags = [
            call.kwargs["allow_miss_hint"]
            for call in recog.await_args_list
        ]
        assert flags == [False, False, True]
        near_flags = [
            call.kwargs["allow_near_followup"]
            for call in recog.await_args_list
        ]
        assert near_flags == [False, False, True]
