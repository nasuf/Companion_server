"""用户头像存储 + 展示身份解析。

这两块的错误都是静默的: 裁剪坐标算错只是头像偏了, key 校验放松只是多开了一条
读文件的路 —— 都不会抛异常提醒任何人。
"""

import io

import pytest
from fastapi import HTTPException
from PIL import Image
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.api.public import users as users_api
from app.services import user_avatars as ua
from app.services import user_profile as up

_USER_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def avatar_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ua, "_AVATAR_DIR", tmp_path)
    return tmp_path


def _jpeg(width: int, height: int, patch: tuple[int, int, int, int] | None = None) -> bytes:
    """A dark image with an optional red square at ``(x, y, w, h)``."""
    image = Image.new("RGB", (width, height), (10, 20, 30))
    if patch:
        x, y, w, h = patch
        image.paste(Image.new("RGB", (w, h), (255, 0, 0)), (x, y))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class TestClampCropBox:
    def test_in_range_box_is_untouched(self):
        assert ua._clamp_crop_box((512, 300), (200, 100, 100)) == (200, 100, 300, 200)

    def test_oversized_edge_shrinks_to_the_short_side(self):
        assert ua._clamp_crop_box((512, 300), (0, 0, 900)) == (0, 0, 300, 300)

    def test_negative_origin_clamps_to_zero(self):
        assert ua._clamp_crop_box((512, 300), (-50, -50, 100)) == (0, 0, 100, 100)

    def test_origin_past_the_edge_slides_the_box_back_in(self):
        # 边长先收进图内, 再收原点 —— 顺序反了这里会给出越界的 right/bottom。
        assert ua._clamp_crop_box((512, 300), (500, 290, 100)) == (412, 200, 512, 300)


class TestSaveAvatar:
    def test_crop_selects_the_framed_square(self, avatar_dir):
        blob = _jpeg(1024, 600, patch=(300, 100, 200, 200))

        key = ua.save_avatar(
            user_id=_USER_ID, blob=blob, mime="image/jpeg", crop=(300, 100, 200)
        )

        stored = Image.open(ua.avatar_path(key))
        assert stored.size == (ua._AVATAR_EDGE, ua._AVATAR_EDGE)
        red, green, blue = stored.getpixel((256, 256))
        assert red > 200 and green < 40 and blue < 40

    def test_without_crop_falls_back_to_center(self, avatar_dir):
        """旧版本客户端不带 crop_* 字段, 必须仍然出一张能看的方图。"""
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(1024, 600), mime="image/jpeg")

        assert Image.open(ua.avatar_path(key)).size == (ua._AVATAR_EDGE, ua._AVATAR_EDGE)

    def test_thumbnail_sibling_is_written(self, avatar_dir):
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(800, 800), mime="image/jpeg")

        thumb = ua.avatar_path(ua.thumb_key(key))
        assert Image.open(thumb).size == (ua._THUMB_EDGE, ua._THUMB_EDGE)

    def test_stored_avatar_is_far_smaller_than_the_upload(self, avatar_dir):
        """头像每屏都在渲染 —— 存原图等于每次拉取都付一次相机原图的带宽。"""
        blob = _jpeg(1024, 1024)

        key = ua.save_avatar(user_id=_USER_ID, blob=blob, mime="image/jpeg")

        assert ua.avatar_path(key).stat().st_size < len(blob)

    def test_key_is_prefixed_with_the_user_id(self, avatar_dir):
        """注销时按 `{user_id}_` 前缀清文件 (data_reset), 前缀没了就清不掉。"""
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(400, 400), mime="image/jpeg")

        assert key.startswith(f"{_USER_ID}_")
        assert ua.thumb_key(key).startswith(f"{_USER_ID}_")

    @pytest.mark.parametrize(
        "kwargs, detail",
        [
            ({"blob": b"", "mime": "image/jpeg"}, "空"),
            ({"blob": b"x" * 16, "mime": "image/gif"}, "格式"),
            ({"blob": b"not an image", "mime": "image/jpeg"}, "解析"),
        ],
    )
    def test_unusable_input_is_rejected(self, avatar_dir, kwargs, detail):
        with pytest.raises(HTTPException) as excinfo:
            ua.save_avatar(user_id=_USER_ID, **kwargs)

        assert excinfo.value.status_code == 400
        assert detail in excinfo.value.detail

    def test_oversized_upload_is_rejected(self, avatar_dir):
        with pytest.raises(HTTPException) as excinfo:
            ua.save_avatar(
                user_id=_USER_ID,
                blob=b"x" * (ua.MAX_UPLOAD_BYTES + 1),
                mime="image/jpeg",
            )

        assert excinfo.value.status_code == 400


class TestAvatarPath:
    def test_accepts_a_generated_key(self, avatar_dir):
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(300, 300), mime="image/jpeg")

        assert ua.avatar_path(key).parent == avatar_dir

    @pytest.mark.parametrize(
        "key",
        [
            "../../etc/passwd",
            "x/../y.jpg",
            "..%2Fetc.jpg",
            "x\\y.jpg",
            "..jpg.jpg/../x.jpg",
            "",
            f"{_USER_ID}_{'0' * 32}.png",
            "a" * 200 + ".jpg",
        ],
    )
    def test_rejects_anything_that_could_escape_the_directory(self, key):
        """正则是唯一的防线 —— 这个端点是公开的, 放松一点就是任意文件读。"""
        with pytest.raises(HTTPException) as excinfo:
            ua.avatar_path(key)

        assert excinfo.value.status_code == 404


class TestDeleteAvatar:
    def test_removes_both_variants(self, avatar_dir):
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(300, 300), mime="image/jpeg")

        ua.delete_avatar(key)

        assert not ua.avatar_path(key).exists()
        assert not ua.avatar_path(ua.thumb_key(key)).exists()

    def test_missing_files_are_not_an_error(self, avatar_dir):
        ua.delete_avatar(f"{_USER_ID}_{'a' * 32}.jpg")
        ua.delete_avatar(None)


class TestServeAvatar:
    def test_small_variant_serves_the_thumbnail(self, avatar_dir):
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(600, 600), mime="image/jpeg")

        assert ua.serve_avatar(key, small=True).path == ua.avatar_path(
            ua.thumb_key(key)
        )

    def test_missing_thumbnail_falls_back_to_the_original(self, avatar_dir):
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(600, 600), mime="image/jpeg")
        ua.avatar_path(ua.thumb_key(key)).unlink()

        assert ua.serve_avatar(key, small=True).path == ua.avatar_path(key)

    def test_response_is_cacheable_forever(self, avatar_dir):
        """key 里嵌 uuid, 换头像就是换 URL —— 所以可以 immutable 缓存。"""
        key = ua.save_avatar(user_id=_USER_ID, blob=_jpeg(300, 300), mime="image/jpeg")

        assert "immutable" in ua.serve_avatar(key).headers["cache-control"]

    def test_unknown_key_is_404(self, avatar_dir):
        with pytest.raises(HTTPException) as excinfo:
            ua.serve_avatar(f"{_USER_ID}_{'b' * 32}.jpg")

        assert excinfo.value.status_code == 404


class TestAvatarEndpoint:
    """文件与 DB 行的两步写入 —— 中间失败的那半步是这里唯一有意思的地方。"""

    @staticmethod
    def _upload(client, header, **fields):
        return client.post(
            "/users/me/avatar",
            headers=header("user-1"),
            files={"file": ("a.jpg", _jpeg(600, 600), "image/jpeg")},
            data=fields,
        )

    def test_replacing_an_avatar_deletes_the_previous_files(
        self, api_client, auth_header, avatar_dir, monkeypatch
    ):
        old_key = ua.save_avatar(user_id="user-1", blob=_jpeg(300, 300), mime="image/jpeg")
        self._mock_user(monkeypatch, avatar_key=old_key)

        response = self._upload(api_client, auth_header)

        assert response.status_code == 200
        assert not ua.avatar_path(old_key).exists()
        assert not ua.avatar_path(ua.thumb_key(old_key)).exists()

    def test_failed_db_write_removes_the_orphan_file(
        self, api_client, auth_header, avatar_dir, monkeypatch
    ):
        """行没写成就没人引用得到这个文件 —— 留着就是永久垃圾。"""
        self._mock_user(monkeypatch, avatar_key=None, update_raises=True)

        with pytest.raises(RuntimeError):
            self._upload(api_client, auth_header)

        assert list(avatar_dir.iterdir()) == []

    def test_crop_fields_reach_the_storage_layer(
        self, api_client, auth_header, avatar_dir, monkeypatch
    ):
        self._mock_user(monkeypatch, avatar_key=None)

        seen = self._spy_on_save(monkeypatch)
        self._upload(api_client, auth_header, crop_x=10, crop_y=20, crop_size=200)

        assert seen["crop"] == (10, 20, 200)

    def test_partial_crop_fields_degrade_to_center_crop(
        self, api_client, auth_header, avatar_dir, monkeypatch
    ):
        """旧客户端一个都不带; 只带一半是坏请求, 不该拿它当坐标用。"""
        self._mock_user(monkeypatch, avatar_key=None)

        seen = self._spy_on_save(monkeypatch)
        self._upload(api_client, auth_header, crop_x=10)

        assert seen["crop"] is None

    @staticmethod
    def _spy_on_save(monkeypatch) -> dict:
        seen: dict = {}
        real_save = ua.save_avatar
        monkeypatch.setattr(
            up,
            "save_avatar",
            lambda **kwargs: (seen.update(kwargs), real_save(**kwargs))[1],
        )
        return seen

    @staticmethod
    def _mock_user(monkeypatch, *, avatar_key, update_raises=False):
        row = SimpleNamespace(
            id="user-1", displayName="小明", avatarKey=avatar_key, role="user"
        )

        async def _update(**kwargs):
            if update_raises:
                raise RuntimeError("db down")
            return SimpleNamespace(
                id="user-1",
                displayName="小明",
                avatarKey=kwargs["data"].get("avatarKey", avatar_key),
            )

        # 端点自己只读 user 行, 写在 user_profile 里 —— 两个模块各持一份 db 引用。
        monkeypatch.setattr(
            users_api,
            "db",
            SimpleNamespace(user=SimpleNamespace(find_unique=AsyncMock(return_value=row))),
        )
        monkeypatch.setattr(
            up, "db", SimpleNamespace(user=SimpleNamespace(update=_update))
        )
        monkeypatch.setattr(
            users_api, "resolve_display_identity", AsyncMock(return_value=("小明", "/u/1"))
        )


class TestVisibleTextOrNone:
    """判空必须挡住"看不见但不是空白"的字符。

    生产上真有一个微信昵称是 U+3164 (HANGUL FILLER) 拼的: 它的 Unicode 分类是 Lo
    (字母), `.strip()` 判不出空, 于是通过了所有 `or None`, 在每个界面渲染成空白。
    """

    @pytest.mark.parametrize(
        "value",
        [
            "ㅤ          ㅤ",  # U+3164 + 空格 —— 生产实例
            "",
            "   ",
            "​​",  # ZERO WIDTH SPACE
            "﻿",  # BOM
            "⠀",  # BRAILLE PATTERN BLANK
            None,
            123,
        ],
    )
    def test_invisible_input_is_none(self, value):
        assert up.visible_text_or_none(value) is None

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("干饭！", "干饭！"),
            ("  Eling  ", "Eling"),
            ("L", "L"),
            ("💤", "💤"),
            # 有一个可见字就算有名字, 且返回**原文**而不是清洗过的版本。
            ("a​", "a​"),
        ],
    )
    def test_visible_input_is_returned_trimmed(self, value, expected):
        assert up.visible_text_or_none(value) == expected


class TestResolveDisplayIdentity:
    """优先级: display_name → 微信昵称 → 用户{手机尾号}。"""

    @staticmethod
    def _patch_identities(monkeypatch, *, wechat: dict | None = None, phone: str | None = None):
        identities = []
        if wechat is not None:
            identities.append(
                SimpleNamespace(
                    provider="wechat", rawProfile=wechat, updatedAt="2026-01-01T00:00:00"
                )
            )
        if phone is not None:
            identities.append(
                SimpleNamespace(
                    provider="phone", providerAccountId=phone, updatedAt="2026-01-01T00:00:00"
                )
            )
        find_many = AsyncMock(return_value=identities)
        monkeypatch.setattr(
            up, "db", SimpleNamespace(authidentity=SimpleNamespace(find_many=find_many))
        )
        return find_many

    @pytest.mark.asyncio
    async def test_local_columns_win_over_wechat(self, monkeypatch):
        """微信登录会不断刷新 rawProfile —— 让它赢就等于用户改不动昵称。"""
        self._patch_identities(
            monkeypatch, wechat={"nickname": "微信名", "headimgurl": "https://x/1"}
        )
        user = SimpleNamespace(
            id="u1", displayName="我自己取的", avatarKey=f"{_USER_ID}_{'c' * 32}.jpg"
        )

        name, avatar = await up.resolve_display_identity(user)

        assert name == "我自己取的"
        assert avatar == f"/users/avatar/{_USER_ID}_{'c' * 32}.jpg"

    @pytest.mark.asyncio
    async def test_halves_fall_back_independently(self, monkeypatch):
        """只换了头像的用户应该继续用微信昵称, 而不是掉回 username。"""
        self._patch_identities(
            monkeypatch, wechat={"nickname": "微信名", "headimgurl": "https://x/1"}
        )
        user = SimpleNamespace(
            id="u1", displayName=None, avatarKey=f"{_USER_ID}_{'d' * 32}.jpg"
        )

        name, avatar = await up.resolve_display_identity(user)

        assert name == "微信名"
        assert avatar == f"/users/avatar/{_USER_ID}_{'d' * 32}.jpg"

    @pytest.mark.asyncio
    async def test_wechat_nickname_beats_the_phone_placeholder(self, monkeypatch):
        """双绑用户的关键顺序: 昵称是人起的名字, 尾号只是占位符。

        反过来的话, 一个微信用户绑了手机号之后会从「Eling」变成「用户5678」——
        用户会认为我们把他的名字弄丢了。
        """
        self._patch_identities(
            monkeypatch, wechat={"nickname": "Eling"}, phone="13812345678"
        )
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        name, _ = await up.resolve_display_identity(user)

        assert name == "Eling"

    @pytest.mark.asyncio
    async def test_phone_tail_is_the_last_rung(self, monkeypatch):
        self._patch_identities(monkeypatch, phone="13812345678")
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        name, avatar = await up.resolve_display_identity(user)

        assert name == "用户5678"
        assert avatar is None

    @pytest.mark.asyncio
    async def test_blank_local_name_is_treated_as_unset(self, monkeypatch):
        self._patch_identities(monkeypatch, wechat={"nickname": "微信名"})
        user = SimpleNamespace(id="u1", displayName="   ", avatarKey=None)

        name, _ = await up.resolve_display_identity(user)

        assert name == "微信名"

    @pytest.mark.asyncio
    async def test_invisible_wechat_nickname_falls_through_to_the_phone_tail(
        self, monkeypatch
    ):
        """那个 U+3164 昵称的用户不该到处显示空白。"""
        self._patch_identities(
            monkeypatch, wechat={"nickname": "ㅤ          ㅤ"}, phone="13800001234"
        )
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        name, _ = await up.resolve_display_identity(user)

        assert name == "用户1234"

    @pytest.mark.asyncio
    async def test_no_identity_and_no_local_values(self, monkeypatch):
        """存量密码账号被 migration 回填, 新注册在 register 里预写 —— 真落到这里
        只剩模板系统账号, 客户端按场景兜底词处理。"""
        self._patch_identities(monkeypatch)
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        assert await up.resolve_display_identity(user) == (None, None)

    @pytest.mark.asyncio
    async def test_wechat_avatar_is_upgraded_to_https(self, monkeypatch):
        self._patch_identities(
            monkeypatch,
            wechat={"nickname": "小明", "headimgurl": "http://thirdwx.qlogo.cn/x"},
        )
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        _, avatar = await up.resolve_display_identity(user)

        assert avatar == "https://thirdwx.qlogo.cn/x"

    @pytest.mark.asyncio
    async def test_identity_lookup_is_skipped_when_both_locals_are_set(self, monkeypatch):
        """两边都自设了就不该再查 auth_identities —— 每次登录/取 me 都会走这里。"""
        find_many = self._patch_identities(monkeypatch)
        user = SimpleNamespace(
            id="u1", displayName="名字", avatarKey=f"{_USER_ID}_{'e' * 32}.jpg"
        )

        await up.resolve_display_identity(user)

        find_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_identity_table_is_queried_at_most_once(self, monkeypatch):
        """昵称和头像两个回落共用一次查询, 不是各查一遍。"""
        find_many = self._patch_identities(
            monkeypatch, wechat={"nickname": "微信名"}, phone="13812345678"
        )
        user = SimpleNamespace(id="u1", displayName=None, avatarKey=None)

        await up.resolve_display_identity(user)

        assert find_many.await_count == 1
