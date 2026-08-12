"""用户展示身份 (昵称 + 头像) 的读写口。

读取优先级 —— 整条链只在 `resolve_display_identity` 里存在一份:

    users.display_name  →  微信昵称  →  用户{手机尾号4位}

顺序不是随意的: 微信昵称是**人自己起的名字**, `用户5678` 是**机器占位符**, 占位符
必须垫最底。双绑流程 (bind_wechat_to_user / bind_phone_to_user) 是通的, 一个微信
用户绑了手机号之后如果占位符能盖过昵称, 他会觉得我们把他的名字弄丢了。

**只有密码账号在建号时预写 display_name** (= username, 见 auth.register)。微信和
手机号都不预写, 保持"算出来"而不是"复制一份":

* 微信昵称每次登录都会被 `_merged_raw_profile` 刷新。建号时复制一份进 display_name
  会冻结跟随 —— 用户改了微信昵称, App 里永远停在建号那天。而且事后无法区分
  "这是他自己设的" 还是 "这是一份过期拷贝", 要区分就得再加一个 is_custom 标志位,
  比懒回落多出更多状态。用户**主动编辑**一次就是他明确的 opt-out。
* 手机尾号同理有活来源 (改绑手机号应当跟着变), 而且预写之后这人再绑微信,
  `用户5678` 会压住真昵称。

密码账号是唯一没有活来源的类型, 所以那一份复制不会过期 —— 也正因如此, 这条链对
任何真实用户都不会返回 None, 客户端不需要自己拼来源, 只需要在为空时选一个场景
兜底词。
"""

from __future__ import annotations

import unicodedata

from fastapi import HTTPException

from app.db import db
from app.models.user import MAX_DISPLAY_NAME_LENGTH
from app.services.user_avatars import build_avatar_url, delete_avatar, save_avatar

_WECHAT_PROVIDER = "wechat"
_PHONE_PROVIDER = "phone"
# 看不见但不是空白的字符。生产上真有一个微信昵称是 "ㅤ          ㅤ" (U+3164
# HANGUL FILLER + 空格): 它通不过 `.strip()` 的判空 (U+3164 的 Unicode 分类是
# Lo, 字母), 于是一路通过所有 `or None` 检查, 在每个界面上渲染成一片空白。
_INVISIBLE_CHARS = frozenset("ᅟᅠㅤ⠀​‌‍⁠﻿᠎")


def visible_text_or_none(value: object) -> str | None:
    """Return the trimmed text, or None when it has no visible glyph.

    Whitespace, control/format codepoints and the invisible-letter set above all
    count as "not visible". The returned value is the ORIGINAL trimmed text, not
    a scrubbed one — a name like "干饭！" must survive untouched.
    """
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    for char in trimmed:
        if char.isspace() or char in _INVISIBLE_CHARS:
            continue
        if unicodedata.category(char) in {"Cc", "Cf"}:
            continue
        return trimmed
    return None


async def resolve_display_identity(user) -> tuple[str | None, str | None]:
    """Return ``(display_name, avatar_url)`` for a user row.

    昵称和头像各自独立回落: 只换了头像没改昵称的用户继续显示微信昵称。

    身份表最多查一次 —— 两个字段都已自设时完全不查。
    """
    local_name = visible_text_or_none(getattr(user, "displayName", None))
    local_avatar = build_avatar_url(getattr(user, "avatarKey", None))
    if local_name and local_avatar:
        return local_name, local_avatar

    identities = await db.authidentity.find_many(where={"userId": user.id})
    wechat_name, wechat_avatar = _wechat_identity_profile(identities)
    display_name = local_name or wechat_name
    if not display_name:
        # 最后一级: 手机号尾号。总比把 `wx_89b939bc004` 这种登录标识摆给用户看好。
        phone = _phone_number(identities)
        if phone and len(phone) >= 4:
            display_name = f"用户{phone[-4:]}"
    return display_name, local_avatar or wechat_avatar


async def apply_profile_update(
    user,
    *,
    display_name: str | None = None,
    avatar: tuple[bytes, str | None, tuple[int, int, int] | None] | None = None,
):
    """Write nickname / avatar and return the updated user row.

    头像是"先落盘再写库"的两步操作, 两个写入端点 (App 的 /users/me/avatar 与小程序
    的 /auth/wechat/profile) 都要处理中间失败: 库没写成就没人引用得到那个文件,
    留着就是永久垃圾; 而旧文件要等新行写成之后再删, 反过来做中途失败会让用户连旧
    头像都没了。这两条顺序在两处各写一遍迟早会漏掉一处。

    没有任何字段要改时原样返回传入的 ``user`` —— 空更新不该产生一次写库。
    """
    update: dict[str, str] = {}
    cleaned_name = visible_text_or_none(display_name)
    if cleaned_name:
        update["displayName"] = cleaned_name[:MAX_DISPLAY_NAME_LENGTH]
    if avatar is not None:
        blob, mime, crop = avatar
        update["avatarKey"] = save_avatar(
            user_id=user.id, blob=blob, mime=mime, crop=crop
        )
    if not update:
        return user

    try:
        updated = await db.user.update(where={"id": user.id}, data=update)
    except Exception:
        delete_avatar(update.get("avatarKey"))
        raise
    if updated is None:
        delete_avatar(update.get("avatarKey"))
        raise HTTPException(status_code=404, detail="User not found")

    previous_key = getattr(user, "avatarKey", None)
    if "avatarKey" in update and previous_key != update["avatarKey"]:
        delete_avatar(previous_key)
    return updated


def _wechat_identity_profile(identities) -> tuple[str | None, str | None]:
    """微信身份里的 (nickname, headimgurl)。多条身份时取最近更新的那条。"""
    candidates = [
        identity
        for identity in identities
        if getattr(identity, "provider", None) == _WECHAT_PROVIDER
    ]
    if not candidates:
        return None, None
    # 登录流程对同一 user 只维护一条微信身份 (find_first + update, 没有解绑路径),
    # 所以这里几乎总是只有一条。真出现多条时取最后更新的那条: 按 str() 比较而不是
    # 直接比 datetime —— 同一列出来的 ISO 串字典序即时序, 而缺时间戳的行会得到 ""
    # 从而永远不胜出, 不会让 None 参与比较。
    identity = max(candidates, key=lambda item: str(getattr(item, "updatedAt", "") or ""))
    profile = getattr(identity, "rawProfile", None)
    if not isinstance(profile, dict):
        return None, None

    nickname = visible_text_or_none(profile.get("nickname"))
    avatar = visible_text_or_none(profile.get("headimgurl"))
    if avatar and avatar.startswith("http://"):
        # 微信头像 CDN (qlogo.cn) 常回 http://, 在 https 页面里会被浏览器按
        # mixed-content 拦截导致头像不显示; CDN 本身支持 https, 读取侧统一升级.
        avatar = "https://" + avatar[len("http://"):]
    return nickname, avatar


def _phone_number(identities) -> str | None:
    for identity in identities:
        if getattr(identity, "provider", None) == _PHONE_PROVIDER:
            return visible_text_or_none(getattr(identity, "providerAccountId", None))
    return None
