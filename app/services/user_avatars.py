"""用户自设头像的存储与读取。

跟 chat_media / capsule_media 的区别是 **不鉴权**, 走的是 agent 头像那条路
(`/agents/avatar/{key}.png`, public + immutable)。原因: 头像会被十几处纯展示
widget 渲染 (个人页、棋类对局双方头像…), 那些地方拿的是裸 image URL 没有机会
挂 Bearer header。小程序为了显示鉴权版头像已经不得不写了一个"先带 token 下载成
本地文件再 <image>"的绕行 (utils/api.js:resolveUserAvatar) —— 每多一个端就要
重写一遍。头像本身是低敏感数据 (今天在用的微信 CDN 直链本来就是全公开的), key
里嵌 uuid4 不可枚举, 换取的是三端零特殊处理。

文件落 USER_AVATAR_DIR, 与聊天媒体分目录: 生命周期不同 (头像随用户在, 聊天图片
有 orphan 清理), 混在一起会让任何一侧的清理策略误伤另一侧。
"""

from __future__ import annotations

import io
import logging
import os
import re
import uuid
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_AVATAR_DIR = Path(os.getenv("USER_AVATAR_DIR", "var/user_avatar"))
_AVATAR_PUBLIC_PREFIX = (
    os.getenv("USER_AVATAR_PUBLIC_PREFIX", "/users/avatar").strip().rstrip("/")
    or "/users/avatar"
)
# 客户端上传前已经裁成正方形并压过一轮; 这里的上限只用来挡异常/恶意载荷, 正常
# 路径远远碰不到。
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# 头像的最大展示尺寸是 64-96 逻辑像素, 3x 屏也就 288。存 512 已经留足余量, 再大
# 只是白白占带宽 —— 而头像是每屏都在渲染的东西。
_AVATAR_EDGE = 512
_AVATAR_QUALITY = 85
# 列表 / 游戏对局里的小头像 (24-40 逻辑像素) 用这个变体, 省掉解码全尺寸位图。
_THUMB_EDGE = 128
_THUMB_QUALITY = 80
_THUMB_SUFFIX = "_s.jpg"
_ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
# key 形如 `{user_id}_{uuid4hex}.jpg` (缩略图多一个 `_s`)。这里刻意不去匹配那个
# 结构而只限定字符集: 结构式正则会把"用户 id 是 uuid"焊进读取路径, 换一种 id
# 格式就是全站头像 404。防目录穿越只需要挡住 `/` `\` 和多余的 `.`, 剩下的由文件
# 存不存在决定。
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}\.jpg$")
_IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


def build_avatar_url(key: str | None) -> str | None:
    trimmed = (key or "").strip()
    if not trimmed:
        return None
    return f"{_AVATAR_PUBLIC_PREFIX}/{trimmed}"


def avatar_path(key: str) -> Path:
    """Resolve a storage key to a path, rejecting anything not shaped like a key.

    The regex is the whole defense against traversal: it admits neither `/` nor
    `.` beyond the extension.
    """
    if not _KEY_RE.fullmatch(key.strip()):
        raise HTTPException(status_code=404, detail="Avatar not found")
    return _AVATAR_DIR / key.strip()


def thumb_key(key: str) -> str:
    return f"{Path(key).stem}{_THUMB_SUFFIX}"


def save_avatar(
    *,
    user_id: str,
    blob: bytes,
    mime: str | None,
    crop: tuple[int, int, int] | None = None,
) -> str:
    """Normalize an uploaded avatar to a square JPEG and store it + a thumbnail.

    ``crop`` is ``(x, y, size)`` in **source pixels after EXIF orientation** —
    the square the user framed in the client's circular cropper. The client
    sends the picked JPEG plus this rectangle rather than a pre-rasterized
    square: Flutter can only encode PNG, which would put ~400KB on the wire for
    an image we are about to re-encode anyway.

    Out-of-range values are clamped rather than rejected: a request that got the
    rectangle slightly wrong should still produce a reasonable avatar, and the
    clamp is what makes the parameter safe to accept from a client at all.

    Returns the storage key. Raises HTTPException on unusable input — an avatar
    we cannot decode is a client bug worth surfacing, unlike chat images where
    the historical behaviour is to store the raw bytes untouched.
    """
    if not blob:
        raise HTTPException(status_code=400, detail="头像内容为空")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="头像需要小于 10MB")
    normalized_mime = (mime or "image/jpeg").strip().lower()
    if normalized_mime not in _ALLOWED_MIMES:
        raise HTTPException(status_code=400, detail="不支持该图片格式")

    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="图片无法解析") from exc
    # EXIF 方向必须在裁剪之前应用: 客户端量出来的坐标是它显示的那张图上的,
    # 而 Flutter 的解码器同样会应用 EXIF。顺序反了 iOS 竖拍照片会裁错位置。
    oriented = (ImageOps.exif_transpose(image) or image).convert("RGB")
    if crop is not None:
        oriented = oriented.crop(_clamp_crop_box(oriented.size, crop))
    # 客户端跳过裁剪 (旧版本) 时 fit 会 center-crop 兜底; 已经是正方形时它只是
    # 一次缩放。
    square = ImageOps.fit(
        oriented,
        (_AVATAR_EDGE, _AVATAR_EDGE),
        method=Image.Resampling.LANCZOS,
    )

    _AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{user_id}_{uuid.uuid4().hex}.jpg"
    _encode_jpeg(square, _AVATAR_QUALITY, avatar_path(key))
    thumb = square.resize((_THUMB_EDGE, _THUMB_EDGE), Image.Resampling.LANCZOS)
    _encode_jpeg(thumb, _THUMB_QUALITY, avatar_path(thumb_key(key)))
    return key


def delete_avatar(key: str | None) -> None:
    """Best-effort removal of an avatar and its thumbnail.

    Called after a successful replace: failing to delete the old file must never
    fail the request the user actually asked for (the new avatar is already
    stored and recorded).
    """
    if not key:
        return
    for candidate in (key, thumb_key(key)):
        try:
            path = avatar_path(candidate)
        except HTTPException:
            return
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            logger.warning("[user-avatar] delete failed key=%s", candidate, exc_info=True)


def serve_avatar(key: str, *, small: bool = False) -> FileResponse:
    path = avatar_path(thumb_key(key) if small else key)
    if not path.is_file():
        # 缩略图是后加的变体, 缺失时回落原图而不是 404。
        path = avatar_path(key)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Avatar not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": _IMMUTABLE_CACHE_CONTROL},
    )


def _clamp_crop_box(
    size: tuple[int, int],
    crop: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    """Clamp a client-supplied ``(x, y, size)`` square into the image bounds."""
    width, height = size
    x, y, edge = crop
    # 先把边长收进图内, 再把原点收进 [0, 图长-边长] —— 反过来做的话边长仍可能
    # 越界。
    edge = max(1, min(edge, width, height))
    x = max(0, min(x, width - edge))
    y = max(0, min(y, height - edge))
    return x, y, x + edge, y + edge


def _encode_jpeg(image: Image.Image, quality: int, path: Path) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    path.write_bytes(buffer.getvalue())
