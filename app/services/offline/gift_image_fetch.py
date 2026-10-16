"""后台填写的礼物图片链接：下载并转存为收件用户的私有图片。

为什么要转存而不是直接引用外链：
- 运营常贴的是商品页（如 .../products/memo-2402）而不是图片地址，客户端加载必然失败；
  这里会从商品页的 og:image / twitter:image 取主图。
- 电商图床普遍防盗链、链接会过期，转存后图片稳定，也不会向第三方暴露用户的访问。

只有管理员接口会调用；仍按 SSRF 的标准防护：只允许 http(s)、目标必须是公网地址
（每一跳重定向都重新校验）、限制大小与超时。
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from html import unescape
from urllib.parse import urljoin, urlsplit

import httpx

MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_PAGE_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 3
_TIMEOUT_S = 10.0
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_META_IMAGE = re.compile(
    r"""<meta[^>]+(?:property|name)\s*=\s*["'](?:og:image(?::secure_url)?|twitter:image)["'][^>]*>""",
    re.IGNORECASE,
)
_CONTENT_ATTR = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


class GiftImageFetchError(ValueError):
    """链接取不到可用图片；message 直接展示给运营。"""


async def fetch_gift_image(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> tuple[bytes, str]:
    """返回 (图片字节, mime)。商品页会退而取页面主图。"""
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_S,
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT, "Accept": "image/*,text/html;q=0.8,*/*;q=0.5"},
        transport=transport,
        trust_env=False,
    ) as client:
        body, content_type, final_url = await _get(client, url, limit=MAX_IMAGE_BYTES)
        if content_type.startswith("image/"):
            return body, content_type
        if "html" in content_type:
            image_url = _page_image_url(body[:_MAX_PAGE_BYTES].decode("utf-8", "ignore"), final_url)
            if not image_url:
                raise GiftImageFetchError("这是一个网页链接，但没找到商品主图，请复制图片地址或直接上传图片")
            body, content_type, _ = await _get(client, image_url, limit=MAX_IMAGE_BYTES)
            if content_type.startswith("image/"):
                return body, content_type
        raise GiftImageFetchError("链接返回的不是图片，请复制图片地址或直接上传图片")


async def _get(client: httpx.AsyncClient, url: str, *, limit: int) -> tuple[bytes, str, str]:
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        await _ensure_public_http_url(current)
        try:
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise GiftImageFetchError("图片链接跳转异常")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise GiftImageFetchError(f"图片链接打不开（HTTP {response.status_code}）")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise GiftImageFetchError("图片超过 10MB，请压缩后上传")
                    chunks.append(chunk)
                return b"".join(chunks), content_type, str(response.url)
        except httpx.HTTPError as exc:
            raise GiftImageFetchError("图片链接暂时打不开，请稍后重试或直接上传图片") from exc
    raise GiftImageFetchError("图片链接跳转次数过多")


async def _ensure_public_http_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise GiftImageFetchError("图片地址需要是 http(s) 链接")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, parts.port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise GiftImageFetchError("图片链接的域名解析失败") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise GiftImageFetchError("不允许访问内网地址")


def _page_image_url(html: str, base_url: str) -> str | None:
    for tag in _META_IMAGE.findall(html):
        match = _CONTENT_ATTR.search(tag)
        if match:
            return urljoin(base_url, unescape(match.group(1).strip()))
    return None
