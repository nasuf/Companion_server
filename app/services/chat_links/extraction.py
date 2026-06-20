from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Companion/0.1"
)
_SUPPORTED_BARE_HOSTS = {
    "xhslink.com",
    "xiaohongshu.com",
    "douyin.com",
    "v.douyin.com",
    "iesdouyin.com",
    "weibo.com",
    "weibo.cn",
    "t.cn",
    "toutiao.com",
    "snssdk.com",
    "zhihu.com",
    "zhuanlan.zhihu.com",
}
_URL_TRAILING_PUNCT = ",.;:)]}，。；：）】》"
_APP_ACCENTS = {
    "小红书": "#F43F5E",
    "微博": "#FF8A00",
    "今日头条": "#D7262E",
    "抖音": "#111827",
    "知乎": "#1772F6",
}


@dataclass(frozen=True)
class LinkMetadata:
    source_url: str
    final_url: str
    platform: str
    title: str
    description: str = ""
    author: str | None = None
    image_url: str | None = None
    content_text: str = ""
    original_text: str = ""
    summary: str = ""
    status: str = "ready"
    error: str | None = None


def extract_first_url(input_text: str) -> str | None:
    urls = extract_urls(input_text)
    return urls[0] if urls else None


def extract_urls(input_text: str) -> list[str]:
    text = input_text or ""
    candidates: list[tuple[int, str]] = []
    for match in re.finditer(r"""https?://[^\s<>"'，。；：）】》]+""", text):
        candidates.append((match.start(), _clean_shared_url(match.group(0))))
    for match in re.finditer(r"""\bwww\.[^\s<>"'，。；：）】》]+""", text, re.I):
        candidates.append((match.start(), "https://" + _clean_shared_url(match.group(0))))
    bare_pattern = re.compile(
        r"""(?i)\b[a-z0-9-]+(?:\.[a-z0-9-]+)+/[^\s<>"'，。；：）】》]+"""
    )
    for match in bare_pattern.finditer(text):
        cleaned = _clean_shared_url(match.group(0))
        host = cleaned.split("/", 1)[0].lower().removeprefix("www.")
        if _is_supported_bare_host(host):
            candidates.append((match.start(), f"https://{cleaned}"))
    candidates.sort(key=lambda item: item[0])
    urls: list[str] = []
    for _, url in candidates:
        if url not in urls:
            urls.append(url)
    return urls


async def extract_link_metadata(
    *,
    url: str | None,
    shared_text: str | None,
    timeout: float = 12.0,
) -> LinkMetadata:
    source_url = _resolve_source_url(url=url, shared_text=shared_text)
    shared = (shared_text or "").strip()
    fallback_title = _first_meaningful_line(shared) or "未命名链接"
    base = LinkMetadata(
        source_url=source_url,
        final_url=source_url,
        platform=platform_for_url(source_url),
        title=fallback_title,
        description=shared,
        content_text=shared,
        original_text=shared,
        summary=_summary_from_text(shared),
    )
    try:
        headers = {
            "user-agent": _USER_AGENT,
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.6",
        }
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:
            response = await client.get(source_url)
            final_url = str(response.url)
            platform = platform_for_url(final_url)
            if response.status_code >= 400:
                return _with_error(
                    base,
                    final_url=final_url,
                    platform=platform,
                    error=f"页面返回 HTTP {response.status_code}",
                )
            html = response.text[:1_500_000]
    except Exception as exc:
        return _with_error(base, error=f"请求页面失败: {str(exc)[:180]}")

    parsed = _HtmlMetadataParser()
    try:
        parsed.feed(html)
    except Exception:
        logger.debug("[chat-links] html parser failed", exc_info=True)

    platform_data = _platform_json_metadata(html, platform_for_url(str(response.url)))
    title = _first_non_empty(
        platform_data.get("title"),
        parsed.meta.get("og:title"),
        parsed.meta.get("twitter:title"),
        parsed.title,
        base.title,
    )
    description = _first_non_empty(
        platform_data.get("description"),
        parsed.meta.get("og:description"),
        parsed.meta.get("description"),
        parsed.meta.get("twitter:description"),
        base.description,
    ) or ""
    author = _first_non_empty(
        platform_data.get("author"),
        parsed.meta.get("author"),
        parsed.meta.get("article:author"),
    )
    image_url = _first_non_empty(
        platform_data.get("image_url"),
        parsed.meta.get("og:image"),
        parsed.meta.get("twitter:image"),
    )
    image_url = _absolute_http_url(image_url, str(response.url))
    visible_text = _clean_text(platform_data.get("content_text") or parsed.visible_text)
    content_text = visible_text or _clean_text("\n".join([title or "", description, shared]))
    original_text = _clean_text(platform_data.get("original_text") or content_text or shared)
    summary = _summary_from_text(content_text or description or title or shared)
    return LinkMetadata(
        source_url=source_url,
        final_url=str(response.url),
        platform=platform_for_url(str(response.url)),
        title=_clean_text(title or "未命名链接")[:240],
        description=_clean_text(description)[:1000],
        author=_clean_text(author)[:120] if author else None,
        image_url=_clean_text(image_url)[:2000] if image_url else None,
        content_text=content_text[:6000],
        original_text=original_text[:6000],
        summary=summary,
        status="ready",
        error=None,
    )


def platform_for_url(raw_url: str) -> str:
    host = (urlparse(raw_url).hostname or "").lower().removeprefix("www.")
    if host.endswith("xhslink.com") or host.endswith("xiaohongshu.com"):
        return "小红书"
    if host.endswith("weibo.com") or host.endswith("weibo.cn") or host == "t.cn":
        return "微博"
    if host.endswith("toutiao.com") or host.endswith("snssdk.com"):
        return "今日头条"
    if host.endswith("douyin.com") or host.endswith("iesdouyin.com"):
        return "抖音"
    if host.endswith("zhihu.com"):
        return "知乎"
    return "链接"


def accent_for_platform(platform: str) -> str:
    return _APP_ACCENTS.get(platform, "#177DDC")


def _resolve_source_url(*, url: str | None, shared_text: str | None) -> str:
    candidate = (url or "").strip() or extract_first_url(shared_text or "") or ""
    if not candidate:
        raise ValueError("request must include a supported URL")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid URL")
    return candidate


def _clean_shared_url(value: str) -> str:
    return value.strip().strip("<>\"'“”‘’").rstrip(_URL_TRAILING_PUNCT)


def _is_supported_bare_host(host: str) -> bool:
    host = host.lower().removeprefix("www.")
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in _SUPPORTED_BARE_HOSTS)


def _first_meaningful_line(text: str) -> str | None:
    for line in text.splitlines():
        cleaned = _clean_text(line)
        if cleaned and not extract_first_url(cleaned) == cleaned:
            return cleaned[:120]
    return None


def _summary_from_text(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    return cleaned[:360]


def _with_error(
    base: LinkMetadata,
    *,
    final_url: str | None = None,
    platform: str | None = None,
    error: str,
) -> LinkMetadata:
    content = base.content_text or base.description or base.title
    return LinkMetadata(
        source_url=base.source_url,
        final_url=final_url or base.final_url,
        platform=platform or base.platform,
        title=base.title,
        description=base.description,
        author=base.author,
        image_url=base.image_url,
        content_text=content,
        original_text=base.original_text or content,
        summary=base.summary or _summary_from_text(content),
        status="partial",
        error=error,
    )


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _clean_text(value: str | None) -> str:
    text = unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _absolute_http_url(raw_url: str | None, base_url: str) -> str | None:
    cleaned = _clean_text(raw_url)
    if not cleaned:
        return None
    resolved = urljoin(base_url, cleaned)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return resolved


class _HtmlMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.visible_chunks: list[str] = []
        self._tag_stack: list[str] = []
        self._in_title = False

    @property
    def visible_text(self) -> str:
        return _clean_text(" ".join(self.visible_chunks))[:6000]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = attrs_dict.get("property") or attrs_dict.get("name")
            content = attrs_dict.get("content")
            if key and content:
                self.meta[key.lower()] = content

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self._in_title:
            self.title += data
            return
        if any(tag in {"script", "style", "noscript", "svg"} for tag in self._tag_stack):
            return
        cleaned = _clean_text(data)
        if len(cleaned) >= 2:
            self.visible_chunks.append(cleaned)


def _platform_json_metadata(html: str, platform: str) -> dict[str, str]:
    values = _json_ld_values(html)
    values.extend(_script_json_values(html, ("RENDER_DATA", "js-initialData")))
    values.extend(_json_after_markers(
        html,
        (
            "window.__INITIAL_STATE__=",
            "window.__INIT_PROPS__=",
            "window._ROUTER_DATA=",
            "window.__data=",
            "$render_data =",
            "render_data =",
        ),
    ))
    fields = {
        "title": ("title", "displayTitle", "questionTitle", "headline", "text_raw"),
        "description": ("desc", "description", "excerpt", "content", "text", "articleBody"),
        "author": ("nickname", "nickName", "screen_name", "authorName", "name"),
        "image_url": ("urlDefault", "urlPre", "cover", "poster", "thumbnailUrl", "image", "pic"),
    }
    data: dict[str, str] = {}
    for output_key, keys in fields.items():
        found = _find_first_json_string(values, keys)
        if found:
            data[output_key] = found
    if "description" in data:
        data["content_text"] = "\n\n".join(
            part for part in (data.get("title"), data["description"]) if part
        )
    if platform == "小红书" and "description" in data:
        data["original_text"] = data["description"]
    return data


def _json_ld_values(html: str) -> list[Any]:
    values: list[Any] = []
    pattern = re.compile(
        r"""<script[^>]+type=["']application/ld\+json["'][^>]*>(.*?)</script>""",
        re.I | re.S,
    )
    for match in pattern.finditer(html):
        value = _loads_jsonish(match.group(1))
        if value is not None:
            values.append(value)
    return values


def _script_json_values(html: str, ids: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for script_id in ids:
        pattern = re.compile(
            rf"""<script[^>]+id=["']{re.escape(script_id)}["'][^>]*>(.*?)</script>""",
            re.I | re.S,
        )
        for match in pattern.finditer(html):
            value = _loads_jsonish(unescape(match.group(1)))
            if value is not None:
                values.append(value)
    return values


def _json_after_markers(html: str, markers: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for marker in markers:
        start = html.find(marker)
        if start < 0:
            continue
        start += len(marker)
        raw = _balanced_json_slice(html[start : start + 500_000])
        value = _loads_jsonish(raw)
        if value is not None:
            values.append(value)
    return values


def _balanced_json_slice(text: str) -> str:
    text = text.lstrip()
    if not text:
        return ""
    opening = text[0]
    closing = "}" if opening == "{" else "]" if opening == "[" else ""
    if not closing:
        return text.split(";", 1)[0]
    depth = 0
    in_string = False
    escape = False
    for index, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return text.split(";", 1)[0]


def _loads_jsonish(raw: str | None) -> Any:
    if not raw:
        return None
    text = raw.strip().rstrip(";")
    try:
        return json.loads(text)
    except Exception:
        try:
            return json.loads(unescape(text))
        except Exception:
            return None


def _find_first_json_string(values: list[Any], keys: tuple[str, ...]) -> str | None:
    seen: set[int] = set()

    def walk(value: Any) -> str | None:
        marker = id(value)
        if marker in seen:
            return None
        seen.add(marker)
        if isinstance(value, dict):
            for key in keys:
                raw = value.get(key)
                if isinstance(raw, str) and _clean_text(raw):
                    return _clean_text(raw)
                if isinstance(raw, list):
                    nested = walk(raw)
                    if nested:
                        return nested
                if isinstance(raw, dict):
                    nested = walk(raw)
                    if nested:
                        return nested
            for raw in value.values():
                nested = walk(raw)
                if nested:
                    return nested
        elif isinstance(value, list):
            for item in value:
                nested = walk(item)
                if nested:
                    return nested
        return None

    for value in values:
        found = walk(value)
        if found:
            return found
    return None
