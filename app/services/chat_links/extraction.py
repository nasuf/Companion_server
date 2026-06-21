from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
import logging
import re
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlparse

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
    "bilibili.com",
    "b23.tv",
}
_URL_TRAILING_PUNCT = ",.;:)]}，。；：）】》"
_APP_ACCENTS = {
    "小红书": "#F43F5E",
    "微博": "#FF8A00",
    "今日头条": "#D7262E",
    "抖音": "#111827",
    "知乎": "#1772F6",
    "B站": "#00A1D6",
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
    if base.platform == "B站":
        bilibili_metadata = await _fetch_bilibili_video_metadata(
            source_url=source_url,
            shared_text=shared,
            timeout=timeout,
        )
        if bilibili_metadata:
            return bilibili_metadata
    try:
        headers = {
            "user-agent": _USER_AGENT,
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.6",
        }
        api_platform_data: dict[str, str] = {}
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        ) as client:
            response = await client.get(source_url)
            final_url = str(response.url)
            platform = platform_for_url(final_url)
            if response.status_code >= 400:
                if platform_for_url(source_url) == "微博" or platform == "微博":
                    api_platform_data = await _fetch_weibo_status_metadata(
                        client,
                        source_url,
                        final_url,
                    )
                    if api_platform_data:
                        return _metadata_from_platform_data(
                            base=base,
                            platform_data=api_platform_data,
                            final_url=_safe_final_url(source_url, final_url),
                            platform="微博",
                        )
                return _with_error(
                    base,
                    final_url=final_url,
                    platform=platform,
                    error=f"页面返回 HTTP {response.status_code}",
                )
            html = response.text[:1_500_000]
            if platform_for_url(source_url) == "微博" or platform == "微博":
                api_platform_data = await _fetch_weibo_status_metadata(
                    client,
                    source_url,
                    final_url,
                )
            if platform_for_url(source_url) == "今日头条" or platform == "今日头条":
                api_platform_data.update(
                    await _fetch_toutiao_article_metadata(client, source_url, final_url)
                )
            if _is_weibo_visitor_page(html, source_url=source_url, final_url=final_url):
                final_url = _safe_final_url(source_url, final_url)
                if api_platform_data:
                    return _metadata_from_platform_data(
                        base=base,
                        platform_data=api_platform_data,
                        final_url=final_url,
                        platform="微博",
                    )
                return _with_error(
                    base,
                    final_url=final_url,
                    platform="微博",
                    error="微博访客系统拦截，未能读取正文",
                )
    except Exception as exc:
        return _with_error(base, error=f"请求页面失败: {str(exc)[:180]}")

    parsed = _HtmlMetadataParser()
    try:
        parsed.feed(html)
    except Exception:
        logger.debug("[chat-links] html parser failed", exc_info=True)

    final_url = _safe_final_url(source_url, str(response.url))
    platform = platform_for_url(final_url)
    platform_data = _platform_json_metadata(html, platform)
    if api_platform_data:
        platform_data.update(api_platform_data)
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
    raw_body = _first_non_empty(
        platform_data.get("original_text"),
        platform_data.get("description"),
        description,
        platform_data.get("content_text"),
        parsed.visible_text,
        shared,
    ) or ""
    body_text = _normalize_post_body_text(raw_body, platform=platform, author=author)
    description = body_text or _normalize_post_body_text(
        description,
        platform=platform,
        author=author,
    )
    content_text = body_text or _clean_text("\n".join([title or "", description, shared]))
    original_text = _clean_text(platform_data.get("original_text") or raw_body or shared)
    summary = _summary_from_text(content_text or description or title or shared, platform=platform, author=author)
    clean_title = _clean_text(title or "未命名链接")[:240]
    clean_author = _clean_link_author(author, title=clean_title)
    return LinkMetadata(
        source_url=source_url,
        final_url=final_url,
        platform=platform,
        title=clean_title,
        description=_clean_text(description)[:1000],
        author=clean_author,
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
    if host.endswith("bilibili.com") or host == "b23.tv":
        return "B站"
    return "链接"


def accent_for_platform(platform: str) -> str:
    return _APP_ACCENTS.get(platform, "#177DDC")


def app_url_for_link(*, platform: str, source_url: str, final_url: str) -> str | None:
    raw_url = final_url or source_url
    if platform == "今日头条":
        article_id = _first_regex_group(raw_url, r"/article/(\d+)")
        if article_id:
            return f"snssdk141://detail?groupid={article_id}"
    if platform == "微博":
        status_id = _weibo_status_id(raw_url) or _weibo_status_id(source_url)
        if status_id:
            return f"sinaweibo://detail?mblogid={status_id}"
    if platform == "抖音":
        video_id = _first_regex_group(raw_url, r"/video/(\d+)")
        if video_id:
            return f"snssdk1128://aweme/detail/{video_id}"
    if platform == "知乎":
        answer_id = _first_regex_group(raw_url, r"/answer/(\d+)")
        if answer_id:
            return f"zhihu://answers/{answer_id}"
        question_id = _first_regex_group(raw_url, r"/question/(\d+)")
        if question_id:
            return f"zhihu://questions/{question_id}"
    if platform == "小红书":
        note_id = (
            _first_regex_group(raw_url, r"/(?:explore|discovery/item)/([0-9a-fA-F]+)")
            or _first_regex_group(source_url, r"/(?:explore|discovery/item)/([0-9a-fA-F]+)")
        )
        if note_id:
            return f"xhsdiscover://item/{note_id}"
    if platform == "B站":
        bvid = _bilibili_bvid(raw_url) or _bilibili_bvid(source_url)
        if bvid:
            return f"bilibili://video/{bvid}"
    return None


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


def _summary_from_text(
    text: str,
    *,
    platform: str = "",
    author: str | None = None,
) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    normalized = _normalize_post_body_text(cleaned, platform=platform, author=author)
    return (normalized or cleaned)[:360]


def _metadata_from_platform_data(
    *,
    base: LinkMetadata,
    platform_data: dict[str, str],
    final_url: str,
    platform: str,
) -> LinkMetadata:
    title = _first_non_empty(platform_data.get("title"), base.title) or "未命名链接"
    description = _first_non_empty(platform_data.get("description"), base.description) or ""
    content_text = _normalize_post_body_text(
        platform_data.get("content_text")
        or "\n\n".join(part for part in (title, description) if part)
        or base.content_text,
        platform=platform,
        author=platform_data.get("author") or base.author,
    )
    description = _normalize_post_body_text(
        description,
        platform=platform,
        author=platform_data.get("author") or base.author,
    )
    original_text = _clean_text(platform_data.get("original_text") or content_text)
    return LinkMetadata(
        source_url=base.source_url,
        final_url=final_url,
        platform=platform,
        title=_clean_text(title)[:240],
        description=_clean_text(description)[:1000],
        author=_clean_text(platform_data.get("author"))[:120]
        if platform_data.get("author")
        else base.author,
        image_url=_clean_text(platform_data.get("image_url"))[:2000]
        if platform_data.get("image_url")
        else base.image_url,
        content_text=content_text[:6000],
        original_text=original_text[:6000],
        summary=_summary_from_text(
            content_text or description or title,
            platform=platform,
            author=platform_data.get("author") or base.author,
        ),
        status="ready",
        error=None,
    )


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


def _safe_final_url(source_url: str, final_url: str) -> str:
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if host in {"passport.weibo.com", "visitor.passport.weibo.cn"}:
        target = parse_qs(parsed.query).get("url", [""])[0]
        if target:
            return target
        return source_url
    return final_url


def _is_weibo_visitor_page(
    html: str | None,
    *,
    source_url: str = "",
    final_url: str = "",
) -> bool:
    if platform_for_url(source_url) != "微博" and platform_for_url(final_url) != "微博":
        parsed = urlparse(final_url)
        host = (parsed.hostname or "").lower()
        if host not in {"passport.weibo.com", "visitor.passport.weibo.cn"}:
            return False
    text = (html or "")[:20_000]
    return (
        "Sina Visitor System" in text
        or "visitor/visitor" in final_url
        or "visitor.passport.weibo.cn" in final_url
    )


def _first_regex_group(raw: str, pattern: str) -> str | None:
    match = re.search(pattern, raw)
    return match.group(1) if match else None


def _bilibili_bvid(raw_url: str) -> str | None:
    return _first_regex_group(raw_url, r"/video/(BV[0-9A-Za-z]+)") or _first_regex_group(
        raw_url,
        r"\b(BV[0-9A-Za-z]{8,})\b",
    )


def _weibo_status_id(raw_url: str) -> str | None:
    parsed = urlparse(raw_url)
    path_matches = [
        segment
        for segment in parsed.path.split("/")
        if re.fullmatch(r"\d{10,}", segment)
    ]
    if path_matches:
        return path_matches[-1]
    query = parse_qs(parsed.query)
    for key in ("id", "mid", "mblogid"):
        raw = query.get(key, [""])[0]
        if re.fullmatch(r"\d{10,}", raw):
            return raw
    matches = re.findall(r"\b\d{10,}\b", raw_url)
    return matches[-1] if matches else None


async def _fetch_weibo_status_metadata(
    client: httpx.AsyncClient,
    source_url: str,
    final_url: str,
) -> dict[str, str]:
    status_id = _weibo_status_id(source_url) or _weibo_status_id(final_url)
    if not status_id:
        return {}
    cookie = await _weibo_visitor_cookie(client)
    if not cookie:
        return {}
    try:
        response = await client.get(
            f"https://weibo.com/ajax/statuses/show?id={quote(status_id)}",
            headers={
                "user-agent": "Mozilla/5.0",
                "referer": "https://weibo.com/",
                "accept": "application/json",
                "cookie": cookie,
            },
        )
        response.raise_for_status()
        value = response.json()
    except Exception:
        logger.debug("[chat-links] weibo ajax metadata failed", exc_info=True)
        return {}
    return _weibo_metadata_from_api_value(value)


async def _fetch_toutiao_article_metadata(
    client: httpx.AsyncClient,
    source_url: str,
    final_url: str,
) -> dict[str, str]:
    article_id = (
        _first_regex_group(source_url, r"/article/(\d+)")
        or _first_regex_group(final_url, r"/article/(\d+)")
        or _first_regex_group(source_url, r"/i(\d+)")
        or _first_regex_group(final_url, r"/i(\d+)")
    )
    if not article_id:
        return {}
    urls = [
        f"https://m.toutiao.com/i{article_id}/info/",
        f"https://www.toutiao.com/api/pc/detail/?group_id={article_id}",
    ]
    for api_url in urls:
        try:
            response = await client.get(
                api_url,
                headers={
                    "user-agent": _USER_AGENT,
                    "referer": "https://www.toutiao.com/",
                    "accept": "application/json",
                },
            )
            response.raise_for_status()
            value = response.json()
        except Exception:
            logger.debug("[chat-links] toutiao article metadata failed", exc_info=True)
            continue
        data = value.get("data") if isinstance(value, dict) else None
        if not isinstance(data, dict):
            continue
        metadata = _toutiao_metadata_from_api_value(data)
        if metadata:
            return metadata
    return {}


async def _fetch_bilibili_video_metadata(
    *,
    source_url: str,
    shared_text: str,
    timeout: float,
) -> LinkMetadata | None:
    bvid = _bilibili_bvid(source_url) or _bilibili_bvid(shared_text)
    final_url = source_url
    if not bvid:
        final_url = await _resolve_bilibili_final_url(source_url, timeout=timeout)
        bvid = _bilibili_bvid(final_url)
    if not bvid:
        return None
    api_data = await _fetch_bilibili_api_metadata(bvid=bvid, timeout=timeout)
    if not api_data:
        title = _bilibili_title_from_shared_text(shared_text) or f"B站视频 {bvid}"
        canonical_url = final_url if _bilibili_bvid(final_url) else f"https://www.bilibili.com/video/{bvid}"
        return LinkMetadata(
            source_url=source_url,
            final_url=canonical_url,
            platform="B站",
            title=title[:240],
            description=title,
            content_text=title,
            original_text=_clean_text(shared_text),
            summary=_summary_from_text(title, platform="B站"),
            status="ready",
            error=None,
        )
    final_url = final_url if _bilibili_bvid(final_url) else f"https://www.bilibili.com/video/{bvid}"
    return _metadata_from_platform_data(
        base=LinkMetadata(
            source_url=source_url,
            final_url=source_url,
            platform="B站",
            title=_first_meaningful_line(shared_text) or api_data.get("title") or "B站视频",
            description=shared_text,
            content_text=shared_text,
            original_text=shared_text,
            summary=_summary_from_text(shared_text),
        ),
        platform_data=api_data,
        final_url=final_url,
        platform="B站",
    )


async def _resolve_bilibili_final_url(source_url: str, *, timeout: float) -> str:
    headers = {
        "user-agent": _USER_AGENT,
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=min(max(timeout, 3.0), 12.0),
            headers=headers,
            trust_env=False,
        ) as client:
            for method in ("HEAD", "GET"):
                response = await client.request(method, source_url)
                location = response.headers.get("location")
                if location:
                    return urljoin(source_url, location)
                if _bilibili_bvid(str(response.url)):
                    return str(response.url)
    except Exception:
        logger.debug("[chat-links] bilibili short url resolve failed", exc_info=True)
    return source_url


async def _fetch_bilibili_api_metadata(*, bvid: str, timeout: float) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(
            timeout=min(max(timeout, 3.0), 12.0),
            headers={
                "user-agent": _USER_AGENT,
                "referer": "https://www.bilibili.com/",
                "accept": "application/json",
            },
            trust_env=False,
        ) as client:
            response = await client.get(
                "https://api.bilibili.com/x/web-interface/view",
                params={"bvid": bvid},
            )
            response.raise_for_status()
            value = response.json()
    except Exception:
        logger.debug("[chat-links] bilibili api metadata failed", exc_info=True)
        return {}
    data = value.get("data") if isinstance(value, dict) else None
    if not isinstance(data, dict) or value.get("code") != 0:
        return {}
    return _bilibili_metadata_from_api_value(data)


def _bilibili_metadata_from_api_value(value: dict[str, Any]) -> dict[str, str]:
    title = _json_string(value.get("title")) or ""
    description = _bilibili_description(value)
    owner = value.get("owner")
    author = ""
    if isinstance(owner, dict):
        author = _json_string(owner.get("name")) or ""
    image_url = _json_string(value.get("pic")) or ""
    if image_url.startswith("http://"):
        image_url = "https://" + image_url.removeprefix("http://")
    data = {
        "title": title,
        "description": description,
        "author": author,
        "image_url": image_url,
        "content_text": description or title,
        "original_text": description or title,
    }
    return {key: val for key, val in data.items() if val}


def _bilibili_title_from_shared_text(shared_text: str | None) -> str:
    text = _clean_text(shared_text)
    if not text:
        return ""
    text = re.sub(r"https?://\S+", " ", text).strip()
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1].strip()
    text = re.sub(r"[-－—]\s*(?:哔哩哔哩|bilibili|B站)\s*$", "", text).strip()
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1].strip()
    return text


def _bilibili_description(value: dict[str, Any]) -> str:
    desc_v2 = value.get("desc_v2")
    if isinstance(desc_v2, list):
        parts = [
            _json_string(item.get("raw_text"))
            for item in desc_v2
            if isinstance(item, dict)
        ]
        text = _clean_text("\n".join(part for part in parts if part))
        if text:
            return text
    return _json_string(value.get("desc")) or ""


def _toutiao_metadata_from_api_value(value: dict[str, Any]) -> dict[str, str]:
    raw_content = value.get("content")
    content = raw_content if isinstance(raw_content, str) else ""
    title = _first_non_empty(
        _json_string(value.get("title")),
        _first_img_alt(content),
    )
    description = _clean_text(content)
    author = _first_non_empty(
        _json_string(value.get("source")),
        _json_string(value.get("detail_source")),
        _json_string(value.get("media_name")),
    )
    media_user = value.get("media_user")
    if not author and isinstance(media_user, dict):
        author = _first_non_empty(
            _json_string(media_user.get("screen_name")),
            _json_string(media_user.get("name")),
        )
    image_url = _first_img_src(content) or _json_string(value.get("poster_url"))
    data = {
        "title": title or "",
        "description": description,
        "author": author or "",
        "image_url": image_url or "",
        "content_text": "\n\n".join(part for part in (title, description) if part),
        "original_text": description,
    }
    return {key: val for key, val in data.items() if val}


async def _weibo_visitor_cookie(client: httpx.AsyncClient) -> str | None:
    visitor_url = (
        "https://passport.weibo.com/visitor/genvisitor?cb=gen_callback&fp="
        "%7B%22os%22%3A%221%22%2C%22browser%22%3A%22Chrome%22%2C"
        "%22fonts%22%3A%22undefined%22%2C%22screenInfo%22%3A%221920*1080*24%22%2C"
        "%22plugins%22%3A%22%22%7D"
    )
    try:
        visitor_response = await client.get(
            visitor_url,
            headers={"user-agent": "Mozilla/5.0"},
        )
        visitor = _jsonp_value(visitor_response.text)
        tid = (
            visitor.get("data", {}).get("tid")
            if isinstance(visitor.get("data"), dict)
            else None
        )
        if not tid:
            return None
        incarnate_url = (
            "https://passport.weibo.com/visitor/visitor"
            f"?a=incarnate&t={quote(str(tid))}&w=2&c=095&gc=&cb=cross_domain&from=weibo"
        )
        incarnate_response = await client.get(
            incarnate_url,
            headers={"user-agent": "Mozilla/5.0"},
        )
        incarnate = _jsonp_value(incarnate_response.text)
        data = incarnate.get("data") if isinstance(incarnate.get("data"), dict) else {}
        sub = data.get("sub")
        subp = data.get("subp")
        if isinstance(sub, str) and isinstance(subp, str):
            return f"SUB={sub}; SUBP={subp}"
    except Exception:
        logger.debug("[chat-links] weibo visitor token failed", exc_info=True)
    return None


def _jsonp_value(raw: str) -> dict[str, Any]:
    match = re.search(r"\((\{.*\})\)\s*;?\s*$", raw.strip(), re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(1))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _weibo_metadata_from_api_value(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or value.get("error"):
        return {}
    description = _first_non_empty(
        _json_string(value.get("text_raw")),
        _json_string(value.get("text")),
    )
    author = None
    user = value.get("user")
    if isinstance(user, dict):
        author = _json_string(user.get("screen_name"))
    image_url = _weibo_image_url(value)
    title = _summary_from_text(description or "", platform="微博", author=author)
    data = {
        "title": title[:80] if title else "",
        "description": description or "",
        "author": author or "",
        "image_url": image_url or "",
        "content_text": description or "",
        "original_text": description or "",
    }
    return {key: val for key, val in data.items() if val}


def _first_img_src(html: str) -> str | None:
    match = re.search(r"""<img\b[^>]*\bsrc=["']([^"']+)["']""", html, re.I)
    return unescape(match.group(1)) if match else None


def _first_img_alt(html: str) -> str | None:
    match = re.search(r"""<img\b[^>]*\balt=["']([^"']+)["']""", html, re.I)
    return _clean_text(unescape(match.group(1))) if match else None


def _json_string(value: Any) -> str | None:
    if isinstance(value, str):
        return _clean_text(value)
    return None


def _weibo_image_url(value: dict[str, Any]) -> str | None:
    pic_infos = value.get("pic_infos")
    if not isinstance(pic_infos, dict):
        return None
    for pic in pic_infos.values():
        if not isinstance(pic, dict):
            continue
        for key in ("largest", "large", "original", "bmiddle", "thumbnail"):
            image = pic.get(key)
            if not isinstance(image, dict):
                continue
            url = image.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url
    return None


def _clean_text(value: str | None) -> str:
    text = unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_link_author(value: str | None, *, title: str = "") -> str | None:
    author = _clean_text(value)
    if not author:
        return None
    if title and _compact_compare(author) == _compact_compare(title):
        return None
    return author[:120]


def _compact_compare(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def _normalize_post_body_text(
    value: str | None,
    *,
    platform: str = "",
    author: str | None = None,
) -> str:
    original = _clean_text(value)
    if not original:
        return ""
    text = re.sub(r"^(?:图\s*)?\d+\s*/\s*\d+\s+", "", original).strip()
    text = _strip_leading_author_follow(text, author)
    text = _strip_platform_noise(text)
    text = _strip_trailing_topic_tags(text, aggressive=platform in {"小红书", "抖音"})
    text = re.sub(r"\s+", " ", text).strip()
    return text or original


def _strip_leading_author_follow(text: str, author: str | None) -> str:
    author_text = _clean_text(author)
    if author_text:
        pattern = rf"^{re.escape(author_text)}\s+(?:关注|已关注|Follow|follow)\s+"
        text = re.sub(pattern, "", text).strip()
    match = re.match(r"^(.{1,40}?)\s+(?:关注|已关注|Follow|follow)\s+", text)
    if match:
        prefix = match.group(1)
        if not re.search(r"[，。！？!?：:；;#]", prefix):
            text = text[match.end() :].strip()
    return text


def _strip_platform_noise(text: str) -> str:
    text = re.sub(r"\s*(?:展开|收起|全文|更多)\s*$", "", text)
    text = re.sub(r"\s*(?:点击|打开).{0,12}(?:App|APP|网页|原文)\s*$", "", text)
    return text.strip()


def _strip_trailing_topic_tags(text: str, *, aggressive: bool) -> str:
    if aggressive:
        match = re.search(r"\s#[^#]+", text)
        if match and len(text[: match.start()].strip()) >= 4:
            return text[: match.start()].strip()
    cleaned = re.sub(r"(?:\s*#[^\s#]+#?)+\s*$", "", text).strip()
    return cleaned or text


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
        "image_url": (
            "urlDefault",
            "urlPre",
            "url",
            "url_list",
            "imageList",
            "image_list",
            "large_image_url",
            "middle_image",
            "thumb_image",
            "thumbnail",
            "thumbnailUrl",
            "cover",
            "cover_url",
            "coverUrl",
            "poster",
            "image",
            "pic",
            "pics",
        ),
    }
    data: dict[str, str] = {}
    for output_key, keys in fields.items():
        found = _find_first_json_string(values, keys)
        if found:
            data[output_key] = found
    if "description" in data:
        body = _normalize_post_body_text(
            data["description"],
            platform=platform,
            author=data.get("author"),
        )
        data["description"] = body
        data["content_text"] = "\n\n".join(
            part for part in (data.get("title"), body) if part
        )
    if "description" in data:
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
                    for item in raw:
                        if isinstance(item, str) and _clean_text(item):
                            return _clean_text(item)
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
