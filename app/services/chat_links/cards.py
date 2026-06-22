from __future__ import annotations

from typing import Any

from app.services.chat_links.extraction import accent_for_platform, app_url_for_link


def component_card_for_link(link: Any) -> dict[str, Any]:
    payload = {
        "link_id": link.id,
        "source_url": link.source_url,
        "final_url": link.final_url,
        "platform": link.platform,
        "status": link.status,
    }
    if link.author:
        payload["author"] = link.author
    if link.image_url:
        payload["image_url"] = link.image_url
    if link.original_text:
        payload["original_text"] = link.original_text
    if link.content_text:
        payload["content_text"] = link.content_text
    if link.summary:
        payload["summary"] = link.summary
    app_url = app_url_for_link(
        platform=link.platform,
        source_url=link.source_url,
        final_url=link.final_url,
    )
    if app_url:
        payload["app_url"] = app_url
    if link.error:
        payload["error"] = link.error
    platform = _platform_name(link)
    return {
        "version": 1,
        "type": "external_link",
        "title": platform,
        "subtitle": "",
        "body": _body(link),
        "footer": _footer(platform),
        "accent": accent_for_platform(link.platform),
        "payload": payload,
    }


def metadata_for_link_card(link: Any) -> dict[str, Any]:
    return {
        "id": link.id,
        "role": link.role,
        "source_app": link.source_app,
        "source_url": link.source_url,
        "final_url": link.final_url,
        "platform": link.platform,
        "title": link.title,
        "description": link.description,
        "author": link.author,
        "image_url": link.image_url,
        "content_text": link.content_text,
        "summary": link.summary,
        "status": link.status,
        "error": link.error,
        "component_card": component_card_for_link(link),
    }


def _platform_name(link: Any) -> str:
    platform = str(getattr(link, "platform", "") or "").strip()
    return platform or "链接"


def _body(link: Any) -> str:
    for value in (
        getattr(link, "original_text", ""),
        getattr(link, "content_text", ""),
        getattr(link, "summary", ""),
        getattr(link, "description", ""),
        getattr(link, "title", ""),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _footer(platform: str) -> str:
    label = "原" if platform == "链接" else platform
    return f"点击打开{label}app/网页"
