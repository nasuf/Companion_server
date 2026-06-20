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
    return {
        "version": 1,
        "type": "external_link",
        "title": link.title or "未命名链接",
        "subtitle": _subtitle(link),
        "body": link.summary or link.description or link.content_text or "",
        "footer": "点击打开原 App / 网页",
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


def _subtitle(link: Any) -> str:
    parts = [str(link.platform or "链接")]
    author = str(link.author or "").strip()
    if author:
        parts.append(author)
    if link.status == "partial":
        parts.append("内容读取不完整")
    return " · ".join(parts)
