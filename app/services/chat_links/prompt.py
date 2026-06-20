from __future__ import annotations

from typing import Any

_LINK_PLACEHOLDER = "用户分享了一个链接"


def render_message_content_for_prompt(
    content: str,
    metadata: dict[str, Any] | None,
) -> str:
    text = (content or "").strip()
    link = _metadata_link(metadata)
    if not link:
        return content
    lines = _link_lines(link)
    if not lines:
        return content
    prefix = text or _LINK_PLACEHOLDER
    return f"{prefix}\n\n[链接卡片内容]\n" + "\n".join(lines)


def render_user_message_with_link(
    content: str,
    link_metadata: dict[str, Any] | None,
) -> str:
    if not link_metadata:
        return content
    return render_message_content_for_prompt(content, {"link_card": link_metadata})


def _metadata_link(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("link_card")
    return dict(raw) if isinstance(raw, dict) else None


def _link_lines(link: dict[str, Any]) -> list[str]:
    title = str(link.get("title") or "").strip()
    platform = str(link.get("platform") or "链接").strip()
    author = str(link.get("author") or "").strip()
    summary = str(link.get("summary") or "").strip()
    content = str(link.get("content_text") or "").strip()
    description = str(link.get("description") or "").strip()
    final_url = str(link.get("final_url") or link.get("source_url") or "").strip()
    status = str(link.get("status") or "ready").strip()
    error = str(link.get("error") or "").strip()

    lines = [f"平台：{platform}"]
    if title:
        lines.append(f"标题：{title}")
    if author:
        lines.append(f"作者：{author}")
    body = summary or description
    if body:
        lines.append(f"摘要：{body[:1000]}")
    if content and not _same_or_contained(content, body):
        lines.append(f"正文：{content[:2000]}")
    if final_url:
        lines.append(f"链接：{final_url}")
    if status == "partial" and error:
        lines.append(f"读取状态：内容读取不完整（{error}），回复时不要编造未读取到的细节。")
    return lines


def _same_or_contained(value: str, other: str | None) -> bool:
    if not other:
        return False
    value_norm = " ".join(value.split())
    other_norm = " ".join(other.split())
    return value_norm == other_norm or value_norm in other_norm
