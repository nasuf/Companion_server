from __future__ import annotations

from typing import Any

_IMAGE_PLACEHOLDER = "用户发送了一张图片"


def attachment_to_metadata(attachment: Any) -> dict[str, Any]:
    data = {
        "id": attachment.id,
        "kind": attachment.kind,
        "name": attachment.name,
        "mime": attachment.mime,
        "size": attachment.size,
        "width": attachment.width,
        "height": attachment.height,
        "url": attachment.url,
        "vision_status": attachment.vision_status,
    }
    if attachment.vision_summary:
        data["vision_summary"] = attachment.vision_summary
    return data


def render_message_content_for_prompt(
    content: str,
    metadata: dict[str, Any] | None,
) -> str:
    text = (content or "").strip()
    attachments = _metadata_attachments(metadata)
    if not attachments:
        return content

    image_lines: list[str] = []
    for index, attachment in enumerate(attachments, start=1):
        if attachment.get("kind") != "image":
            continue
        summary = str(attachment.get("vision_summary") or "").strip()
        if summary:
            image_lines.append(f"图片{index}：{summary}")
        else:
            image_lines.append(f"图片{index}：当前无法识别图片内容，回复时不要猜测图片细节。")

    if not image_lines:
        return content
    prefix = text or _IMAGE_PLACEHOLDER
    return f"{prefix}\n\n[图片内容]\n" + "\n".join(image_lines)


def render_user_message_with_attachments(
    content: str,
    attachments: list[dict[str, Any]],
) -> str:
    return render_message_content_for_prompt(content, {"attachments": attachments})


def _metadata_attachments(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("attachments")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]
