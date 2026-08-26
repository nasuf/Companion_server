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
        "duration_seconds": attachment.duration_seconds,
        "url": attachment.url,
        "vision_status": attachment.vision_status,
        "transcription_status": attachment.transcription_status,
    }
    if attachment.vision_summary:
        data["vision_summary"] = attachment.vision_summary
    if attachment.transcription_text:
        data["transcription_text"] = attachment.transcription_text
    if attachment.transcription_model:
        data["transcription_model"] = attachment.transcription_model
    if attachment.transcription_request_id:
        data["transcription_request_id"] = attachment.transcription_request_id
    return data


def render_message_content_for_prompt(
    content: str,
    metadata: dict[str, Any] | None,
) -> str:
    text = (content or "").strip()
    attachments = _metadata_attachments(metadata)
    link_card = _metadata_link_card(metadata)
    rendered = content

    if attachments:
        image_lines: list[str] = []
        for index, attachment in enumerate(attachments, start=1):
            if attachment.get("kind") != "image":
                continue
            summary = str(attachment.get("vision_summary") or "").strip()
            if summary:
                image_lines.append(f"图片{index}：{summary}")
            else:
                image_lines.append(f"图片{index}：当前无法识别图片内容，回复时不要猜测图片细节。")

        if image_lines:
            prefix = text or _IMAGE_PLACEHOLDER
            rendered = f"{prefix}\n\n[图片内容]\n" + "\n".join(image_lines)

    if link_card:
        from app.services.chat_links.prompt import render_message_content_for_prompt as render_link

        rendered = render_link(rendered, {"link_card": link_card})

    component_card = _metadata_component_card(metadata)
    if component_card:
        from app.services.offerings_memory_text import render_component_card_line

        rendered = render_component_card_line(rendered, component_card)
    return rendered


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


def _metadata_link_card(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("link_card")
    return dict(raw) if isinstance(raw, dict) else None


def _metadata_component_card(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("component_card")
    return dict(raw) if isinstance(raw, dict) else None
