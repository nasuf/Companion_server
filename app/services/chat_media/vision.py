from __future__ import annotations

from typing import Any
import logging

import httpx

from app.config import settings
from app.services.chat_media import repo, storage
from app.services.chat_media.prompt import attachment_to_metadata

logger = logging.getLogger(__name__)

_VISION_PROMPT = """请分析用户发来的图片，输出一段适合陪伴式聊天使用的中文摘要。
要求：
1. 客观描述画面、可见文字、物体、人物状态和可能的场景。
2. 如果用户附带了文字，请结合用户文字理解图片。
3. 不要臆测隐私身份、疾病诊断、危险结论；不确定就说不确定。
4. 控制在 120 字以内。"""


async def ensure_vision_summaries(
    attachments: list[repo.ChatAttachment],
    *,
    user_text: str,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for attachment in attachments:
        current = attachment
        if attachment.kind == "image" and not attachment.vision_summary:
            summary, status, error = await _analyze_attachment(attachment, user_text=user_text)
            if status != attachment.vision_status or summary:
                await repo.update_vision_result(
                    attachment.id,
                    status=status,
                    summary=summary,
                    error=error,
                )
                current = repo.ChatAttachment(
                    **{
                        **attachment.__dict__,
                        "vision_status": status,
                        "vision_summary": summary,
                        "vision_error": error,
                    }
                )
        metadata.append(attachment_to_metadata(current))
    return metadata


async def _analyze_attachment(
    attachment: repo.ChatAttachment,
    *,
    user_text: str,
) -> tuple[str | None, str, str | None]:
    if not settings.ark_api_key:
        return None, "skipped", "ARK_API_KEY is not configured"
    try:
        data_url = storage.read_image_base64(attachment.storage_key, attachment.mime)
        content = await _call_doubao_vision(data_url=data_url, user_text=user_text)
        summary = _clean_summary(content)
        if not summary:
            return None, "failed", "empty vision response"
        return summary, "ready", None
    except Exception as exc:
        logger.warning(
            "[chat-media] vision analysis failed attachment=%s: %s",
            attachment.id,
            exc,
            exc_info=True,
        )
        return None, "failed", str(exc)[:300]


async def _call_doubao_vision(*, data_url: str, user_text: str) -> str:
    endpoint = settings.ark_base_url.rstrip("/") + "/chat/completions"
    user_hint = user_text.strip() or "用户没有附加文字，只发送了图片。"
    payload = {
        "model": settings.doubao_vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{_VISION_PROMPT}\n\n用户文字：{user_hint}"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.ark_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_text(response)
            if response.status_code == 404:
                raise RuntimeError(
                    "Doubao vision endpoint/model not found. "
                    f"Check ARK_BASE_URL={settings.ark_base_url!r} and "
                    f"DOUBAO_VISION_MODEL={settings.doubao_vision_model!r}. "
                    f"response={detail}"
                ) from exc
            raise RuntimeError(
                f"Doubao vision request failed status={response.status_code} "
                f"response={detail}"
            ) from exc
        data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return str(content or "")


def _clean_summary(value: str) -> str:
    text = " ".join((value or "").strip().split())
    return text[:500]


def _response_error_text(response: httpx.Response) -> str:
    text = (response.text or "").strip()
    return text[:500] if text else "<empty>"
