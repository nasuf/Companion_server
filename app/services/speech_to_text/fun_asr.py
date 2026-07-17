from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_CONTEXT_MESSAGE_LIMIT = 10
_CONTEXT_MESSAGE_MAX_CHARS = 200
_CONTEXT_TOTAL_MAX_CHARS = 400


class SpeechTranscriptionNotConfigured(Exception):
    pass


class SpeechTranscriptionTimeout(Exception):
    pass


class SpeechTranscriptionRateLimited(Exception):
    pass


class SpeechTranscriptionProviderError(Exception):
    pass


class SpeechTranscriptionEmpty(Exception):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    request_id: str | None
    model: str


def _context_messages(
    context: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str]] = []
    for role, raw_text in context:
        if role not in {"user", "assistant"}:
            continue
        text = raw_text.strip()
        if not text:
            continue
        candidates.append((role, text[-_CONTEXT_MESSAGE_MAX_CHARS:]))

    remaining = _CONTEXT_TOTAL_MAX_CHARS
    messages: list[dict[str, Any]] = []
    for role, text in reversed(candidates[-_CONTEXT_MESSAGE_LIMIT:]):
        if remaining <= 0:
            break
        clipped = text[-remaining:]
        remaining -= len(clipped)
        content_type = "input_text" if role == "user" else "text"
        messages.append(
            {
                "role": role,
                "content": [
                    {
                        "type": content_type,
                        "text": clipped,
                    }
                ],
            }
        )
    messages.reverse()
    return messages


def build_request_payload(
    *,
    audio: bytes,
    mime: str,
    audio_format: str,
    context: Iterable[tuple[str, str]] = (),
    model: str | None = None,
) -> dict[str, Any]:
    encoded = base64.b64encode(audio).decode("ascii")
    messages = _context_messages(context)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": f"data:{mime};base64,{encoded}"},
                }
            ],
        }
    )
    return {
        "model": model or settings.dashscope_asr_model,
        "input": {"messages": messages},
        "parameters": {"format": audio_format, "sample_rate": "16000"},
    }


def _extract_text(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, dict):
        return ""
    nested_output = output.get("output")
    if isinstance(nested_output, dict):
        sentence = nested_output.get("sentence")
        if isinstance(sentence, dict):
            nested_text = sentence.get("text")
            if isinstance(nested_text, str) and nested_text.strip():
                return nested_text.strip()
    output_text = output.get("text")
    return output_text.strip() if isinstance(output_text, str) else ""


async def transcribe_audio(
    *,
    audio: bytes,
    mime: str,
    audio_format: str,
    context: Iterable[tuple[str, str]] = (),
    client: httpx.AsyncClient | None = None,
) -> TranscriptionResult:
    api_key = settings.dashscope_api_key.strip()
    endpoint = settings.dashscope_asr_endpoint.strip()
    model = settings.dashscope_asr_model.strip()
    if not api_key or not endpoint or not model:
        raise SpeechTranscriptionNotConfigured

    payload = build_request_payload(
        audio=audio,
        mime=mime,
        audio_format=audio_format,
        context=context,
        model=model,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }

    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=settings.dashscope_asr_timeout_s)
    try:
        response = await http_client.post(endpoint, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise SpeechTranscriptionTimeout from exc
    except httpx.HTTPError as exc:
        raise SpeechTranscriptionProviderError from exc
    finally:
        if owns_client:
            await http_client.aclose()

    request_id = response.headers.get("x-request-id")
    response_payload: Any = None
    try:
        response_payload = response.json()
    except ValueError:
        pass

    if isinstance(response_payload, dict):
        request_id = str(response_payload.get("request_id") or request_id or "") or None
    if response.status_code == 429:
        raise SpeechTranscriptionRateLimited
    if response.status_code in {401, 403}:
        logger.error(
            "[speech-to-text] provider credentials rejected status=%s request_id=%s",
            response.status_code,
            request_id,
        )
        raise SpeechTranscriptionNotConfigured
    if response.status_code < 200 or response.status_code >= 300:
        logger.warning(
            "[speech-to-text] provider request failed status=%s request_id=%s",
            response.status_code,
            request_id,
        )
        raise SpeechTranscriptionProviderError
    if not isinstance(response_payload, dict):
        logger.warning(
            "[speech-to-text] invalid provider response status=%s request_id=%s",
            response.status_code,
            request_id,
        )
        raise SpeechTranscriptionProviderError

    text = _extract_text(response_payload)
    if not text:
        raise SpeechTranscriptionEmpty
    logger.info(
        "[speech-to-text] transcription completed model=%s audio_bytes=%s request_id=%s",
        model,
        len(audio),
        request_id,
    )
    return TranscriptionResult(text=text, request_id=request_id, model=model)
