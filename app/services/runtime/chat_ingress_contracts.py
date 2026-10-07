"""Prepared, server-only inputs for the staged SQL chat ingress.

Construct these after authentication and domain validation. They are not HTTP
schemas or permission grants. Original client input defines retry identity;
rendered text, policy and execution snapshots are captured separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any, Literal


AggregationMode = Literal["immediate", "fragment_window", "turn_window"]
MAX_JSON_BYTES = 262_144
MAX_TURN_MESSAGES = 32
MAX_TURN_CHARS = 32_768


def canonical_object(value: dict[str, Any]) -> str:
    """Capture JSON with strict string keys and finite numbers, without aliases."""
    def validate(item: Any) -> None:
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for key in item:
                validate(key)
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) is str:
            if "\x00" in item:
                raise ValueError("JSON strings cannot contain null characters")
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise ValueError("Input must contain JSON values only")

    if type(value) is not dict:
        raise ValueError("Expected a JSON object")
    try:
        validate(value)
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("JSON input is too large")
        return encoded
    except (TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid JSON input") from exc


def _identity(value: str, *, maximum: int = 256) -> str:
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > maximum or any(ord(char) < 32 for char in value)):
        raise ValueError("Invalid request identity")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Invalid request identity") from exc
    return value


@dataclass(frozen=True, slots=True)
class ChatRequestInput:
    request_key: str
    source: Literal["client", "wechat"]
    input_json: str
    client_id: str | None = None

    def __post_init__(self) -> None:
        _identity(self.request_key)
        if self.source not in {"client", "wechat"}:
            raise ValueError("Unsupported request source")
        payload = json.loads(self.input_json)
        if canonical_object(payload) != self.input_json:
            raise ValueError("Input must be canonical JSON")
        if set(payload) != {"text", "attachment_ids", "component_card"}:
            raise ValueError("Unsupported chat input")
        text, ids, card = payload["text"], payload["attachment_ids"], payload["component_card"]
        if type(text) is not str or len(text) > MAX_TURN_CHARS:
            raise ValueError("Invalid message text")
        if type(ids) is not list or len(ids) > 20:
            raise ValueError("Invalid attachment list")
        for item in ids:
            _identity(item)
        if len(set(ids)) != len(ids) or (card is not None and type(card) is not dict):
            raise ValueError("Invalid attachment/card input")
        if not text.strip() and not ids and not card:
            raise ValueError("Empty chat input")
        if self.source == "client":
            _identity(self.client_id, maximum=249)
            if self.request_key != "client:" + self.client_id:
                raise ValueError("Client identity mismatch")
        else:
            if self.client_id is not None or not self.request_key.startswith("wechat:"):
                raise ValueError("Provider identity mismatch")
            _identity(self.request_key[7:], maximum=249)

    @classmethod
    def from_client(cls, *, client_id: str, text: str,
                    attachment_ids: list[str] | None = None,
                    component_card: dict | None = None) -> ChatRequestInput:
        return cls("client:" + _identity(client_id, maximum=249), "client",
                   canonical_object({"text": text, "attachment_ids": attachment_ids if attachment_ids is not None else [],
                                     "component_card": component_card}), client_id)

    @classmethod
    def from_wechat(cls, *, message_id: str, text: str,
                    attachment_ids: list[str] | None = None) -> ChatRequestInput:
        # The message ID must come from the verified provider envelope.
        return cls("wechat:" + _identity(message_id, maximum=249), "wechat",
                   canonical_object({"text": text, "attachment_ids": attachment_ids if attachment_ids is not None else [],
                                     "component_card": None}))

    @property
    def fingerprint(self) -> str:
        return sha256(self.input_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedChatMessage:
    persisted_text: str
    prompt_text: str
    metadata_json: str
    reply_context_json: str
    received_at: datetime

    def __post_init__(self) -> None:
        for text in (self.persisted_text, self.prompt_text):
            if type(text) is not str or len(text) > MAX_TURN_CHARS or "\x00" in text:
                raise ValueError("Invalid prepared text")
            try:
                text.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("Invalid prepared text") from exc
        for value in (self.metadata_json, self.reply_context_json):
            if canonical_object(json.loads(value)) != value:
                raise ValueError("Prepared input must be canonical JSON")
        if not isinstance(self.received_at, datetime) or self.received_at.utcoffset() is None:
            raise ValueError("Receipt time must have a timezone")
        object.__setattr__(self, "received_at", self.received_at.astimezone(timezone.utc))

    @classmethod
    def capture(cls, *, persisted_text: str, prompt_text: str,
                metadata: dict, reply_context: dict, received_at: datetime) -> PreparedChatMessage:
        return cls(persisted_text, prompt_text, canonical_object(metadata),
                   canonical_object(reply_context), received_at)


@dataclass(frozen=True, slots=True)
class ChatExecutionSnapshot:
    executor: Literal["legacy", "langgraph"]
    graph_version: str
    state_version: int
    config_json: str
    prompts_json: str
    budget_json: str
    deadline_at: datetime | None = None
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.executor not in {"legacy", "langgraph"}:
            raise ValueError("Unsupported executor")
        _identity(self.graph_version)
        if type(self.state_version) is not int or self.state_version < 1:
            raise ValueError("Invalid state version")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("Invalid execution attempt budget")
        for value in (self.config_json, self.prompts_json, self.budget_json):
            if canonical_object(json.loads(value)) != value:
                raise ValueError("Snapshot must be canonical JSON")
        if self.deadline_at is not None:
            if not isinstance(self.deadline_at, datetime) or self.deadline_at.utcoffset() is None:
                raise ValueError("Deadline must have a timezone")
            object.__setattr__(self, "deadline_at", self.deadline_at.astimezone(timezone.utc))

    @classmethod
    def capture(cls, *, executor: str, graph_version: str, state_version: int,
                config: dict, prompts: dict, budget: dict,
                deadline_at: datetime | None = None, max_attempts: int = 3) -> ChatExecutionSnapshot:
        return cls(executor, graph_version, state_version, canonical_object(config),
                   canonical_object(prompts), canonical_object(budget), deadline_at, max_attempts)


@dataclass(frozen=True, slots=True)
class ChatAggregationPolicy:
    mode: AggregationMode
    quiet_seconds: float = 0
    max_wait_seconds: float | None = None
    delay_seconds: float = 0
    allow_join: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"immediate", "fragment_window", "turn_window"}:
            raise ValueError("Unsupported aggregation mode")
        if self.quiet_seconds is None or self.delay_seconds is None:
            raise ValueError("Invalid timing policy")
        for value in (self.quiet_seconds, self.delay_seconds, self.max_wait_seconds):
            if value is not None and (type(value) not in {int, float}
                    or not math.isfinite(value) or not 0 <= value <= 86_400):
                raise ValueError("Invalid timing policy")
        if type(self.allow_join) is not bool:
            raise ValueError("Invalid join policy")
        if self.mode == "immediate":
            if self.quiet_seconds != 0 or self.max_wait_seconds is not None:
                raise ValueError("Immediate input cannot open an aggregation window")
        elif self.quiet_seconds <= 0:
            raise ValueError("Aggregation window must be positive")
        if self.max_wait_seconds is not None and self.max_wait_seconds < self.quiet_seconds:
            raise ValueError("Maximum wait must cover the quiet window")
