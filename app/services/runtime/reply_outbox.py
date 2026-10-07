"""SQL-only prepared reply commits. Use with SqlJobQueue.finish, never stream first."""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import NAMESPACE_URL, uuid5

from app.services.chat.reply_formatting import strip_system_markers
from app.services.runtime.chat_ingress_contracts import MAX_JSON_BYTES, canonical_object
from app.services.runtime.sql_job_contracts import ClaimedJob, BoundSqlCommit
from app.services.runtime.sql_job_consumer import PreparedJobResult


@dataclass(frozen=True, slots=True)
class PreparedReply:
    text: str
    data_json: str

    def __post_init__(self):
        if (
            type(self.text) is not str
            or not self.text.strip()
            or len(self.text) > 32768
        ):
            raise ValueError("Invalid reply text")
        if strip_system_markers(self.text) != self.text:
            raise ValueError("Reply contains system markers")
        if canonical_object(json.loads(self.data_json)) != self.data_json:
            raise ValueError("Reply data must be canonical JSON")
        if set(json.loads(self.data_json)) & {
            "text",
            "index",
            "message_id",
            "event_id",
            "run_id",
            "sequence",
            "delivery_token",
        }:
            raise ValueError("Reply data cannot override delivery identity")

    @classmethod
    def capture(cls, text: str, **data) -> PreparedReply:
        return cls(strip_system_markers(text), canonical_object(data))


def prepare_chat_result(
    claim: ClaimedJob, replies: tuple[PreparedReply, ...], *, done: dict | None = None
) -> PreparedJobResult:
    """Capture immutable output before entering the queue's final fenced transaction.

    The callback writes messages, events and the returned result together with
    Run/Job success. No notifications, achievements or background tasks here:
    those require their own idempotent SQL jobs at endpoint activation.
    """
    claim.require_active()
    if (
        type(replies) is not tuple
        or not 1 <= len(replies) <= 32
        or any(not isinstance(r, PreparedReply) for r in replies)
    ):
        raise ValueError("A bounded nonempty reply tuple is required")
    done_json = canonical_object({} if done is None else done)
    if set(json.loads(done_json)) & {
        "message_id",
        "run_id",
        "event_id",
        "sequence",
        "delivery_token",
    }:
        raise ValueError("Done data cannot override delivery identity")

    def identity(key):
        return str(uuid5(NAMESPACE_URL, "companion:outbox:" + claim.run_id + ":" + key))

    message_ids = [identity("message:" + str(i)) for i in range(len(replies))]
    records = []
    for i, reply in enumerate(replies):
        metadata = {**json.loads(reply.data_json), "reply_index": i}
        payload = {
            **json.loads(reply.data_json),
            "text": reply.text,
            "index": i,
            "message_id": message_ids[i],
        }
        records.append(
            (
                identity("reply:" + str(i)),
                "reply:" + str(i),
                i,
                "reply",
                canonical_object(payload),
                message_ids[i],
                reply.text,
                canonical_object(metadata),
            )
        )
    records.append(
        (
            identity("done"),
            "done",
            len(replies),
            "done",
            canonical_object({**json.loads(done_json), "message_id": message_ids[0]}),
            None,
            None,
            None,
        )
    )
    if (
        sum(len(r[4].encode()) + len((r[7] or "").encode()) for r in records)
        > MAX_JSON_BYTES
    ):
        raise ValueError("Prepared turn output is too large")
    result = {"message_ids": message_ids, "event_ids": [r[0] for r in records]}

    async def commit(tx):
        claim.require_active()
        # One common SQL publish time, after the conversation fence was acquired.
        # Replay is acknowledgement-driven; no timestamp cursor can miss late commits.
        stamp = (
            await tx.query_raw(
                "SELECT GREATEST(clock_timestamp(),COALESCE((SELECT max(created_at)+interval '1 millisecond' FROM messages WHERE conversation_id=$1),'-infinity'::timestamptz)) AS now",
                claim.scope.conversation_id,
            )
        )[0]["now"]
        for event_id, key, seq, kind, payload, message_id, text, metadata in records:
            if message_id is not None:
                await tx.execute_raw(
                    "INSERT INTO messages (id,conversation_id,role,content,metadata,created_at) VALUES ($1,$2,'assistant',$3,$4::jsonb,$5::timestamptz+$6::integer*interval '1 millisecond')",
                    message_id,
                    claim.scope.conversation_id,
                    text,
                    metadata,
                    stamp,
                    seq,
                )
            await tx.execute_raw(
                "INSERT INTO runtime_outbox (id,run_id,event_key,sequence,event_type,payload,message_id,created_at,updated_at) VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8::timestamptz,$8::timestamptz)",
                event_id,
                claim.run_id,
                key,
                seq,
                kind,
                payload,
                message_id,
                stamp,
            )
        claim.require_active()

    return PreparedJobResult(result, BoundSqlCommit(claim, commit))
