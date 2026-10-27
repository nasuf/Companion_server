"""One event per logical send, independent of its card/text bubble count."""

import json
from uuid import NAMESPACE_URL, uuid4, uuid5


def proactive_event_id(*, conversation_id: str, kind: str, source_id: str | None = None) -> str:
    # A durable operation ID makes a retried multi-part send keep its identity.
    # Do not use just a user/agent/trigger: different sends must remain distinct.
    if source_id:
        key = json.dumps(["companion:proactive:v1", conversation_id, kind, source_id])
        return str(uuid5(NAMESPACE_URL, key))
    return str(uuid4())
