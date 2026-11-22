"""Select an exact reply-producing Trace input, never a similar conversation.

Historical cases exercise the model against its original complete input. They
do not claim to rerender old business state under today's template versions.
Missing or ambiguous evidence remains unknown rather than using a fake persona.
"""

from app.services.emoji import _EMOJI_UNIT_RE


def _comparable(text):
    # Delivery appends an emoji and stores split bubbles joined by spaces; undo
    # only those two steps so the stored reply still has to match exactly.
    return " ".join(_EMOJI_UNIT_RE.sub("", text.replace("||", " ")).split())


# Shorter bubbles ("嗯", "哈哈") also occur in classifier and helper outputs.
MIN_MATCH_CHARS = 4


def reply_messages(traces, bubbles):
    # The first stored bubble can be a topic-continuation line from its own model
    # call; the longest bubble is the one most specific to the main reply, and on
    # a tie the later one, since continuation lines always come first.
    if isinstance(bubbles, str):
        bubbles = [bubbles]
    comparable = [_comparable(b or "") for b in bubbles]
    wanted = max(reversed(comparable), key=len, default="")
    if len(wanted) < MIN_MATCH_CHARS:
        return None
    matched = 0
    matches = []
    for trace in traces:
        inputs, outputs = trace.get("inputs_json"), trace.get("outputs_json")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        generations = outputs.get("generations")
        if not isinstance(generations, list) or len(generations) != 1:
            continue
        row = generations[0]
        if not isinstance(row, list) or len(row) != 1 or not isinstance(row[0], dict):
            continue
        text = row[0].get("text")
        # Exact substring after undoing delivery decoration. No fuzzy match,
        # time-nearest call, or judgement-model input qualifies.
        if not isinstance(text, str) or wanted not in (produced := _comparable(text)):
            continue
        # Ambiguity counts every call whose output matches, parsable or not.
        matched += 1
        # Only leading continuation lines may come from another call. A later
        # bubble missing here belongs to another reply fragment, whose problem
        # this input would not reproduce (2026-11-19 sample 262aaefc).
        first = next(i for i, c in enumerate(comparable) if c and c in produced)
        if any(c and c not in produced for c in comparable[first:]):
            continue
        messages = inputs.get("messages")
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], list)
        ):
            messages = messages[0]
        # Main prompts carry up to 50 history rounds plus system parts; size is
        # bounded by the 200KB input filter in sampling.
        if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
            continue
        parsed = []
        for message in messages:
            if not isinstance(message, dict):
                break
            m = message.get("kwargs", message)
            role = m.get("role", m.get("type"))
            role = {"human": "user", "ai": "assistant"}.get(role, role)
            if role not in {"system", "user", "assistant"} or not isinstance(
                m.get("content"), str
            ):
                break
            parsed.append(dict(role=role, content=m["content"]))
        if len(parsed) == len(messages) and any(m["role"] == "user" for m in parsed):
            matches.append(parsed)
    return matches[0] if matched == 1 and len(matches) == 1 else None
