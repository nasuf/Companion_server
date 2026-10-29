"""Select an exact reply-producing Trace input, never a similar conversation.

Historical cases exercise the model against its original complete input. They
do not claim to rerender old business state under today's template versions.
Missing or ambiguous evidence remains unknown rather than using a fake persona.
"""


def reply_messages(traces, first_bubble):
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
        # Exact substring permits only downstream bubble splitting. No fuzzy
        # match, time-nearest call, or judgement-model input qualifies.
        if not isinstance(text, str) or not first_bubble or first_bubble not in text:
            continue
        messages = inputs.get("messages")
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], list)
        ):
            messages = messages[0]
        if not isinstance(messages, list) or not 1 <= len(messages) <= 80:
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
    return matches[0] if len(matches) == 1 else None
