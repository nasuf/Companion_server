"""AI Identity service.

Generates identity narrative from personality + background + values.
"""

from app.services.personality import big_five_to_description


def build_identity_narrative(agent) -> str:
    """Build a natural identity narrative for the AI agent."""
    parts = []

    name = getattr(agent, "name", "AI")
    parts.append(f"My name is {name}.")

    personality = getattr(agent, "personality", None)
    if personality:
        desc = big_five_to_description(personality)
        parts.append(f"I am {desc}.")

    background = getattr(agent, "background", None)
    if background:
        parts.append(background)

    values = getattr(agent, "values", None)
    if values:
        if isinstance(values, list):
            val_str = ", ".join(str(v) for v in values)
        elif isinstance(values, dict):
            val_str = ", ".join(str(v) for v in values.values())
        else:
            val_str = str(values)
        parts.append(f"I value {val_str}.")

    return " ".join(parts)
