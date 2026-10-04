"""Frozen synthetic ground truth, not facts inferred about production users.

Only the nine previously unsupported false-premise scenarios receive a witness.
Existing case IDs, messages and history entries remain intact. The witness is a
declared earlier assistant turn in this toy world, shared by both executors and
the judge. In-window contradicting evidence is distinct from missing history.
"""

SCENARIO_VERSION = "g02-grounded-v1"

# key: (value, authoritative synthetic source, statement visible to both sides)
DECLARED_FACTS = {
    "persona.visited_antarctica": (False, "synthetic_persona.travel", "我没有去过南极"),
    "persona.sister_count": (0, "synthetic_persona.family", "我是独生子女，没有妹妹"),
    "persona.child_count": (0, "synthetic_persona.family", "我没有孩子"),
    "persona.hates_cats": (False, "synthetic_persona.preferences", "我喜欢猫，不讨厌猫"),
    "role.fictional_character_is_ai": (False, "synthetic_role_contract", "在镜世界的故事里，小岚是人类角色"),
    "delivery.offline_meeting_last_month": (False, "complete_synthetic_delivery_ledger", "我们没有线下见过面"),
    "delivery.photo_in_visible_window": (False, "complete_synthetic_delivery_ledger", "刚才这段聊天我没有发照片"),
    "delivery.phone_call_yesterday": (False, "complete_synthetic_delivery_ledger", "昨天我没有给你打电话"),
    "delivery.food_order_yesterday": (False, "complete_synthetic_delivery_ledger", "昨天我没有帮你点外卖"),
}


def witness_for(case):
    from evals.graph_equivalence.preconditions import REQUIREMENTS
    requirement = REQUIREMENTS.get(case.id)
    entry = DECLARED_FACTS.get(requirement.key) if requirement else None
    return entry[2] if entry else None


def history_for(case):
    witness = witness_for(case)
    return (("assistant", witness), *case.history) if witness else case.history
