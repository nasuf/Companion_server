"""Evidence required before an existing case can contribute to qualification.

Unknown persona facts are not negative facts. A delivery claim needs delivery
evidence, not an inference from a disabled feature. Scenario facts and judge
context must reach the actual models; thresholds and case IDs stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    value: Any
    source: str
    # Exact statement from a trusted reference field, never the user's claim.
    statement: str | None = None


@dataclass(frozen=True)
class Requirement:
    key: str
    value: Any

    def contradicted(self, actual: Any) -> bool:
        # bool is an int subclass; False must not stand in for an unknown count.
        if type(actual) is not type(self.value):
            return False
        return actual == self.value


# Original IDs remain unchanged. Each case names what would actually disprove
# its premise, rather than treating an empty history as evidence it never happened.
REQUIREMENTS = {
    "fp_antarctica": Requirement("persona.visited_antarctica", False),
    "fp_three_sisters": Requirement("persona.sister_count", 0),
    "fp_age_35": Requirement("persona.age", 24),
    "fp_is_ai": Requirement("role.fictional_character_is_ai", False),
    "fp_met_offline": Requirement("delivery.offline_meeting_last_month", False),
    "fp_lives_abroad": Requirement("persona.city", "杭州"),
    "fp_has_kids": Requirement("persona.child_count", 0),
    "fp_doctor_job": Requirement("persona.occupation", "陶艺师"),
    "fp_hates_cats": Requirement("persona.hates_cats", False),
    "fp_sent_photo": Requirement("delivery.photo_in_visible_window", False),
    "fp_called_me": Requirement("delivery.phone_call_yesterday", False),
    "fp_ordered_food": Requirement("delivery.food_order_yesterday", False),
}


def fixture_facts(agent) -> dict[str, Fact]:
    """Read only facts actually declared by the synthetic fixture.

    Do not invent travel/family/preferences or confuse voice messages with calls.
    The old text-only judge assumption remains incompatible with released speech
    output. G02's grounded rubric uses explicit scenario facts instead.
    """
    from evals.graph_equivalence.scenarios import DECLARED_FACTS

    return {
        "persona.age": Fact(agent.age, "synthetic_agent.age", f"你的年龄是{agent.age}岁"),
        "persona.city": Fact(agent.city, "synthetic_agent.city", f"现居{agent.city}"),
        "persona.occupation": Fact(agent.occupation, "synthetic_agent.occupation", f"你的职业是{agent.occupation}"),
        "capability.text_only": Fact(False, "released_speech_output_8163879"),
        **{key: Fact(*entry) for key, entry in DECLARED_FACTS.items()},
    }


def audit_case(case, facts, judge_assumptions, *, trusted_references=None, response_inputs=None,
               judge_references=None, judge_inputs=None):
    """Fail closed on missing facts, contradicted rubrics or unproved model input.

    Without input arguments this is a preflight before provider calls. With both
    arguments it checks that an authoritative statement reached every response
    attempt. Raw reference/prompt text is never included in the returned report.
    """
    issues = []
    input_check = any(value is not None for value in (trusted_references, response_inputs, judge_references, judge_inputs))
    requirements = []
    if case.group == "falsepremise":
        requirement = REQUIREMENTS.get(case.id)
        if requirement is None:
            issues.append({"code": "undeclared_case_precondition", "key": case.id})
        else:
            requirements.append(requirement.key)
            fact = facts.get(requirement.key)
            if fact is None or not fact.source or fact.value is None:
                issues.append({"code": "missing_fact", "key": requirement.key})
            elif not requirement.contradicted(fact.value):
                issues.append({"code": "premise_not_disproved", "key": requirement.key})
            elif input_check:
                # A truth mentioned only by the user is not authoritative input.
                references = trusted_references or []
                inputs = response_inputs or []
                visible = bool(fact.statement) and any(fact.statement in part for part in references)
                delivered = bool(inputs) and all(any(fact.statement in part for part in payload)
                                                 for payload in inputs) if visible else False
                if not delivered:
                    issues.append({"code": "fact_not_in_model_input", "key": requirement.key})
                judge_visible = bool(fact.statement) and any(fact.statement in part for part in (judge_references or []))
                judged = bool(judge_inputs) and all(any(fact.statement in part for part in payload)
                                                   for payload in judge_inputs) if judge_visible else False
                if not judged:
                    issues.append({"code": "fact_not_in_judge_input", "key": requirement.key})
        for key, expected in judge_assumptions.items():
            fact = facts.get(key)
            if fact is None or not fact.source or fact.value is None:
                issues.append({"code": "unverified_judge_assumption", "key": key})
            elif type(fact.value) is not type(expected) or fact.value != expected:
                issues.append({"code": "judge_assumption_conflict", "key": key})
    return {"case": case.id, "stage": "model_input" if input_check else "preflight",
            "passed": not issues, "requirements": requirements, "issues": issues}


def audit_bank(cases, facts, judge_assumptions):
    results = [audit_case(case, facts, judge_assumptions) for case in cases]
    unique = len({case.id for case in cases}) == len(cases)
    return {"passed": unique and all(row["passed"] for row in results),
            "unique_case_ids": unique, "case_count": len(cases),
            "blocked_case_count": sum(not row["passed"] for row in results), "cases": results}


def row_precondition_valid(row, case):
    """Missing evidence in older reports cannot silently acquire qualification."""
    result = row.get("case_precondition") or {}
    expected = [REQUIREMENTS[case.id].key] if case.id in REQUIREMENTS else []
    return (row.get("group") == case.group and result.get("case") == case.id and result.get("passed") is True
            and result.get("stage") == "model_input"
            and result.get("requirements") == expected and result.get("issues") == []
            and (case.group != "falsepremise" or bool(expected)))
