"""Deterministic graders for Companion agent evals.

The graders intentionally avoid LLM calls. They are meant to catch high-signal
regressions in CI and local smoke runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssertionResult:
    passed: bool
    assertion_type: str
    message: str


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _contains_any(reply: str, values: list[str]) -> bool:
    haystack = normalize_text(reply)
    return any(normalize_text(value) in haystack for value in values)


def evaluate_assertion(reply: str, assertion: dict[str, Any]) -> AssertionResult:
    assertion_type = str(assertion.get("type") or "").strip()
    if not assertion_type:
        return AssertionResult(False, "", "assertion is missing type")

    if assertion_type == "must_contain":
        value = str(assertion.get("value") or "")
        passed = normalize_text(value) in normalize_text(reply)
        return AssertionResult(
            passed,
            assertion_type,
            f"expected reply to contain {value!r}",
        )

    if assertion_type == "must_not_contain":
        value = str(assertion.get("value") or "")
        passed = normalize_text(value) not in normalize_text(reply)
        return AssertionResult(
            passed,
            assertion_type,
            f"expected reply not to contain {value!r}",
        )

    if assertion_type == "should_contain_any":
        values = [str(v) for v in assertion.get("values") or [] if str(v)]
        passed = _contains_any(reply, values)
        return AssertionResult(
            passed,
            assertion_type,
            f"expected reply to contain one of {values!r}",
        )

    if assertion_type == "must_not_contain_any":
        values = [str(v) for v in assertion.get("values") or [] if str(v)]
        passed = not _contains_any(reply, values)
        return AssertionResult(
            passed,
            assertion_type,
            f"expected reply not to contain any of {values!r}",
        )

    if assertion_type == "max_chars":
        try:
            limit = int(assertion.get("value"))
        except (TypeError, ValueError):
            return AssertionResult(False, assertion_type, "max_chars requires integer value")
        passed = len(reply) <= limit
        return AssertionResult(
            passed,
            assertion_type,
            f"expected reply length <= {limit}, got {len(reply)}",
        )

    return AssertionResult(False, assertion_type, f"unknown assertion type {assertion_type!r}")


def grade_reply(reply: str, assertions: list[dict[str, Any]]) -> dict[str, Any]:
    results = [evaluate_assertion(reply, assertion) for assertion in assertions]
    failed = [r for r in results if not r.passed]
    return {
        "passed": not failed,
        "n_assertions": len(results),
        "n_failed": len(failed),
        "failures": [
            {
                "type": r.assertion_type,
                "message": r.message,
            }
            for r in failed
        ],
    }


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("id must be a non-empty string")
    if case.get("priority") not in {"P0", "P1", "P2"}:
        errors.append("priority must be P0, P1, or P2")
    if not isinstance(case.get("category"), str) or not case.get("category"):
        errors.append("category must be a non-empty string")
    grade_target = case.get("grade_target", "all_replies")
    if grade_target not in {"all_replies", "last_reply"}:
        errors.append("grade_target must be all_replies or last_reply")

    turns = case.get("turns")
    if not isinstance(turns, list) or not turns:
        errors.append("turns must be a non-empty list")
    else:
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                errors.append(f"turns[{index}] must be an object")
                continue
            if turn.get("role") != "user":
                errors.append(f"turns[{index}].role must be 'user'")
            if not isinstance(turn.get("content"), str) or not turn.get("content").strip():
                errors.append(f"turns[{index}].content must be a non-empty string")

    assertions = case.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        errors.append("assertions must be a non-empty list")
    else:
        for index, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                errors.append(f"assertions[{index}] must be an object")
                continue
            assertion_type = assertion.get("type")
            if assertion_type in {"must_contain", "must_not_contain"}:
                if not isinstance(assertion.get("value"), str) or not assertion["value"].strip():
                    errors.append(f"assertions[{index}].value must be a non-empty string")
            if assertion_type in {"should_contain_any", "must_not_contain_any"}:
                values = assertion.get("values")
                if (
                    not isinstance(values, list)
                    or not values
                    or not all(isinstance(value, str) and value.strip() for value in values)
                ):
                    errors.append(f"assertions[{index}].values must be a non-empty string list")
            if assertion_type == "max_chars":
                try:
                    limit = int(assertion.get("value"))
                except (TypeError, ValueError):
                    errors.append(f"assertions[{index}].value must be an integer")
                else:
                    if limit <= 0:
                        errors.append(f"assertions[{index}].value must be positive")
            result = evaluate_assertion("", assertion)
            if result.assertion_type == "" or result.message.startswith("unknown assertion type"):
                errors.append(f"assertions[{index}] has invalid type")

    return errors
