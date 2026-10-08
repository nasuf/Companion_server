"""Explicit process contracts. Importing a role never starts work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoleContract:
    name: str
    serves_api: bool
    starts_scheduler: bool
    consumes_redis_jobs: bool


ROLES = {
    "integrated": RoleContract("integrated", True, True, True),
    "api": RoleContract("api", True, False, False),
    "scheduler": RoleContract("scheduler", False, True, False),
    "background": RoleContract("background", False, False, True),
}


def require_role(name: str, *, api: bool) -> RoleContract:
    role = ROLES.get(name)
    if role is None or role.serves_api != api:
        raise ValueError("Runtime role does not match this process entry point")
    return role


def validate_process_budget(settings) -> None:
    """Never silently multiply provider capacity when adding non-API processes."""
    count = settings.llm_process_count
    if count is not None and (type(count) is not int or count < settings.web_concurrency):
        raise ValueError("LLM_PROCESS_COUNT must cover all API workers")
    if settings.app_runtime_role in {"scheduler", "background"}:
        if count is None or count <= settings.web_concurrency:
            raise ValueError("Independent roles require an explicit total LLM_PROCESS_COUNT")
