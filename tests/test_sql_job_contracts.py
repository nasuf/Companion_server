import math
import pytest
from app.services.runtime.sql_job_contracts import HandlerSpec, LeasePolicy


def spec(**changes):
    return HandlerSpec(
        **{
            "name": "chat.execute.v1",
            "job_key": "chat:0",
            "versions": (("v1", 1),),
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"lease_seconds": 0},
        {"heartbeat_seconds": 30},
        {"heartbeat_seconds": math.nan},
        {"lease_seconds": True},
        {"retry_seconds": -1},
        {"scan_limit": True},
        {"scan_limit": 257},
        {"cancellation_seconds": math.inf},
    ],
)
def test_policy_rejects_unbounded_or_unsafe_timing(changes):
    with pytest.raises(ValueError):
        LeasePolicy(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": " x"},
        {"job_key": "\x00"},
        {"versions": []},
        {"versions": (("v1", True),)},
        {"versions": (("v1", 1), ("v1", 1))},
        {"payload_version": 0},
        {"retry_safe": 1},
        {"max_execution_seconds": math.inf},
        {"kind": "provider_payload"},
        {"executor": "unregistered"},
    ],
)
def test_handler_requires_explicit_compatible_versions_and_retry_authority(changes):
    with pytest.raises(ValueError):
        spec(**changes)


def test_default_retry_is_fail_closed_and_policy_matches_roadmap():
    assert not spec().retry_safe
    assert (LeasePolicy().lease_seconds, LeasePolicy().heartbeat_seconds) == (60, 15)
