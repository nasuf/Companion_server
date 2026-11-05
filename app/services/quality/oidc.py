"""GitHub workflow identity, pinned to this repository, main and two workflows.

Each workflow controls only its own kind of evaluation job (see quality_ci).
"""

import asyncio
import re
import jwt
from fastapi import Depends, Header, HTTPException
from app.config import settings

ISSUER = "https://token.actions.githubusercontent.com"
AUDIENCE = "companion-quality-evaluation"
_jwks = jwt.PyJWKClient(ISSUER + "/.well-known/jwks", timeout=5, lifespan=300)
# workflow file -> (job kind it may control, events allowed to start it)
WORKFLOWS = {
    "quality-nightly.yml": ("nightly", {"schedule", "workflow_dispatch"}),
    # Manual only: a scheduled billed gate run would compare nothing new.
    "quality-repetition-gate.yml": ("repetition_gate", {"workflow_dispatch"}),
}


def verify(token):
    key = _jwks.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        key.key,
        algorithms=["RS256"],
        issuer=ISSUER,
        audience=AUDIENCE,
        options={
            "require": [
                "exp",
                "iat",
                "nbf",
                "sub",
                "repository",
                "repository_id",
                "ref",
                "sha",
                "run_id",
                "run_attempt",
                "workflow_ref",
                "event_name",
            ]
        },
        leeway=10,
    )
    repo = settings.quality_oidc_repository
    prefix, _, suffix = claims["workflow_ref"].partition("@")
    name = prefix.removeprefix(f"{repo}/.github/workflows/")
    kind, events = WORKFLOWS.get(name, (None, set()))
    if (
        claims["repository"] != repo
        or claims["repository_id"] != settings.quality_oidc_repository_id
        or claims["ref"] != "refs/heads/main"
        or claims["sub"] != f"repo:{repo}:ref:refs/heads/main"
        or kind is None
        or prefix != f"{repo}/.github/workflows/{name}"
        or suffix != "refs/heads/main"
        or claims["event_name"] not in events
        or not re.fullmatch("[a-f0-9]{40}", claims["sha"])
        or not all(str(claims[k]).isdigit() for k in ("run_id", "run_attempt"))
    ):
        raise ValueError("untrusted_quality_workflow")
    return {**claims, "quality_workflow": kind}


async def identity(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer ") or len(authorization) > 16000:
        raise HTTPException(401, "quality_workflow_identity_required")
    try:
        return await asyncio.to_thread(verify, authorization[7:])
    except Exception as exc:
        raise HTTPException(401, "quality_workflow_identity_invalid") from exc


def require(kind):
    """Dependency: a verified identity from the workflow that owns `kind` jobs."""

    async def dependency(claims=Depends(identity)):
        if claims.get("quality_workflow") != kind:
            raise HTTPException(403, "quality_workflow_not_permitted")
        return claims

    return dependency


NIGHTLY = require("nightly")
REPETITION_GATE = require("repetition_gate")


def source_key(claims):
    return "github:" + claims["run_id"] + ":" + claims["run_attempt"]
