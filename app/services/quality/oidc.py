"""GitHub workflow identity, pinned to this repository, main and one workflow."""

import asyncio
import re
import jwt
from fastapi import Header, HTTPException
from app.config import settings

ISSUER = "https://token.actions.githubusercontent.com"
AUDIENCE = "companion-quality-evaluation"
_jwks = jwt.PyJWKClient(ISSUER + "/.well-known/jwks", timeout=5, lifespan=300)


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
    expected = f"{repo}/.github/workflows/quality-nightly.yml@refs/heads/main"
    if (
        claims["repository"] != repo
        or claims["repository_id"] != settings.quality_oidc_repository_id
        or claims["ref"] != "refs/heads/main"
        or claims["sub"] != f"repo:{repo}:ref:refs/heads/main"
        or claims["workflow_ref"] != expected
        or claims["event_name"] not in {"schedule", "workflow_dispatch"}
        or not re.fullmatch("[a-f0-9]{40}", claims["sha"])
        or not all(str(claims[k]).isdigit() for k in ("run_id", "run_attempt"))
    ):
        raise ValueError("untrusted_quality_workflow")
    return claims


async def identity(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer ") or len(authorization) > 16000:
        raise HTTPException(401, "quality_workflow_identity_required")
    try:
        return await asyncio.to_thread(verify, authorization[7:])
    except Exception as exc:
        raise HTTPException(401, "quality_workflow_identity_invalid") from exc


def source_key(claims):
    return "github:" + claims["run_id"] + ":" + claims["run_attempt"]
