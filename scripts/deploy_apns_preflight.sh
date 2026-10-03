#!/usr/bin/env bash
# Sourced by deploy.yml so CANDIDATE_IMAGE remains available for startup checks.
# The caller sets DOCKER and uses set -euo pipefail.
: "${DOCKER:?DOCKER must be set by the deployment caller}"

# Keep the previous image referenced by a stopped, networkless
# container so image prune -a cannot remove the rollback target.
PREVIOUS_IMAGE="$($DOCKER inspect --format '{{.Image}}' companion-server 2>/dev/null || true)"
if [ -n "$PREVIOUS_IMAGE" ]; then
  RETENTION_CONTAINER="companion-server-rollback-retention"
  if $DOCKER container inspect "$RETENTION_CONTAINER" >/dev/null 2>&1; then
    test "$($DOCKER inspect --format '{{index .Config.Labels "com.companion.rollback"}}' "$RETENTION_CONTAINER")" = "true"
    $DOCKER rm "$RETENTION_CONTAINER"
  fi
  $DOCKER create --name "$RETENTION_CONTAINER" --network none \
    --label com.companion.rollback=true --entrypoint /bin/true "$PREVIOUS_IMAGE"
  echo "Rollback image: $PREVIOUS_IMAGE"
fi

echo "==> Building server image"
$DOCKER compose -f docker-compose.deploy.yml build server
CANDIDATE_IMAGE="$($DOCKER image inspect --format '{{.Id}}' companion-server:deploy)"
echo "==> Checking the exact candidate image before stopping production: $CANDIDATE_IMAGE"
$DOCKER run --rm --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --entrypoint python "$CANDIDATE_IMAGE" -m scripts.check_apns_log_redaction
