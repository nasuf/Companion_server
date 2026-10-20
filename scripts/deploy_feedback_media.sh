# Sourced with set -euo pipefail and DOCKER set by the deployment workflow.
# Stop uploads/deletions before copying old container-only screenshots.
FEEDBACK_CHECK=0
python3 scripts/persist_feedback_media.py --docker "$DOCKER" \
  --destination /mnt/datadisk0/companion/feedback_media --check || FEEDBACK_CHECK=$?
if [ "$FEEDBACK_CHECK" -eq 1 ]; then
  $DOCKER compose -f docker-compose.deploy.yml stop server
elif [ "$FEEDBACK_CHECK" -ne 0 ]; then
  exit "$FEEDBACK_CHECK"
fi
python3 scripts/persist_feedback_media.py --docker "$DOCKER" \
  --destination /mnt/datadisk0/companion/feedback_media
