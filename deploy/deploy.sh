#!/usr/bin/env bash
# Runs ON the EC2 host (sent by .github/workflows/deploy.yml via SSM, as root).
# Usage: deploy.sh <ecr-image-ref>   expects docker-compose.yml + docker-compose.aws.yml next to it.
# Pulls the image, restarts, waits for /health, rolls back to the previous image on failure.
set -euo pipefail
IMAGE="${1:?image ref}"
REGION="${AWS_REGION:-us-east-1}"
cd /opt/nutribot

# First deploy on a fresh host: install the Compose v2 plugin if Docker lacks it.
if ! docker compose version >/dev/null 2>&1; then
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -fsSL "https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# .env is the single source of runtime config: a raw dotenv stored in Secrets Manager (one-time setup).
aws secretsmanager get-secret-value --region "$REGION" --secret-id nutribot/dotenv \
  --query SecretString --output text > .env
chmod 600 .env

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${IMAGE%%/*}"

PREV=$(cat .current_image 2>/dev/null || true)
up() { NUTRIBOT_IMAGE="$1" docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d --pull always --no-build; }

up "$IMAGE"
for _ in $(seq 1 45); do   # up to ~90 s (model load)
  if curl -fsS http://localhost/health >/dev/null; then
    echo "$IMAGE" > .current_image
    echo "healthy: $IMAGE"
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done

echo "unhealthy after deploy: $IMAGE" >&2
docker compose logs --tail 50 nutribot >&2 || true
if [ -n "$PREV" ]; then
  echo "rolling back to $PREV" >&2
  up "$PREV"
fi
exit 1
