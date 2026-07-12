#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${IMAGE:-ghcr.io/lee101/omniserve:latest}"
cog build -t "$IMAGE"
if [ "${PUSH:-1}" = "1" ]; then
  docker push "$IMAGE"
fi
echo "built $IMAGE"
