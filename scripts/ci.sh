#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/env.sh

uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format --check .
uv run --frozen --no-sync pytest
uv run --frozen --no-sync python scripts/check_secrets.py --tracked-only
if [[ -f compose.yaml ]] && command -v docker >/dev/null && docker compose version >/dev/null; then
    docker compose --env-file /dev/null --profile monitoring config --no-env-resolution -q
fi
