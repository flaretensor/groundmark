#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/env.sh
export DOCKER_CONFIG="$PWD/.cache/docker"
mkdir -p "$DOCKER_CONFIG"

compose=(docker compose --project-name groundmark)
ports=.cache/compose-ports.env

case "${1:-}" in
    up)
        python3 scripts/compose_env.py
        if ! docker info --format '{{.DockerRootDir}}' >/dev/null 2>&1; then
            docker compose config -q
            printf '%s\n' 'Docker daemon unavailable' >&2
            exit 1
        fi
        mkdir -p -m 700 .artifacts .cache/compose-runtime "$HF_HOME"
        ss -ltn >/dev/null
        if [[ ! -f $ports ]] || [[ -z $(docker compose --project-name groundmark ps -q --status running) ]]; then
            declare -A selected=()
            pick_port() {
                local name="$1" port="$2"
                while [[ -n "$(ss -H -ltn "sport = :$port")" || -v selected[$port] ]]; do
                    port=$((port + 1))
                done
                selected[$port]=1
                printf '%s=%s\n' "$name" "$port"
            }
            {
                printf 'GROUNDMARK_UID=%s\nGROUNDMARK_GID=%s\n' "$(id -u)" "$(id -g)"
                pick_port GROUNDMARK_API_PORT 8100
                pick_port GROUNDMARK_UI_PORT 8601
                pick_port GROUNDMARK_MLFLOW_PORT 5100
                pick_port GROUNDMARK_POSTGRES_PORT 5433
                pick_port GROUNDMARK_PROMETHEUS_PORT 9091
                pick_port GROUNDMARK_GRAFANA_PORT 3100
            } > "$ports"
        fi
        compose+=(--env-file .env.compose --env-file "$ports")
        if [[ "$#" == 2 && "$2" == "--monitoring" ]]; then
            compose+=(--profile monitoring)
        fi
        exec "${compose[@]}" up -d --build --wait --wait-timeout 150
        ;;
    down|status)
        [[ ! -f .env.compose ]] || compose+=(--env-file .env.compose)
        [[ ! -f $ports ]] || compose+=(--env-file "$ports")
        compose+=(--profile monitoring)
        if [[ "$1" == down ]]; then
            exec "${compose[@]}" down
        fi
        exec "${compose[@]}" ps
        ;;
    *)
        printf '%s\n' 'Usage: scripts/compose.sh up [--monitoring] | down | status' >&2
        exit 2
        ;;
esac
