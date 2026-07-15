#!/usr/bin/env bash

# Portable SecurityClaw launcher for Linux, macOS, WSL, and POSIX-compatible shells.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENSEARCH_CONTAINER="${SECURITYCLAW_OPENSEARCH_CONTAINER:-securityclaw-opensearch}"
OPENSEARCH_VOLUME="${SECURITYCLAW_OPENSEARCH_VOLUME:-securityclaw-opensearch-data}"
OPENSEARCH_IMAGE="${SECURITYCLAW_OPENSEARCH_IMAGE:-opensearchproject/opensearch:2}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
APP_URL="${SECURITYCLAW_URL:-http://127.0.0.1:7799}"

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

compose() {
    if docker compose version >/dev/null 2>&1; then docker compose "$@"
    elif has docker-compose; then docker-compose "$@"
    else die "Docker Compose is required for container mode."
    fi
}

wait_for_url() {
    local name="$1" url="$2" attempts="${3:-45}" i
    for ((i = 1; i <= attempts; i++)); do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            printf '%s is ready.\n' "$name"
            return 0
        fi
        sleep 2
    done
    die "$name did not become ready at $url"
}

ensure_docker() {
    has docker || die "Docker is required. Install Docker Engine or Docker Desktop."
    if ! docker info >/dev/null 2>&1; then
        case "$(uname -s 2>/dev/null || true)" in
            Linux*)
                if has systemctl && has sudo; then sudo systemctl start docker || true; fi
                ;;
            Darwin*)
                has open && open -a Docker >/dev/null 2>&1 || true
                ;;
        esac
    fi
    for _ in {1..30}; do docker info >/dev/null 2>&1 && return 0; sleep 2; done
    die "The Docker daemon is not available. Start Docker Desktop or Docker Engine."
}

ensure_opensearch() {
    ensure_docker
    if docker container inspect "$OPENSEARCH_CONTAINER" >/dev/null 2>&1; then
        docker update --restart=no "$OPENSEARCH_CONTAINER" >/dev/null
        if [[ "$(docker inspect -f '{{.State.Running}}' "$OPENSEARCH_CONTAINER")" != "true" ]]; then
            docker start "$OPENSEARCH_CONTAINER" >/dev/null
        fi
    else
        printf 'Creating the minimal OpenSearch dependency...\n'
        docker volume create "$OPENSEARCH_VOLUME" >/dev/null
        docker run -d --name "$OPENSEARCH_CONTAINER" --restart=no \
            -p 127.0.0.1:9200:9200 -p 127.0.0.1:9600:9600 \
            -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
            -e OPENSEARCH_JAVA_OPTS='-Xms1g -Xmx1g' \
            -v "$OPENSEARCH_VOLUME:/usr/share/opensearch/data" "$OPENSEARCH_IMAGE" >/dev/null
    fi
    wait_for_url "OpenSearch" "http://127.0.0.1:9200" 60
}

check_configuration() {
    [[ -f "$ROOT_DIR/config.yaml" ]] || die "config.yaml is missing. Run: .venv/bin/python main.py onboard"
    has curl || die "curl is required."
}

check_ollama() {
    if [[ "$OLLAMA_URL" == http://host.docker.internal* ]]; then return 0; fi
    wait_for_url "Ollama" "${OLLAMA_URL%/}/api/tags" 15
}

run_local() {
    check_configuration
    [[ -x "$ROOT_DIR/.venv/bin/python" ]] || die ".venv is missing. Create it and install requirements.txt first."
    ensure_opensearch
    check_ollama
    printf 'SecurityClaw is available at %s (Ctrl+C stops the application).\n' "$APP_URL"
    exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/main.py" service
}

run_containers() {
    check_configuration
    ensure_opensearch
    check_ollama
    export OLLAMA_BASE_URL OLLAMA_MODEL="${OLLAMA_MODEL:-}" OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-}"
    compose up --build
}

show_status() {
    printf '%-18s %s\n' "Platform" "$(uname -s 2>/dev/null || echo unknown)"
    printf '%-18s %s\n' "Application" "$(curl -fsS --max-time 2 "$APP_URL" >/dev/null 2>&1 && echo online || echo stopped)"
    printf '%-18s %s\n' "Ollama" "$(curl -fsS --max-time 2 "${OLLAMA_URL%/}/api/tags" >/dev/null 2>&1 && echo online || echo unavailable)"
    if has docker && docker container inspect "$OPENSEARCH_CONTAINER" >/dev/null 2>&1; then
        docker inspect -f 'OpenSearch         {{.State.Status}} (restart={{.HostConfig.RestartPolicy.Name}})' "$OPENSEARCH_CONTAINER"
    else
        printf '%-18s %s\n' "OpenSearch" "not created"
    fi
}

show_help() {
    cat <<'HELP'
SecurityClaw portable launcher

Usage: ./securityclaw.sh [command]

  start, local       Run with the local virtual environment.
  docker, containers Run the application through Docker Compose.
  stop               Stop Compose and the managed OpenSearch container.
  status             Show application, Ollama, and OpenSearch availability.
  chat               Open terminal chat.
  skills             List skills compatible with the current platform.
  logs               Follow OpenSearch logs.
  -h, --help          Show this help.

Environment: OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBED_MODEL,
SECURITYCLAW_URL, SECURITYCLAW_OPENSEARCH_CONTAINER,
SECURITYCLAW_OPENSEARCH_VOLUME, SECURITYCLAW_OPENSEARCH_IMAGE.

The launcher never enables a container restart-at-boot policy.
HELP
}

cd "$ROOT_DIR"
case "${1:-start}" in
    start|local) run_local ;;
    docker|containers) run_containers ;;
    stop)
        if has docker; then
            compose down 2>/dev/null || true
            docker stop "$OPENSEARCH_CONTAINER" >/dev/null 2>&1 || true
        fi
        printf 'SecurityClaw dependencies stopped. No restart-at-boot policy was enabled.\n'
        ;;
    status) show_status ;;
    chat) check_configuration; exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/main.py" chat ;;
    skills) check_configuration; exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/main.py" list-skills ;;
    logs) ensure_docker; docker logs --tail=200 -f "$OPENSEARCH_CONTAINER" ;;
    -h|--help|help) show_help ;;
    *) printf 'Usage: %s {start|local|docker|stop|status|chat|skills|logs}\n' "$0" >&2; exit 2 ;;
esac
