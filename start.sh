#!/usr/bin/env bash
# ============================================================
# RLVD Full Stack — start everything with Docker
# Usage:
#   ./start.sh            # build + up all services
#   ./start.sh --no-build # skip image rebuild
#   ./start.sh down       # stop + remove containers
#   ./start.sh logs       # tail all logs
#   ./start.sh status     # show container status
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.full.yml"
ACTION="${1:-up}"
NO_BUILD=false
[[ "${2:-}" == "--no-build" ]] && NO_BUILD=true
[[ "${1:-}" == "--no-build" ]] && { ACTION="up"; NO_BUILD=true; }

# ---------- colors ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- helpers ----------
check_prereqs() {
    command -v docker >/dev/null 2>&1 || { err "docker not found"; exit 1; }
    docker compose version >/dev/null 2>&1 || { err "docker compose plugin not found"; exit 1; }
    docker info >/dev/null 2>&1 || { err "Docker daemon not running"; exit 1; }
    ok "Docker ready"
}

check_models() {
    local missing=0
    for m in edge_node/models/yolo26m.pt; do
        [[ -f "$m" ]] || { err "Missing model: $m"; missing=1; }
    done
    [[ $missing -eq 0 ]] && ok "Models present" || exit 1
}

check_video() {
    local vid="${VIDEO_INPUT:-edge_node/data/videos/traffic_video_modified.mp4}"
    if [[ ! -f "$vid" ]]; then
        warn "Video not found: $vid"
        warn "Set VIDEO_INPUT env var or place a video there."
        warn "Edge pipeline will fail to start without a valid video."
    else
        ok "Video: $vid"
    fi
}

load_env() {
    # Source central_server/.env for GEMINI_API_KEY etc. (secrets stay local)
    if [[ -f central_server/.env ]]; then
        set -a
        # shellcheck disable=SC1091
        source central_server/.env
        set +a
        ok "Loaded central_server/.env"
    else
        warn "central_server/.env not found — GEMINI_API_KEY will be empty"
    fi
}

wait_healthy() {
    local url="$1" name="$2" timeout="${3:-90}"
    info "Waiting for $name ..."
    local elapsed=0 code
    while (( elapsed < timeout )); do
        # Any HTTP response (even 503 "degraded") means the service is listening.
        # code 000 = connection refused / not up yet.
        code=$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo 000)
        if [[ "$code" != "000" ]]; then
            ok "$name is up (HTTP $code)"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    warn "$name did not respond within ${timeout}s (may still be starting)"
    return 1
}

print_summary() {
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  RLVD Stack is running${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Web Dashboard${NC}      http://localhost:3000"
    echo -e "  ${CYAN}Central Server${NC}     http://localhost:8002/api/health"
    echo -e "  ${CYAN}Edge Pipeline API${NC}  http://localhost:8082/health"
    echo -e "  ${CYAN}Camera HLS${NC}         http://localhost:8082/api/cameras"
    echo -e "  ${CYAN}MinIO Console${NC}      http://localhost:9011  (minioadmin/minioadmin)"
    echo ""
    echo -e "  ${YELLOW}Logs:${NC}    ./start.sh logs"
    echo -e "  ${YELLOW}Status:${NC}  ./start.sh status"
    echo -e "  ${YELLOW}Stop:${NC}    ./start.sh down"
    echo ""
}

# ---------- actions ----------
do_up() {
    check_prereqs
    check_models
    check_video
    load_env

    info "Building & starting full stack ..."
    local build_flag="--build"
    [[ "$NO_BUILD" == true ]] && build_flag="--no-build"

    docker compose -f "$COMPOSE_FILE" up -d $build_flag 2>&1 | tail -20

    echo ""
    wait_healthy "http://localhost:8002/api/health" "Central Server" 120
    wait_healthy "http://localhost:3000" "Web Dashboard" 90
    wait_healthy "http://localhost:8082/health" "Edge Pipeline" 120

    print_summary
}

do_down() {
    info "Stopping all services ..."
    docker compose -f "$COMPOSE_FILE" down
    ok "Stack stopped"
}

do_logs() {
    docker compose -f "$COMPOSE_FILE" logs -f --tail=50
}

do_status() {
    docker compose -f "$COMPOSE_FILE" ps
}

# ---------- dispatch ----------
case "$ACTION" in
    up)     do_up ;;
    down)   do_down ;;
    logs)   do_logs ;;
    status) do_status ;;
    *)
        echo "Usage: $0 [up|down|logs|status] [--no-build]"
        exit 1
        ;;
esac
