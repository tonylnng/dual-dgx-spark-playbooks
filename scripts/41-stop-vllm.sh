#!/usr/bin/env bash
# Stop the vLLM process inside the Ray head container without stopping Ray itself.

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

CN="$(node_container)"
[[ -n "$CN" ]] || fatal "No node-* container found."

log "Sending SIGTERM to vllm serve…"
docker exec "$CN" bash -lc "pkill -TERM -f 'vllm serve' || true"

for i in {1..30}; do
  if ! docker exec "$CN" pgrep -af 'vllm serve' >/dev/null 2>&1; then
    log "vLLM stopped after ${i} check(s)."
    exit 0
  fi
  sleep 2
done

warn "vLLM still running after 60s. Sending SIGKILL…"
docker exec "$CN" bash -lc "pkill -KILL -f 'vllm serve' || true"
log "vLLM force-stopped."
