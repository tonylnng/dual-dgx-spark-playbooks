#!/usr/bin/env bash
# Quick monitoring snapshot: Ray, processes, port, GPU, tail of server log.
# Run on HEAD.

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

CN="$(node_container)"
[[ -n "$CN" ]] || fatal "No node-* container found."

echo "=== ray status ==="
docker exec "$CN" ray status || true

echo
echo "=== processes (ray + vllm) ==="
docker exec "$CN" pgrep -af 'ray|vllm' || true

echo
echo "=== port $VLLM_PORT ==="
ss -lntp | grep ":${VLLM_PORT} " || echo "not listening"

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== server.log (last 100) ==="
tail -n 100 "$LOG_DIR/server.log" 2>/dev/null || echo "no log yet"
