#!/usr/bin/env bash
# Stop the local Ray node. On HEAD, stop vLLM first with 41-stop-vllm.sh.
# The Ray helper's EXIT trap tears the container down when tmux ends.

source "$(dirname "$0")/_lib.sh"
load_env

case "$ROLE" in
  head)   SESSION="ray-head"   ;;
  worker) SESSION="ray-worker" ;;
  *) fatal "ROLE must be head or worker" ;;
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "Killing tmux session '$SESSION' (helper EXIT trap will remove the container)…"
  tmux kill-session -t "$SESSION"
else
  warn "tmux session '$SESSION' not found. Nothing to stop."
fi

log "If a stale node-* container remains, remove it manually:"
log "  docker ps --filter name=^node- --format '{{.Names}}' | xargs -r docker rm -f"
