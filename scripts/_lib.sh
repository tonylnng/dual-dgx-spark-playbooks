# shellcheck shell=bash
# Common helpers for the dual-DGX-Spark playbook scripts.
# Source this file at the top of every script.

set -Eeuo pipefail

# Resolve the repository root regardless of invocation cwd
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LIB_DIR/.." && pwd)"

log()   { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }
warn()  { printf '[%s] WARN: %s\n' "$(date +'%H:%M:%S')" "$*" >&2; }
fatal() { printf '[%s] ERROR: %s\n' "$(date +'%H:%M:%S')" "$*" >&2; exit 1; }

load_env() {
  local env_file="${1:-$REPO_ROOT/config/cluster.env}"
  if [[ ! -f "$env_file" ]]; then
    fatal "Missing $env_file. Copy config/cluster.env.example and edit it."
  fi
  # shellcheck disable=SC1090
  source "$env_file"

  : "${ROLE:?ROLE must be set (head|worker)}"
  : "${MN_IF_NAME:?MN_IF_NAME must be set}"
  : "${HEAD_QSFP_IP:?HEAD_QSFP_IP must be set}"
  : "${WORKER_QSFP_IP:?WORKER_QSFP_IP must be set}"
  : "${VLLM_IMAGE:?VLLM_IMAGE must be set}"
  : "${MODEL_ID:?MODEL_ID must be set}"
  : "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
  : "${VLLM_PORT:?VLLM_PORT must be set}"
  : "${WORKDIR:?WORKDIR must be set}"
  : "${HF_CACHE:?HF_CACHE must be set}"
  : "${LOG_DIR:?LOG_DIR must be set}"

  mkdir -p "$WORKDIR" "$LOG_DIR" "$HF_CACHE"
}

# Print the vLLM container name inside the Ray helper (node-0, node-1, …)
node_container() {
  docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$' | head -1
}

require_role() {
  local wanted="$1"
  if [[ "${ROLE:-}" != "$wanted" ]]; then
    fatal "This script must run on the '$wanted' node (ROLE=$ROLE)"
  fi
}

# Return the first IPv4 address bound to the given interface
iface_ipv4() {
  local iface="$1"
  ip -4 addr show "$iface" 2>/dev/null \
    | grep -oP '(?<=inet\s)\d+(\.\d+){3}' \
    | head -1
}
