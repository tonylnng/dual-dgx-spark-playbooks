#!/usr/bin/env bash
# Start vLLM inside the Ray head container. Run on HEAD only.
# Logs stream to $LOG_DIR/server.log (bind-mounted at /var/log/vllm).

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

CN="$(node_container)"
[[ -n "$CN" ]] || fatal "No node-* container. Start Ray head first."

[[ "$VLLM_API_KEY" != "REPLACE_ME" && -n "$VLLM_API_KEY" ]] \
  || fatal "Set VLLM_API_KEY in config/cluster.env (openssl rand -hex 32)."

mkdir -p "$LOG_DIR"

# Conservative bring-up profile (Red Hat card TP=2 evaluation profile).
TOOL_FLAGS=""
if [[ "${ENABLE_TOOLS:-false}" == "true" ]]; then
  TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER:-hermes}"
  log "Tool calling ENABLED with parser: ${TOOL_CALL_PARSER:-hermes}"
else
  log "Tool calling disabled for bring-up. Validate ordinary chat first."
fi

log "Starting vLLM on port $VLLM_PORT (model: $MODEL_ID)…"
docker exec -d \
  -e MODEL_ID="$MODEL_ID" \
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  -e VLLM_PORT="$VLLM_PORT" \
  -e VLLM_API_KEY="$VLLM_API_KEY" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  "$CN" \
  bash -lc "
    exec vllm serve \"\$MODEL_ID\" \
      --served-model-name \"\$SERVED_MODEL_NAME\" \
      --host 0.0.0.0 \
      --port \"\$VLLM_PORT\" \
      --api-key \"\$VLLM_API_KEY\" \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --dtype auto \
      --trust-remote-code \
      --max-model-len \"\$MAX_MODEL_LEN\" \
      --max-num-seqs \"\$MAX_NUM_SEQS\" \
      --max-num-batched-tokens \"\$MAX_NUM_BATCHED_TOKENS\" \
      --gpu-memory-utilization \"\$GPU_MEMORY_UTILIZATION\" \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --enforce-eager \
      --generation-config vllm \
      $TOOL_FLAGS \
      --disable-log-requests \
      >> /var/log/vllm/server.log 2>&1
  "

log "vLLM launched. Watch startup:"
log "  tail -f $LOG_DIR/server.log"
log "Wait for 'Application startup complete' before probing /health."
