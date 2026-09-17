#!/usr/bin/env bash
# Local acceptance tests: /health, /v1/models, and a short chat completion.
# Run on HEAD.

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

BASE="http://127.0.0.1:${VLLM_PORT}"

log "GET $BASE/health"
curl --fail --silent --show-error "$BASE/health" && echo

log "GET $BASE/v1/models"
curl --fail --silent --show-error \
  "$BASE/v1/models" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" | jq .

log "POST $BASE/v1/chat/completions"
curl --fail --silent --show-error \
  "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a concise enterprise AI assistant.\"},
      {\"role\": \"user\", \"content\": \"Reply with exactly: DGX cluster ready\"}
    ],
    \"temperature\": 0,
    \"max_tokens\": 32,
    \"stream\": false
  }" | jq .

log "Smoke test passed."
