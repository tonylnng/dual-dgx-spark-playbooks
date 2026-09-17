#!/usr/bin/env bash
# Non-streaming tool-call test. Only meaningful when ENABLE_TOOLS=true was
# set for scripts/40-start-vllm.sh.

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

if [[ "${ENABLE_TOOLS:-false}" != "true" ]]; then
  warn "ENABLE_TOOLS=false; this test will not exercise tool parsing."
fi

BASE="http://127.0.0.1:${VLLM_PORT}"

curl --fail --silent --show-error \
  "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is the GPU temperature on spark-1? Use the tool.\"}
    ],
    \"tools\": [
      {
        \"type\": \"function\",
        \"function\": {
          \"name\": \"get_gpu_temperature\",
          \"description\": \"Get the GPU temperature for a host\",
          \"parameters\": {
            \"type\": \"object\",
            \"properties\": {\"hostname\": {\"type\": \"string\"}},
            \"required\": [\"hostname\"]
          }
        }
      }
    ],
    \"tool_choice\": \"auto\",
    \"temperature\": 0,
    \"max_tokens\": 256,
    \"stream\": false
  }" | jq .
