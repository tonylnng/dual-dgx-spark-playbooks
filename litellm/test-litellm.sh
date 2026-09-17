#!/usr/bin/env bash
# End-to-end test through LiteLLM.
#
# Usage:
#   export LITELLM_URL=http://<LITELLM_IP>:4000
#   export LITELLM_MASTER_KEY='<master or virtual key>'
#   ./litellm/test-litellm.sh

set -Eeuo pipefail

: "${LITELLM_URL:?Set LITELLM_URL (e.g. http://192.168.1.60:4000)}"
: "${LITELLM_MASTER_KEY:?Set LITELLM_MASTER_KEY}"

MODEL_ALIAS="${MODEL_ALIAS:-qwen3-235b}"

echo "== GET $LITELLM_URL/v1/models"
curl --fail --silent --show-error \
  "$LITELLM_URL/v1/models" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[] | .id'

echo
echo "== POST $LITELLM_URL/v1/chat/completions  (model=$MODEL_ALIAS)"
curl --fail --silent --show-error \
  "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL_ALIAS\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Reply with exactly: LiteLLM route ready\"}
    ],
    \"temperature\": 0,
    \"max_tokens\": 32,
    \"stream\": false
  }" | jq .
