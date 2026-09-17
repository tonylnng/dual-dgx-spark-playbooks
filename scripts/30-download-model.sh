#!/usr/bin/env bash
# Download the model into the local Hugging Face cache inside the running
# vLLM container. Run this on BOTH Sparks — each host has its own cache.

source "$(dirname "$0")/_lib.sh"
load_env

CN="$(node_container)"
[[ -n "$CN" ]] || fatal "No node-* container found. Start Ray first."

[[ "$HF_TOKEN" != "hf_REPLACE_ME" && -n "$HF_TOKEN" ]] \
  || fatal "Set HF_TOKEN in config/cluster.env before downloading."

log "Downloading $MODEL_ID inside container $CN…"
docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_ID="$MODEL_ID" \
  "$CN" \
  bash -lc 'hf download "$MODEL_ID"'

log "Cache size on $(hostname):"
du -sh "$HF_CACHE/hub/models--$(echo "$MODEL_ID" | tr '/' '-' | sed 's/^/models--/;s/-/--/2')" 2>/dev/null \
  || du -sh "$HF_CACHE/hub/" 2>/dev/null || true

log "Model download complete on $(hostname)."
