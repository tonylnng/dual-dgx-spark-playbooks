#!/usr/bin/env bash
# Download NVIDIA's pinned two-node run_cluster.sh, patch it, and pull the vLLM image.
# Run on BOTH Sparks.

source "$(dirname "$0")/_lib.sh"
load_env

cd "$WORKDIR"

RUN_CLUSTER_URL="https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh"

log "Downloading pinned run_cluster.sh…"
curl -fsSL "$RUN_CLUSTER_URL" -o run_cluster.sh

log "Applying NVIDIA Ray installation patch…"
sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' \
    run_cluster.sh

chmod 700 run_cluster.sh

log "Pulling vLLM image: $VLLM_IMAGE"
docker pull "$VLLM_IMAGE"

log "Recording image digest…"
docker image inspect "$VLLM_IMAGE" \
    --format '{{index .RepoDigests 0}}' | tee vllm-image-digest.txt

log "Bootstrap complete on $(hostname)"
