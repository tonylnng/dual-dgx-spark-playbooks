#!/usr/bin/env bash
# Verify: exactly two live Ray nodes and exactly two GPU resources.
# Run on the HEAD node only.

source "$(dirname "$0")/_lib.sh"
load_env
require_role head

CN="$(node_container)"
[[ -n "$CN" ]] || fatal "No node-* container found. Is the Ray head running in tmux?"

log "Ray status:"
docker exec "$CN" ray status || fatal "ray status failed"

log "Strict node/GPU assertion…"
docker exec "$CN" python - <<'PY'
import ray, sys
ray.init(address="auto")
nodes = [n for n in ray.nodes() if n["Alive"]]
resources = ray.cluster_resources()
print("Live nodes:", len(nodes))
print("Resources:", resources)
print("Addresses:", [n["NodeManagerAddress"] for n in nodes])
if len(nodes) != 2:
    print("FAIL: expected exactly two live Ray nodes", file=sys.stderr)
    sys.exit(1)
if int(resources.get("GPU", 0)) != 2:
    print("FAIL: expected exactly two GPUs", file=sys.stderr)
    sys.exit(1)
print("OK: 2 nodes / 2 GPUs")
PY

log "Cluster ready."
