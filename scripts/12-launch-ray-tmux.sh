#!/usr/bin/env bash
# Start the Ray head (Spark 1) or worker (Spark 2) inside a persistent tmux session.
# The Ray helper's EXIT trap tears the container down if its controlling
# shell exits, so tmux is required.

source "$(dirname "$0")/_lib.sh"
load_env

cd "$WORKDIR"
[[ -x "$WORKDIR/run_cluster.sh" ]] \
  || fatal "run_cluster.sh missing. Run scripts/01-bootstrap.sh first."

case "$ROLE" in
  head)
    SESSION="ray-head"
    MODE_FLAG="--head"
    LOCAL_IP="$HEAD_QSFP_IP"
    RAY_TARGET="$HEAD_QSFP_IP"
    ;;
  worker)
    SESSION="ray-worker"
    MODE_FLAG="--worker"
    LOCAL_IP="$WORKER_QSFP_IP"
    RAY_TARGET="$HEAD_QSFP_IP"
    ;;
  *) fatal "ROLE must be head or worker (got '$ROLE')" ;;
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
  fatal "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
fi

CMD=$(cat <<EOF
set -e
source "$REPO_ROOT/scripts/_lib.sh"
load_env

LOCAL_QSFP_IP=\$(iface_ipv4 "\$MN_IF_NAME")
if [[ "\$LOCAL_QSFP_IP" != "$LOCAL_IP" ]]; then
  echo "QSFP mismatch: \$LOCAL_QSFP_IP != $LOCAL_IP"
  sleep 30
  exit 1
fi

exec bash "\$WORKDIR/run_cluster.sh" \\
  "\$VLLM_IMAGE" \\
  "$RAY_TARGET" \\
  $MODE_FLAG \\
  "\$HF_CACHE" \\
  -v "\$LOG_DIR:/var/log/vllm" \\
  -e VLLM_HOST_IP="$LOCAL_IP" \\
  -e UCX_NET_DEVICES="\$MN_IF_NAME" \\
  -e NCCL_SOCKET_IFNAME="\$MN_IF_NAME" \\
  -e OMPI_MCA_btl_tcp_if_include="\$MN_IF_NAME" \\
  -e GLOO_SOCKET_IFNAME="\$MN_IF_NAME" \\
  -e TP_SOCKET_IFNAME="\$MN_IF_NAME" \\
  -e RAY_memory_monitor_refresh_ms=0 \\
  -e MASTER_ADDR="$HEAD_QSFP_IP"
EOF
)

log "Starting tmux session '$SESSION'…"
tmux new-session -d -s "$SESSION" "bash -lc '$CMD'"

log "Session '$SESSION' launched."
log "Attach:  tmux attach -t $SESSION"
log "Detach:  Ctrl-b then d  (never type 'exit' inside the session)"
