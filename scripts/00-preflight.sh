#!/usr/bin/env bash
# Preflight: verify Docker, GPU, network interface, peer connectivity, disk.
# Run on BOTH Sparks.

source "$(dirname "$0")/_lib.sh"
load_env

log "Host: $(hostname)  User: $(whoami)  Role: $ROLE"

command -v docker  >/dev/null || fatal "docker not installed"
command -v nvidia-smi >/dev/null || fatal "nvidia-smi not installed"
command -v tmux    >/dev/null || fatal "tmux not installed (apt install tmux)"
command -v curl    >/dev/null || fatal "curl not installed"
command -v jq      >/dev/null || fatal "jq not installed"
command -v openssl >/dev/null || fatal "openssl not installed"

log "Docker version:"
docker version --format '{{.Server.Version}}' || fatal "Docker daemon unreachable"

log "GPU visibility:"
nvidia-smi -L || fatal "nvidia-smi failed"

log "QSFP interface: $MN_IF_NAME"
ip -4 addr show "$MN_IF_NAME" >/dev/null 2>&1 \
  || fatal "Interface $MN_IF_NAME not found. Run: ibdev2netdev ; ip -4 -o addr show"

LOCAL_QSFP_IP="$(iface_ipv4 "$MN_IF_NAME")"
[[ -n "$LOCAL_QSFP_IP" ]] || fatal "No IPv4 on $MN_IF_NAME"
log "Local QSFP IP: $LOCAL_QSFP_IP"

case "$ROLE" in
  head)
    [[ "$LOCAL_QSFP_IP" == "$HEAD_QSFP_IP" ]] \
      || fatal "Local QSFP IP $LOCAL_QSFP_IP != HEAD_QSFP_IP $HEAD_QSFP_IP"
    PEER_IP="$WORKER_QSFP_IP"
    ;;
  worker)
    [[ "$LOCAL_QSFP_IP" == "$WORKER_QSFP_IP" ]] \
      || fatal "Local QSFP IP $LOCAL_QSFP_IP != WORKER_QSFP_IP $WORKER_QSFP_IP"
    PEER_IP="$HEAD_QSFP_IP"
    ;;
  *) fatal "ROLE must be head or worker (got '$ROLE')" ;;
esac

log "Pinging peer $PEER_IP over QSFP…"
ping -c 3 -W 2 "$PEER_IP" >/dev/null || fatal "Peer $PEER_IP unreachable over QSFP"

log "Passwordless SSH to peer $PEER_IP…"
ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
    "$PEER_IP" hostname \
  || warn "Passwordless SSH to peer failed (only needed if you use SSH-based helpers)"

log "Port 8000 availability:"
if ss -lntp 2>/dev/null | grep -q ":${VLLM_PORT} "; then
  warn "Port ${VLLM_PORT} already listening — stop the previous vLLM process before Phase 10"
else
  log "Port ${VLLM_PORT} free"
fi

log "Disk space for HF cache ($HF_CACHE):"
df -h "$HF_CACHE"

log "Preflight OK on $(hostname) (role=$ROLE)"
