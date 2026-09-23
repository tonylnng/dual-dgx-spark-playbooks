#!/usr/bin/env bash
# Load testing for the 2x DGX Spark vLLM pair -- the two questions people actually
# ask about a local model, from one entrypoint:
#
#   ./load-test.sh check     is the pair healthy and safe to test right now?
#   ./load-test.sh load      how long does the model take to load, and what did
#                            each phase cost? (reads logs; restarts nothing)
#   ./load-test.sh bench     how much traffic can it serve? (concurrency ladder,
#                            TTFT / tok/s / KV pressure, client and engine side)
#   ./load-test.sh all       check -> load -> bench
#
# NON-DESTRUCTIVE BY CONSTRUCTION. Nothing here stops, restarts, or re-creates a
# container, and nothing writes to /proc/sys/vm/drop_caches -- on a GB10 with
# 121 GiB of unified memory holding 65 GiB of live weights, a page-cache drop is
# not a neutral act. `load` measures the boot that already happened by reading its
# log. To get a genuinely cold number you must restart the pair yourself;
# `./load-test.sh load --cmd` prints exactly that command instead of running it.
#
# Everything is read from dual-spark/dual.env, so it follows the same config as
# run.sh / verify.sh / status.sh.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/dual-spark/dual.env"
# shellcheck source=/dev/null
source "$HERE/dual-spark/_common.sh"

BASE="${BASE:-http://127.0.0.1:${HOST_PORT}}"
AUTH="Authorization: Bearer ${API_KEY}"
SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$WORKER")
rsh(){ timeout 40 "${SSH[@]}" "$@"; }
LT="$HERE/loadtest"
# DOCKER is deliberately NOT defaulted here: resolve_docker() in _common.sh picks
# 'ssh localhost docker' when this process has a stale docker group, and presetting
# the variable would make that fallback unreachable.

usage(){ sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- status accumulation for `check` ------------------------------------------
FAILS=0; WARNS=0
# Colour only on a terminal: escape codes inside the padded column would otherwise
# corrupt alignment in logs and CI captures. NO_COLOR is honoured.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; B=$'\033[0m'
else
  G=; Y=; R=; B=
fi
ok(){   printf ' %s%-6s%s %-34s %s\n' "$G" PASS "$B" "$1" "$2"; }
warn(){ printf ' %s%-6s%s %-34s %s\n' "$Y" WARN "$B" "$1" "$2"; WARNS=$((WARNS+1)); }
fail(){ printf ' %s%-6s%s %-34s %s\n' "$R" FAIL "$B" "$1" "$2"; FAILS=$((FAILS+1)); }
note(){ printf ' %-41s %s\n' "     $1" "$2"; }

# ==============================================================================
check(){
  echo "══════════════════════════════════════════════════════════════════════════════"
  echo " PREFLIGHT — ${SERVED_NAME} @ ${BASE}"
  echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "══════════════════════════════════════════════════════════════════════════════"
  printf ' %-7s %-34s %s\n' "STATE" "CHECK" "DETAIL"
  echo "------------------------------------------------------------------------"

  if ! resolve_docker; then
    fail "docker reachable" "neither direct socket nor 'ssh localhost docker' works"
  else
    ok "docker reachable" "${DOCKER%% *}"
    local st
    st=$($DOCKER inspect -f '{{.State.Status}} since={{.State.StartedAt}} restart={{.HostConfig.RestartPolicy.Name}}' \
         "$CONTAINER" 2>/dev/null)
    case "$st" in
      running*) ok "head container" "${st/running/running}"
                [ "${st#*restart=}" = "unless-stopped" ] || \
                  warn "head restart policy" "not unless-stopped -- a reboot leaves rank 0 down (§17)" ;;
      "")       fail "head container" "$CONTAINER does not exist" ;;
      *)        fail "head container" "$st" ;;
    esac
  fi

  # The pair, not the container: a half-up TP2 cluster looks exactly like a hang
  # from the API side, and this is the check that catches it in one line.
  local wst
  wst=$(rsh "docker inspect -f '{{.State.Status}}' $CONTAINER" 2>/dev/null)
  case "$wst" in
    running) ok "worker container" "$WORKER running" ;;
    "")      warn "worker container" "cannot read state over ssh ($WORKER)" ;;
    *)       fail "worker container" "$WORKER: $wst" ;;
  esac

  local code models
  code=$(curl -s -o /tmp/lt_models.json -w '%{http_code}' -m 8 -H "$AUTH" "$BASE/v1/models" 2>/dev/null)
  if [ "$code" = 200 ]; then
    models=$(python3 -c "import json;print(','.join(m['id'] for m in json.load(open('/tmp/lt_models.json'))['data']))" 2>/dev/null)
    [ "$models" = "$SERVED_NAME" ] && ok "API serving" "model '$models'" \
      || warn "API serving" "serving '$models', dual.env says '$SERVED_NAME'"
  else
    fail "API serving" "GET /v1/models -> ${code:-no answer}"
  fi

  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 "$BASE/v1/models" 2>/dev/null)
  [ "$code" = 401 ] && ok "auth enforced" "401 without a key" \
                    || warn "auth enforced" "unauthenticated GET returned ${code:-no answer}, expected 401"

  if curl -s -m 8 -H "$AUTH" "$BASE/metrics" -o /tmp/lt_metrics.txt 2>/dev/null; then
    local run wait kv preempt
    run=$(awk '$1 ~ /^vllm:num_requests_running/ {s+=$NF} END{printf "%.0f", s+0}' /tmp/lt_metrics.txt)
    wait=$(awk '$1 ~ /^vllm:num_requests_waiting/ && $1 !~ /by_reason/ {s+=$NF} END{printf "%.0f", s+0}' /tmp/lt_metrics.txt)
    kv=$(awk '$1 ~ /^vllm:kv_cache_usage_perc/ {printf "%.0f", $NF*100}' /tmp/lt_metrics.txt)
    preempt=$(awk '$1 ~ /^vllm:num_preemptions_total/ {s+=$NF} END{printf "%.0f", s+0}' /tmp/lt_metrics.txt)
    if [ "${run:-0}" -gt 0 ] || [ "${wait:-0}" -gt 0 ]; then
      warn "engine idle" "$run running / $wait waiting -- someone is using it; bench will refuse without --force"
    else
      ok "engine idle" "0 running, 0 waiting"
    fi
    note "KV at rest" "${kv:-?}%"
    [ "${preempt:-0}" -gt 0 ] && warn "preemptions since boot" "$preempt -- the pair has been under KV pressure" \
                              || ok "preemptions since boot" "0"
  else
    fail "engine metrics" "GET /metrics failed"
  fi

  # GB10: unified memory is the model, the KV cache and the page cache at once.
  # This is the check that predicts the NVRM OOM in the dual.env note.
  local avail swap_total swap_used
  avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
  swap_used=$(awk '/^SwapTotal/{t=$2} /^SwapFree/{f=$2} END{printf "%.1f", (t-f)/1048576}' /proc/meminfo)
  swap_total=$(awk '/^SwapTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
  if awk -v a="$avail" 'BEGIN{exit !(a<8)}'; then
    fail "host memory headroom" "only ${avail} GiB available -- a long generation can OOM the GPU context"
  else
    ok "host memory headroom" "${avail} GiB available"
  fi
  if awk -v s="$swap_used" 'BEGIN{exit !(s>2)}'; then
    warn "swap in use" "${swap_used} GiB of ${swap_total} GiB -- memory is drifting toward the cliff"
  else
    note "swap in use" "${swap_used} GiB"
  fi

  if [ -f "$MODEL_DIR/config.json" ]; then
    local shards size
    shards=$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    # metadata.total_size is the declared byte count. The index is one enormous
    # line, so it has to be parsed as JSON -- awk field splitting reads garbage here.
    size=$(python3 -c "
import json
d = json.load(open('$MODEL_DIR/model.safetensors.index.json'))
print(round(d.get('metadata', {}).get('total_size', 0) / 2**30))
" 2>/dev/null)
    ok "checkpoint (head)" "${shards} shards, ${size:-?} GiB at $MODEL_DIR"
  else
    fail "checkpoint (head)" "$MODEL_DIR/config.json missing"
  fi
  rsh "test -f $REMOTE_MODEL_DIR/config.json" 2>/dev/null \
    && ok "checkpoint (worker)" "$REMOTE_MODEL_DIR" \
    || warn "checkpoint (worker)" "cannot confirm $REMOTE_MODEL_DIR over ssh"

  # A warm compile cache is the difference between ~80 s and several minutes of
  # engine init. If it is empty, every load test measures torch.compile, not the model.
  local ncached
  ncached=$(find "$CACHE_DIR"/{vllm,triton,inductor} -type f 2>/dev/null | wc -l)
  [ "${ncached:-0}" -gt 50 ] && ok "compile cache (head)" "$ncached files in $CACHE_DIR" \
    || warn "compile cache (head)" "only ${ncached:-0} cached artefacts -- engine init will pay full compile cost"
  local wncached
  wncached=$(rsh "find $REMOTE_CACHE_DIR/vllm $REMOTE_CACHE_DIR/triton -type f 2>/dev/null | wc -l" 2>/dev/null)
  [ "${wncached:-0}" -gt 50 ] && ok "compile cache (worker)" "$wncached files" \
    || warn "compile cache (worker)" "only ${wncached:-0} cached artefacts on $WORKER_HOST"

  # `load` can only measure what the daemon still has. --tail defaults to all, but
  # a json-log rotation can have eaten the boot. grep -q would close the pipe early
  # and SIGPIPE docker, which pipefail then reports as a failure: count, do not quit.
  local nlog=0
  if [ -n "${DOCKER:-}" ]; then
    nlog=$($DOCKER logs "$CONTAINER" 2>&1 | grep -c "Loading weights took" || true)
  fi
  if [ "${nlog:-0}" -gt 0 ]; then
    ok "boot log retained" "$nlog weight-load line(s) present"
  else
    warn "boot log retained" "no weight-load lines in the daemon log -- 'load' has nothing to measure"
  fi

  echo "------------------------------------------------------------------------"
  if   [ "$FAILS" -gt 0 ]; then echo " >> $FAILS FAIL, $WARNS WARN -- fix before testing"
  elif [ "$WARNS" -gt 0 ]; then echo " >> 0 FAIL, $WARNS WARN -- safe to test, read the warnings"
  else echo " >> all checks pass -- safe to test"; fi
  return $(( FAILS > 0 ? 1 : 0 ))
}

# ==============================================================================
load(){
  local watch=0 pair=0 show_cmd=0 rest=()
  # $DOCKER is only set once resolve_docker runs, and --cmd returns before that.
  local DK="${DOCKER:-docker}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --watch) watch=1; shift ;;
      --pair|--worker) pair=1; shift ;;
      --cmd|--show-cmd) show_cmd=1; shift ;;
      *) rest+=("$1"); shift ;;
    esac
  done

  if [ "$show_cmd" = 1 ]; then
    cat <<EOF
A cold load means a real restart of BOTH ranks plus a page-cache drop, in this
order -- worker first, head second (RUNBOOK §11). This script will not do it for
you; copying these in is the whole procedure:

  cd $HERE/dual-spark

  # 1. worker (rank 1, headless) FIRST, then the head. Wrong order = the head
  #    waits on a peer that is not there yet and looks exactly like a hang.
  ./run.sh up            # stage_up drops the caches and starts BOTH ranks in order

  # By hand instead, if run.sh is not how you bring this pair up:
  #   worker: ssh $WORKER docker rm -f $CONTAINER \\
  #             && ssh $WORKER docker run -d --name $CONTAINER <same flags as run.sh> -e NODE_RANK=1 -e HEADLESS=1
  #   head:   $DK rm -f $CONTAINER \\
  #             && $DK run -d --name $CONTAINER <same flags> -e NODE_RANK=0

  # 2. page-cache drop -- run.sh up already does this on both nodes, so only do it
  #    yourself if you restarted by hand. It is not free on a live GB10: 65 GiB of
  #    weights are resident, and this forces 126 GiB to be re-read from NVMe.
  #    Run it with no traffic in flight.
  ssh $WORKER 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

  # 3. watch the boot through this same tool (read-only):
  $HERE/load-test.sh load --watch --pair

Then expect, from the 2026-09-18 validated pair: weights ~552 s, engine init
~45 s (a cold compile cache is far worse), total to ready ~12 min.
EOF
    return 0
  fi

  resolve_docker || { echo "!! docker unreachable" >&2; return 3; }

  if [ "$watch" = 1 ]; then
    echo ">> following $CONTAINER. Start the boot yourself; this only reads."
    set -o pipefail
    # shellcheck disable=SC2086
    $DOCKER logs -f --timestamps "$CONTAINER" 2>&1 \
      | python3 "$LT/load_report.py" --watch "${rest[@]}"
    return $?
  fi

  local tmp rc=0
  tmp=$(mktemp -d) || { echo "!! mktemp failed" >&2; return 3; }
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  $DOCKER logs --timestamps "$CONTAINER" >"$tmp/head.log" 2>&1 \
    || { echo "!! cannot read logs of $CONTAINER" >&2; return 3; }
  local args=(--logs "head=$tmp/head.log")
  if [ "$pair" = 1 ]; then
    if rsh "docker logs --timestamps $CONTAINER" >"$tmp/worker.log" 2>&1; then
      args+=(--logs "worker=$tmp/worker.log")
    else
      echo "!! worker log unreadable over ssh; reporting rank 0 only" >&2
    fi
  fi
  python3 "$LT/load_report.py" "${args[@]}" "${rest[@]}"
  rc=$?
  [ $rc -eq 0 ] && echo ">> note: these are the timings of the boot currently in the log." && \
                  echo ">>       for a cold number:  $0 load --cmd"
  return $rc
}

# ==============================================================================
bench(){
  # Our defaults come first so anything the caller passes wins on a duplicate.
  python3 "$LT/bench.py" \
    --base "$BASE" --api-key "$API_KEY" --model "$SERVED_NAME" \
    --levels "${LEVELS:-1,2,4,8}" "$@"
}

# ==============================================================================
CMD="${1:-check}"; shift || true
case "$CMD" in
  check|preflight) check ;;
  load|cold|boot)  load "$@" ;;
  bench|loadgen)   bench "$@" ;;
  all)   check; echo; load --pair "$@"; echo; bench ;;
  -h|--help|help) usage ;;
  *) echo "unknown subcommand: $CMD" >&2; echo; usage >&2; exit 2 ;;
esac
