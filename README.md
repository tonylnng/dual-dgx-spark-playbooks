# Dual DGX Spark Runbook: Qwen3-235B-A22B NVFP4 with vLLM and LiteLLM

## Purpose

This runbook deploys `RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4` across two paired NVIDIA DGX Spark systems using Ray, NCCL, and vLLM tensor parallelism (`TP=2`). It then registers Spark 1's OpenAI-compatible vLLM endpoint with LiteLLM running on a third machine.

The two Sparks provide 256 GB of aggregate unified memory, not one cache-coherent 256 GB address space. vLLM must shard the model across the two GPU nodes, and both Sparks participate in every inference request.[cite:4][cite:7]

The selected Red Hat checkpoint uses FP4 weights and activations, requires vLLM 0.9.1 or later, and has a published TP=2 evaluation configuration using a 4,096-token model length, chunked prefill, and eager execution.[cite:50]

## Target architecture

```text
Applications / agents
         |
         | OpenAI-compatible API
         v
LiteLLM machine :4000
  - Authentication and virtual keys
  - Model alias and routing
  - Budgets, logging and policy
         |
         | LAN HTTP, authenticated
         v
DGX Spark 1 :8000
  - vLLM OpenAI API server
  - Ray head
  - Tensor-parallel shard 1
         |
         | QSFP / ConnectX-7 / NCCL
         v
DGX Spark 2
  - Ray worker
  - Tensor-parallel shard 2
```

Use the dedicated QSFP addresses for Ray/NCCL traffic. Use Spark 1's ordinary LAN address for LiteLLM-to-vLLM API traffic.

## Deployment baseline

| Component | Baseline |
|---|---|
| Model | `RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4` |
| vLLM container | `nvcr.io/nvidia/vllm:26.05-py3` |
| Distributed executor | Ray |
| Tensor parallelism | 2 |
| Initial context | 4,096 tokens |
| Initial concurrent sequences | 1 |
| Initial batched tokens | 4,096 |
| Initial memory utilization | 0.70 |
| vLLM model name | `qwen3-235b-a22b` |
| LiteLLM model alias | `qwen3-235b` |
| vLLM API port | 8000 |
| LiteLLM API port | 4000 |

NVIDIA's two-node vLLM guide currently uses the NGC vLLM container and a pinned `run_cluster.sh` helper. NVIDIA also warns that the helper's exit trap removes the associated Ray container when its controlling shell exits, so the head and worker processes must run in persistent `tmux` sessions.[cite:7]

## Information worksheet

Complete this table before starting.

| Variable | Example | Actual value |
|---|---|---|
| Spark 1 hostname | `spark-1` | |
| Spark 2 hostname | `spark-2` | |
| Spark 1 LAN IP | `192.168.1.50` | |
| Spark 2 LAN IP | `192.168.1.51` | |
| LiteLLM machine IP | `192.168.1.60` | |
| QSFP interface | `enp1s0f1np1` | |
| Spark 1 QSFP IP | `192.168.100.10` | |
| Spark 2 QSFP IP | `192.168.100.11` | |
| Linux username | `nvidia` | |
| Hugging Face token | `hf_...` | |
| vLLM API key | generated secret | |

## Phase 1: Preconditions

The NVIDIA two-Spark pairing procedure must already be complete. Verify that both systems meet these conditions:

- The QSFP cable is connected and its network interface is up.
- Each Spark has a persistent, unique QSFP IP address.
- Both Sparks can ping each other through the QSFP addresses.
- Passwordless SSH works between the two systems.
- Docker can access the local GPU.
- Both systems can pull images from NVIDIA NGC.
- Both systems have enough local storage for their own model cache.
- The LiteLLM machine can reach Spark 1's LAN IP.

Install host utilities on both Sparks:

```bash
sudo apt-get update
sudo apt-get install -y curl jq tmux openssl
```

Create the required local directories on both Sparks:

```bash
mkdir -p "$HOME/dgx-qwen3" \
         "$HOME/dgx-qwen3/logs" \
         "$HOME/.cache/huggingface" \
         "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
```

## Phase 2: Define variables

### Spark 1

Create `~/dgx-qwen3/env.sh`:

```bash
cat > "$HOME/dgx-qwen3/env.sh" <<'ENV'
export ROLE=head
export MN_IF_NAME=enp1s0f1np1
export HEAD_QSFP_IP=192.168.100.10
export WORKER_QSFP_IP=192.168.100.11
export HEAD_LAN_IP=192.168.1.50
export LITELLM_IP=192.168.1.60

export WORKDIR="$HOME/dgx-qwen3"
export HF_CACHE="$HOME/.cache/huggingface"
export LOG_DIR="$HOME/dgx-qwen3/logs"
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
export MODEL_ID=RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4
export SERVED_MODEL_NAME=qwen3-235b-a22b
export VLLM_PORT=8000

export GPU_MEMORY_UTILIZATION=0.70
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=4096

export HF_TOKEN=hf_REPLACE_ME
export VLLM_API_KEY=REPLACE_ME
ENV
chmod 600 "$HOME/dgx-qwen3/env.sh"
```

### Spark 2

Create the same file on Spark 2, but change `ROLE` to `worker`. All network values, tokens, model values, and API-key values should otherwise match:

```bash
sed -i 's/export ROLE=head/export ROLE=worker/' "$HOME/dgx-qwen3/env.sh"
chmod 600 "$HOME/dgx-qwen3/env.sh"
```

Generate the vLLM API key once on Spark 1:

```bash
openssl rand -hex 32
```

Put the generated value into `VLLM_API_KEY` on both systems. Put a Hugging Face read token into `HF_TOKEN`. Never commit either secret to Git.

Load the variables whenever opening a new administrative shell:

```bash
source "$HOME/dgx-qwen3/env.sh"
```

## Phase 3: Network preflight

Run on both Sparks:

```bash
source "$HOME/dgx-qwen3/env.sh"

hostname
whoami
nvidia-smi
docker version
ibdev2netdev
ip -4 -o addr show
ip -4 addr show "$MN_IF_NAME"
```

NVIDIA's example QSFP interface name is `enp1s0f1np1`, but the active interface depends on which ConnectX-7 port is connected.[cite:17]

On Spark 1:

```bash
ping -c 4 "$WORKER_QSFP_IP"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER_QSFP_IP" hostname
```

On Spark 2:

```bash
ping -c 4 "$HEAD_QSFP_IP"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$HEAD_QSFP_IP" hostname
```

Confirm that each configured QSFP IP belongs to the selected interface:

```bash
LOCAL_QSFP_IP=$(ip -4 addr show "$MN_IF_NAME" | \
  grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
echo "$LOCAL_QSFP_IP"
```

Check storage on both systems:

```bash
df -h "$HOME/.cache/huggingface"
```

## Phase 4: Validate NCCL

Before loading the model, validate the distributed communication path using NVIDIA's NCCL playbook. Run the following on Spark 1 using the systems' normal management/LAN IPs where requested by the helper:[cite:26]

```bash
cd "$HOME/dgx-qwen3"

curl -fsSL \
  https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/setup.sh \
  -o setup-nccl.sh

curl -fsSL \
  https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/nccl/assets/launch.sh \
  -o launch-nccl.sh

chmod +x setup-nccl.sh launch-nccl.sh

bash setup-nccl.sh <SPARK_2_LAN_IP>
bash launch-nccl.sh --topology direct \
  <SPARK_1_LAN_IP> <SPARK_2_LAN_IP>
```

Do not proceed if the NCCL test hangs, reports transport errors, or unexpectedly selects Wi-Fi. Save the successful result as the infrastructure baseline.

## Phase 5: Bootstrap vLLM

Run on both Sparks:

```bash
source "$HOME/dgx-qwen3/env.sh"
cd "$WORKDIR"

curl -fsSL \
  https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh \
  -o run_cluster.sh

sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' \
  run_cluster.sh

chmod 700 run_cluster.sh
docker pull "$VLLM_IMAGE"
docker image inspect "$VLLM_IMAGE" \
  --format '{{index .RepoDigests 0}}' | tee vllm-image-digest.txt
```

The helper uses host networking, mounts the local Hugging Face cache at `/root/.cache/huggingface`, grants access to all GPUs, and starts Ray on port 6379.[cite:7]

## Phase 6: Start Ray head

On Spark 1:

```bash
source "$HOME/dgx-qwen3/env.sh"

tmux new-session -s ray-head
```

Inside the new tmux session:

```bash
source "$HOME/dgx-qwen3/env.sh"

LOCAL_QSFP_IP=$(ip -4 addr show "$MN_IF_NAME" | \
  grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

if [[ "$LOCAL_QSFP_IP" != "$HEAD_QSFP_IP" ]]; then
  echo "QSFP mismatch: $LOCAL_QSFP_IP != $HEAD_QSFP_IP"
  exit 1
fi

exec bash "$WORKDIR/run_cluster.sh" \
  "$VLLM_IMAGE" \
  "$HEAD_QSFP_IP" \
  --head \
  "$HF_CACHE" \
  -v "$LOG_DIR:/var/log/vllm" \
  -e VLLM_HOST_IP="$HEAD_QSFP_IP" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_QSFP_IP"
```

Detach from tmux without stopping the process:

```text
Ctrl-b, then d
```

Do not type `exit` inside the tmux session. The helper's EXIT trap stops and removes the Ray container.[cite:7]

## Phase 7: Start Ray worker

On Spark 2:

```bash
source "$HOME/dgx-qwen3/env.sh"
tmux new-session -s ray-worker
```

Inside the worker tmux session:

```bash
source "$HOME/dgx-qwen3/env.sh"

LOCAL_QSFP_IP=$(ip -4 addr show "$MN_IF_NAME" | \
  grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

if [[ "$LOCAL_QSFP_IP" != "$WORKER_QSFP_IP" ]]; then
  echo "QSFP mismatch: $LOCAL_QSFP_IP != $WORKER_QSFP_IP"
  exit 1
fi

exec bash "$WORKDIR/run_cluster.sh" \
  "$VLLM_IMAGE" \
  "$HEAD_QSFP_IP" \
  --worker \
  "$HF_CACHE" \
  -v "$LOG_DIR:/var/log/vllm" \
  -e VLLM_HOST_IP="$WORKER_QSFP_IP" \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e GLOO_SOCKET_IFNAME="$MN_IF_NAME" \
  -e TP_SOCKET_IFNAME="$MN_IF_NAME" \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR="$HEAD_QSFP_IP"
```

Detach with `Ctrl-b`, then `d`.

## Phase 8: Verify Ray

On Spark 1:

```bash
HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

echo "$HEAD_CONTAINER"
docker exec "$HEAD_CONTAINER" ray status
```

Ray must report two live nodes and two GPU resources before continuing.[cite:7]

Perform a strict check:

```bash
docker exec "$HEAD_CONTAINER" python - <<'PY'
import ray
ray.init(address="auto")

nodes = [n for n in ray.nodes() if n["Alive"]]
resources = ray.cluster_resources()

print("Live nodes:", len(nodes))
print("Resources:", resources)
print("Addresses:", [n["NodeManagerAddress"] for n in nodes])

assert len(nodes) == 2, "Expected exactly two live Ray nodes"
assert int(resources.get("GPU", 0)) == 2, "Expected exactly two GPUs"
PY
```

## Phase 9: Download model

The Ray helper mounts a different host-local cache on each Spark. Download the model independently on both systems so every tensor-parallel worker has local access to the same checkpoint revision.

On Spark 1 and Spark 2:

```bash
source "$HOME/dgx-qwen3/env.sh"

NODE_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MODEL_ID="$MODEL_ID" \
  "$NODE_CONTAINER" \
  bash -lc 'hf download "$MODEL_ID"'
```

Confirm the cache on both systems:

```bash
du -sh "$HOME/.cache/huggingface/hub/models--RedHatAI--Qwen3-235B-A22B-Instruct-2507-NVFP4"
```

The model card says this checkpoint is ready for vLLM 0.9.1 or later; its published TP=2 reproduction profile uses `dtype=auto`, `max_model_len=4096`, chunked prefill, and eager execution.[cite:50]

## Phase 10: Start vLLM

Run only on Spark 1:

```bash
source "$HOME/dgx-qwen3/env.sh"

HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

mkdir -p "$LOG_DIR"

docker exec -d \
  -e MODEL_ID="$MODEL_ID" \
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  -e VLLM_PORT="$VLLM_PORT" \
  -e VLLM_API_KEY="$VLLM_API_KEY" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  "$HEAD_CONTAINER" \
  bash -lc '
    exec vllm serve "$MODEL_ID" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$VLLM_PORT" \
      --api-key "$VLLM_API_KEY" \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --dtype auto \
      --trust-remote-code \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --enforce-eager \
      --generation-config vllm \
      --disable-log-requests \
      >> /var/log/vllm/server.log 2>&1
  '
```

The FP4 format should be detected from the checkpoint metadata. Do not add a quantization override unless the pinned vLLM image fails to detect it and its documentation explicitly requires one.

Follow startup:

```bash
tail -f "$HOME/dgx-qwen3/logs/server.log"
```

Wait until the log reports that application startup is complete. Initial model loading can take a long time; inspect the log instead of repeatedly restarting.

## Phase 11: Local acceptance

### Health endpoint

On Spark 1:

```bash
source "$HOME/dgx-qwen3/env.sh"
curl --fail --silent --show-error \
  "http://127.0.0.1:${VLLM_PORT}/health"
echo
```

### Model list

```bash
curl --fail --silent --show-error \
  "http://127.0.0.1:${VLLM_PORT}/v1/models" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" | jq .
```

### Chat completion

```bash
curl --fail --silent --show-error \
  "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-235b-a22b",
    "messages": [
      {
        "role": "system",
        "content": "You are a concise enterprise AI architecture assistant."
      },
      {
        "role": "user",
        "content": "Reply with exactly: DGX cluster ready"
      }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "stream": false
  }' | jq .
```

vLLM exposes an OpenAI-compatible HTTP server supporting model, chat-completion, and completion APIs.[cite:59]

## Phase 12: Test remote access

From the LiteLLM machine:

```bash
export SPARK_HEAD_IP=<SPARK_1_LAN_IP>
export VLLM_API_KEY='<VLLM_API_KEY_FROM_SPARK_1>'

curl --fail --silent --show-error \
  "http://${SPARK_HEAD_IP}:8000/v1/models" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" | jq .
```

If the local test passes but the remote test fails, check Spark 1:

```bash
ss -lntp | grep ':8000'
ip -4 addr
sudo ufw status verbose
```

If UFW is active, permit only the LiteLLM machine:

```bash
sudo ufw allow from <LITELLM_MACHINE_IP> \
  to any port 8000 proto tcp
```

Do not expose vLLM port 8000, Ray port 6379, or the Ray dashboard directly to the Internet.

## Phase 13: Configure LiteLLM

Merge this entry into the LiteLLM `config.yaml`:

```yaml
model_list:
  - model_name: qwen3-235b
    litellm_params:
      model: hosted_vllm/qwen3-235b-a22b
      api_base: os.environ/QWEN_VLLM_API_BASE
      api_key: os.environ/QWEN_VLLM_API_KEY
      timeout: 600
      stream_timeout: 600
      max_retries: 1
    model_info:
      mode: chat
      max_input_tokens: 3584
      max_output_tokens: 512

litellm_settings:
  drop_params: true
  request_timeout: 600
  num_retries: 1
```

LiteLLM documents the `hosted_vllm/` provider prefix for routing to a separately hosted vLLM OpenAI-compatible server. `model_name` is the client-facing alias, while the value after `hosted_vllm/` should match vLLM's served model name.[cite:30][cite:33]

Set these variables on the LiteLLM machine:

```bash
export QWEN_VLLM_API_BASE=http://<SPARK_1_LAN_IP>:8000/v1
export QWEN_VLLM_API_KEY='<SAME_VLLM_API_KEY>'
```

For Docker Compose, add the variables to the LiteLLM service:

```yaml
services:
  litellm:
    environment:
      QWEN_VLLM_API_BASE: ${QWEN_VLLM_API_BASE}
      QWEN_VLLM_API_KEY: ${QWEN_VLLM_API_KEY}
```

Restart LiteLLM using the existing deployment method:

```bash
# Example for Docker Compose
docker compose up -d --force-recreate litellm
docker compose logs -f litellm
```

## Phase 14: End-to-end acceptance

On the LiteLLM machine:

```bash
export LITELLM_URL=http://<LITELLM_MACHINE_IP>:4000
export LITELLM_MASTER_KEY='<LITELLM_MASTER_OR_VIRTUAL_KEY>'

curl --fail --silent --show-error \
  "${LITELLM_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-235b",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: LiteLLM route ready"
      }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "stream": false
  }' | jq .
```

Expected routing:

```text
Client model: qwen3-235b
      -> LiteLLM mapping: hosted_vllm/qwen3-235b-a22b
      -> vLLM model: qwen3-235b-a22b
      -> Spark 1 and Spark 2 TP=2 execution
```

## Optional: tool calling

First complete ordinary chat acceptance without tool-calling flags. Tool parsers and their registered names have changed between vLLM releases, and some Qwen3 streaming/parser combinations have returned raw tool-call markup.[cite:69][cite:82]

Inspect the parsers in the installed container:

```bash
HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

docker exec "$HEAD_CONTAINER" \
  bash -lc "vllm serve --help=all | grep -i -A8 tool-call-parser"
```

Qwen recommends Hermes-style tool use for Qwen3, but use the exact Qwen-compatible parser name registered by the pinned vLLM image.[cite:73]

After identifying the parser, stop vLLM and add these flags to the launch command:

```bash
--enable-auto-tool-choice \
--tool-call-parser <PARSER_REPORTED_BY_VLLM>
```

Validate tool calls with `stream: false` first:

```bash
curl --fail --silent --show-error \
  "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-235b-a22b",
    "messages": [
      {
        "role": "user",
        "content": "What is the GPU temperature on spark-1? Use the tool."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_gpu_temperature",
          "description": "Get the GPU temperature for a host",
          "parameters": {
            "type": "object",
            "properties": {
              "hostname": {"type": "string"}
            },
            "required": ["hostname"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
    "stream": false
  }' | jq .
```

## Performance tuning

Use the published 4,096-token TP=2 profile as the bring-up baseline.[cite:50] Change one dimension at a time and retain test evidence.

| Profile | Context | Sequences | Batched tokens | Memory utilization | Eager mode |
|---|---:|---:|---:|---:|---|
| Bring-up | 4,096 | 1 | 4,096 | 0.70 | Enabled |
| RAG trial | 8,192 | 1 | 4,096 | 0.75 | Enabled |
| Concurrent RAG | 8,192 | 2 | 4,096 | 0.75 | Enabled |
| Extended context | 16,384 | 1 | 8,192 | 0.80 | Enabled |
| Performance test | Last stable | Last stable | Last stable | Last stable | Disabled |

For each profile:

1. Stop incoming LiteLLM traffic.
2. Stop vLLM without stopping Ray.
3. Change the values in Spark 1's `env.sh`.
4. Start vLLM and wait for `/health`.
5. Run local and LiteLLM acceptance tests.
6. Run realistic RAG, concurrency, long-context, and soak tests.
7. Record startup time, time to first token, output tokens per second, peak memory, error rate, and NCCL/interface counters.

If vLLM reports OOM:

1. Reduce `MAX_MODEL_LEN`.
2. Keep `MAX_NUM_SEQS=1`.
3. Reduce `MAX_NUM_BATCHED_TOKENS`.
4. Stop unrelated containers, browsers, desktops, and development workloads.
5. Reduce `GPU_MEMORY_UTILIZATION` if Ray, NCCL, CUDA, or the OS lacks operational headroom.
6. Increase memory utilization only after proving adequate system headroom.

Do not assume that the model's advertised native context will fit on two Sparks. KV-cache capacity and distributed runtime overhead determine the usable context.

## Monitoring

### Cluster and process status

On Spark 1:

```bash
source "$HOME/dgx-qwen3/env.sh"
HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

docker exec "$HEAD_CONTAINER" ray status
docker exec "$HEAD_CONTAINER" pgrep -af 'ray|vllm' || true
nvidia-smi
ss -lntp | grep ":${VLLM_PORT}" || true
tail -n 100 "$LOG_DIR/server.log"
```

Some `nvidia-smi --query-gpu` memory fields may return `N/A` on unified-memory systems; NVIDIA recommends normal `nvidia-smi` output for inspection.[cite:7]

### Health probe

```bash
curl --fail --max-time 10 \
  http://127.0.0.1:8000/health
```

### Ray dashboard

Tunnel the dashboard rather than exposing it:

```bash
ssh -L 8265:localhost:8265 \
  <USER>@<SPARK_1_LAN_IP>
```

Open `http://localhost:8265` locally. NVIDIA documents the head dashboard on port 8265 in this host-network configuration.[cite:7]

## Stop vLLM only

On Spark 1:

```bash
HEAD_CONTAINER=$(docker ps --format '{{.Names}}' | \
  grep -E '^node-[0-9]+$' | head -1)

docker exec "$HEAD_CONTAINER" \
  bash -lc "pkill -TERM -f 'vllm serve' || true"

for i in {1..30}; do
  if ! docker exec "$HEAD_CONTAINER" \
    pgrep -af 'vllm serve' >/dev/null 2>&1; then
    echo "vLLM stopped"
    break
  fi
  sleep 2
done
```

If the process remains after the grace period:

```bash
docker exec "$HEAD_CONTAINER" \
  bash -lc "pkill -KILL -f 'vllm serve' || true"
```

## Controlled shutdown

1. Remove `qwen3-235b` from LiteLLM rotation or stop client traffic.
2. Stop vLLM on Spark 1.
3. Stop the Ray worker on Spark 2.
4. Stop the Ray head on Spark 1.

On Spark 2:

```bash
tmux kill-session -t ray-worker
```

On Spark 1:

```bash
tmux kill-session -t ray-head
```

The helper's EXIT trap stops and removes each corresponding `node-*` container. Host model caches and log files remain persistent.[cite:7]

## Restart sequence

1. Run network preflight on both Sparks.
2. Start `ray-head` on Spark 1.
3. Start `ray-worker` on Spark 2.
4. Verify exactly two live Ray nodes and two GPUs.
5. Start vLLM from Spark 1.
6. Wait for `/health` and run local acceptance.
7. Run the LiteLLM end-to-end test.
8. Restore application traffic.

## Troubleshooting

| Symptom | Likely cause | Corrective action |
|---|---|---|
| Ray reports one GPU | Worker failed to join | Inspect `tmux attach -t ray-worker`; confirm Spark 1's QSFP IP and Ray port 6379. |
| Ray/NCCL uses the wrong NIC | Interface variables are incorrect | Check `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and the OpenMPI include value. |
| Worker downloads model at startup | Worker cache was not pre-staged | Run the model-download command on Spark 2 and verify the exact model repository. |
| Model initialization OOM | Insufficient runtime headroom | Return to 4K context, one sequence, 4K batched tokens, and 0.70 memory utilization. |
| `/health` never appears | Model is loading, process crashed, or port conflicts | Inspect `server.log`, `pgrep -af vllm`, and `ss -lntp`. |
| Local API works but LiteLLM cannot connect | LAN route or firewall issue | Test Spark 1 `/v1/models` directly from the LiteLLM machine and restrict UFW to the LiteLLM IP. |
| LiteLLM returns model not found | Alias mismatch | Client: `qwen3-235b`; LiteLLM upstream: `hosted_vllm/qwen3-235b-a22b`; vLLM served name: `qwen3-235b-a22b`. |
| vLLM returns HTTP 401 | API keys differ | Ensure LiteLLM's `QWEN_VLLM_API_KEY` exactly matches Spark 1's `VLLM_API_KEY`. |
| Raw tool XML appears | Parser or streaming mismatch | Test with `stream:false`, inspect registered parsers, or disable automatic tools.[cite:82] |
| Cluster disappears after logout | Ray helper shell exited | Start it in `tmux`; do not exit the session.[cite:7] |
| Performance is unexpectedly poor | Cross-node traffic uses the wrong interface or communication dominates | Re-run NCCL, inspect interface counters, and verify every distributed interface variable. |

## Production hardening

Before production use:

- Pin the vLLM image by digest rather than a floating tag.
- Pin the model repository to an explicit Hugging Face revision.
- Store API keys in a secrets manager or protected environment file.
- Permit port 8000 only from the LiteLLM machine.
- Do not expose Ray ports externally.
- Add readiness probes for Ray node count, vLLM `/health`, `/v1/models`, and a deterministic completion.
- Add restart ordering: network, Ray head, Ray worker, quorum check, vLLM, local acceptance, LiteLLM routing.
- Monitor both nodes, not only Spark 1.
- Retain NCCL and inference baselines after every software upgrade.
- Put configuration and operational changes under Git change control without committing secrets.
- Keep a last-known-good context, concurrency, container digest, and model revision for rollback.

## Acceptance checklist

- [ ] QSFP peer ping passes in both directions.
- [ ] Passwordless SSH passes between Sparks.
- [ ] NCCL validation passes.
- [ ] Both systems use the same pinned vLLM image.
- [ ] Ray reports exactly two nodes and two GPUs.
- [ ] The model is cached locally on both systems.
- [ ] vLLM starts with TP=2.
- [ ] `/health` returns success.
- [ ] `/v1/models` returns `qwen3-235b-a22b`.
- [ ] A local chat completion succeeds.
- [ ] The LiteLLM machine reaches Spark 1 port 8000.
- [ ] LiteLLM maps `qwen3-235b` to the vLLM endpoint.
- [ ] End-to-end chat through LiteLLM succeeds.
- [ ] Tool calling passes non-streaming tests, if enabled.
- [ ] Shutdown and restart procedures are tested.
- [ ] Last-known-good settings and versions are recorded.
