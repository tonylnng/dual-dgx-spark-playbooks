# Quick setup

The full runbook is `README.md`. This page is the ~5-minute overview of the
scripts in this repo, which correspond one-to-one to the phases in the runbook.

## Layout

```
config/cluster.env.example      Copy to config/cluster.env on both Sparks
scripts/_lib.sh                 Common helpers (do not run directly)
scripts/00-preflight.sh         Phase 3
scripts/01-bootstrap.sh         Phase 5
scripts/12-launch-ray-tmux.sh   Phase 6 (head) / Phase 7 (worker)
scripts/20-cluster-status.sh    Phase 8
scripts/30-download-model.sh    Phase 9
scripts/40-start-vllm.sh        Phase 10
scripts/41-stop-vllm.sh         Stop vLLM only
scripts/50-smoke-test.sh        Phase 11
scripts/51-tool-test.sh         Optional tool-calling test
scripts/60-monitor.sh           Monitoring snapshot
scripts/90-stop-cluster.sh      Controlled shutdown
litellm/config-snippet.yaml     Merge into LiteLLM config.yaml
litellm/test-litellm.sh         Phase 14
```

## Prepare both Sparks

```bash
git clone https://github.com/tonylnng/dual-dgx-spark-playbooks.git
cd dual-dgx-spark-playbooks

cp config/cluster.env.example config/cluster.env
chmod 600 config/cluster.env
# Edit config/cluster.env: set ROLE (head|worker), interface/IPs, HF_TOKEN,
# and a shared VLLM_API_KEY generated once with: openssl rand -hex 32
```

## Bring-up (Makefile)

```bash
# Both Sparks
make preflight
make bootstrap
make ray-up

# Head only
make cluster-status

# Both Sparks
make download-model

# Head only
make vllm-up
tail -f ~/dgx-qwen3/logs/server.log   # wait for "Application startup complete"
make smoke
```

## Shutdown

```bash
# Head first
make vllm-down
make stop

# Then worker
make stop
```

## Never commit

- `config/cluster.env` (real HF/VLLM secrets)
- Anything containing an actual `hf_...` token or generated `VLLM_API_KEY`

`.gitignore` and `.github/workflows/validate.yml` enforce this.
