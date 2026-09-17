SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help preflight bootstrap ray-up cluster-status download-model \
        vllm-up vllm-down smoke tool-test monitor stop

help:
	@echo "Dual DGX Spark playbook — common targets"
	@echo ""
	@echo "  make preflight         # Verify Docker, GPU, network, disk (both Sparks)"
	@echo "  make bootstrap         # Fetch run_cluster.sh and pull vLLM image (both Sparks)"
	@echo "  make ray-up            # Start Ray head/worker in tmux (both Sparks)"
	@echo "  make cluster-status    # Strict 2-nodes/2-GPUs check (HEAD only)"
	@echo "  make download-model    # Cache the model in each container (both Sparks)"
	@echo "  make vllm-up           # Start vLLM (HEAD only)"
	@echo "  make smoke             # /health + /v1/models + short chat (HEAD only)"
	@echo "  make tool-test         # Non-streaming tool-call test (HEAD only)"
	@echo "  make monitor           # Ray + procs + port + GPU + log tail"
	@echo "  make vllm-down         # Stop vLLM but leave Ray running (HEAD)"
	@echo "  make stop              # Kill the tmux Ray session on this node"

preflight:      ; @bash scripts/00-preflight.sh
bootstrap:      ; @bash scripts/01-bootstrap.sh
ray-up:         ; @bash scripts/12-launch-ray-tmux.sh
cluster-status: ; @bash scripts/20-cluster-status.sh
download-model: ; @bash scripts/30-download-model.sh
vllm-up:        ; @bash scripts/40-start-vllm.sh
vllm-down:      ; @bash scripts/41-stop-vllm.sh
smoke:          ; @bash scripts/50-smoke-test.sh
tool-test:      ; @bash scripts/51-tool-test.sh
monitor:        ; @bash scripts/60-monitor.sh
stop:           ; @bash scripts/90-stop-cluster.sh
