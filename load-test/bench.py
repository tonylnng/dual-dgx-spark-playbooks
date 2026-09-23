#!/usr/bin/env python3
"""Closed-loop load generator for the dual-Spark vLLM endpoint.

Stdlib only (threads + http.client). The host has no torch/vllm, and reaching for
`vllm bench serve` would mean starting the 30 GB image with a GPU attached just to
send HTTP requests -- and it measures from inside the box it is loading, which is
not where the caller lives.

Two things are measured and kept visibly separate:

  client side  -- what a caller experiences: TTFT, inter-token gap, E2E
  engine side  -- what vLLM says about itself from /metrics over the same window

The two *disagreeing* is the interesting result. Client latency climbing while
`vllm:num_preemptions_total` climbs too means KV pressure, not a slow GPU. Client
latency climbing with a flat engine is a problem on this box, not in the model.

Requests are closed-loop (`conc` in flight, the next sent when one completes):
the right model for "how many concurrent agents can this pair serve". Open-loop
Poisson arrival answers a different question ("can it survive a traffic shape")
and is not what this measures.

Prompts carry a per-request nonce by default. Prefix caching is ON in this
deployment, so a repeated prompt is not a load test -- it is a cache hit, and the
TTFT numbers become fiction. Pass --no-cache-bust to measure the opposite thing:
how well the prefix cache holds under concurrent load.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import random
import re
import socket
import ssl
import sys
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor

# Small fixed vocabulary: prompt length is what matters here, not its content.
WORDS = (
    "the plan records every transfer between the two storage nodes and the "
    "cluster keeps a ledger of each shard revision while operators review the "
    "timeline of events that produced this report without changing anything in "
    "the running system or its cached compiled graphs and kernels".split()
)
TAIL = "\n\nSummarise the passage above in one short line. Do not reason out loud."
MAX_SEQS_HINT = 8  # MAX_NUM_SEQS in dual.env; above this the scheduler queues


# --- prompt construction ------------------------------------------------------
def build_prompt(tokens: int, bust: bool) -> str:
    """~`tokens` prompt tokens. 0.75 words/token is close enough for English."""
    n_words = max(4, int(tokens * 0.75))
    rng = random.Random(0xC0FFEE)  # stable body, so only the nonce varies
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    # A trailing nonce survives the chat template but can still land in the
    # tail the prefix cache reuses; a leading one cannot. 16 hex chars ~= 9 tok.
    head = f"[req {uuid.uuid4().hex[:16]}]\n" if bust else ""
    return head + body + TAIL


# --- HTTP ---------------------------------------------------------------------
class Target:
    def __init__(self, base: str, timeout: float):
        u = urllib.parse.urlparse(base)
        if u.scheme not in ("http", "https"):
            sys.exit(f"!! --base must be http(s), got {base!r}")
        self.tls = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.tls else 80)
        self.prefix = (u.path or "").rstrip("/")
        self.timeout = timeout

    def connect(self):
        # One connection per request, closed afterwards. Sharing keep-alive
        # sockets across a thread pool is where "spurious ECONNRESET at
        # concurrency 8" bugs are born, and a TCP handshake is free next to a
        # 128-token generation.
        if self.tls:
            # context= is HTTPSConnection-only; passing it to HTTPConnection is a
            # TypeError that surfaces as a bogus "server is down" message.
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def path(self, api: str) -> str:
        return self.prefix + api


def post_stream(target: Target, api: str, key: str, body: dict):
    """Send a completion request; return the live HTTPResponse (caller drains)."""
    conn = target.connect()
    try:
        conn.request(
            "POST", target.path(api), body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}",
                     "Accept": "text/event-stream" if body.get("stream") else "application/json",
                     "Connection": "close"},
        )
        return conn.getresponse()
    except Exception:
        conn.close()
        raise


def get_text(target: Target, api: str, key: str) -> str:
    conn = target.connect()
    try:
        conn.request("GET", target.path(api), headers={"Authorization": f"Bearer {key}"})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", "replace")
        if resp.status != 200:
            raise RuntimeError(f"GET {api} -> HTTP {resp.status}")
        return data
    finally:
        conn.close()


# --- one request --------------------------------------------------------------
def do_request(target: Target, args, prompt: str) -> dict:
    """One completion round trip. Never raises: a failed request is a data point."""
    body = {"model": args.model, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "stream": args.stream}
    if args.ignore_eos:
        body["ignore_eos"] = True  # vLLM-specific extra param; keeps decode comparable
    if args.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if args.stream:
        # Without include_usage the stream reports no usage and the output count
        # silently falls back to "count SSE chunks", which over-counts every
        # chunk that carries an empty delta.
        body["stream_options"] = {"include_usage": True}

    api = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    t0 = time.perf_counter()
    s = {"ok": False, "status": None, "error": None, "first": None, "last": None,
         "out_tokens": 0, "in_tokens": 0, "finish": None, "chunks": 0,
         "ttft": None, "tpot": None, "e2e": None}
    resp = None
    try:
        resp = post_stream(target, api, args.api_key, body)
        s["status"] = resp.status
        if resp.status != 200:
            s["error"] = f"HTTP {resp.status}: {resp.read(400).decode('utf-8', 'replace').strip()[:200]}"
            s["e2e"] = time.perf_counter() - t0
            return s

        if args.stream:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    s["error"] = f"stream error: {obj['error']}"
                    break
                usage = obj.get("usage")
                if usage:
                    s["out_tokens"] = int(usage.get("completion_tokens") or 0)
                    s["in_tokens"] = int(usage.get("prompt_tokens") or 0)
                for choice in obj.get("choices") or []:
                    if choice.get("finish_reason"):
                        s["finish"] = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    # Three spellings across vLLM builds: this one streams the
                    # chain of thought as `reasoning`; upstream uses
                    # `reasoning_content`. Counting only `content` would call a
                    # 200-token reasoned answer a 12-token one -- or, as it first
                    # did here, "no content delta at all".
                    text = (delta.get("content") or "") + (delta.get("reasoning") or "") \
                        + (delta.get("reasoning_content") or "")
                    if text:
                        now = time.perf_counter()
                        s["chunks"] += 1
                        if s["first"] is None:
                            # First non-empty delta. The role-only chunk vLLM
                            # sends first would otherwise flatter TTFT.
                            s["first"] = now
                        s["last"] = now
            if s["out_tokens"] == 0:
                s["out_tokens"] = s["chunks"]
        else:
            obj = json.loads(resp.read().decode("utf-8", "replace"))
            if obj.get("error"):
                s["error"] = f"error: {obj['error']}"
                return s
            usage = obj.get("usage") or {}
            s["out_tokens"] = int(usage.get("completion_tokens") or 0)
            s["in_tokens"] = int(usage.get("prompt_tokens") or 0)
            s["finish"] = (obj.get("choices") or [{}])[0].get("finish_reason")
            s["first"] = s["last"] = time.perf_counter()  # TTFT collapses onto E2E

        if s["first"] is None:
            s["error"] = "no content delta in response"
            return s
        s["e2e"] = time.perf_counter() - t0
        s["ttft"] = s["first"] - t0
        # Mean inter-token gap over this request. True per-interval percentiles
        # would need every timestamp kept; see /metrics ITL for the engine view.
        s["tpot"] = (s["last"] - s["first"]) / max(1, s["out_tokens"] - 1)
        s["ok"] = True
        return s
    except (socket.timeout, TimeoutError) as exc:
        s["error"] = f"timeout after {target.timeout:.0f}s ({type(exc).__name__})"
        s["e2e"] = time.perf_counter() - t0
        return s
    except Exception as exc:  # reset, DNS, broken JSON: record, do not abort the run
        s["error"] = f"{type(exc).__name__}: {exc}"[:200]
        s["e2e"] = time.perf_counter() - t0
        return s
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


# --- engine metrics -----------------------------------------------------------
LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)$")


def parse_metrics(text: str) -> dict[str, list[tuple[dict, float]]]:
    out: dict[str, list[tuple[dict, float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, raw_labels, value = m.groups()
        if name.endswith("_created"):  # "when was this counter made" noise
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', raw_labels or ""))
        try:
            out.setdefault(name, []).append((labels, float(value)))
        except ValueError:
            pass
    return out


def msum(sample: dict, name: str, **want) -> float:
    return sum(v for labels, v in (sample.get(name) or [])
               if all(labels.get(k) == val for k, val in want.items()))


class EngineSampler(threading.Thread):
    """Poll /metrics during a level, so peaks are visible and not averaged away."""

    def __init__(self, target: Target, key: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.target, self.key, self.interval = target, key, max(0.2, interval)
        self.stop_evt = threading.Event()
        self.peak = {"running": 0.0, "waiting": 0.0, "kv_pct": 0.0}
        self.polls = 0
        self.errors = 0

    def run(self):
        while not self.stop_evt.is_set():
            try:
                s = parse_metrics(get_text(self.target, "/metrics", self.key))
                self.peak["running"] = max(self.peak["running"], msum(s, "vllm:num_requests_running"))
                self.peak["waiting"] = max(self.peak["waiting"], msum(s, "vllm:num_requests_waiting"))
                self.peak["kv_pct"] = max(self.peak["kv_pct"], 100 * msum(s, "vllm:kv_cache_usage_perc"))
                self.polls += 1
            except Exception:
                self.errors += 1
            self.stop_evt.wait(self.interval)

    def snapshot(self) -> dict:
        try:
            return parse_metrics(get_text(self.target, "/metrics", self.key))
        except Exception:
            return {}


def engine_delta(before: dict, after: dict, peak: dict) -> dict:
    def mean(sum_name: str, count_name: str) -> float | None:
        n = msum(after, count_name) - msum(before, count_name)
        if n <= 0:
            return None
        return (msum(after, sum_name) - msum(before, sum_name)) / n

    q = msum(after, "vllm:prefix_cache_queries_total") - msum(before, "vllm:prefix_cache_queries_total")
    h = msum(after, "vllm:prefix_cache_hits_total") - msum(before, "vllm:prefix_cache_hits_total")
    drafts = (msum(after, "vllm:spec_decode_num_draft_tokens_total")
              - msum(before, "vllm:spec_decode_num_draft_tokens_total"))
    acc = (msum(after, "vllm:spec_decode_num_accepted_tokens_total")
           - msum(before, "vllm:spec_decode_num_accepted_tokens_total"))
    return {
        "prompt_tokens": msum(after, "vllm:prompt_tokens_total") - msum(before, "vllm:prompt_tokens_total"),
        "generation_tokens": msum(after, "vllm:generation_tokens_total") - msum(before, "vllm:generation_tokens_total"),
        "requests_ok": msum(after, "vllm:request_success_total") - msum(before, "vllm:request_success_total"),
        "requests_error": (msum(after, "vllm:request_success_total", finished_reason="error")
                           - msum(before, "vllm:request_success_total", finished_reason="error")),
        "requests_abort": (msum(after, "vllm:request_success_total", finished_reason="abort")
                           - msum(before, "vllm:request_success_total", finished_reason="abort")),
        "preemptions": msum(after, "vllm:num_preemptions_total") - msum(before, "vllm:num_preemptions_total"),
        "prefix_hit_pct": (100.0 * h / q) if q > 0 else None,
        "spec_accept_pct": (100.0 * acc / drafts) if drafts > 0 else None,
        "ttft_mean_s": mean("vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count"),
        "itl_mean_s": mean("vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count"),
        # Per output token, which is what a client's TPOT actually corresponds to.
        # ITL is not that number under speculative decoding: a step emits several
        # accepted tokens at once, so ITL (per step) runs 3-4x a real token gap.
        "rtpt_mean_s": mean("vllm:request_time_per_output_token_seconds_sum",
                            "vllm:request_time_per_output_token_seconds_count"),
        "queue_mean_s": mean("vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count"),
        "prefill_mean_s": mean("vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count"),
        "peak_running": peak.get("running"),
        "peak_waiting": peak.get("waiting"),
        "peak_kv_pct": peak.get("kv_pct"),
        "sampler_errors": None,
    }


# --- aggregation --------------------------------------------------------------
def pct(values: list, p: float) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    xs = sorted(vals)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def pct_ms(values: list, p: float) -> float | None:
    v = pct(values, p)
    return None if v is None else 1000 * v


def fmt(v, spec=".2f", unit=""):
    if not isinstance(v, (int, float)) or v != v:
        return "-"
    return format(v, spec) + unit


def run_level(target: Target, args, conc: int, n_requests: int) -> dict:
    samples: list[dict] = []
    lock = threading.Lock()
    counter = itertools.count()
    deadline = time.perf_counter() + args.duration if args.duration else None

    def worker():
        while True:
            with lock:
                if deadline is not None:
                    if time.perf_counter() >= deadline:
                        return
                elif next(counter) >= n_requests:
                    return
            prompt = build_prompt(args.input_tokens, not args.no_cache_bust)
            samples.append(do_request(target, args, prompt))

    sampler = EngineSampler(target, args.api_key, args.metric_interval)
    before = sampler.snapshot()
    sampler.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        for _ in range(conc):
            pool.submit(worker)
    wall = time.perf_counter() - t0
    sampler.stop_evt.set()
    time.sleep(0.1)
    after = sampler.snapshot()
    return {"concurrency": conc, "wall_s": wall, "samples": samples,
            "before": before, "after": after, "peak": dict(sampler.peak),
            "sampler_errors": sampler.errors, "sampler_polls": sampler.polls}


def level_row(res: dict) -> dict:
    ok = [s for s in res["samples"] if s["ok"]]
    err = [s for s in res["samples"] if not s["ok"]]
    wall = max(res["wall_s"], 1e-9)
    out_tok = sum(s["out_tokens"] for s in ok)
    in_tok = sum(s["in_tokens"] for s in ok)
    eng = engine_delta(res["before"], res["after"], res["peak"])
    eng["sampler_errors"] = res["sampler_errors"]
    return {
        "concurrency": res["concurrency"], "sent": len(res["samples"]),
        "ok": len(ok), "errors": len(err),
        "error_ratio": (len(err) / len(res["samples"])) if res["samples"] else 0.0,
        "error_samples": [s["error"] for s in err][:5],
        "finish_reasons": {r: sum(1 for s in ok if s["finish"] == r)
                           for r in {s["finish"] for s in ok}},
        "wall_s": res["wall_s"],
        "req_per_s": len(ok) / wall,
        "out_tok_per_s": out_tok / wall,
        "total_tok_per_s": (out_tok + in_tok) / wall,
        # Aggregate decode throughput divided by streams actually offered: what
        # one agent out of `conc` experiences on average.
        "per_stream_tok_per_s": (out_tok / wall / res["concurrency"]) if ok else None,
        "mean_out_tokens": (out_tok / len(ok)) if ok else None,
        "mean_in_tokens": (in_tok / len(ok)) if ok else None,
        "ttft_ms": {"p50": pct_ms([s["ttft"] for s in ok], 50),
                    "p90": pct_ms([s["ttft"] for s in ok], 90),
                    "p99": pct_ms([s["ttft"] for s in ok], 99),
                    "max": pct_ms([s["ttft"] for s in ok], 100)},
        "tpot_ms": {"p50": pct_ms([s["tpot"] for s in ok], 50),
                    "p90": pct_ms([s["tpot"] for s in ok], 90),
                    "p99": pct_ms([s["tpot"] for s in ok], 99)},
        "e2e_s": {"p50": pct([s["e2e"] for s in ok], 50), "p90": pct([s["e2e"] for s in ok], 90),
                  "p99": pct([s["e2e"] for s in ok], 99)},
        "engine": eng,
        "warnings": [],
    }


def judge(row: dict, args) -> None:
    w, g = row["warnings"], row["engine"]
    if row["error_ratio"] > args.max_error_ratio:
        w.append(f"ERROR {row['errors']}/{row['sent']} requests failed "
                 f"({100 * row['error_ratio']:.0f}% > budget {100 * args.max_error_ratio:.0f}%)")
    if (g.get("preemptions") or 0) > 0:
        w.append(f"WARN {g['preemptions']:.0f} scheduler preemptions -- KV pressure, not GPU speed; "
                 "see the GPU_MEMORY_UTILIZATION note in dual.env")
    if (g.get("requests_error") or 0) > 0:
        w.append(f"WARN the engine itself counted {g['requests_error']:.0f} errored requests")
    if (g.get("requests_abort") or 0) > 0:
        w.append(f"WARN {g['requests_abort']:.0f} requests aborted (client-side timeout or disconnect)")
    if (g.get("peak_kv_pct") or 0) >= 95:
        w.append(f"WARN KV cache peaked at {g['peak_kv_pct']:.0f}% -- at the eviction edge")
    if row["ttft_ms"]["p99"] is not None and row["ttft_ms"]["p99"] > args.ttft_budget_ms:
        w.append(f"WARN TTFT p99 {row['ttft_ms']['p99']:.0f} ms over the "
                 f"{args.ttft_budget_ms:.0f} ms budget")
    if (g.get("peak_waiting") or 0) >= max(2, row["concurrency"]):
        w.append(f"WARN scheduler queue peaked at {g['peak_waiting']:.0f} waiting -- "
                 "offered load exceeds capacity at this concurrency")
    if row["sent"] and g.get("peak_running") and g["peak_running"] < row["concurrency"]:
        w.append(f"note engine peaked at {g['peak_running']:.0f} running of {row['concurrency']} offered "
                 "-- the client, not the engine, was the limit (or admissions lagged)")
    elif (g.get("peak_running") or 0) > row["concurrency"]:
        # Observed, not hypothetical: LiteLLM traffic lands on this engine while a
        # ladder is running. The idle check at startup cannot see requests that
        # arrive mid-level, and aggregate tok/s silently includes their tokens.
        w.append(f"note the engine ran {g['peak_running']:.0f} requests though only "
                 f"{row['concurrency']} were offered -- something else used it during this "
                 "level, so aggregate tok/s is partly theirs")
    mo = row["mean_out_tokens"]
    if mo is not None and mo < 0.5 * args.max_tokens:
        w.append(f"note mean output {mo:.0f} tok vs max_tokens {args.max_tokens} -- the model "
                 "stopped early, so decode numbers rest on a shorter sample")


def render(rows: list[dict]) -> None:
    bar = "─" * 128
    head1 = (f"{'conc':>5} {'ok':>5} {'err':>4} {'wall':>7} {'req/s':>7} {'out tok/s':>10} "
             f"{'tot tok/s':>10} {'TTFT p50':>9} {'p90':>7} {'p99':>8} {'TPOT p50':>9} {'p90':>7} {'E2E p99':>8}")
    head2 = (f"{'conc':>5} {'vLLM TTFT':>10} {'/step':>7} {'ms/tok':>7} {'queue':>7} {'prefill':>8} "
             f"{'peak run':>9} {'peak wait':>10} {'peak KV':>8} {'preempt':>8} "
             f"{'prefix hit':>11} {'MTP accept':>11}")
    print()
    print(bar); print(head1); print(bar)
    for r in rows:
        t, tp, e = r["ttft_ms"], r["tpot_ms"], r["e2e_s"]
        print(f"{r['concurrency']:>5} {r['ok']:>5} {r['errors']:>4} {r['wall_s']:>7.1f} "
              f"{fmt(r['req_per_s'], '.2f'):>7} {fmt(r['out_tok_per_s'], '.1f'):>10} "
              f"{fmt(r['total_tok_per_s'], '.1f'):>10} "
              f"{fmt(t['p50'], '.0f', 'ms'):>9} {fmt(t['p90'], '.0f'):>7} {fmt(t['p99'], '.0f'):>8} "
              f"{fmt(tp['p50'], '.1f', 'ms'):>9} {fmt(tp['p90'], '.1f'):>7} {fmt(e['p99'], '.1f', 's'):>8}")
    print(bar); print(head2); print(bar)
    for r in rows:
        g = r["engine"]
        rtpt = None if g.get("rtpt_mean_s") is None else 1000 * g["rtpt_mean_s"]
        print(f"{r['concurrency']:>5} "
              f"{fmt(g['ttft_mean_s'], '.2f', 's'):>10} {fmt(g['itl_mean_s'], '.3f', 's'):>7} "
              f"{fmt(rtpt, '.1f'):>7} "
              f"{fmt(g['queue_mean_s'], '.2f', 's'):>7} {fmt(g['prefill_mean_s'], '.2f', 's'):>8} "
              f"{fmt(g['peak_running'], '.0f'):>9} {fmt(g['peak_waiting'], '.0f'):>10} "
              f"{fmt(g['peak_kv_pct'], '.0f', '%'):>8} {fmt(g['preemptions'], '.0f'):>8} "
              f"{fmt(g['prefix_hit_pct'], '.0f', '%'):>11} {fmt(g['spec_accept_pct'], '.0f', '%'):>11}")
    print(bar)
    if any(r["engine"].get("spec_accept_pct") for r in rows):
        print(" note: MTP speculative decoding is on. '/step' is vLLM's inter-token latency, which is\n"
              "       per ENGINE STEP -- a step emits several accepted tokens at once, so it is 3-4x a\n"
              "       real token gap. 'ms/tok' (request_time_per_output_token) is the per-token figure\n"
              "       comparable with the client-side TPOT columns above.")


# --- cli ----------------------------------------------------------------------
def parse_levels(raw: str) -> list[int]:
    try:
        out = [int(x) for x in raw.replace(" ", "").split(",") if x]
    except ValueError:
        sys.exit(f"!! --levels must be comma-separated ints, got {raw!r}")
    if not out or any(x < 1 or x > 128 for x in out):
        sys.exit(f"!! --levels values must be within 1..128, got {raw!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="vLLM base URL")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="Bearer token")
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME", ""), help="served model name")
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--levels", default="1,2,4,8", help="concurrency ladder, e.g. '1,2,4,8'")
    ap.add_argument("--requests", type=int, default=0,
                    help="requests per level (default max(6, 3*concurrency))")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds per level instead of a request count (sustained mode)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--input-tokens", type=int, default=128, help="approx prompt size")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="force full max_tokens output so decode is comparable across levels")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="no SSE; TTFT then collapses onto E2E -- only to test non-streaming clients")
    ap.add_argument("--no-cache-bust", action="store_true",
                    help="repeat identical prompts: measure the prefix cache instead of beating it")
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests at conc=1 first")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--metric-interval", type=float, default=1.0)
    ap.add_argument("--ttft-budget-ms", type=float, default=10000.0)
    ap.add_argument("--max-error-ratio", type=float, default=0.02)
    ap.add_argument("--abort-after-levels", type=int, default=2,
                    help="stop the ladder after this many consecutive levels over the error budget")
    ap.add_argument("--max-requests", type=int, default=3000, help="runaway guard (request-count mode)")
    ap.add_argument("--max-minutes", type=float, default=45.0, help="runaway guard (sustained mode)")
    ap.add_argument("--force", action="store_true",
                    help="skip the idle-server and volume guards -- someone else is on this box")
    ap.add_argument("--yes", action="store_true", help="skip the countdown")
    ap.add_argument("--label", default="", help="tag stored in the JSON, e.g. 'nvfp4-mtp4'")
    ap.add_argument("--json", dest="json_path", default="", help="write results JSON here")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()

    if not args.model:
        sys.exit("!! --model required (or export SERVED_NAME)")
    levels = parse_levels(args.levels)
    per_level = [args.requests or max(6, 3 * c) for c in levels]
    planned = sum(per_level)

    target = Target(args.base, args.timeout)

    # ---- guards. This pair also serves interactive traffic; do not join unnoticed.
    try:
        live = parse_metrics(get_text(target, "/metrics", args.api_key))
    except Exception as exc:
        sys.exit(f"!! cannot read {args.base}/metrics -- is vLLM up? ({exc})")
    busy = msum(live, "vllm:num_requests_running") + msum(live, "vllm:num_requests_waiting")
    kv_rest = 100 * msum(live, "vllm:kv_cache_usage_perc")
    if not args.force:
        if busy > 0:
            sys.exit(f"!! server already busy ({busy:.0f} running/waiting) -- someone is using it.\n"
                     "   Let it drain, or pass --force to load on top of them.")
        if args.duration and args.duration * len(levels) > args.max_minutes * 60:
            sys.exit(f"!! sustained plan is {args.duration * len(levels) / 60:.0f} h of load, above "
                     f"--max-minutes {args.max_minutes}")
        if not args.duration and planned > args.max_requests:
            sys.exit(f"!! plan is {planned} requests, above the {args.max_requests} guard; "
                     "raise --max-requests or pass --force")

    print(f">> target  {args.base}  model '{args.model}'  "
          f"{'/v1/chat/completions' if args.endpoint == 'chat' else '/v1/completions'}")
    print(f">> ladder  {' -> '.join(str(c) for c in levels)}")
    print(f">> load    " + (f"{args.duration:.0f}s per level, closed-loop" if args.duration
                           else f"{', '.join(str(n) for n in per_level)} requests per level"))
    print(f">> shape   ~{args.input_tokens} tok in, {args.max_tokens} tok out max, "
          f"{'stream' if args.stream else 'non-stream'}, "
          f"{'shared' if args.no_cache_bust else 'cache-busted'} prompts, temp={args.temperature}")
    print(f">> engine  {msum(live, 'vllm:num_requests_running'):.0f} running, "
          f"{kv_rest:.1f}% KV at rest")
    if max(levels) > MAX_SEQS_HINT:
        print(f">> note    ladder exceeds MAX_NUM_SEQS={MAX_SEQS_HINT}: past it, requests queue. "
              "That is the point of testing beyond it.")

    if not args.yes and sys.stdin.isatty():
        for i in range(5, 0, -1):
            print(f"\r>> starting in {i}s (Ctrl-C to abort)   ", end="", flush=True)
            time.sleep(1)
        print()

    if args.warmup > 0:
        print(f">> warming up ({args.warmup} untimed request{'s' if args.warmup > 1 else ''})")
        for _ in range(args.warmup):
            s = do_request(target, args, build_prompt(args.input_tokens, not args.no_cache_bust))
            if not s["ok"]:
                sys.exit(f"!! warmup request failed: {s['error']}\n   Not loading -- server is not answering.")
        print(f">> warm ok: TTFT {1000 * s['ttft']:.0f} ms, {s['out_tokens']} tok in {s['e2e']:.2f}s")

    rows: list[dict] = []
    breach = 0
    for idx, conc in enumerate(levels):
        mode = f"{args.duration:.0f}s" if args.duration else f"{per_level[idx]} reqs"
        print(f">> level concurrency={conc} ({mode}) ...", flush=True)
        row = level_row(run_level(target, args, conc, per_level[idx]))
        judge(row, args)
        rows.append(row)
        print(f"   ok={row['ok']} err={row['errors']}  out={fmt(row['out_tok_per_s'], '.1f')} tok/s  "
              f"TTFT p50={fmt(row['ttft_ms']['p50'], '.0f')} ms  "
              f"TPOT p50={fmt(row['tpot_ms']['p50'], '.1f')} ms")
        for msg in row["warnings"]:
            print(f"   !! {msg}")
        if row["sent"] == 0:
            # A level that issued nothing says nothing. Report it instead of
            # printing a row of dashes that reads like a very fast server.
            print("!! zero requests issued at this level, stopping the ladder")
            break
        breach = breach + 1 if row["error_ratio"] > args.max_error_ratio else 0
        if args.abort_after_levels and breach >= args.abort_after_levels:
            print("!! error budget breached on consecutive levels, stopping the ladder")
            break

    render(rows)

    total_err = sum(r["errors"] for r in rows)
    total_sent = sum(r["sent"] for r in rows)
    good = [r for r in rows if r["ok"]]
    single = min(rows, key=lambda r: r["concurrency"]) if rows else None
    best = max(good, key=lambda r: r["out_tok_per_s"], default=None)
    print(f"\n>> {total_sent} requests, {total_err} failed, "
          f"{sum(r['wall_s'] for r in rows):.0f}s under load")
    if best and single and len(rows) > 1 and best is not single:
        print(f">> peak aggregate decode {best['out_tok_per_s']:.1f} tok/s at concurrency "
              f"{best['concurrency']} vs {single['out_tok_per_s']:.1f} tok/s single-stream "
              f"({best['out_tok_per_s'] / max(single['out_tok_per_s'], 1e-9):.1f}x)")
    fails = [r for r in rows if any(m.startswith("ERROR") for m in r["warnings"])]
    if fails:
        print(f">> FAIL: {len(fails)} level(s) over the error budget")

    if not args.no_json:
        path = args.json_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results",
            time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-bench.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "label": args.label or None,
            "target": {"base": args.base, "model": args.model, "endpoint": args.endpoint},
            "config": {k: getattr(args, k) for k in
                       ("levels", "requests", "duration", "max_tokens", "input_tokens", "temperature",
                        "stream", "ignore_eos", "no_cache_bust", "timeout", "ttft_budget_ms",
                        "max_error_ratio")},
            "kv_pct_at_rest": kv_rest,
            "levels": rows,
            "ok": not fails,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f">> results written to {path}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
