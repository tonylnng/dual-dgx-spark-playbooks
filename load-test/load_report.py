#!/usr/bin/env python3
"""Turn a vLLM container log into a boot timeline: where the load time actually went.

Feed it `docker logs --timestamps` on stdin (or --logs LABEL=PATH). It reads log
text only, so it is safe against a live server -- and it works on a log from a
previous boot, which is why this exists instead of a restart: the answer to "how
long does the model take to load" is usually already sitting in the log.

Phases match the RUNBOOK §12 measured-boot-timeline table, so a regression shows
up as a delta against those numbers rather than as a bare wall clock. It also
knows the failure signatures this pair has already paid hours for (Pitfalls 2, 3,
6) and names them when it sees one, because the interesting result of a load test
is often "it never came up, and here is which pitfall".

  docker logs --timestamps CTN | load_report.py --label head
  docker logs -f --timestamps CTN | load_report.py --watch
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

# RUNBOOK §12 "Measured boot timeline" -- the 2026-09-18 validated pair at TP2.
# Timings only. KV size deliberately has no baseline: it tracks
# GPU_MEMORY_UTILIZATION, which has moved since that measurement.
BASELINE = {
    "weights_s": 552.0,      # 206 shards -> 65.0 GiB per rank
    "init_engine_s": 44.5,   # profile + KV allocation + warmup
    "total_s": 720.0,        # container start -> "Application startup complete."
}

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z?\s+(.*)$")
# vLLM's own clock, "[INFO 09-22 07:35:11]" -- the fallback when a log was taken
# without --timestamps. It carries no year, so --year supplies one.
CLOCK_RE = re.compile(r"\b(?:INFO|WARN|ERROR)\s+(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\b")

LABELS = {
    "banner": "vLLM banner",
    "arch": "architecture resolved",
    "scratch": "worker: loading model from scratch",
    "weights": "weights loaded",
    "model_loaded": "model loaded (memory checkpoint)",
    "kv_mem": "KV cache memory sized",
    "kv_size": "KV cache tokens",
    "concurrency": "KV concurrency limit",
    "capturing": "CUDA graph capture",
    "headless": "rank joined headless (no API server here)",
    "mem_usage": "memory accounted",
    "free_mem": "device memory at startup",
    "init_engine": "engine initialised",
    "server_start": "HTTP server listening",
    "ready": "Application startup complete",
}

PATTERNS = [
    # Anchored to a dotted number on purpose: "Using FlashAttention version 2"
    # otherwise masquerades as the vLLM version.
    ("banner",       re.compile(r"version\s+(\d+\.\S+)")),
    ("args",         re.compile(r"non-default args:\s*(\{.*\})\s*$")),
    ("arch",         re.compile(r"Resolved architecture:\s*(\S+)")),
    ("scratch",      re.compile(r"Loading model from scratch")),
    ("weights",      re.compile(r"Loading weights took\s+([\d.]+)\s+seconds")),
    ("model_loaded", re.compile(r"Model loading took\s+([\d.]+)\s+GiB\s+memory\s+and\s+([\d.]+)\s+seconds")),
    ("kv_mem",       re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")),
    ("kv_size",      re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")),
    ("concurrency",  re.compile(r"Maximum concurrency for\s+([\d,]+)\s*tokens per request:\s*([\d.]+)x")),
    ("capturing",    re.compile(r"[Cc]apturing.*graph|[Gg]raph capturing")),
    # The headless rank never logs "Application startup complete" -- it has no API
    # server. This is the marker that says "this rank is rank 1, judge it differently".
    ("headless",     re.compile(r"[Hh]eadless multiproc executor")),
    # Memory accounting the loader prints once: the number to watch against the
    # GPU_MEMORY_UTILIZATION ceiling in dual.env.
    ("mem_usage",    re.compile(r"Actual usage is\s+([\d.]+)\s+GiB for consumed memory[^,]*,\s*([\d.]+)\s+GiB for peak activation")),
    ("free_mem",     re.compile(r"Free memory on device\s*\(([\d.]+)/([\d.]+)\s*GiB\)")),
    ("init_engine",  re.compile(r"init engine \(.*\) took\s+([\d.]+)\s*s")),
    ("server_start", re.compile(r"Starting vLLM server on\s+(\S+)")),
    ("ready",        re.compile(r"Application startup complete")),
]

# Signatures this pair has already burned hours on, keyed to RUNBOOK sections.
# Every one of these also appears in a HEALTHY boot, so each carries the hit count
# at which it stops being noise. A single shm_broadcast line during a 9-minute
# weight load is normal: the queue has no reader yet. Pitfall 2 was pathological
# at ~20 minutes of them, once per minute.
PITFALLS = [
    ("Pitfall 2", "vLLM deadlocks on its inter-node shm broadcast",
     re.compile(r"No available shared memory broadcast block found", re.I),
     "point VLLM_HOST_IP at the fabric address (WORKER_FABRIC_IP in dual.env)",
     10, "one-off stalls here are normal while a rank is still loading weights"),
    ("Pitfall 3", "FlashInfer autotune hangs the boot",
     re.compile(r"\[AutoTuner\]:\s*Tuning", re.I),
     "keep --no-enable-flashinfer-autotune in EXTRA_VLLM_ARGS",
     1, "autotune is disabled for a reason on GB10"),
    ("Pitfall 6", "RDMA registration wedges; only a cold boot clears it",
     re.compile(r"world_size=\d+", re.I),
     "repeated rendezvous lines with no progress means the pair is wedged",
     4, "a healthy TP2 boot logs this twice (DP leader + parallel_state)"),
    ("§3", "GPU_MEMORY_UTILIZATION ceiling on GB10",
     re.compile(r"NVRM:.*Out of memory|CUDA out of memory|NV_ERR_NO_MEMORY", re.I),
     "0.75 is the safe ceiling here -- raising it drifts into swap",
     1, "driver-level allocation failure"),
    ("§17", "a reboot leaves the pair half-up",
     re.compile(r"Broken pipe|TCPStore.*closed|Connection refused.*29501", re.I),
     "one rank died: restart the worker first, then the head",
     3, "a lone transport error at teardown is unremarkable"),
]


def match_milestone(text: str):
    for key, rx in PATTERNS:
        m = rx.search(text)
        if m:
            return key, m
    return None, None


def parse_ts(line: str, year: int):
    """(epoch_seconds, payload) for one docker log line, or None."""
    m = TS_RE.match(line)
    if m:
        base, frac, rest = m.groups()
        dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp() + float("0." + frac if frac else "0"), rest.strip()
    m = CLOCK_RE.search(line)
    if m:
        mo, day, hh, mm, ss = (int(x) for x in m.groups())
        try:
            dt = datetime(year, mo, day, hh, mm, ss, tzinfo=timezone.utc)
        except ValueError:
            return None
        return dt.timestamp(), line.strip()
    return None


def scan(lines: list[str], year: int) -> dict:
    """One pass over the log -> milestones, engine shape, pitfall hits."""
    events: list[dict] = []
    shape: dict = {}
    hits: list[dict] = []
    t0: float | None = None
    t: float | None = None

    for raw in lines:
        if not raw.strip():
            continue
        parsed = parse_ts(raw, year)
        if parsed:
            t, text = parsed
            if t0 is None:
                t0 = t
        else:
            text = raw.strip()
        rel = (t - t0) if (t is not None and t0 is not None) else 0.0

        key, m = match_milestone(text)
        if key and key != "args":
            if key == "banner":
                # First banner wins: later lines match the same shape.
                shape.setdefault("version", m.group(1))
                if any(e["key"] == "banner" for e in events):
                    key = None
            if key:
                events.append({"key": key, "t": rel, "abs": t,
                               "values": [g for g in m.groups() if g is not None],
                               "line": text[:200]})
        elif key == "args":
            try:
                # It is a python repr with single quotes, not JSON.
                shape.update(ast.literal_eval(m.group(1)))
            except (ValueError, SyntaxError):
                pass

        for section, title, rx, remedy, min_hits, quiet in PITFALLS:
            if rx.search(text):
                hits.append({"section": section, "title": title, "remedy": remedy,
                             "min_hits": min_hits, "quiet": quiet,
                             "t": rel, "line": text[:160]})
                break

    return {"t0": t0, "events": events, "shape": shape, "pitfalls": dedupe(hits)}


def dedupe(hits: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for h in hits:
        cur = out.get(h["section"])
        if cur:
            cur["count"] += 1
        else:
            out[h["section"]] = dict(h, count=1)
    return list(out.values())


# --- derived phases -----------------------------------------------------------
def first_t(events: list[dict], key: str):
    return next((e["t"] for e in events if e["key"] == key), None)


def last_t(events: list[dict], key: str):
    vals = [e["t"] for e in events if e["key"] == key]
    return vals[-1] if vals else None


def last_val(events: list[dict], key: str, idx: int = 0):
    vals = [e["values"] for e in events if e["key"] == key and len(e["values"]) > idx]
    if not vals:
        return None
    try:
        return float(vals[-1][idx].replace(",", ""))
    except ValueError:
        return None


def phases(events: list[dict]) -> dict:
    weights = [float(e["values"][0]) for e in events if e["key"] == "weights" and e["values"]]
    init_t = last_t(events, "init_engine")
    ready_t = last_t(events, "ready")
    keys = [e["key"] for e in events]
    p = {
        # Everything before the first loader line is not model I/O: the API
        # process booting, args parsing, architecture resolution, TP rendezvous.
        "startup_s": first_t(events, "scratch") or first_t(events, "weights"),
        # vLLM's own "Model loading took N seconds" is the phase the RUNBOOK
        # baselines (552 s): it spans every loader pass plus loader overhead. The
        # passes are 476.8 s of main weights and 62.1 s of the MTP draft -- the
        # breakdown is in the timeline table, the phase total is not where you
        # would look for a slow draft.
        "weights_s": last_val(events, "model_loaded", 1) or (sum(weights) or None),
        "loader_passes_s": sum(weights) or None,
        "model_loaded_s": last_val(events, "model_loaded", 1),
        "weights_gib": last_val(events, "model_loaded", 0),
        "weight_passes": len(weights),
        "init_engine_s": last_val(events, "init_engine", 0),
        "api_s": (ready_t - init_t) if (ready_t is not None and init_t is not None) else None,
        "total_s": ready_t,
        "kv_mem_gib": last_val(events, "kv_mem", 0),
        "kv_tokens": last_val(events, "kv_size", 0),
        "max_concurrency_x": last_val(events, "concurrency", 1),
        # A headless rank has no API server, so 'ready' never appears. Its own
        # work ends at its last milestone; only rank 0 can be judged on startup.
        "headless": "headless" in keys,
        "rank_done_s": events[-1]["t"] if events else None,
        "mem_consumed_gib": last_val(events, "mem_usage", 0),
        "mem_activation_gib": last_val(events, "mem_usage", 1),
        "device_total_gib": last_val(events, "free_mem", 1),
    }
    if p["weights_gib"] and p["weights_s"]:
        # Per rank: each rank reads its own shard copy off its own NVMe, so this
        # is the single-disk rate the loader achieves -- not a cluster figure.
        p["read_gib_s"] = p["weights_gib"] / p["weights_s"]
    return p


def compare(value, baseline, unit="s"):
    if value is None or baseline is None:
        return ""
    d = value - baseline
    pctd = 100 * d / baseline if baseline else 0
    if abs(pctd) < 3:
        return f"within 3% of baseline {baseline:g}{unit}"
    return f"{abs(d):.0f}{unit} {'slower' if d > 0 else 'faster'} than baseline {baseline:g}{unit} ({pctd:+.0f}%)"


def fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = float(s)
    return f"{s:.0f}s" if s < 90 else f"{s / 60:.1f}m"


def render(label: str, data: dict, args) -> dict:
    events, shape, pitfalls = data["events"], data["shape"], data["pitfalls"]
    bar = "─" * 78
    print(bar)
    print(f" BOOT TIMELINE — {label}")
    print(bar)
    bits = []
    if shape.get("version"):
        bits.append(f"vLLM {shape['version']}")
    if shape.get("tensor_parallel_size"):
        bits.append(f"TP{shape['tensor_parallel_size']}")
    if shape.get("max_model_len"):
        bits.append(f"ctx {shape['max_model_len']:,}")
    if shape.get("gpu_memory_utilization") is not None:
        bits.append(f"gpu_mem {shape['gpu_memory_utilization']}")
    if "speculative_config" in shape:  # only when the args banner was in this log
        spec = (shape.get("speculative_config") or {}).get("num_speculative_tokens")
        bits.append(f"MTP {spec}" if spec else "MTP off")
    if shape.get("quantization"):
        bits.append(str(shape["quantization"]))
    print(" " + "  ·  ".join(bits)) if bits else print(
        "  (no API arg banner in this log -- a headless rank prints none)")
    if data["t0"]:
        print(" " + datetime.fromtimestamp(data["t0"], timezone.utc).strftime(
            "log opens %Y-%m-%d %H:%M:%S UTC"))
    if not events:
        print("\n !! no vLLM milestones in this log. Wrong container, or captured")
        print("    without `docker logs --timestamps` and --year is off?")
        return {}

    weights_total = len([e for e in events if e["key"] == "weights"])
    print()
    print(f"   {'t+':>7} {'Δ':>7}  milestone")
    prev, seen_w = 0.0, 0
    for e in events:
        note = ""
        if e["key"] == "weights":
            seen_w += 1
            which = "  (MTP draft pass)" if weights_total > 1 and seen_w > 1 else ""
            note = f"  {float(e['values'][0]):.1f}s{which}"
        elif e["key"] == "model_loaded":
            note = f"  {float(e['values'][0]):.1f} GiB in {float(e['values'][1]):.1f}s"
        elif e["key"] == "kv_mem":
            note = f"  {e['values'][0]} GiB left for KV"
        elif e["key"] == "kv_size":
            note = f"  {e['values'][0]} tokens"
        elif e["key"] == "concurrency":
            note = f"  {e['values'][1]}x at {e['values'][0]} tok/request"
        elif e["key"] == "init_engine":
            note = f"  profile + KV + warmup {float(e['values'][0]):.1f}s"
        elif e["key"] == "arch":
            note = f"  {e['values'][0]}"
        elif e["key"] == "mem_usage":
            note = (f"  {float(e['values'][0]):.1f} GiB weights+non-torch, "
                    f"{float(e['values'][1]):.2f} GiB peak activation")
        elif e["key"] == "free_mem":
            note = f"  {e['values'][0]} of {e['values'][1]} GiB free"
        print(f"   {fmt_secs(e['t']):>7} {fmt_secs(e['t'] - prev):>7}  "
              f"{LABELS.get(e['key'], e['key'])}{note}")
        prev = e["t"]

    p = phases(events)
    print()
    print(f"   {'phase':<26} {'seconds':>9}   vs RUNBOOK measured baseline")
    print("   " + "-" * 72)
    base = BASELINE if args.baseline else {}
    rows = [("boot → loader starts", p["startup_s"], None),
            ("weights loaded", p["weights_s"], "weights_s"),
            ("profile + KV + warmup", p["init_engine_s"], "init_engine_s"),
            ("→ API serving", p["api_s"], None),
            ("TOTAL to ready", p["total_s"], "total_s")]
    if p["headless"]:
        rows[-1] = ("rank finished (headless)", p["rank_done_s"], None)
    for name, val, key in rows:
        if val is None:
            continue
        mark = ">> " if name.startswith("TOTAL") else "   "
        print(f" {mark}{name:<24} {fmt_secs(val):>9}   {compare(val, base.get(key) if key else None)}")
    if p.get("read_gib_s"):
        print(f"   {'NVMe read rate':<26} {p['read_gib_s']:>7.3f} GiB/s"
              f"   ({p['weights_gib']:.1f} GiB/rank, {p['weight_passes']} loader pass(es))")

    if p["total_s"] is None and not p["headless"]:
        print("\n !! this log never reached 'Application startup complete'.")
    elif p["headless"]:
        print("   (headless rank: it never logs 'startup complete' -- rank 0 owns the API)")
    elif args.budget and p["total_s"] > args.budget:
        print(f"\n !! FAIL total {p['total_s']:.0f}s exceeds the {args.budget:.0f}s budget")

    loud = [h for h in pitfalls if h["count"] >= h["min_hits"]]
    quiet = [h for h in pitfalls if h["count"] < h["min_hits"]]
    for h in loud:
        print(f"\n !! {h['section']} — {h['title']}  (seen x{h['count']}, "
              f"threshold {h['min_hits']})")
        print(f"    at t+{fmt_secs(h['t'])}: {h['line'][:120]}")
        print(f"    remedy: {h['remedy']}")
    if quiet:
        print()
        for h in quiet:
            print(f"   note: {h['section']} signature seen x{h['count']} "
                  f"(below the x{h['min_hits']} threshold) -- {h['quiet']}")
    return p


def watch(stream, args) -> int:
    """Follow a live boot, printing milestones as they land. Read-only."""
    t0 = last_evt = None
    warned: set[str] = set()
    seen_pitfalls: dict[str, int] = {}
    for raw in stream:
        parsed = parse_ts(raw.rstrip("\n"), args.year)
        if not parsed:
            continue
        t, text = parsed
        if t0 is None:
            t0 = t
            print(f">> following the boot from "
                  f"{datetime.fromtimestamp(t0, timezone.utc):%H:%M:%S} UTC — nothing here "
                  f"restarts or touches the server (Ctrl-C to stop)")
        rel = t - t0
        key, m = match_milestone(text)
        if key == "ready":
            print(f"   {fmt_secs(rel):>7}  {LABELS['ready']}")
            print(f"\n>> READY in {fmt_secs(rel)}  vs baseline {BASELINE['total_s']:.0f}s: "
                  f"{compare(rel, BASELINE['total_s'])}")
            if args.budget and rel > args.budget:
                print(f">> FAIL: over the {args.budget:.0f}s budget")
                return 1
            return 0
        if key and key not in ("args", "banner"):
            val = ""
            if key == "weights":
                val = f"  {float(m.group(1)):.1f}s"
            elif key == "model_loaded":
                val = f"  {float(m.group(1)):.1f} GiB in {float(m.group(2)):.1f}s"
            elif key == "init_engine":
                val = f"  {float(m.group(1)):.1f}s"
            elif key == "kv_size" and m.groups():
                val = f"  {m.group(1)} tokens"
            gap = f" (+{rel - last_evt:.0f}s)" if last_evt is not None else ""
            print(f"   t+{fmt_secs(rel):>6}{gap:>8}  {LABELS.get(key, key)}{val}", flush=True)
            last_evt = rel
        for section, title, rx, remedy, min_hits, quiet in PITFALLS:
            if not rx.search(text):
                continue
            seen_pitfalls[section] = seen_pitfalls.get(section, 0) + 1
            # Same threshold as the offline report: the first shm_broadcast stall
            # during a weight load is normal, the tenth is the Pitfall 2 deadlock.
            if seen_pitfalls[section] == min_hits and section not in warned:
                warned.add(section)
                print(f"   !! {section} — {title}: {remedy}", flush=True)
            break
        if args.stall and last_evt is not None and rel - last_evt > args.stall:
            stamp = f"stall-{int((rel - last_evt) // args.stall)}"
            if stamp not in warned:
                warned.add(stamp)
                print(f"   !! no new milestone for {(rel - last_evt) / 60:.1f} min. A normal "
                      f"weight load here takes ~9 min; past that suspect Pitfall 2 or 3.", flush=True)
    print("!! log ended before 'Application startup complete'")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", action="append", default=[], metavar="LABEL=PATH",
                    help="log file; repeatable (--logs head=a --logs worker=b). Default: stdin.")
    ap.add_argument("--label", default="head", help="label used when reading stdin")
    ap.add_argument("--year", type=int, default=time.gmtime().tm_year,
                    help="year assumed for logs captured without --timestamps")
    ap.add_argument("--budget", type=float, default=0,
                    help="exit 1 if total time to ready exceeds this many seconds")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="skip the RUNBOOK measured-boot-timeline comparison")
    ap.add_argument("--watch", action="store_true",
                    help="live: read stdin, print milestones as they arrive")
    ap.add_argument("--stall", type=float, default=600,
                    help="--watch: warn when no new milestone for this many seconds (0 = never)")
    ap.add_argument("--json", dest="json_path", default="",
                    help="also write machine-readable phases here")
    args = ap.parse_args()

    if args.watch:
        return watch(sys.stdin, args)

    streams: list[tuple[str, list[str]]] = []
    if args.logs:
        for spec in args.logs:
            label, _, path = spec.partition("=")
            if not path:
                sys.exit(f"!! --logs expects LABEL=PATH, got {spec!r}")
            if not os.path.exists(path):
                sys.exit(f"!! {path} is not readable")
            with open(path, errors="replace") as fh:
                streams.append((label, fh.readlines()))
    else:
        streams.append((args.label, sys.stdin.readlines()))

    summary: dict = {}
    ok = True
    for label, lines in streams:
        p = render(label, scan(lines, args.year), args)
        if not p:
            ok = False
            continue
        summary[label] = p
        if p.get("total_s") is None:
            # Only a rank that owns the API can be judged on 'startup complete'.
            if not p.get("headless"):
                ok = False
        elif args.budget and p["total_s"] > args.budget:
            ok = False

    if len(summary) > 1:
        # TP2 boots only as fast as its slowest rank; rank 0's number alone flatters.
        print("─" * 78)
        print(" TP2 PAIR — the boot is as slow as its slowest rank")
        for k, v in sorted(summary.items(), key=lambda kv: -(kv[1].get("total_s")
                                                            or kv[1].get("rank_done_s") or 0)):
            total = v.get("total_s")
            print(f"   {k:<10} weights {fmt_secs(v.get('weights_s')):>7}   "
                  f"init {fmt_secs(v.get('init_engine_s')):>7}   "
                  f"total {fmt_secs(total if total is not None else v.get('rank_done_s')):>7}"
                  f"{'  (headless: no API)' if total is None else ''}")
        worst = max(((v.get("total_s") or v.get("rank_done_s") or 0) for v in summary.values()),
                    default=0)
        if worst:
            print(f"   effective boot time {fmt_secs(worst)}")
            if args.budget and worst > args.budget:
                ok = False

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "baseline": BASELINE, "budget_s": args.budget or None,
                       "ranks": summary}, fh, indent=2)
        print(f"\n>> machine-readable copy: {args.json_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
