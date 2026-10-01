# scripts/benchmark_cache.py
"""Measure what the cache actually buys.

Runs against a live backend and reports, per phase, client-observed latency
(p50 / p95 / max), the server's own stage timings, and how many upstream
OpenAI calls were avoided. Phases:

  cold           every distinct question once      -> expect MISS
  warm-exact     the same questions again          -> expect HIT-EXACT
  warm-semantic  paraphrases of those questions    -> HIT-SEMANTIC if similar enough
  mixed          N requests drawn Zipf-style from   -> realistic repeat traffic; the
                 questions + paraphrases              hit rate here backs any "% fewer
                                                      API calls" claim

Usage (backend running on :8000, Redis reachable):

    python -m scripts.benchmark_cache --flush            # start from an empty cache
    python -m scripts.benchmark_cache --mixed 300 --concurrency 4

Results are printed as a table and written to eval_results/cache_benchmark.json.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "benchmark_questions.txt"
PARAPHRASES_FILE = HERE / "benchmark_paraphrases.txt"
OUT_FILE = HERE.parent / "eval_results" / "cache_benchmark.json"

# Upstream OpenAI requests a query costs, by how it was served. A miss pays
# moderation + embedding + chat completion; a semantic hit still needs the
# first two (to find the neighbour) but skips the chat completion, which is
# by far the slowest and most expensive of the three.
UPSTREAM_CALLS = {"MISS": 3, "HIT-SEMANTIC": 2, "HIT-EXACT": 0}


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def one_request(base_url: str, question: str) -> Dict:
    start = time.perf_counter()
    try:
        res = requests.post(f"{base_url}/api/query", json={"question": question}, timeout=120)
        elapsed = (time.perf_counter() - start) * 1000
        body = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        meta = body.get("meta") or {}
        return {
            "question": question,
            "status": res.status_code,
            "cache": res.headers.get("X-Cache", "?"),
            "client_ms": elapsed,
            "server_ms": meta.get("timings_ms", {}).get("total"),
            "timings_ms": meta.get("timings_ms", {}),
            "embedding_cached": meta.get("embedding_cached"),
            "semantic_similarity": meta.get("semantic_similarity"),
            "verses": len(body.get("verses") or []),
        }
    except requests.RequestException as e:
        return {"question": question, "status": 0, "cache": "ERROR", "client_ms": (time.perf_counter() - start) * 1000,
                "server_ms": None, "timings_ms": {}, "error": str(e), "verses": 0}


def run_phase(name: str, base_url: str, questions: List[str], concurrency: int) -> Dict:
    print(f"\n▶ {name}: {len(questions)} requests, concurrency={concurrency}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda q: one_request(base_url, q), questions))
    wall = time.perf_counter() - started
    summary = summarize(name, rows, wall)
    print_summary(summary)
    return {"summary": summary, "rows": rows}


def pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


def summarize(name: str, rows: List[Dict], wall_seconds: float) -> Dict:
    ok = [r for r in rows if r["status"] == 200]
    client = [r["client_ms"] for r in ok]
    server = [r["server_ms"] for r in ok if r["server_ms"] is not None]
    outcomes = Counter(r["cache"] for r in rows)

    # Average stage breakdown, only over requests that ran that stage.
    stages: Dict[str, List[float]] = defaultdict(list)
    for r in ok:
        for stage, ms in (r.get("timings_ms") or {}).items():
            if stage != "total":
                stages[stage].append(ms)

    upstream = sum(UPSTREAM_CALLS.get(r["cache"], 3) for r in ok)
    baseline = 3 * len(ok)
    return {
        "phase": name,
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "outcomes": dict(outcomes),
        "hit_rate": round(sum(v for k, v in outcomes.items() if k.startswith("HIT")) / len(ok), 4) if ok else None,
        "client_ms": {"p50": pct(client, 50), "p95": pct(client, 95), "max": max(client) if client else None,
                      "mean": statistics.fmean(client) if client else None},
        "server_ms": {"p50": pct(server, 50), "p95": pct(server, 95)},
        "stage_mean_ms": {k: round(statistics.fmean(v), 1) for k, v in sorted(stages.items())},
        "upstream_calls": upstream,
        "upstream_calls_without_cache": baseline,
        "upstream_calls_avoided_pct": round(100 * (1 - upstream / baseline), 1) if baseline else None,
        "wall_seconds": round(wall_seconds, 2),
        "throughput_rps": round(len(rows) / wall_seconds, 2) if wall_seconds else None,
    }


def fmt_ms(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:,.0f}"


def print_summary(s: Dict) -> None:
    c = s["client_ms"]
    print(f"  outcomes: {s['outcomes']}   hit rate: {s['hit_rate']}")
    print(f"  latency (client) p50={fmt_ms(c['p50'])}ms  p95={fmt_ms(c['p95'])}ms  max={fmt_ms(c['max'])}ms  "
          f"| server p50={fmt_ms(s['server_ms']['p50'])}ms")
    if s["stage_mean_ms"]:
        print("  stage means: " + "  ".join(f"{k}={v:.0f}ms" for k, v in s["stage_mean_ms"].items()))
    print(f"  upstream OpenAI calls: {s['upstream_calls']} vs {s['upstream_calls_without_cache']} uncached "
          f"({s['upstream_calls_avoided_pct']}% avoided)   throughput={s['throughput_rps']} req/s")
    if s["errors"]:
        print(f"  ⚠ {s['errors']} errors")


def markdown_table(summaries: List[Dict]) -> str:
    head = "| phase | n | outcomes | hit rate | p50 | p95 | upstream calls avoided |\n|---|---|---|---|---|---|---|"
    rows = [
        f"| {s['phase']} | {s['ok']} | {s['outcomes']} | {s['hit_rate']} | {fmt_ms(s['client_ms']['p50'])} ms "
        f"| {fmt_ms(s['client_ms']['p95'])} ms | {s['upstream_calls_avoided_pct']}% |"
        for s in summaries
    ]
    return "\n".join([head, *rows])


def zipf_sample(pool: List[str], n: int, seed: int, exponent: float = 1.0) -> List[str]:
    """Real question traffic is heavy-tailed: a few questions are asked constantly."""
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** exponent for i in range(len(pool))]
    return rng.choices(pool, weights=weights, k=n)


def flush_caches(flush_cmd: Optional[str] = None) -> None:
    # Imports the app to reuse its configured cache objects (reads backend/.env).
    from app.main import cache, semantic_cache

    if flush_cmd:
        # The backend under test keeps its Redis somewhere this process cannot
        # reach (e.g. inside a cluster): run the caller's command instead, such as
        #   kubectl -n bible-reference-ai exec deploy/bible-redis -- redis-cli FLUSHALL
        import subprocess
        subprocess.run(flush_cmd, shell=True, check=True, capture_output=True)
        print(f"🧹 ran flush command: {flush_cmd}")
    else:
        deleted = cache.flush()
        print(f"🧹 flushed {deleted} Redis keys")
    semantic_cache.flush()  # Pinecone is shared, so this reaches the backend under test too
    print("🧹 flushed the semantic-cache namespace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--mixed", type=int, default=0, help="number of Zipf-sampled requests for the mixed phase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flush", action="store_true", help="empty both cache tiers before starting")
    ap.add_argument("--flush-cmd", default=None,
                    help="shell command that empties the backend's Redis when it is not the local one "
                         "(e.g. 'kubectl -n bible-reference-ai exec deploy/bible-redis -- redis-cli FLUSHALL')")
    ap.add_argument("--out", type=Path, default=OUT_FILE, help=f"where to write the results (default {OUT_FILE})")
    ap.add_argument("--skip-semantic", action="store_true")
    args = ap.parse_args()

    questions = read_lines(QUESTIONS_FILE)
    paraphrases = read_lines(PARAPHRASES_FILE)
    if not questions:
        print(f"no questions in {QUESTIONS_FILE}")
        return 1

    health = requests.get(f"{args.base_url}/health", timeout=10).json()
    print(f"backend: {args.base_url}  health={health}")
    if health.get("redis") != "up":
        print("⚠ Redis is down: every request will be a MISS and the numbers below only show the uncached path.")

    if args.flush:
        flush_caches(args.flush_cmd)

    phases = []
    phases.append(run_phase("cold", args.base_url, questions, args.concurrency))
    phases.append(run_phase("warm-exact", args.base_url, questions, args.concurrency))
    if paraphrases and not args.skip_semantic:
        phases.append(run_phase("warm-semantic", args.base_url, paraphrases, args.concurrency))
    if args.mixed:
        # Start from an empty cache so the hit rate reflects repetition in the
        # traffic itself, not what the earlier phases left behind.
        if args.flush:
            flush_caches(args.flush_cmd)
        pool = questions + ([] if args.skip_semantic else paraphrases)
        sample = zipf_sample(pool, args.mixed, args.seed)
        print(f"  ({len(set(sample))} distinct questions in {len(sample)} requests)")
        phases.append(run_phase("mixed", args.base_url, sample, args.concurrency))

    summaries = [p["summary"] for p in phases]
    print("\n" + markdown_table(summaries))

    stats = requests.get(f"{args.base_url}/api/cache/stats", timeout=10).json()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "ran_at": datetime.now(timezone.utc).isoformat(),
                "base_url": args.base_url,
                "concurrency": args.concurrency,
                "config": stats.get("config"),
                "semantic_cache": stats.get("semantic_cache"),
                "server_stats_after": {k: stats.get(k) for k in ("exact_hits", "semantic_hits", "misses", "requests", "hit_rate")},
                "phases": [p["summary"] for p in phases],
                "requests": [r for p in phases for r in p["rows"]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n📄 wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
