# scripts/compare_models.py
"""Compare chat models on identical retrieved context: latency, tokens, cost.

Retrieval is done once per question (it depends only on the embedding model),
then every candidate model answers from the same verses, so the comparison
isolates the generation step. Each model's answers are written as a JSONL file
in the same shape as app/logs/query_log.jsonl, so quality can be scored with
the existing Ragas script:

    python -m scripts.compare_models --models gpt-4o-mini gpt-6-luna gpt-5-nano --limit 20
    python -m scripts.eval_with_ragas --log eval_results/model_compare/gpt-6-luna.jsonl \
                                      --out eval_results/model_compare/gpt-6-luna_ragas.csv

Prices are per 1M tokens from https://developers.openai.com/api/docs/pricing
(fetched 2026-09-30); pass --prices prices.json to override.
"""

import argparse
import dataclasses
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import default_reasoning_effort, settings
from app.language import detect_language
from app.services.rag_service import RAGService

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "benchmark_questions.txt"
OUT_DIR = HERE.parent / "eval_results" / "model_compare"

# (input, output) USD per 1M tokens
PRICES: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6.1-sol": (2.00, 10.00),
}


def read_questions(limit: Optional[int]) -> List[str]:
    lines = [ln.strip() for ln in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()]
    qs = [ln for ln in lines if ln and not ln.startswith("#")]
    return qs[:limit] if limit else qs


def pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def run_model(model: str, effort: Optional[str], contexts: List[dict], prices: Dict[str, tuple]) -> dict:
    cfg = dataclasses.replace(settings, chat_model=model, chat_reasoning_effort=effort)
    svc = RAGService(cfg)
    price_in, price_out = prices.get(model, (None, None))

    rows = []
    errors = 0
    for ctx in contexts:
        if not ctx["docs"]:
            continue
        start = time.perf_counter()
        try:
            msg = svc.generate_message(ctx["question"], ctx["docs"], ctx["lang"])
        except Exception as e:  # a wrong model id or rejected parameter should not kill the run
            errors += 1
            print(f"    ✗ {model}: {type(e).__name__}: {str(e)[:160]}")
            if errors >= 3:
                print(f"    giving up on {model} after 3 errors")
                break
            continue
        latency = (time.perf_counter() - start) * 1000
        usage = msg.usage_metadata or {}
        tokens_in = usage.get("input_tokens")
        tokens_out = usage.get("output_tokens")
        cost = None
        if price_in is not None and tokens_in is not None and tokens_out is not None:
            cost = (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
        rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": ctx["question"],
                "answer": answer.strip(),
                "verses": ctx["verses"],
                "cache": None,
                "model": model,
                "reasoning_effort": effort,
                "latency_ms": round(latency, 1),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost_usd": cost,
            }
        )
        print(f"    {latency:6.0f}ms  in={tokens_in} out={tokens_out}  {ctx['question'][:60]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{model}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lat = [r["latency_ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    outs = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    unsure = sum(1 for r in rows if "not sure" in r["answer"].lower())
    return {
        "model": model,
        "reasoning_effort": effort,
        "answered": len(rows),
        "errors": errors,
        "latency_p50_ms": pct(lat, 50),
        "latency_p95_ms": pct(lat, 95),
        "latency_mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        "output_tokens_mean": round(statistics.fmean(outs), 1) if outs else None,
        "cost_per_1k_queries_usd": round(statistics.fmean(costs) * 1000, 4) if costs else None,
        "not_sure_rate": round(unsure / len(rows), 3) if rows else None,
    }


def markdown_table(summaries: List[dict]) -> str:
    head = ("| model | effort | n | p50 | p95 | out tokens | $ / 1k queries | \"not sure\" rate | errors |\n"
            "|---|---|---|---|---|---|---|---|---|")
    fmt = lambda v, suffix="": "-" if v is None else f"{v:,.0f}{suffix}" if isinstance(v, (int, float)) and suffix == " ms" else (f"{v}" if v is not None else "-")
    rows = [
        f"| {s['model']} | {s['reasoning_effort'] or '-'} | {s['answered']} | {fmt(s['latency_p50_ms'], ' ms')} "
        f"| {fmt(s['latency_p95_ms'], ' ms')} | {s['output_tokens_mean'] if s['output_tokens_mean'] is not None else '-'} "
        f"| {('$%.4f' % s['cost_per_1k_queries_usd']) if s['cost_per_1k_queries_usd'] is not None else '-'} "
        f"| {s['not_sure_rate'] if s['not_sure_rate'] is not None else '-'} | {s['errors']} |"
        for s in summaries
    ]
    return "\n".join([head, *rows])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-6-luna", "gpt-5-nano"])
    ap.add_argument("--effort", default=None,
                    help="reasoning_effort for every model (default: each family's cheapest, see app.config)")
    ap.add_argument("--limit", type=int, default=20, help="number of questions from benchmark_questions.txt")
    ap.add_argument("--prices", type=Path, default=None, help="JSON {model: [input, output]} per 1M tokens")
    args = ap.parse_args()

    prices = dict(PRICES)
    if args.prices:
        prices.update({k: tuple(v) for k, v in json.loads(args.prices.read_text()).items()})

    questions = read_questions(args.limit)
    print(f"retrieving context for {len(questions)} questions with {settings.embedding_model} ...")
    base = RAGService(settings)
    contexts = []
    for q in questions:
        lang = detect_language(q)
        docs = base.retrieve(base.embed(q), settings.translation_for(lang))
        contexts.append({"question": q, "lang": lang, "docs": docs, "verses": RAGService.to_verses(docs, lang)})
    with_context = sum(1 for c in contexts if c["docs"])
    print(f"  {with_context}/{len(questions)} questions have verses above SCORE_THRESHOLD={settings.score_threshold}")

    summaries = []
    for model in args.models:
        effort = args.effort or default_reasoning_effort(model)
        print(f"\n▶ {model} (reasoning_effort={effort})")
        summaries.append(run_model(model, effort, contexts, prices))

    print("\n" + markdown_table(summaries))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"ran_at": datetime.now(timezone.utc).isoformat(), "questions": len(questions),
                    "embedding_model": settings.embedding_model, "models": summaries}, indent=2),
        encoding="utf-8",
    )
    print(f"\n📄 per-model answers in {OUT_DIR}/<model>.jsonl — score them with scripts/eval_with_ragas.py --log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
