# scripts/eval_retrieval.py
"""Retrieval quality against the hand-labelled golden set. Deterministic, no LLM judge.

scripts/golden_set.jsonl lists, per question, two label sets (ids in the
index's "Book.Chapter.Verse" form):

    canonical  the famous verses a reader would name from memory — strict
    relevant   canonical plus every other on-topic verse found while reviewing
               the retriever's top-10 — what a reader would accept

Each question is embedded once, the top-K verses are pulled from Pinecone, and
for each label set:

    hit@k                any labelled verse in the top k            <- the headline number
    MRR                  1 / rank of the first labelled verse
    recall@k             share of the labelled verses in the top k   (lower bound: lists are partial)
    precision@threshold  share of verses scoring >= the threshold that are labelled (lower bound)

Report both: the strict number says how often the retriever finds *the*
verse people expect; the curated number says how often it finds *an* on-topic
verse. The curated set was built by looking at retriever output, so on its own
it is not an unbiased test — keep that in mind when quoting it.

    uv run python -m scripts.eval_retrieval            # k = 5 and 10
    uv run python -m scripts.eval_retrieval --k 3 5 10 20

Cost: one embedding call per question (~$0.0001 total). Output goes to
eval_results/retrieval_eval.json, and the questions that miss at k=5 are
printed with what came back instead, which is what you look at to tune
TOP_K / SCORE_THRESHOLD or to spot a wrong label. scripts/eval_retrieval_variants.py
scores query rewriting and hybrid (dense + BM25) retrieval on the same labels.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.language import detect_language
from app.services.rag_service import RAGService, clean_text, verse_reference
from app.verses import load_parallel

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden_set.jsonl"
OUT = HERE.parent / "eval_results" / "retrieval_eval.json"
LABEL_SETS = ("canonical", "relevant")


def load_golden(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows


def evaluate_question(svc: RAGService, item: Dict, ks: List[int]) -> Dict:
    max_k = max(ks)
    lang = detect_language(item["question"])
    vector = svc.embed(item["question"])
    scored = svc.vectorstore.similarity_search_by_vector_with_score(
        vector, k=max_k, namespace=settings.translation_for(lang)
    )
    retrieved = [
        {"id": doc.id or f'{doc.metadata.get("book")}.{doc.metadata.get("chapter")}.{doc.metadata.get("verse")}',
         "ref": verse_reference(doc.metadata), "score": round(float(score), 4),
         "text": clean_text(doc.page_content)[:90]}
        for doc, score in scored
    ]
    above = [r for r in retrieved[: settings.top_k] if r["score"] >= settings.threshold_for(lang)]
    result = {
        "question": item["question"],
        "language": lang,
        "retrieved": retrieved,
        "above_threshold": len(above),
    }
    for label_set in LABEL_SETS:
        labels = set(item.get(label_set) or item["relevant"])
        ranks = [i + 1 for i, r in enumerate(retrieved) if r["id"] in labels]
        metrics = {
            "labels": sorted(labels),
            "first_hit_rank": ranks[0] if ranks else None,
            "mrr": (1 / ranks[0]) if ranks else 0.0,
            "precision_at_threshold": (sum(r["id"] in labels for r in above) / len(above)) if above else None,
        }
        for k in ks:
            hits = sum(r["id"] in labels for r in retrieved[:k])
            metrics[f"hit@{k}"] = hits > 0
            metrics[f"recall@{k}"] = hits / len(labels) if labels else 0.0
        result[label_set] = metrics
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = ap.parse_args()

    items = load_golden(args.golden)
    ks = sorted(set(args.k) | {settings.top_k})
    print(f"{len(items)} questions · embedding={settings.embedding_model} · index={settings.pinecone_index_name} "
          f"· TOP_K={settings.top_k} · thresholds en={settings.score_threshold} zh={settings.score_threshold_zh}")

    # A label that is not a real verse id can never be hit; catch typos before spending anything.
    known = load_parallel()
    unknown = sorted({vid for item in items for ls in LABEL_SETS for vid in item.get(ls, []) if vid not in known})
    if unknown:
        print(f"⚠ {len(unknown)} golden-set ids are not in the vendored translations: {unknown[:10]}")

    svc = RAGService(settings)
    results = [evaluate_question(svc, item, ks) for item in items]

    summary: Dict[str, object] = {
        "questions": len(results),
        "mean_above_threshold": round(statistics.fmean(r["above_threshold"] for r in results), 2),
    }
    for label_set in LABEL_SETS:
        rows = [r[label_set] for r in results]
        block: Dict[str, object] = {"mrr": round(statistics.fmean(m["mrr"] for m in rows), 4)}
        prec = [m["precision_at_threshold"] for m in rows if m["precision_at_threshold"] is not None]
        block["precision_at_threshold"] = round(statistics.fmean(prec), 4) if prec else None
        for k in ks:
            block[f"hit@{k}"] = round(statistics.fmean(m[f"hit@{k}"] for m in rows), 4)
            block[f"recall@{k}"] = round(statistics.fmean(m[f"recall@{k}"] for m in rows), 4)
        summary[label_set] = block

    print(f"\n| metric | canonical (strict) | relevant (curated) |\n|---|---|---|")
    for key in summary["canonical"]:
        print(f"| {key} | {summary['canonical'][key]} | {summary['relevant'][key]} |")
    print(f"| mean verses ≥ threshold (of {settings.top_k}) | {summary['mean_above_threshold']} | |")

    for label_set in LABEL_SETS:
        misses = [r for r in results if not r[label_set][f"hit@{settings.top_k}"]]
        if misses:
            print(f"\n✗ [{label_set}] {len(misses)} question(s) with no labelled verse in the top {settings.top_k}:")
            for r in misses:
                print(f"  · {r['question']}")
                for got in r["retrieved"][: settings.top_k]:
                    print(f"      {got['score']:.3f}  {got['ref']:<22} {got['text']}")
        else:
            print(f"\n✓ [{label_set}] every question has a labelled verse in the top {settings.top_k}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"ran_at": datetime.now(timezone.utc).isoformat(), "config": {
            "embedding_model": settings.embedding_model, "index": settings.pinecone_index_name,
            "top_k": settings.top_k, "score_threshold": settings.score_threshold,
            "score_threshold_zh": settings.score_threshold_zh},
            "summary": summary, "questions": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n📄 wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
