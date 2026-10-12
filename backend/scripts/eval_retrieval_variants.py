# scripts/eval_retrieval_variants.py
"""Query rewriting and hybrid retrieval against the golden set: which strategy finds the verses people expect?

scripts/eval_retrieval.py showed single-vector dense retrieval finds *a* relevant verse
for every golden question but *the* canonical verse only 56% of the time at k=5: it
matches question-shaped verses to question-shaped queries ("Who do you say I am?" for
*Who is Jesus?*) and short formulaic verses ("Grace and peace to you…" for *What is
grace?*). This script runs the same labels through the obvious remedies and reports
quality, added latency and added cost for each, so the choice is a measurement:

    query side                      retriever side
    ─────────────────────────────   ───────────────────────────────────────────
    q        the question as asked  dense    text-embedding-3-small -> Pinecone (production)
    strip    rule-based: drop the   bm25     Okapi BM25 over the vendored text, in-process
             question template               (app/services/lexical.py)
    hyde     LLM writes a short     hybrid   dense ⊕ bm25, reciprocal rank fusion
             passage in the voice
             of Scripture; that is
             the query (HyDE)
    q+hyde   q and the HyDE passage
             searched separately,
             lists fused: q anchors
             the result when the
             passage wanders
    multi    LLM writes 3 alternate
             queries; each is
             searched and the lists
             are fused with q

Every (query side, retriever) pair is an arm. Metrics are those of eval_retrieval.py
(hit@k, MRR, recall@k on the strict "canonical" and curated "relevant" label sets),
plus per arm the median retrieval-path latency and the rewrite cost per 1k queries.

    uv run python -m scripts.eval_retrieval_variants             # the default arms, k = 5 and 10
    uv run python -m scripts.eval_retrieval_variants --arms dense hybrid hyde+hybrid --show-unlabelled hyde+hybrid

LLM rewrites are cached in eval_results/retrieval_variants/rewrites.json (keyed by
strategy, model and prompt version) so re-runs are deterministic and free; pass --fresh
to regenerate them. Cost of a fresh run: 64 short gpt-5-nano calls + ~200 embeddings,
well under $0.05. Results: eval_results/retrieval_variants.json.

Caveat that applies to every number here: the "relevant" labels were curated by reading
the *dense* retriever's top-10, so an arm that surfaces different on-topic verses is
under-credited on that set. --show-unlabelled prints what an arm puts in its top-5 that
carries no label, which is the review queue for extending the labels.
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.config import default_reasoning_effort, settings, supports_temperature
from app.language import detect_language
from app.services.lexical import BM25Index, build_index, dedupe, rrf
from app.services.rag_service import RAGService, clean_text
from app.verses import load_parallel
from scripts.compare_models import PRICES
from scripts.eval_retrieval import GOLDEN, LABEL_SETS, load_golden

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "eval_results"
OUT = OUT_DIR / "retrieval_variants.json"
REWRITES = OUT_DIR / "retrieval_variants" / "rewrites.json"

PROMPT_VERSION = "1"
_STYLE = {"en": ("English", "the Berean Standard Bible"), "zh": ("Chinese (简体中文)", "和合本")}

HYDE_PROMPT = """Write what a Bible verse answering the question below would say.
Respond with 2-3 short sentences in the voice and vocabulary of Scripture ({style}), in {language}.
No references, headings or commentary — only the passage.

Question: {question}"""

MULTI_PROMPT = """Rewrite the question below as 3 alternative search queries for finding Bible verses that answer it.
Use declarative phrasing and the words such verses would actually use; make the three differ from each other.
Write them in {language}, one per line, with no numbering or other text.

Question: {question}"""

# Rule-based "strip": take the question template off and search for what is left.
_STRIP_EN = [
    r"^what does the bible (say|teach) about\s+",
    r"^what (is|are|was|were) (the )?",
    r"^who (is|was|were)\s+",
    r"^why (did|does|do|is|was)\s+",
    r"^how (should|do|does|did|was|were|can|is)\s+(i|we|you)?\s*",
    r"\s+according to the bible",
    r"\?+$",
]
_STRIP_ZH = [r"^圣经(里|中)?(关于|对|论)", r"是?怎么说的?", r"是谁", r"[？?]+$"]


def strip_template(question: str, lang: str) -> str:
    out = question.strip()
    for pattern in (_STRIP_ZH if lang == "zh" else _STRIP_EN):
        out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip()
    return out or question


# ---- the two network stages, memoised so each distinct call is made (and timed) once ----

class Timed:
    """Memo of (result, ms of the first call): arms share calls but are charged the real latency."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.memo: Dict[str, Tuple[object, float]] = {}

    def __call__(self, key: str, *args) -> Tuple[object, float]:
        if key not in self.memo:
            start = time.perf_counter()
            result = self.fn(*args)
            self.memo[key] = (result, (time.perf_counter() - start) * 1000)
        return self.memo[key]


class Retrievers:
    def __init__(self, svc: RAGService, pool: int, stem: bool = True):
        self.svc = svc
        self.pool = pool
        self.bm25: Dict[str, BM25Index] = {
            lang: build_index(settings.translation_for(lang), stem=stem) for lang in ("en", "zh")
        }
        self._embed = Timed(lambda texts: svc.embeddings.embed_documents(texts))
        self._query = Timed(self._pinecone)

    def _pinecone(self, vector: List[float], lang: str) -> List[Tuple[str, float]]:
        scored = self.svc.vectorstore.similarity_search_by_vector_with_score(
            vector, k=self.pool, namespace=settings.translation_for(lang)
        )
        return [(doc.id, round(float(score), 4)) for doc, score in scored]

    def dense(self, texts: Sequence[str], lang: str) -> Tuple[List[List[str]], float]:
        """One embedding call for all texts, then one Pinecone query each. Returns id lists and ms."""
        vectors, ms = self._embed("|".join(texts), list(texts))
        lists = []
        for text, vector in zip(texts, vectors):
            hits, q_ms = self._query(f"{lang}|{text}", vector, lang)
            lists.append([vid for vid, _ in hits])
            ms += q_ms
        return lists, ms

    def lexical(self, texts: Sequence[str], lang: str) -> Tuple[List[List[str]], float]:
        start = time.perf_counter()
        lists = [[vid for vid, _ in self.bm25[lang].search(text, k=self.pool)] for text in texts]
        return lists, (time.perf_counter() - start) * 1000


# ---- LLM rewrites, cached on disk --------------------------------------------------------

class Rewriter:
    def __init__(self, model: str, cache_path: Path, fresh: bool = False):
        from app.services.openai_client import client
        self.client = client
        self.model = model
        self.effort = default_reasoning_effort(model)
        self.path = cache_path
        self.cache: Dict = {} if fresh or not cache_path.exists() else json.loads(cache_path.read_text(encoding="utf-8"))
        self.calls = 0

    def _key(self, strategy: str) -> str:
        return f"{strategy}|{self.model}|v{PROMPT_VERSION}"

    def rewrite(self, strategy: str, question: str, lang: str) -> Dict:
        bucket = self.cache.setdefault(self._key(strategy), {})
        if question in bucket:
            return bucket[question]
        language, style = _STYLE[lang]
        template = HYDE_PROMPT if strategy == "hyde" else MULTI_PROMPT
        kwargs = {"model": self.model, "messages": [
            {"role": "user", "content": template.format(question=question, language=language, style=style)}]}
        if supports_temperature(self.model):
            kwargs["temperature"] = 0
        if self.effort:
            kwargs["reasoning_effort"] = self.effort
        start = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        ms = (time.perf_counter() - start) * 1000
        self.calls += 1
        text = (resp.choices[0].message.content or "").strip()
        bucket[question] = {
            "text": text, "ms": round(ms, 1),
            "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        return bucket[question]

    def cost_usd(self, entry: Dict) -> Optional[float]:
        price = PRICES.get(self.model)
        if not price:
            return None
        return (entry["tokens_in"] * price[0] + entry["tokens_out"] * price[1]) / 1_000_000


def split_queries(text: str) -> List[str]:
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln).strip().strip('"“”') for ln in text.splitlines()]
    return [ln for ln in lines if ln][:3]


# ---- arms --------------------------------------------------------------------------------

QUERY_SIDES = ("q", "strip", "hyde", "q+hyde", "multi")
RETRIEVERS = ("dense", "bm25", "hybrid")
# The default report. Any side+retriever pair is accepted by --arms; these are the ones that answer a question.
DEFAULT_ARMS = ("dense", "bm25", "hybrid", "strip+dense",
                "hyde+dense", "hyde+bm25", "hyde+hybrid", "q+hyde+dense", "q+hyde+hybrid",
                "multi+dense", "multi+hybrid")


def arm_name(side: str, retriever: str) -> str:
    return retriever if side == "q" else f"{side}+{retriever}"


def run_arm(side: str, retriever: str, question: str, lang: str, ret: Retrievers,
            rewriter: Optional[Rewriter], rrf_k: int, bm25_weight: float, bm25_skip_question: bool = False) -> Dict:
    """Ordered verse ids for one arm on one question, with the latency and cost it would add."""
    ms = 0.0
    cost = 0.0
    rewrite_ms = 0.0
    queries: List[str]
    if side == "q":
        queries = [question]
    elif side == "strip":
        queries = [strip_template(question, lang)]
    else:
        strategy = "hyde" if side.endswith("hyde") else side
        entry = rewriter.rewrite(strategy, question, lang)
        rewrite_ms = entry["ms"]
        ms += rewrite_ms
        cost += rewriter.cost_usd(entry) or 0.0
        if side == "hyde":
            queries = [entry["text"]]
        elif side == "q+hyde":
            queries = [question, entry["text"]]
        else:
            queries = [question] + split_queries(entry["text"])
    # When the question is fused with rewrites, BM25 over the question itself can be left out:
    # the question form is the lexical retriever's weak spot (see the bm25 arm).
    lexical_queries = queries[1:] if bm25_skip_question and side in ("q+hyde", "multi") else queries

    rankings: Dict[str, List[str]] = {}
    weights: Dict[str, float] = {}
    if retriever in ("dense", "hybrid"):
        lists, d_ms = ret.dense(queries, lang)
        ms += d_ms
        for i, ids in enumerate(lists):
            rankings[f"dense{i}"] = ids
    if retriever in ("bm25", "hybrid"):
        lists, l_ms = ret.lexical(lexical_queries, lang)
        ms += l_ms
        for i, ids in enumerate(lists):
            rankings[f"bm25{i}"] = ids
            weights[f"bm25{i}"] = bm25_weight
    if len(rankings) == 1:
        ids = next(iter(rankings.values()))
    else:
        ids = [vid for vid, _ in rrf(rankings, k=rrf_k, weights=weights)]
    return {"ids": dedupe(ids), "queries": queries, "ms": round(ms, 1), "rewrite_ms": round(rewrite_ms, 1),
            "cost_usd": cost}


def score(ids: List[str], labels: set, ks: Sequence[int]) -> Dict:
    max_k = max(ks)
    ranks = [i + 1 for i, vid in enumerate(ids[:max_k]) if vid in labels]
    out: Dict = {"first_hit_rank": ranks[0] if ranks else None, "mrr": (1 / ranks[0]) if ranks else 0.0}
    for k in ks:
        hits = sum(vid in labels for vid in ids[:k])
        out[f"hit@{k}"] = hits > 0
        out[f"recall@{k}"] = hits / len(labels) if labels else 0.0
    return out


def summarise(rows: List[Dict], ks: Sequence[int]) -> Dict:
    block = {"mrr": round(statistics.fmean(r["mrr"] for r in rows), 4)}
    for k in ks:
        block[f"hit@{k}"] = round(statistics.fmean(r[f"hit@{k}"] for r in rows), 4)
        block[f"recall@{k}"] = round(statistics.fmean(r[f"recall@{k}"] for r in rows), 4)
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--pool", type=int, default=20, help="candidates taken from each retriever before fusion")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--bm25-weight", type=float, default=1.0, help="RRF weight of the BM25 lists (dense = 1)")
    ap.add_argument("--no-stem", action="store_true", help="BM25 without Snowball stemming (English)")
    ap.add_argument("--bm25-skip-question", action="store_true",
                    help="in q+hyde / multi arms, run BM25 only over the rewrites, not the question itself")
    ap.add_argument("--rewrite-model", default=settings.chat_model)
    ap.add_argument("--fresh", action="store_true", help="ignore cached LLM rewrites")
    ap.add_argument("--arms", nargs="+", default=None,
                    help=f"any query side × retriever, e.g. dense hybrid hyde+hybrid (default: {' '.join(DEFAULT_ARMS)})")
    ap.add_argument("--show-unlabelled", metavar="ARM", default=None,
                    help="print this arm's top-5 verses that carry no label (default: best arm by canonical hit@5)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ks = sorted(set(args.k))
    top_k = settings.top_k
    arms = args.arms or list(DEFAULT_ARMS)
    unknown = [a for a in arms if a not in {arm_name(s, r) for s in QUERY_SIDES for r in RETRIEVERS}]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; sides={QUERY_SIDES} retrievers={RETRIEVERS}")
    items = load_golden(args.golden)
    known = load_parallel()
    texts = {lang: {vid: v.texts[lang] for vid, v in known.items() if lang in v.texts} for lang in ("en", "zh")}

    print(f"{len(items)} questions · {len(arms)} arms · pool={args.pool} · rrf_k={args.rrf_k} · bm25_weight={args.bm25_weight}"
          f" · rewrite model={args.rewrite_model} ({default_reasoning_effort(args.rewrite_model) or 'no effort param'})")
    t0 = time.perf_counter()
    ret = Retrievers(RAGService(settings), pool=args.pool, stem=not args.no_stem)
    print(f"BM25 indexes built in {time.perf_counter() - t0:.1f}s "
          f"({len(ret.bm25['en'])} en / {len(ret.bm25['zh'])} zh verses)")
    rewriter = Rewriter(args.rewrite_model, REWRITES, fresh=args.fresh) if any("hyde" in a or "multi" in a for a in arms) else None

    per_question: List[Dict] = []
    for item in items:
        lang = detect_language(item["question"])
        row: Dict = {"question": item["question"], "language": lang, "arms": {}}
        for arm in arms:
            side, _, retriever = arm.rpartition("+") if "+" in arm else ("q", "", arm)
            result = run_arm(side, retriever, item["question"], lang, ret, rewriter, args.rrf_k, args.bm25_weight,
                             args.bm25_skip_question)
            for label_set in LABEL_SETS:
                result[label_set] = score(result["ids"], set(item.get(label_set) or item["relevant"]), ks)
            result["top"] = [{"id": vid, "text": clean_text(texts[lang].get(vid, ""))[:80]} for vid in result["ids"][:max(ks)]]
            result["ids"] = result["ids"][:max(ks)]
            row["arms"][arm] = result
        per_question.append(row)
        print(f"  · {item['question']}")

    # ---- summary --------------------------------------------------------------------------
    summary: Dict[str, Dict] = {}
    for arm in arms:
        results = [row["arms"][arm] for row in per_question]
        block = {label_set: summarise([r[label_set] for r in results], ks) for label_set in LABEL_SETS}
        block["latency_ms_p50"] = round(statistics.median(r["ms"] for r in results), 0)
        block["rewrite_ms_p50"] = round(statistics.median(r["rewrite_ms"] for r in results), 0)
        block["cost_per_1k_queries_usd"] = round(1000 * statistics.fmean(r["cost_usd"] for r in results), 4)
        summary[arm] = block

    head = "| arm | canonical hit@5 | hit@10 | MRR | relevant hit@5 | recall@5 | MRR | retrieval p50 | rewrite p50 | $ / 1k |"
    print(f"\n{head}\n|" + "---|" * (head.count("|") - 1))
    base = summary.get("dense")
    for arm, s in summary.items():
        c, r = s["canonical"], s["relevant"]
        delta = f" ({c['hit@5'] - base['canonical']['hit@5']:+.0%})" if base and arm != "dense" else ""
        print(f"| {arm} | {c['hit@5']:.0%}{delta} | {c['hit@10']:.0%} | {c['mrr']:.2f} "
              f"| {r['hit@5']:.0%} | {r['recall@5']:.0%} | {r['mrr']:.2f} "
              f"| {s['latency_ms_p50']:.0f} ms | {s['rewrite_ms_p50']:.0f} ms | ${s['cost_per_1k_queries_usd']:.3f} |")

    # Per-question canonical first-hit rank, one column per arm: where each arm wins or loses.
    print(f"\ncanonical first-hit rank (within top {max(ks)}; '-' = none):")
    print("| question | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for row in per_question:
        cells = [row["arms"][a]["canonical"]["first_hit_rank"] for a in arms]
        print(f"| {row['question'][:44]} | " + " | ".join("-" if c is None else str(c) for c in cells) + " |")

    show = args.show_unlabelled or max(summary, key=lambda a: (summary[a]["canonical"]["hit@5"], summary[a]["canonical"]["mrr"]))
    if show in summary:
        print(f"\nunlabelled verses in the top {top_k} of [{show}] — the review queue for the 'relevant' labels:")
        for row, item in zip(per_question, items):
            labelled = set(item["relevant"]) | set(item.get("canonical", []))
            new = [t for t in row["arms"][show]["top"][:top_k] if t["id"] not in labelled]
            if new:
                print(f"  · {row['question']}")
                for t in new:
                    print(f"      {t['id']:<22} {t['text']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "config": {"embedding_model": settings.embedding_model, "index": settings.pinecone_index_name,
                   "pool": args.pool, "rrf_k": args.rrf_k, "bm25_weight": args.bm25_weight, "stem": not args.no_stem,
                   "bm25_skip_question": args.bm25_skip_question,
                   "rewrite_model": args.rewrite_model, "rewrite_effort": default_reasoning_effort(args.rewrite_model),
                   "prompt_version": PROMPT_VERSION, "ks": ks},
        "summary": summary, "questions": per_question,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📄 wrote {args.out}" + (f" · {rewriter.calls} new LLM rewrites" if rewriter else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
