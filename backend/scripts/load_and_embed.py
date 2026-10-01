# scripts/load_and_embed.py
"""Embed the vendored translations and upsert them into Pinecone, one namespace each.

Every vector carries both parallel texts in its metadata, so a hit in either
namespace can be shown bilingually without a second lookup:

    namespace "bsb":  vector = embedding(English text)   metadata.text = English,  text_en, text_zh
    namespace "cuv":  vector = embedding(Chinese text)   metadata.text = Chinese,  text_en, text_zh

Verse ids ("John.3.16") are identical across namespaces.

    uv run python -m scripts.load_and_embed --dry-run              # parse, align, count, estimate cost
    uv run python -m scripts.load_and_embed                        # both translations (~10 min, ~$0.05)
    uv run python -m scripts.load_and_embed --translations cuv --purge
    uv run python -m scripts.load_and_embed --delete-namespace ""  # drop the legacy KJV vectors
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from app.config import settings
from app.verses import TRANSLATIONS, ParallelVerse, load_parallel

BATCH_SIZE = 100          # verses per embedding request / upsert
DEFAULT_CONCURRENCY = 6   # batches in flight at once; each is one OpenAI call + one Pinecone upsert


def chunk(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def progress(current: int, total: int, bar_len: int = 24) -> str:
    filled = int(bar_len * current / total)
    return f"[{'█' * filled}{'░' * (bar_len - filled)}] {100 * current / total:5.1f}%"


def records_for(code: str, verses: Dict[str, ParallelVerse]) -> List[dict]:
    """The verses this translation has, in canonical order, with parallel texts attached."""
    lang = TRANSLATIONS[code].lang
    out = []
    for v in verses.values():
        if lang not in v.texts:
            continue
        out.append(
            {
                "id": v.id,
                "text": v.texts[lang],
                "metadata": {
                    "book": v.book,
                    "chapter": v.chapter,
                    "verse": v.verse,
                    "usfm": v.usfm,
                    "translation": code,
                    "text": v.texts[lang],
                    **({"text_en": v.texts["en"]} if "en" in v.texts else {}),
                    **({"text_zh": v.texts["zh"]} if "zh" in v.texts else {}),
                },
            }
        )
    return out


def estimate_tokens(records: List[dict], lang: str) -> int:
    # Rough: ~4 chars/token for English, ~1.3 tokens/char for Chinese.
    chars = sum(len(r["text"]) for r in records)
    return int(chars / 4) if lang == "en" else int(chars * 1.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--translations", nargs="+", default=list(TRANSLATIONS), choices=list(TRANSLATIONS))
    ap.add_argument("--dry-run", action="store_true", help="parse and report; no OpenAI or Pinecone calls")
    ap.add_argument("--limit", type=int, default=0, help="only the first N verses per translation (smoke test)")
    ap.add_argument("--purge", action="store_true", help="delete each target namespace before loading")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"batches embedded and upserted in parallel (default {DEFAULT_CONCURRENCY})")
    ap.add_argument("--delete-namespace", default=None, metavar="NS",
                    help='delete this namespace and exit ("" = the default namespace, where the old KJV vectors live)')
    args = ap.parse_args()

    if args.delete_namespace is not None:
        from app.services.pinecone_client import get_index
        get_index().delete(delete_all=True, namespace=args.delete_namespace)
        print(f"🗑  deleted namespace {args.delete_namespace!r}")
        return 0

    print(f"🔍 loading {', '.join(TRANSLATIONS[c].name for c in args.translations)} ...")
    verses = load_parallel(tuple(TRANSLATIONS))  # always align against all translations
    print(f"✅ {len(verses)} distinct verse ids across translations")

    plan = {}
    for code in args.translations:
        recs = records_for(code, verses)
        if args.limit:
            recs = recs[: args.limit]
        both = sum(1 for r in recs if "text_en" in r["metadata"] and "text_zh" in r["metadata"])
        tokens = estimate_tokens(recs, TRANSLATIONS[code].lang)
        print(f"   {code}: {len(recs)} verses ({both} with a parallel text), ~{tokens:,} tokens "
              f"≈ ${tokens / 1e6 * 0.02:.3f} with {settings.embedding_model}")
        plan[code] = recs

    if args.dry_run:
        print("dry run — nothing sent")
        return 0

    from app.services.openai_client import embed_texts
    from app.services.pinecone_client import get_index

    index = get_index()

    def embed_and_upsert(code: str, batch: List[dict], attempts: int = 6) -> int:
        # One embedding request and one upsert; upserts are idempotent, so a
        # retry or a rerun simply overwrites. The OpenAI client already retries
        # 429s; this loop rides out short network / DNS blips (up to ~1 minute).
        for attempt in range(1, attempts + 1):
            try:
                embeddings = embed_texts([r["text"] for r in batch])
                index.upsert(
                    vectors=[{"id": r["id"], "values": emb, "metadata": r["metadata"]} for r, emb in zip(batch, embeddings)],
                    namespace=code,
                )
                return len(batch)
            except Exception as e:
                if attempt == attempts:
                    raise
                wait = 2 ** attempt
                print(f"   ⚠ batch starting {batch[0]['id']}: {type(e).__name__}; retry {attempt}/{attempts - 1} in {wait}s", flush=True)
                time.sleep(wait)
        return 0  # unreachable

    for code, recs in plan.items():
        if args.purge:
            try:
                index.delete(delete_all=True, namespace=code)
                print(f"🗑  purged namespace {code!r}", flush=True)
            except Exception as e:  # a namespace that never existed
                print(f"   (purge skipped: {type(e).__name__})", flush=True)
        batches = list(chunk(recs, BATCH_SIZE))
        print(f"\n📦 {code}: {len(batches)} batches of {BATCH_SIZE}, {args.concurrency} in flight", flush=True)
        started = time.perf_counter()
        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(embed_and_upsert, code, batch) for batch in batches]
            try:
                for fut in as_completed(futures):
                    fut.result()
                    done += 1
                    if done % 10 == 0 or done == len(batches):
                        elapsed = time.perf_counter() - started
                        print(f"   {progress(done, len(batches))}  {done}/{len(batches)}  {elapsed:5.0f}s  "
                              f"({done * BATCH_SIZE / elapsed:,.0f} verses/s)", flush=True)
            except Exception:
                pool.shutdown(cancel_futures=True)
                raise
    print("\n🎉 done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
