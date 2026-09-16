# scripts/load_and_embed_kjv.py

import json
import os
from pathlib import Path
from typing import List, Dict

from app.services.openai_client import embed_texts
from app.services.pinecone_client import get_index


DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "bible-json" / "JSON"


def load_all_verses() -> List[Dict]:
    """
    从 bible-json/JSON 目录里读取所有章节，扁平化成单条 verse。
    返回的每一条 dict 格式：
    {
        "id": "Genesis.1.1",
        "book": "Genesis",
        "chapter": 1,
        "verse": 1,
        "text": "..."
    }
    """
    verses: List[Dict] = []

    if not DATA_ROOT.exists():
        raise RuntimeError(f"JSON data directory not found: {DATA_ROOT}")

    # 遍历每一本书
    for book_dir in sorted(DATA_ROOT.iterdir()):
        if not book_dir.is_dir():
            continue

        book_name = book_dir.name  # e.g. "Genesis"

        # 遍历该书的每一章
        for chapter_file in sorted(book_dir.glob("*.json")):
            with chapter_file.open("r", encoding="utf-8") as f:
                chapter_data = json.load(f)

            chapter_num = chapter_data.get("chapter")
            verses_list = chapter_data.get("verses", [])

            for v in verses_list:
                verse_num = v.get("verse")
                text = v.get("text", "").strip()

                if not text:
                    continue

                verse_id = f"{book_name}.{chapter_num}.{verse_num}"

                verses.append(
                    {
                        "id": verse_id,
                        "book": book_name,
                        "chapter": int(chapter_num),
                        "verse": int(verse_num),
                        "text": text,
                    }
                )

    return verses


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """简单的分批函数，把列表切成若干小块。"""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def print_progress(current, total, bar_len=24):
    """打印一个简单的进度条。"""
    filled_len = int(bar_len * current / total)
    bar = "█" * filled_len + "░" * (bar_len - filled_len)
    percent = (current / total) * 100
    return f"[{bar}]  {percent:4.1f}%"


def main():
    print("🔍 Loading all KJV verses from JSON...")
    verses = load_all_verses()
    total = len(verses)
    print(f"✅ Loaded {total} verses.")

    index = get_index()
    print("✅ Connected to Pinecone index")

    batch_size = 100
    batches = list(chunk_list(verses, batch_size))
    total_batches = len(batches)

    print(f"📦 Total batches: {total_batches} (batch_size={batch_size})\n")

    for i, batch in enumerate(batches, 1):
        texts = [v["text"] for v in batch]
        ids = [v["id"] for v in batch]
        metadatas = [
            {
                "book": v["book"],
                "chapter": v["chapter"],
                "verse": v["verse"],
                "text": v["text"],
            }
            for v in batch
        ]

        # Embedding
        embeddings = embed_texts(texts)

        # Prepare vectors
        vectors = [
            {"id": vid, "values": emb, "metadata": meta}
            for vid, emb, meta in zip(ids, embeddings, metadatas)
        ]

        # Upsert
        index.upsert(vectors=vectors)

        # Print progress
        bar = print_progress(i, total_batches)
        print(f"Batch {i}/{total_batches}  {bar}")

    print("\n🎉 All verses embedded and upserted into Pinecone!")

if __name__ == "__main__":
    main()