import os
import sys
import json
from pathlib import Path

# 把 backend 根目录加进 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.services.openai_client import embed_texts
from app.services.pinecone_client import get_index


DATA_PATH = Path(BACKEND_ROOT) / "data" / "bible_sample.json"


def main():
    print("Using data file:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Bible data file not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        verses = json.load(f)

    print(f"Loaded {len(verses)} verses from JSON")
    texts = [v["text"] for v in verses]

    # 1) 生成 embeddings
    print("Generating embeddings with OpenAI...")
    embeddings = embed_texts(texts)
    print(f"Got {len(embeddings)} embeddings")

    # 2) 写入 Pinecone
    index = get_index()

    vectors = []
    for verse, emb in zip(verses, embeddings):
        vectors.append(
            {
                "id": verse["id"],
                "values": emb,
                "metadata": {
                    "book": verse["book"],
                    "chapter": verse["chapter"],
                    "verse": verse["verse"],
                    "text": verse["text"],
                },
            }
        )

    print("Upserting to Pinecone...")
    upsert_response = index.upsert(vectors=vectors)
    print("Upsert response:", upsert_response)

    # 3) 再查看一下 stats
    stats = index.describe_index_stats()
    print("Index stats after upsert:", stats)


if __name__ == "__main__":
    main()