"""
使用 LangChain 进行嵌入和存储的脚本（可选）
这个脚本展示了如何使用 LangChain 的 PineconeVectorStore 来存储文档
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# 把 backend 根目录加进 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

DATA_PATH = Path(BACKEND_ROOT) / "data" / "bible_sample.json"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bible-verses")


def main():
    print("Using data file:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Bible data file not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        verses = json.load(f)

    print(f"Loaded {len(verses)} verses from JSON")

    # 初始化 embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 创建 LangChain Documents
    documents = []
    metadatas = []
    ids = []
    
    for verse in verses:
        # 创建文档文本（包含引用信息）
        text = f"[{verse['book']} {verse['chapter']}:{verse['verse']}] {verse['text']}"
        documents.append(Document(page_content=text))
        
        # 保存元数据
        metadatas.append({
            "id": verse["id"],
            "book": verse["book"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "text": verse["text"],
        })
        ids.append(verse["id"])

    print(f"Created {len(documents)} documents")

    # 使用 LangChain PineconeVectorStore 存储
    print("Storing documents in Pinecone using LangChain...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        ids=ids,
        metadatas=metadatas
    )

    print("✅ Documents stored successfully!")

    # 查看 stats（需要通过 Pinecone 客户端）
    from app.services.pinecone_client import get_index
    index = get_index()
    stats = index.describe_index_stats()
    print("Index stats:", stats)


if __name__ == "__main__":
    main()

