import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 加载 .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bible-verses")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_index(dimension: int = 1536):
    # 如果 index 不存在，则创建
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT,
            ),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def get_index():
    return pc.Index(PINECONE_INDEX_NAME)