import os
from dotenv import load_dotenv
from openai import OpenAI

from app.config import settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

# Retries with backoff cover the occasional 429 when ingestion runs several batches at once.
client = OpenAI(api_key=OPENAI_API_KEY, max_retries=5)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量生成 embedding，返回二维 list，每个元素是一个向量
    """
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def generate_answer(question: str, context: str) -> str:
    """
    用 GPT 生成带经文上下文的回答
    """
    system_prompt = (
        "You are a careful Bible study assistant. "
        "You can ONLY answer based on the provided Bible verses. "
        "If the context is insufficient, say you are not sure and encourage the user to read the Bible directly."
    )

    user_content = f"Question:\n{question}\n\nRelevant Bible verses:\n{context}"

    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content.strip()


def is_safe_content(text: str) -> bool:
    """
    调 OpenAI Moderation 检查内容是否安全
    返回 True 表示安全，False 表示被 flag
    """
    resp = client.moderations.create(
        model=settings.moderation_model,
        input=text,
    )
    result = resp.results[0]
    return not result.flagged