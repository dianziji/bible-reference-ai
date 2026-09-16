from app.services.logger import log_query

from typing import List
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.rag_service import get_rag_service
from app.services.openai_client import is_safe_content
from app.services.messages import MODERATION_REFUSED, INSUFFICIENT_CONTEXT

app = FastAPI()
logger = logging.getLogger("bible-backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://bible.local",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Verse(BaseModel):
    id: str
    book: str
    chapter: int
    verse: int
    text: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    verses: List[Verse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Moderation 检查
        if not is_safe_content(question):
            return QueryResponse(
                answer=MODERATION_REFUSED,
                verses=[],
            )

        # 使用 LangChain RAG 服务查询
        rag_service = get_rag_service()
        result = rag_service.query(question)
    except Exception as e:
        # 让你在本地调试时能直接看到具体报错原因（比如 OpenAI/Pinecone 配置问题）
        logger.exception("Failed to handle /api/query")
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {type(e).__name__}: {e}",
        )

    # 转换结果为 Verse 对象
    verses: List[Verse] = [
        Verse(
            id=v["id"],
            book=v["book"],
            chapter=v["chapter"],
            verse=v["verse"],
            text=v["text"],
        )
        for v in result["verses"]
    ]

    # 没有 verse 的情况
    if not verses:
        answer = (
            INSUFFICIENT_CONTEXT
        )
        result = QueryResponse(answer=answer, verses=[])
        # 也可以记录日志，这里依然写入
        log_query(question, result.answer, [])
        return result

    # 正常情况：记录日志再返回
    log_query(question, result["answer"], [v.model_dump() for v in verses])

    return QueryResponse(
        answer=result["answer"],
        verses=verses,
    )