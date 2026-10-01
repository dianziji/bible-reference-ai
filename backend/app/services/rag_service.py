"""Retrieval-augmented answering over the verse index.

The service exposes three explicit stages so the caller can cache, time and
parallelise them independently:

    embed(question)              -> vector                 OpenAI embeddings
    retrieve(vector, translation)-> [(Document, score)]    Pinecone namespace of that translation,
                                                           cosine, filtered by SCORE_THRESHOLD
    generate(question, docs, lang) -> answer               chat model, grounded on the retrieved
                                                           verses only, in the question's language

Retrieval filters on the raw Pinecone cosine score. LangChain's retriever
wrapper silently drops ``score_threshold`` for Pinecone (it is forwarded to
``index.query(**kwargs)`` and ignored), so the threshold is applied here.
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.books import book_name
from app.config import Settings, settings as default_settings, supports_temperature
from app.language import LANGUAGE_NAMES
from app.verses import TRANSLATIONS

load_dotenv()

logger = logging.getLogger("bible-backend.rag")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")

ScoredDoc = Tuple[Document, float]

SYSTEM_PROMPT = "You are a careful Bible study assistant."

# Bump app.config.PROMPT_VERSION when this wording changes.
QA_TEMPLATE = """You are a careful Bible study assistant.
You can ONLY answer based on the provided Bible verses.
If the context is insufficient, say you are not sure and encourage the user to read the Bible directly.
When you draw on a verse, cite it in parentheses using its reference exactly as given in the context, e.g. {citation_example}.
Answer in {language_name}, the language of the question.

Use the following pieces of context to answer the question. If you don't know the answer, say you are not sure and encourage the user to read the Bible directly.

{context}

Question: {question}

Answer:"""

_CITATION_EXAMPLE = {"en": "(John 3:16)", "zh": "（约翰福音 3:16）"}


def _content_to_text(content: Any) -> str:
    # langchain 1.x may return a list of content blocks instead of a string.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _as_int(value: Any) -> Any:
    # Pinecone hands numeric metadata back as floats (3.0); references must read "3:16".
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def verse_reference(metadata: Dict[str, Any], lang: str = "en") -> str:
    book = book_name(str(metadata.get("book", "")), lang)
    return f'{book} {_as_int(metadata.get("chapter", ""))}:{_as_int(metadata.get("verse", ""))}'


_SOURCE_MARKUP = re.compile(r"</?i>|¶\s*")


def clean_text(text: str) -> str:
    """Drop source typography some editions carry: <i>…</i> around supplied words and ¶ paragraph marks."""
    return _SOURCE_MARKUP.sub("", text).strip()


class RAGService:
    """Embed -> retrieve -> generate, each callable on its own."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg

        self.embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=OPENAI_API_KEY,
        )

        self.vectorstore = PineconeVectorStore(
            index_name=cfg.pinecone_index_name,
            embedding=self.embeddings,
            pinecone_api_key=PINECONE_API_KEY,
        )

        llm_kwargs: Dict[str, Any] = {"model": cfg.chat_model, "api_key": OPENAI_API_KEY}
        if supports_temperature(cfg.chat_model):
            llm_kwargs["temperature"] = 0
        if cfg.chat_reasoning_effort:
            llm_kwargs["reasoning_effort"] = cfg.chat_reasoning_effort
        self.llm = ChatOpenAI(**llm_kwargs)

    # ---- stages -------------------------------------------------------------

    def embed(self, question: str) -> List[float]:
        return self.embeddings.embed_query(question)

    def retrieve(self, vector: List[float], translation: str) -> List[ScoredDoc]:
        """Top-k verses of one translation by cosine similarity, keeping only those at or above
        that language's threshold."""
        lang = TRANSLATIONS[translation].lang if translation in TRANSLATIONS else "en"
        threshold = self.cfg.threshold_for(lang)
        scored = self.vectorstore.similarity_search_by_vector_with_score(
            vector, k=self.cfg.top_k, namespace=translation
        )
        return [(doc, float(score)) for doc, score in scored if float(score) >= threshold]

    def generate_message(self, question: str, docs: List[ScoredDoc], lang: str = "en") -> AIMessage:
        """Raw model reply; ``usage_metadata`` carries the token counts (see scripts/compare_models.py)."""
        context = "\n\n".join(
            f"[{verse_reference(doc.metadata, lang)}] {clean_text(doc.page_content)}" for doc, _ in docs
        )
        prompt_text = QA_TEMPLATE.format(
            context=context,
            question=question,
            language_name=LANGUAGE_NAMES.get(lang, "English"),
            citation_example=_CITATION_EXAMPLE.get(lang, _CITATION_EXAMPLE["en"]),
        )
        return self.llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt_text)])

    def generate(self, question: str, docs: List[ScoredDoc], lang: str = "en") -> str:
        return _content_to_text(self.generate_message(question, docs, lang).content).strip()

    # ---- output shaping -----------------------------------------------------

    @staticmethod
    def to_verses(docs: List[ScoredDoc], lang: str = "en") -> List[Dict[str, Any]]:
        """API shape: the verse in the question's language plus both parallel texts."""
        verses: List[Dict[str, Any]] = []
        seen_ids = set()
        for doc, score in docs:
            metadata = doc.metadata
            book = str(metadata.get("book", ""))
            chapter = int(metadata.get("chapter", 0))
            verse = int(metadata.get("verse", 0))

            verse_id = doc.id or (f"{book}.{chapter}.{verse}" if book and chapter and verse else "")
            if not verse_id:
                verse_id = f"verse-{len(verses)}"
            elif verse_id in seen_ids:
                verse_id = f"{verse_id}-{len(verses)}"
            seen_ids.add(verse_id)

            text_en = metadata.get("text_en")
            text_zh = metadata.get("text_zh")
            primary = metadata.get("text", doc.page_content)
            verses.append(
                {
                    "id": verse_id,
                    "usfm": metadata.get("usfm"),
                    "book": book,
                    "book_zh": book_name(book, "zh") if book else None,
                    "chapter": chapter,
                    "verse": verse,
                    "text": clean_text(primary),
                    "text_en": clean_text(text_en) if text_en else None,
                    "text_zh": clean_text(text_zh) if text_zh else None,
                    "translation": metadata.get("translation"),
                    "score": round(score, 4),
                }
            )
        return verses


# 全局实例
_rag_service = None


def get_rag_service() -> RAGService:
    """获取 RAG 服务单例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
