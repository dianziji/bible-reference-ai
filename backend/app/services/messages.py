"""User-facing fallback messages, bilingual (English / 中文).

Kept in one place so the API and the RAG service return identical wording.
"""

MODERATION_REFUSED = (
    "Sorry, this question isn't something I can discuss here. / "
    "抱歉，这个问题内容不适合在这里讨论。"
)

NO_VERSES_FOUND = (
    "I couldn't find relevant verses for this question. Try rephrasing it. / "
    "目前没有找到相关的经文，请尝试换一种提问方式。"
)

INSUFFICIENT_CONTEXT = (
    "I couldn't find enough relevant verses to answer this reliably. "
    "Try rephrasing the question, or read the related chapters directly. / "
    "目前没有找到足够相关的经文来可靠地回答这个问题，请尝试换一种提问方式或阅读相关章节。"
)
