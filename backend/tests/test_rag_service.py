from langchain_core.documents import Document

from app.services.rag_service import RAGService, clean_text, verse_reference


def test_clean_text_strips_kjv_source_markup():
    assert clean_text("And the second <i>is</i> like unto it") == "And the second is like unto it"
    assert clean_text("¶ If thou turn away thy foot from the sabbath") == "If thou turn away thy foot from the sabbath"
    assert clean_text("no tags here") == "no tags here"


def test_verse_reference_in_both_languages():
    assert verse_reference({"book": "1 John", "chapter": 4, "verse": 8}) == "1 John 4:8"
    assert verse_reference({"book": "1 John", "chapter": 4, "verse": 8}, "zh") == "约翰一书 4:8"
    # Pinecone returns numeric metadata as floats; the reference must not read "4.0:8.0".
    assert verse_reference({"book": "1 John", "chapter": 4.0, "verse": 8.0}) == "1 John 4:8"


def test_to_verses_uses_index_id_cleans_text_and_carries_parallel_texts():
    docs = [
        (Document(id="Matthew.22.39", page_content="And the second <i>is</i> like unto it",
                  metadata={"book": "Matthew", "chapter": 22, "verse": 39, "usfm": "MAT", "translation": "bsb",
                            "text_en": "And the second <i>is</i> like unto it", "text_zh": "其次也相仿"}), 0.7123456),
        (Document(id=None, page_content="dup", metadata={"book": "Mark", "chapter": 12, "verse": 31}), 0.6),
        (Document(id=None, page_content="dup again", metadata={"book": "Mark", "chapter": 12, "verse": 31}), 0.5),
    ]
    verses = RAGService.to_verses(docs)
    assert verses[0] == {"id": "Matthew.22.39", "usfm": "MAT", "book": "Matthew", "book_zh": "马太福音",
                         "chapter": 22, "verse": 39, "text": "And the second is like unto it",
                         "text_en": "And the second is like unto it", "text_zh": "其次也相仿",
                         "translation": "bsb", "score": 0.7123}
    assert verses[1]["id"] == "Mark.12.31"       # built from metadata when the index id is missing
    assert verses[1]["text_en"] is None          # legacy vectors without parallel texts still work
    assert verses[2]["id"] == "Mark.12.31-2"     # duplicates stay unique for React keys
