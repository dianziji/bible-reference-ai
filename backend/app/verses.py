"""The vendored translations: what they are, and how to load and align them.

Both texts ship as ebible.org verse-per-line files (``GEN 1:1 In the
beginning …``). They share the Protestant versification closely but not
exactly — the 和合本 merges a few dozen verse pairs and keeps a handful the
BSB's critical text omits — so alignment is by verse id and a missing side is
simply absent, never guessed.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from app.books import book_from_code

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Translation:
    code: str    # Pinecone namespace and the `translation` value in API responses
    lang: str    # "en" | "zh"
    name: str
    file: Path
    source: str
    licence: str


TRANSLATIONS: Dict[str, Translation] = {
    "bsb": Translation(
        code="bsb", lang="en", name="Berean Standard Bible",
        file=DATA_DIR / "bsb" / "engbsb_vpl.txt",
        source="https://ebible.org/find/details.php?id=engbsb",
        licence="Public domain (released by BSB Publishing, 2023-04-30)",
    ),
    "cuv": Translation(
        code="cuv", lang="zh", name="和合本（神版）",
        file=DATA_DIR / "cuv" / "cmn-cu89s_vpl.txt",
        source="https://ebible.org/find/details.php?id=cmn-cu89s",
        licence="Public domain",
    ),
}

_LINE = re.compile(r"^([1-3A-Z]{3}) (\d+):(\d+)\s+(.*)$")
_ASCII_WS = re.compile(r"[ \t\r\n]+")
# ebible.org's 和合本 wraps translator-supplied words in [brackets]; the printed
# text has none, so they are dropped while the words are kept.
_SUPPLIED = re.compile(r"\[([^\[\]]*)\]")


def clean_source_text(text: str) -> str:
    """Normalise a verse as it comes from the file.

    Only ASCII whitespace is collapsed: the 神版 writes the divine name as
    "　神" with a full-width space (U+3000) and that must survive intact.
    """
    text = _SUPPLIED.sub(r"\1", text)
    return _ASCII_WS.sub(" ", text).strip()


@dataclass(frozen=True)
class Verse:
    id: str      # "John.3.16" — translation-independent
    usfm: str    # "JHN"
    book: str    # "John"
    chapter: int
    verse: int
    text: str


def parse_vpl(lines: Iterable[str]) -> List[Verse]:
    """Parse verse-per-line text. Books outside the 66 are skipped; a malformed line is an error."""
    verses: List[Verse] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        match = _LINE.match(line)
        if not match:
            raise ValueError(f"line {number} is not 'BOOK c:v text': {line[:60]!r}")
        code, chapter, verse, text = match.groups()
        book = book_from_code(code)
        if book is None:
            continue
        chapter_n, verse_n = int(chapter), int(verse)
        verses.append(Verse(
            id=f"{book.en}.{chapter_n}.{verse_n}", usfm=book.usfm, book=book.en,
            chapter=chapter_n, verse=verse_n, text=clean_source_text(text),
        ))
    return verses


def load_translation(code: str) -> Dict[str, Verse]:
    """All verses of one translation, keyed by verse id."""
    translation = TRANSLATIONS[code]
    with translation.file.open(encoding="utf-8") as f:
        return {v.id: v for v in parse_vpl(f)}


@dataclass(frozen=True)
class ParallelVerse:
    id: str
    usfm: str
    book: str
    chapter: int
    verse: int
    texts: Dict[str, str]  # lang -> text, only for the translations that have this verse


def align(by_code: Dict[str, Dict[str, Verse]]) -> Dict[str, ParallelVerse]:
    """Join translations by verse id, in canonical order."""
    merged: Dict[str, ParallelVerse] = {}
    for code, verses in by_code.items():
        lang = TRANSLATIONS[code].lang
        for vid, v in verses.items():
            existing = merged.get(vid)
            texts = dict(existing.texts) if existing else {}
            texts[lang] = v.text
            merged[vid] = ParallelVerse(vid, v.usfm, v.book, v.chapter, v.verse, texts)
    return dict(sorted(merged.items(), key=lambda kv: _canonical_key(kv[1])))


def load_parallel(codes: Sequence[str] = ("bsb", "cuv")) -> Dict[str, ParallelVerse]:
    return align({code: load_translation(code) for code in codes})


def _canonical_key(v: ParallelVerse):
    from app.books import BY_USFM, BOOKS  # local import keeps module import order simple
    return (BOOKS.index(BY_USFM[v.usfm]), v.chapter, v.verse)
