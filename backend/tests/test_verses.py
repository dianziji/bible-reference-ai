import pytest

from app.books import BOOKS, BY_USFM, book_from_code, book_name
from app.language import detect_language
from app.verses import TRANSLATIONS, align, load_translation, parse_vpl


def test_books_table_is_the_66_book_canon():
    assert len(BOOKS) == 66
    assert len({b.usfm for b in BOOKS}) == 66
    assert len({b.en for b in BOOKS}) == 66
    assert len({b.zh for b in BOOKS}) == 66
    assert BOOKS[0].en == "Genesis" and BOOKS[-1].en == "Revelation"
    assert BOOKS[38].en == "Malachi" and BOOKS[39].en == "Matthew"


def test_book_lookup_accepts_usfm_and_bibleworks_codes():
    assert book_from_code("JHN").en == "John"
    assert book_from_code("JOH").en == "John"          # BibleWorks alias used by ebible.org files
    assert book_from_code("1jo").zh == "约翰一书"
    assert book_from_code("SOL").en == "Song of Solomon"
    assert book_from_code("TOB") is None                # deuterocanon: not part of the index


def test_book_name_by_language():
    assert book_name("John", "zh") == "约翰福音"
    assert book_name("John", "en") == "John"
    assert book_name("Unknown Book", "zh") == "Unknown Book"


@pytest.mark.parametrize("text, lang", [
    ("What is love?", "en"),
    ("耶稣是谁？", "zh"),
    ("What does 爱 mean?", "zh"),   # any Han character routes to the Chinese translation
    ("   ", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_parse_vpl_normalises_and_skips_unknown_books():
    lines = [
        "GEN 1:1 In the beginning God created the heavens and the earth.",
        "",
        "JOH 3:16   For God so loved   the world",   # BibleWorks code + stray whitespace
        "TOB 1:1 skipped, not in the canon",
        "GEN 1:16 于是　神造了两个大光，又[造]众星，",  # 神版 full-width space + supplied-word brackets
    ]
    verses = parse_vpl(lines)
    assert [v.id for v in verses] == ["Genesis.1.1", "John.3.16", "Genesis.1.16"]
    assert verses[1].usfm == "JHN"
    assert verses[1].text == "For God so loved the world"
    assert verses[2].text == "于是　神造了两个大光，又造众星，"   # U+3000 kept, brackets gone, words kept
    with pytest.raises(ValueError):
        parse_vpl(["not a verse line"])


def test_align_keeps_missing_sides_empty_and_orders_canonically():
    en = {v.id: v for v in parse_vpl(["ROM 16:23 Gaius", "MAR 9:43 hand", "GEN 1:1 beginning"])}
    zh = {v.id: v for v in parse_vpl(["ROM 16:24 恩惠", "ROM 16:23 该犹", "GEN 1:1 起初"])}
    merged = align({"bsb": en, "cuv": zh})
    assert list(merged) == ["Genesis.1.1", "Mark.9.43", "Romans.16.23", "Romans.16.24"]
    assert merged["Genesis.1.1"].texts == {"en": "beginning", "zh": "起初"}
    assert merged["Mark.9.43"].texts == {"en": "hand"}
    assert merged["Romans.16.24"].texts == {"zh": "恩惠"}


def test_vendored_files_load_and_are_the_expected_editions():
    bsb = load_translation("bsb")
    cuv = load_translation("cuv")
    assert len(bsb) == 31086 and len(cuv) == 31021
    assert bsb["John.3.16"].text.startswith("For God so loved the world")
    assert "　神爱世人，甚至将他的独生子赐给他们" in cuv["John.3.16"].text   # 神版 spacing kept, [brackets] dropped
    assert cuv["Genesis.1.1"].text == "起初，　神创造天地。"
    assert not any("上帝" in v.text for v in cuv.values())
    assert not any("[" in v.text or "]" in v.text for v in cuv.values())
    assert {v.usfm for v in bsb.values()} == set(BY_USFM)
    for t in TRANSLATIONS.values():
        assert t.file.exists() and "Public domain" in t.licence
