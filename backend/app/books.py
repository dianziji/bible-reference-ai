"""The 66 books of the Protestant canon in three namings.

    usfm   three-letter code used by USFM / ebible.org / YouVersion links (GEN, JHN, ...)
    en     English name; also the first segment of a verse id ("Genesis.1.1")
    zh     和合本 book name (简体)

Verse ids are translation-independent — "John.3.16" is the same verse in the
BSB and the 和合本 — which is what lets the two texts sit side by side and lets
one golden set score retrieval in either language.
"""

from typing import Dict, List, NamedTuple, Optional


class Book(NamedTuple):
    usfm: str
    en: str
    zh: str


BOOKS: List[Book] = [
    Book("GEN", "Genesis", "创世记"),
    Book("EXO", "Exodus", "出埃及记"),
    Book("LEV", "Leviticus", "利未记"),
    Book("NUM", "Numbers", "民数记"),
    Book("DEU", "Deuteronomy", "申命记"),
    Book("JOS", "Joshua", "约书亚记"),
    Book("JDG", "Judges", "士师记"),
    Book("RUT", "Ruth", "路得记"),
    Book("1SA", "1 Samuel", "撒母耳记上"),
    Book("2SA", "2 Samuel", "撒母耳记下"),
    Book("1KI", "1 Kings", "列王纪上"),
    Book("2KI", "2 Kings", "列王纪下"),
    Book("1CH", "1 Chronicles", "历代志上"),
    Book("2CH", "2 Chronicles", "历代志下"),
    Book("EZR", "Ezra", "以斯拉记"),
    Book("NEH", "Nehemiah", "尼希米记"),
    Book("EST", "Esther", "以斯帖记"),
    Book("JOB", "Job", "约伯记"),
    Book("PSA", "Psalms", "诗篇"),
    Book("PRO", "Proverbs", "箴言"),
    Book("ECC", "Ecclesiastes", "传道书"),
    Book("SNG", "Song of Solomon", "雅歌"),
    Book("ISA", "Isaiah", "以赛亚书"),
    Book("JER", "Jeremiah", "耶利米书"),
    Book("LAM", "Lamentations", "耶利米哀歌"),
    Book("EZK", "Ezekiel", "以西结书"),
    Book("DAN", "Daniel", "但以理书"),
    Book("HOS", "Hosea", "何西阿书"),
    Book("JOL", "Joel", "约珥书"),
    Book("AMO", "Amos", "阿摩司书"),
    Book("OBA", "Obadiah", "俄巴底亚书"),
    Book("JON", "Jonah", "约拿书"),
    Book("MIC", "Micah", "弥迦书"),
    Book("NAM", "Nahum", "那鸿书"),
    Book("HAB", "Habakkuk", "哈巴谷书"),
    Book("ZEP", "Zephaniah", "西番雅书"),
    Book("HAG", "Haggai", "哈该书"),
    Book("ZEC", "Zechariah", "撒迦利亚书"),
    Book("MAL", "Malachi", "玛拉基书"),
    Book("MAT", "Matthew", "马太福音"),
    Book("MRK", "Mark", "马可福音"),
    Book("LUK", "Luke", "路加福音"),
    Book("JHN", "John", "约翰福音"),
    Book("ACT", "Acts", "使徒行传"),
    Book("ROM", "Romans", "罗马书"),
    Book("1CO", "1 Corinthians", "哥林多前书"),
    Book("2CO", "2 Corinthians", "哥林多后书"),
    Book("GAL", "Galatians", "加拉太书"),
    Book("EPH", "Ephesians", "以弗所书"),
    Book("PHP", "Philippians", "腓立比书"),
    Book("COL", "Colossians", "歌罗西书"),
    Book("1TH", "1 Thessalonians", "帖撒罗尼迦前书"),
    Book("2TH", "2 Thessalonians", "帖撒罗尼迦后书"),
    Book("1TI", "1 Timothy", "提摩太前书"),
    Book("2TI", "2 Timothy", "提摩太后书"),
    Book("TIT", "Titus", "提多书"),
    Book("PHM", "Philemon", "腓利门书"),
    Book("HEB", "Hebrews", "希伯来书"),
    Book("JAS", "James", "雅各书"),
    Book("1PE", "1 Peter", "彼得前书"),
    Book("2PE", "2 Peter", "彼得后书"),
    Book("1JN", "1 John", "约翰一书"),
    Book("2JN", "2 John", "约翰二书"),
    Book("3JN", "3 John", "约翰三书"),
    Book("JUD", "Jude", "犹大书"),
    Book("REV", "Revelation", "启示录"),
]

BY_USFM: Dict[str, Book] = {b.usfm: b for b in BOOKS}
BY_EN: Dict[str, Book] = {b.en: b for b in BOOKS}
BY_ZH: Dict[str, Book] = {b.zh: b for b in BOOKS}

# ebible.org's "_vpl.txt" files use BibleWorks abbreviations for these books.
BIBLEWORKS_ALIASES: Dict[str, str] = {
    "1JO": "1JN", "2JO": "2JN", "3JO": "3JN", "EZE": "EZK", "JAM": "JAS", "JOE": "JOL",
    "JOH": "JHN", "MAR": "MRK", "NAH": "NAM", "PHI": "PHP", "SOL": "SNG",
}


def book_from_code(code: str) -> Optional[Book]:
    """Look a book up by USFM code or BibleWorks alias; None for anything outside the 66."""
    code = code.upper()
    return BY_USFM.get(BIBLEWORKS_ALIASES.get(code, code))


def book_name(en_name: str, lang: str) -> str:
    """Book name for display in ``lang`` ("en" | "zh"); falls back to the English name."""
    book = BY_EN.get(en_name)
    if book is None or lang != "zh":
        return en_name
    return book.zh


def verse_id(en_name: str, chapter: int, verse: int) -> str:
    return f"{en_name}.{chapter}.{verse}"
