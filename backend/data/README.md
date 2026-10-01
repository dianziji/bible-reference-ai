# Scripture data

Two public-domain translations, vendored as ebible.org verse-per-line text (`BOOK c:v text`, BibleWorks book abbreviations) together with the `_about.htm` each archive ships with. Nothing here is under a restrictive licence; nothing here may be swapped for a licensed text (NIV, 新标点和合本 …) without a licence that covers storing the full text and serving it through an AI system — see the root README.

| dir | translation | language | source | licence (verbatim from the archive) |
|---|---|---|---|---|
| `bsb/` | Berean Standard Bible | English | [ebible.org `engbsb`](https://ebible.org/find/details.php?id=engbsb), 2026-08-08 | "Berean Standard Bible — Public Domain — Contributor: BSB Publishing, LLC". Placed in the public domain by its publisher on 2023-04-30 ([berean.bible/licensing](https://berean.bible/licensing.htm)). |
| `cuv/` | 和合本（神版）, simplified | 中文 | [ebible.org `cmn-cu89s`](https://ebible.org/find/details.php?id=cmn-cu89s), 2021-10-15 | "新标点和合本 — Chinese Union Version (simplified) — Public Domain". The 1919 Union Version text; this file renders the divine name as "　神" throughout (0 occurrences of 上帝). |

The 和合本 file wraps translator-supplied words in `[brackets]` (2,651 verses, e.g. `又[造]众星`); `app/verses.py` drops the brackets and keeps the words, matching the printed text, and preserves the full-width space of "　神".

Counts: BSB 31,086 verses, 和合本 31,021. They follow the same versification except where the 和合本 merges neighbouring verses (72 ids present only in the BSB) and where the BSB's critical text omits a verse the 和合本 keeps (7 ids, e.g. Mark 9:44, Romans 16:24). `app/verses.py` aligns them by id and leaves a missing side empty.

Verse ids are `EnglishBook.chapter.verse` (`John.3.16`) in both translations; `app/books.py` maps them to USFM codes and 和合本 book names. `scripts/load_and_embed.py` embeds each translation into its own Pinecone namespace with both texts in the metadata.
