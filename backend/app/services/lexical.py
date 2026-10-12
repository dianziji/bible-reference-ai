"""Lexical (BM25) retrieval over the vendored verses, and rank fusion.

The Pinecone index is dense-only (cosine, no sparse values), so a hybrid
search cannot be pushed into a single Pinecone query. Both translations are a
few megabytes of text, so the lexical side is an in-process inverted index
built from the vendored files at startup (~1 s) and queried in well under a
millisecond; it needs no network and no extra service.

    build_index("bsb")              -> BM25Index over the English verses
    index.search("rise from the dead", k=20)   -> [(verse_id, score)]
    rrf({"dense": [...ids...], "bm25": [...ids...]}) -> fused [(verse_id, score)]

Tokenisation is per language: English is lower-cased, stop-worded and Snowball-
stemmed ("forgiveness" / "forgiven" / "forgive" -> "forgiv"); Chinese is split into
character bigrams, the usual segmenter-free baseline for CJK text.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import snowballstemmer

from app.verses import TRANSLATIONS, load_translation

_EN_TOKEN = re.compile(r"[a-z]+")
_HAN_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

# Function words with no retrieval value; "bible" is kept out of the query path
# instead (it does not occur in the text, so BM25 simply ignores it).
EN_STOPWORDS = frozenset("""
a an the and or but of to in on at by for with from as is are was were be been being am
do does did have has had will would shall should can could may might must this that these
those it its he she they them his her their we us our you your i me my what which who whom
whose why how when where there here then than so if not no nor into onto unto about
""".split())

_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str, lang: str, stem: bool = True) -> List[str]:
    """Query and document tokens for one language.

    English: lower-case words, stop words dropped, Snowball stems (optional).
    Chinese: character bigrams within each run of Han characters; a lone
    character stays as a unigram so one-character queries still match.
    """
    if lang == "zh":
        tokens: List[str] = []
        for run in _HAN_RUN.findall(text):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        return tokens
    words = [w for w in _EN_TOKEN.findall(text.lower().replace("’", "'")) if w not in EN_STOPWORDS]
    return _stemmer.stemWords(words) if stem else words


@dataclass
class BM25Index:
    """Okapi BM25 over short documents (verses), with an inverted index.

    k1 and b are the textbook defaults; verses are short and fairly uniform in
    length, so b matters little here.
    """

    lang: str
    ids: List[str]
    postings: Dict[str, List[Tuple[int, int]]]   # term -> [(doc_no, term_freq)]
    doc_len: List[int]
    avg_len: float
    stem: bool = True
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: Mapping[str, str], lang: str, stem: bool = True) -> "BM25Index":
        ids = list(docs)
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_len: List[int] = []
        for no, vid in enumerate(ids):
            tokens = tokenize(docs[vid], lang, stem)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings[term].append((no, tf))
        avg_len = (sum(doc_len) / len(doc_len)) if doc_len else 0.0
        return cls(lang=lang, ids=ids, postings=dict(postings), doc_len=doc_len, avg_len=avg_len, stem=stem)

    def __len__(self) -> int:
        return len(self.ids)

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        if n == 0:
            return 0.0
        return math.log(1 + (len(self.ids) - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Top-k (verse_id, BM25 score); an empty list when no query term occurs in the index."""
        scores: Dict[int, float] = defaultdict(float)
        for term, qtf in Counter(tokenize(query, self.lang, self.stem)).items():
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for no, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[no] / self.avg_len)
                scores[no] += idf * (tf * (self.k1 + 1)) / (tf + norm)
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[no], score) for no, score in top]


def build_index(translation: str, stem: bool = True) -> BM25Index:
    """BM25 over every verse of one vendored translation (``"bsb"`` | ``"cuv"``)."""
    verses = load_translation(translation)
    return BM25Index.build({vid: v.text for vid, v in verses.items()}, TRANSLATIONS[translation].lang, stem)


def rrf(rankings: Mapping[str, Sequence[str]], k: int = 60,
        weights: Optional[Mapping[str, float]] = None) -> List[Tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = Σ_source w_s / (k + rank_s(d)), rank starting at 1.

    Takes ordered id lists per source (scores from different retrievers are not
    comparable, ranks are). k = 60 is the value from the original paper; it
    damps the advantage of being first in any one list. Ties break on id so the
    result is deterministic.
    """
    weights = weights or {}
    fused: Dict[str, float] = defaultdict(float)
    for source, ids in rankings.items():
        w = weights.get(source, 1.0)
        for rank, vid in enumerate(ids, start=1):
            fused[vid] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def dedupe(ids: Iterable[str]) -> List[str]:
    """Keep the first occurrence of each id, preserving order."""
    seen = set()
    out = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out
