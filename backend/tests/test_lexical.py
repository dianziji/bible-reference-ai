import pytest

from app.services.lexical import BM25Index, build_index, dedupe, rrf, tokenize


def test_english_tokens_are_stemmed_and_stopworded():
    assert tokenize("What does the Bible say about forgiveness?", "en") == ["bibl", "say", "forgiv"]
    assert tokenize("He has forgiven; forgive them", "en") == ["forgiven", "forgiv"]
    assert tokenize("Did Jesus rise from the dead?", "en", stem=False) == ["jesus", "rise", "dead"]


def test_chinese_tokens_are_character_bigrams():
    assert tokenize("耶稣是谁？", "zh") == ["耶稣", "稣是", "是谁"]
    assert tokenize("爱", "zh") == ["爱"]
    assert tokenize("圣经，关于爱", "zh") == ["圣经", "关于", "于爱"]   # punctuation splits the runs


@pytest.fixture
def tiny():
    return BM25Index.build({
        "Ephesians.2.8": "For it is by grace you have been saved through faith, and this not from yourselves; it is the gift of God",
        "Galatians.1.3": "Grace and peace to you from God our Father and the Lord Jesus Christ",
        "Hebrews.11.1": "Now faith is the assurance of what we hope for and the certainty of what we do not see",
        "John.11.35": "Jesus wept",
    }, lang="en")


def test_bm25_ranks_the_verse_with_more_query_terms_first(tiny):
    got = [vid for vid, _ in tiny.search("saved by grace through faith", k=10)]
    assert got[0] == "Ephesians.2.8"
    assert set(got) == {"Ephesians.2.8", "Galatians.1.3", "Hebrews.11.1"}   # "Jesus wept" shares no term


def test_bm25_ignores_terms_absent_from_the_index(tiny):
    assert tiny.search("bible", k=5) == []
    assert tiny.search("", k=5) == []


def test_bm25_search_respects_k(tiny):
    assert len(tiny.search("grace faith God", k=2)) == 2


def test_rrf_rewards_agreement_between_sources():
    fused = rrf({"dense": ["a", "b", "c"], "bm25": ["c", "d", "a"]}, k=60)
    order = [vid for vid, _ in fused]
    assert order[:2] == ["a", "c"]           # both appear in both lists; "a" has the better rank sum
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_weights_and_tie_break():
    only_dense = rrf({"dense": ["x"], "bm25": ["y"]}, k=60, weights={"bm25": 0.0})
    assert only_dense[0] == ("x", pytest.approx(1 / 61))
    assert only_dense[1] == ("y", 0.0)
    tied = rrf({"a": ["q", "p"], "b": ["p", "q"]})
    assert [vid for vid, _ in tied] == ["p", "q"]  # equal scores -> id order


def test_dedupe_keeps_first_occurrence():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_build_index_over_the_vendored_translations():
    en = build_index("bsb")
    assert len(en) == 31086
    top = [vid for vid, _ in en.search("the Sabbath was made for man", k=5)]
    assert "Mark.2.27" in top
    zh = build_index("cuv")
    assert len(zh) == 31021
    assert "John.11.35" in [vid for vid, _ in zh.search("耶稣哭了", k=3)]
