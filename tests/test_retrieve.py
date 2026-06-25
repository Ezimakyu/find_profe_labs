from scraper.config import Settings
from scraper.retrieve import (
    Chunk,
    EmbeddingRetriever,
    build_research_corpus,
    chunk_sources,
    chunk_text,
    lexical_rank,
)


def _settings(**overrides) -> Settings:
    base = dict(openai_api_key="dummy", embedding_chunk_chars=100, embedding_chunk_overlap=20)
    base.update(overrides)
    return Settings(**base)


def test_chunk_text_short_returns_single():
    assert chunk_text("hello world", 100, 20) == ["hello world"]


def test_chunk_text_splits_long_text():
    text = "\n".join(f"line {i} with some words" for i in range(80))
    chunks = chunk_text(text, 120, 20)
    assert len(chunks) > 1
    # Every chunk is within a sane bound of the window size.
    assert all(len(c) <= 160 for c in chunks)


def test_chunk_sources_tags_source_url():
    chunks = chunk_sources([("https://a", "x" * 250)], 100, 10)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.source_url for c in chunks} == {"https://a"}


def test_build_corpus_fallback_when_retrieval_disabled():
    settings = _settings(use_embedding_retrieval=False)
    corpus = build_research_corpus(settings, None, [("https://a", "alpha"), ("https://b", "beta")])
    assert "### SOURCE: https://a" in corpus
    assert "alpha" in corpus and "beta" in corpus


def test_retriever_rank_fallback_when_unavailable():
    # Force the model to be unavailable; rank should degrade to first top_k chunks.
    r = EmbeddingRetriever("nonexistent-model")
    r._failed = True
    chunks = [Chunk("u", f"c{i}") for i in range(10)]
    ranked = r.rank("query", chunks, top_k=3)
    assert ranked == chunks[:3]


def test_build_corpus_empty_sources():
    assert build_research_corpus(_settings(), None, []) == ""


def test_lexical_rank_prefers_research_dense_chunk():
    boilerplate = Chunk("u", "Home About Contact Login Menu Skip to main content footer")
    research = Chunk(
        "u",
        "My research studies learning algorithms and computer vision systems; "
        "recent projects design optimization methods for robot perception.",
    )
    ranked = lexical_rank([boilerplate, research], top_k=1)
    assert ranked == [research]


def test_lexical_rank_handles_empty_and_top_k():
    assert lexical_rank([], top_k=5) == []
    chunks = [Chunk("u", f"research project {i}") for i in range(5)]
    assert len(lexical_rank(chunks, top_k=3)) == 3


def test_build_corpus_lexical_default_keeps_relevant(monkeypatch):
    settings = _settings(use_embedding_retrieval=False, embedding_max_chars=10000)
    sources = [
        ("https://a", "Navigation home about contact. " * 5),
        ("https://b", "Research interests: machine learning, optimization, vision systems."),
    ]
    corpus = build_research_corpus(settings, None, sources)
    assert "Research interests" in corpus
