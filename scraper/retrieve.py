from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from scraper.config import Settings

# A query that pulls research-bearing text to the top of the ranking. Generic
# enough to work across personal homepages, lab pages, and CVs.
RESEARCH_QUERY = (
    "research interests, research directions, projects, methods and systems, "
    "publications and paper topics, problems studied, what this professor works on"
)

# Terms that signal research-bearing prose; used by the lexical ranker (the
# default, dependency-free relevance filter) to push the substantive chunks to
# the top without paying for a local embedding model.
_RESEARCH_TERMS = (
    "research", "project", "projects", "interest", "interests", "publication",
    "publications", "paper", "papers", "method", "methods", "system", "systems",
    "algorithm", "algorithms", "model", "models", "theory", "analysis", "design",
    "learning", "network", "networks", "data", "optimization", "vision", "language",
    "robot", "robotics", "hardware", "circuit", "circuits", "wireless", "signal",
    "control", "security", "privacy", "quantum", "sensor", "sensing", "work",
    "approach", "problem", "problems", "experiment", "thesis", "dissertation",
)
_TERM_SET = set(_RESEARCH_TERMS)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def lexical_rank(chunks: list["Chunk"], top_k: int) -> list["Chunk"]:
    """Rank chunks by research-term density (cheap, CPU-light, no model).

    A keyword score is a crude proxy for relevance, but on a 2-core box it is
    ~1000x cheaper than embedding every chunk and is enough to keep navigation
    chrome and boilerplate out of the LLM prompt.
    """
    if not chunks:
        return []
    scored: list[tuple[float, int, Chunk]] = []
    for i, chunk in enumerate(chunks):
        words = _WORD_RE.findall(chunk.text.lower())
        if not words:
            scored.append((0.0, i, chunk))
            continue
        hits = sum(1 for w in words if w in _TERM_SET)
        # Density (not raw count) so short, dense bios beat long boilerplate.
        score = hits / (len(words) ** 0.5)
        scored.append((score, i, chunk))
    # Sort by score desc, breaking ties by original order for determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]


@dataclass
class Chunk:
    source_url: str
    text: str


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping windows on paragraph-ish boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - max(0, overlap))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        # Prefer to break on a newline near the window end for cleaner chunks.
        if end < n:
            nl = text.rfind("\n", start + step // 2, end)
            if nl != -1 and nl > start:
                end = nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def chunk_sources(sources: list[tuple[str, str]], chunk_chars: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for url, text in sources:
        for piece in chunk_text(text, chunk_chars, overlap):
            out.append(Chunk(source_url=url, text=piece))
    return out


class EmbeddingRetriever:
    """Lazy wrapper around a local fastembed model (no API calls).

    The model is loaded once on first use. If fastembed is unavailable or the
    model fails to load, callers fall back to head-truncation so the pipeline
    keeps working without embeddings.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def available(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                from fastembed import TextEmbedding

                # onnxruntime's CPU memory arena grows with every embed call and
                # never releases, which OOMs the box across hundreds of
                # professors; disabling it keeps memory flat. threads=1 keeps a
                # single embed call from saturating both cores so concurrent
                # worker threads can still make progress.
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    threads=1,
                    enable_cpu_mem_arena=False,
                )
                return True
            except Exception:
                self._failed = True
                return False

    def _embed(self, texts: list[str]):
        import numpy as np

        vecs = list(self._model.embed(texts))  # type: ignore[union-attr]
        arr = np.asarray(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def rank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        if not self.available():
            return chunks[:top_k]
        try:
            import numpy as np

            # onnxruntime's InferenceSession.Run is safe to call concurrently on
            # one session, so we no longer hold a lock here; that lock made every
            # worker queue behind a single slow embed and capped throughput.
            doc_vecs = self._embed([c.text for c in chunks])
            q_vec = self._embed([query])[0]
            scores = doc_vecs @ q_vec
            order = np.argsort(-scores)[:top_k]
            return [chunks[i] for i in order]
        except Exception:
            return chunks[:top_k]


def build_research_corpus(
    settings: Settings,
    retriever: EmbeddingRetriever | None,
    sources: list[tuple[str, str]],
    query: str = RESEARCH_QUERY,
) -> str:
    """Chunk sources, keep only the most research-relevant chunks, and format.

    When retrieval is disabled/unavailable this still bounds total size via the
    character budget, preserving the original head-truncation behavior.
    """
    if not sources:
        return ""

    chunks = chunk_sources(sources, settings.embedding_chunk_chars, settings.embedding_chunk_overlap)
    if not chunks:
        return ""

    # Pick the relevance filter: local embeddings (semantic, but CPU-heavy —
    # best with a local LLM whose context is small) or the cheap lexical ranker
    # (default; keeps the cloud-LLM path fast on a low-core box).
    if settings.use_embedding_retrieval and retriever is not None:
        ranked = retriever.rank(query, chunks, settings.embedding_top_k * 2)
    else:
        ranked = lexical_rank(chunks, settings.embedding_top_k * 2)

    grouped: dict[str, list[str]] = {}
    total = 0
    for chunk in ranked:
        if total + len(chunk.text) > settings.embedding_max_chars:
            continue
        grouped.setdefault(chunk.source_url, []).append(chunk.text)
        total += len(chunk.text)

    if not grouped:
        per = max(2000, settings.embedding_max_chars // max(1, len(sources)))
        return "\n\n".join(f"### SOURCE: {url}\n{text[:per]}" for url, text in sources)

    blocks = [f"### SOURCE: {url}\n" + "\n…\n".join(parts) for url, parts in grouped.items()]
    return "\n\n".join(blocks)
